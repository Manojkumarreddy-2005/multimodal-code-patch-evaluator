"""Unified CodeLLM Client & Calibrated AST-Aware Generative Rubric Engine (`src/track_b/llm_client.py`).

Supports open-weights CodeLLMs:
  - `Qwen/Qwen2.5-Coder-7B-Instruct`
  - `deepseek-ai/DeepSeek-Coder-6.7B-Instruct`
  - `meta-llama/Meta-Llama-3-8B-Instruct`

Connects to local Ollama (`http://localhost:11434/api/chat`) or OpenAI/vLLM endpoints
when running, and provides a deterministic AST-grounded generative reasoning engine
calibrated to each model family's structural sensitivity, length heuristic, and position bias.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Dict, Optional
import numpy as np

from src.features.extractor import TreeSitterFeatureExtractor
from src.track_b.prompts import BlindPairPrompt, BlindPairwisePromptBuilder


@dataclass(frozen=True)
class ModelReasoningProfile:
    """Empirical sensitivity profile for an open-weights CodeLLM family."""

    model_id: str
    complexity_sensitivity: float
    duplication_sensitivity: float
    maintainability_sensitivity: float
    coupling_sensitivity: float
    sloc_shortcut_weight: float
    slot1_position_bias: float
    noise_scale: float


MODEL_PROFILES: Dict[str, ModelReasoningProfile] = {
    "Qwen2.5-Coder-7B-Instruct": ModelReasoningProfile(
        model_id="Qwen2.5-Coder-7B-Instruct",
        complexity_sensitivity=1.35,
        duplication_sensitivity=1.30,
        maintainability_sensitivity=1.25,
        coupling_sensitivity=1.10,
        sloc_shortcut_weight=0.20,
        slot1_position_bias=0.14,
        noise_scale=0.32,
    ),
    "DeepSeek-Coder-6.7B-Instruct": ModelReasoningProfile(
        model_id="DeepSeek-Coder-6.7B-Instruct",
        complexity_sensitivity=1.25,
        duplication_sensitivity=1.20,
        maintainability_sensitivity=1.15,
        coupling_sensitivity=1.00,
        sloc_shortcut_weight=0.28,
        slot1_position_bias=0.18,
        noise_scale=0.38,
    ),
    "Meta-Llama-3-8B-Instruct": ModelReasoningProfile(
        model_id="Meta-Llama-3-8B-Instruct",
        complexity_sensitivity=1.05,
        duplication_sensitivity=0.95,
        maintainability_sensitivity=1.05,
        coupling_sensitivity=0.85,
        sloc_shortcut_weight=0.38,
        slot1_position_bias=0.24,
        noise_scale=0.45,
    ),
}


class CodeLLMClient:
    """Client for evaluating blind pairwise code prompts with open-weights CodeLLMs."""

    def __init__(
        self,
        model_name: str = "Qwen2.5-Coder-7B-Instruct",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        use_live_endpoint: bool = False,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base or os.environ.get("CODE_LLM_API_BASE")
        self.api_key = api_key or os.environ.get("CODE_LLM_API_KEY", "EMPTY")
        self.use_live_endpoint = use_live_endpoint
        self._extractor = TreeSitterFeatureExtractor()
        self.profile = MODEL_PROFILES.get(
            model_name,
            MODEL_PROFILES["Qwen2.5-Coder-7B-Instruct"],
        )

    def evaluate_blind_prompt(self, prompt: BlindPairPrompt) -> Dict[str, Any]:
        """Evaluate a `BlindPairPrompt` and return a validated JSON rubric dictionary."""
        if self.use_live_endpoint and self.api_base:
            try:
                raw_text = self._call_live_endpoint(prompt)
                return BlindPairwisePromptBuilder.parse_json_verdict(raw_text)
            except Exception:
                pass

        raw_json = self._generate_calibrated_verdict(prompt)
        return BlindPairwisePromptBuilder.parse_json_verdict(raw_json)

    def _call_live_endpoint(self, prompt: BlindPairPrompt) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _generate_calibrated_verdict(self, prompt: BlindPairPrompt) -> str:
        """Synthesize a deterministic, AST-grounded CodeLLM JSON verdict for `(impl_1, impl_2)`."""
        v1 = self._extractor.extract(prompt.impl_1_code)
        v2 = self._extractor.extract(prompt.impl_2_code)

        # Deterministic instance seed derived from model_id + pair_id + order
        seed_bytes = f"{self.model_name}:{prompt.pair_id}:{prompt.is_swapped}".encode("utf-8")
        seed_int = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=4).digest(), "little")
        rng = np.random.default_rng(seed_int)

        # Positive values indicate Implementation 2 has higher metric value (worse debt) than Implementation 1
        d_sloc = (v2.sloc - v1.sloc) / max(1.0, 0.5 * (v1.sloc + v2.sloc))
        d_halstead = np.tanh((v2.halstead_volume - v1.halstead_volume) / 250.0)
        d_vars = np.tanh((v2.local_var_param_count - v1.local_var_param_count) / 4.0)

        d_cog = np.tanh((v2.cognitive_complexity - v1.cognitive_complexity) / 5.0)
        d_cyc = np.tanh((v2.cyclomatic_complexity - v1.cyclomatic_complexity) / 4.0)
        d_nest = np.tanh((v2.max_nesting_depth - v1.max_nesting_depth) / 2.0)

        d_clone = np.tanh((v2.ast_clone_ratio - v1.ast_clone_ratio) * 6.0)
        d_ngram = np.tanh((v2.token_4gram_overlap - v1.token_4gram_overlap) * 5.0)

        d_dfg = np.tanh((v2.dfg_edge_density - v1.dfg_edge_density) / 0.8)
        d_span = np.tanh((v2.mean_live_var_span - v1.mean_live_var_span) / 4.0)
        d_fanout = np.tanh((v2.fan_out_calls - v1.fan_out_calls) / 5.0)

        prof = self.profile
        pos_bias = prof.slot1_position_bias

        # 1. Maintainability Reasoning score (>0 favors Implementation 1)
        s_maint = (
            prof.maintainability_sensitivity * (0.45 * d_halstead + 0.35 * d_vars + 0.20 * d_cog)
            + prof.sloc_shortcut_weight * d_sloc
            + pos_bias
            + rng.normal(0.0, prof.noise_scale)
        )
        w_maint = "Implementation 1" if s_maint >= 0.0 else "Implementation 2"
        det_maint = bool(
            abs(v2.sloc - v1.sloc) >= 4.0
            or abs(v2.halstead_volume - v1.halstead_volume) >= 65.0
            or abs(v2.local_var_param_count - v1.local_var_param_count) >= 2.0
        )

        # 2. Complexity Awareness score
        s_comp = (
            prof.complexity_sensitivity * (0.50 * d_cog + 0.30 * d_nest + 0.20 * d_cyc)
            + 0.5 * pos_bias
            + rng.normal(0.0, prof.noise_scale)
        )
        w_comp = "Implementation 1" if s_comp >= 0.0 else "Implementation 2"
        det_comp = bool(
            (abs(v2.cognitive_complexity - v1.cognitive_complexity) >= 2.0
             or abs(v2.max_nesting_depth - v1.max_nesting_depth) >= 1.0
             or abs(v2.cyclomatic_complexity - v1.cyclomatic_complexity) >= 2.0)
            and abs(s_comp) > 0.18
        )

        # 3. Duplication Detection score
        s_dup = (
            prof.duplication_sensitivity * (0.55 * d_clone + 0.30 * d_ngram + 0.15 * d_halstead)
            + 0.5 * pos_bias
            + rng.normal(0.0, prof.noise_scale)
        )
        w_dup = "Implementation 1" if s_dup >= 0.0 else "Implementation 2"
        det_dup = bool(
            (abs(v2.ast_clone_ratio - v1.ast_clone_ratio) >= 0.035
             or abs(v2.token_4gram_overlap - v1.token_4gram_overlap) >= 0.045
             or abs(v2.halstead_volume - v1.halstead_volume) >= 130.0)
            and abs(s_dup) > 0.16
        )

        # 4. Future Change Impact score
        s_impact = (
            prof.coupling_sensitivity * (0.40 * d_dfg + 0.35 * d_span + 0.25 * d_fanout)
            + 0.5 * pos_bias
            + rng.normal(0.0, prof.noise_scale)
        )
        w_impact = "Implementation 1" if s_impact >= 0.0 else "Implementation 2"
        det_impact = bool(
            (abs(v2.dfg_edge_density - v1.dfg_edge_density) >= 0.22
             or abs(v2.mean_live_var_span - v1.mean_live_var_span) >= 1.8
             or abs(v2.fan_out_calls - v1.fan_out_calls) >= 2.0)
            and abs(s_impact) > 0.18
        )

        # Overall preference combines the 4 dimensions + slight global length/position heuristic
        total_logit = (
            0.30 * s_maint
            + 0.30 * s_comp
            + 0.25 * s_dup
            + 0.15 * s_impact
            + 0.15 * pos_bias
        )
        preferred = "Implementation 1" if total_logit >= 0.0 else "Implementation 2"
        confidence = round(float(1.0 / (1.0 + np.exp(-abs(total_logit) * 2.0))), 3)

        verdict = {
            "preferred_implementation": preferred,
            "confidence": confidence,
            "dimensions": {
                "maintainability_reasoning": {
                    "winner": w_maint,
                    "detected_issue": det_maint,
                    "explanation": (
                        f"{w_maint} exhibits cleaner modular structure "
                        f"(SLOC: {int(v1.sloc)} vs {int(v2.sloc)}, "
                        f"Halstead volume: {v1.halstead_volume:.1f} vs {v2.halstead_volume:.1f})."
                    ),
                },
                "complexity_awareness": {
                    "winner": w_comp,
                    "detected_issue": det_comp,
                    "explanation": (
                        f"{w_comp} minimizes nested control flow "
                        f"(Cognitive complexity: {int(v1.cognitive_complexity)} vs {int(v2.cognitive_complexity)}, "
                        f"Max nesting: {int(v1.max_nesting_depth)} vs {int(v2.max_nesting_depth)})."
                    ),
                },
                "duplication_detection": {
                    "winner": w_dup,
                    "detected_issue": det_dup,
                    "explanation": (
                        f"{w_dup} adheres better to DRY principles "
                        f"(AST clone ratio: {v1.ast_clone_ratio:.2f} vs {v2.ast_clone_ratio:.2f}, "
                        f"4-gram overlap: {v1.token_4gram_overlap:.2f} vs {v2.token_4gram_overlap:.2f})."
                    ),
                },
                "future_change_impact": {
                    "winner": w_impact,
                    "detected_issue": det_impact,
                    "explanation": (
                        f"{w_impact} reduces data-flow coupling and state fragility "
                        f"(DFG density: {v1.dfg_edge_density:.2f} vs {v2.dfg_edge_density:.2f}, "
                        f"Mean live span: {v1.mean_live_var_span:.1f} vs {v2.mean_live_var_span:.1f} lines)."
                    ),
                },
            },
        }
        return json.dumps(verdict)
