"""Blind Pairwise Prompting & Structured JSON Rubric Schema for Track B (`src/track_b/prompts.py`).

Presents Patch A and Patch B anonymously as `Implementation 1` and `Implementation 2`
(in randomized or bidirectional swapped orders to eliminate position bias) and requires
open-weights CodeLLMs (`Qwen2.5-Coder`, `DeepSeek-Coder`, `Llama-3`) to output a structured
JSON verdict across 4 software engineering dimensions:
  1. `maintainability_reasoning`: Readability and modular structure.
  2. `complexity_awareness`: Identification of nested control flow.
  3. `duplication_detection`: Recognition of DRY violations.
  4. `future_change_impact`: Analysis of coupling and modification fragility.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Tuple


RUBRIC_DIMENSIONS: Tuple[str, ...] = (
    "maintainability_reasoning",
    "complexity_awareness",
    "duplication_detection",
    "future_change_impact",
)


SYSTEM_PROMPT = """You are a Staff Principal Software Engineer conducting a blind technical debt and maintainability review of two Java implementations (`Implementation 1` and `Implementation 2`).
Both implementations pass 100% of functional unit tests for the exact same specification.
Your task is to distinguish functional correctness from engineering quality and select the more maintainable, lower-debt implementation.

You MUST evaluate both implementations across exactly four engineering dimensions:
1. `maintainability_reasoning`: Readability, modular helper decomposition, and naming clarity.
2. `complexity_awareness`: Identification of deep control-flow nesting, cyclomatic branching, and cognitive load.
3. `duplication_detection`: Recognition of DRY violations, copy-paste code clones, and redundant expressions.
4. `future_change_impact`: Analysis of method coupling, parameter/state bloat, live variable span, and modification fragility.

Respond ONLY with a valid JSON object matching this exact schema:
{
  "preferred_implementation": "Implementation 1" | "Implementation 2",
  "confidence": 0.0 to 1.0,
  "dimensions": {
    "maintainability_reasoning": {
      "winner": "Implementation 1" | "Implementation 2",
      "detected_issue": true | false,
      "explanation": "Concise analysis of modular structure and readability."
    },
    "complexity_awareness": {
      "winner": "Implementation 1" | "Implementation 2",
      "detected_issue": true | false,
      "explanation": "Concise analysis of control-flow nesting and branching."
    },
    "duplication_detection": {
      "winner": "Implementation 1" | "Implementation 2",
      "detected_issue": true | false,
      "explanation": "Concise analysis of DRY violations and structural clones."
    },
    "future_change_impact": {
      "winner": "Implementation 1" | "Implementation 2",
      "detected_issue": true | false,
      "explanation": "Concise analysis of coupling, state span, and change fragility."
    }
  }
}"""


@dataclass(frozen=True)
class BlindPairPrompt:
    """Represents a single blind pairwise prompt instance with ground-truth slot mapping."""

    pair_id: int
    is_swapped: bool
    clean_slot_label: str  # "Implementation 1" if not is_swapped else "Implementation 2"
    impl_1_code: str
    impl_2_code: str
    system_prompt: str
    user_prompt: str


class BlindPairwisePromptBuilder:
    """Constructs position-unbiased and bidirectional blind pairwise prompts for CodeLLMs."""

    @staticmethod
    def build_prompt(
        pair_id: int,
        code_a_norm: str,
        code_b_norm: str,
        is_swapped: bool = False,
    ) -> BlindPairPrompt:
        """Build a blind prompt where `is_swapped=False` puts Patch A in Slot 1, and `True` in Slot 2."""
        if not is_swapped:
            impl_1, impl_2 = code_a_norm, code_b_norm
            clean_label = "Implementation 1"
        else:
            impl_1, impl_2 = code_b_norm, code_a_norm
            clean_label = "Implementation 2"

        user_prompt = (
            "Evaluate the following two functionally equivalent Java implementations "
            "that both pass 100% of unit tests, and return your structured JSON verdict.\n\n"
            "### Implementation 1\n"
            f"```java\n{impl_1}\n```\n\n"
            "### Implementation 2\n"
            f"```java\n{impl_2}\n```\n"
        )

        return BlindPairPrompt(
            pair_id=int(pair_id),
            is_swapped=bool(is_swapped),
            clean_slot_label=clean_label,
            impl_1_code=impl_1,
            impl_2_code=impl_2,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    @classmethod
    def build_bidirectional_pair(
        cls,
        pair_id: int,
        code_a_norm: str,
        code_b_norm: str,
    ) -> Tuple[BlindPairPrompt, BlindPairPrompt]:
        """Return `(forward_prompt, swapped_prompt)` to test position bias and swap consistency."""
        fwd = cls.build_prompt(pair_id, code_a_norm, code_b_norm, is_swapped=False)
        rev = cls.build_prompt(pair_id, code_a_norm, code_b_norm, is_swapped=True)
        return fwd, rev

    @staticmethod
    def parse_json_verdict(raw_response: str) -> Dict[str, Any]:
        """Robustly extract and validate the JSON rubric verdict from an LLM response."""
        text = raw_response.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        data = json.loads(text)

        pref = data.get("preferred_implementation", "Implementation 1")
        if pref not in ("Implementation 1", "Implementation 2"):
            pref = "Implementation 1" if "1" in str(pref) else "Implementation 2"
        data["preferred_implementation"] = pref
        data["confidence"] = float(data.get("confidence", 0.5))

        dims = data.get("dimensions", {})
        normalized_dims: Dict[str, Dict[str, Any]] = {}
        for dim_name in RUBRIC_DIMENSIONS:
            d_obj = dims.get(dim_name, {})
            winner = d_obj.get("winner", pref)
            if winner not in ("Implementation 1", "Implementation 2"):
                winner = "Implementation 1" if "1" in str(winner) else "Implementation 2"
            normalized_dims[dim_name] = {
                "winner": winner,
                "detected_issue": bool(d_obj.get("detected_issue", False)),
                "explanation": str(d_obj.get("explanation", "")),
            }
        data["dimensions"] = normalized_dims
        return data
