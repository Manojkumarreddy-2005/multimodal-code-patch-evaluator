"""Track B Evaluation Engine: Bidirectional Prompting, Rubric Consistency & 4-Axis Recall (`src/track_b/reasoner_eval.py`)."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from src.track_b.llm_client import CodeLLMClient
from src.track_b.prompts import BlindPairwisePromptBuilder, RUBRIC_DIMENSIONS


class TrackBReasoningEvaluator:
    """Evaluates open-weights CodeLLMs on blind pairwise engineering debt reasoning."""

    def __init__(self, llm_client: CodeLLMClient) -> None:
        self.client = llm_client

    def evaluate_pairs(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run bidirectional blind evaluation on `pairs` and compute quantitative & qualitative metrics."""
        if not pairs:
            return {}

        n_pairs = len(pairs)
        randomized_correct = 0
        strict_swap_consistent_correct = 0
        position_biased_flips = 0
        selection_reasoning_consistent = 0

        # Track predictions for Macro-F1 (class 0 = Implementation 1 wins, class 1 = Implementation 2 wins)
        y_true_slot: List[int] = []
        y_pred_slot: List[int] = []

        # Track 4-dimension recall against ground-truth Tree-Sitter triggers
        dim_triggered_counts = {dim: 0 for dim in RUBRIC_DIMENSIONS}
        dim_recalled_counts = {dim: 0 for dim in RUBRIC_DIMENSIONS}

        for idx, pair in enumerate(pairs):
            pid = int(pair["pair_id"])
            code_a = pair["code_a_norm"]
            code_b = pair["code_b_norm"]
            gt_triggers = pair.get("triggered_dimensions", {})

            fwd_prompt, rev_prompt = BlindPairwisePromptBuilder.build_bidirectional_pair(
                pair_id=pid,
                code_a_norm=code_a,
                code_b_norm=code_b,
            )

            verdict_fwd = self.client.evaluate_blind_prompt(fwd_prompt)
            verdict_rev = self.client.evaluate_blind_prompt(rev_prompt)

            fwd_clean_win = verdict_fwd["preferred_implementation"] == fwd_prompt.clean_slot_label
            rev_clean_win = verdict_rev["preferred_implementation"] == rev_prompt.clean_slot_label

            if fwd_clean_win and rev_clean_win:
                strict_swap_consistent_correct += 1
            if fwd_clean_win != rev_clean_win:
                position_biased_flips += 1

            # Deterministic 50/50 randomized slot assignment for standard blind evaluation
            use_swapped = (pid % 2 == 1)
            active_prompt = rev_prompt if use_swapped else fwd_prompt
            active_verdict = verdict_rev if use_swapped else verdict_fwd

            chose_clean = (
                active_verdict["preferred_implementation"] == active_prompt.clean_slot_label
            )
            if chose_clean:
                randomized_correct += 1

            true_slot = 1 if use_swapped else 0
            pred_slot = 0 if active_verdict["preferred_implementation"] == "Implementation 1" else 1
            y_true_slot.append(true_slot)
            y_pred_slot.append(pred_slot)

            # Check Selection-Reasoning Consistency: does overall choice agree with the 4 rubric dimensions?
            dims_dict = active_verdict["dimensions"]
            pref_label = active_verdict["preferred_implementation"]
            agreeing_dims = sum(
                1 for d in RUBRIC_DIMENSIONS if dims_dict[d]["winner"] == pref_label
            )
            if agreeing_dims >= 2:
                selection_reasoning_consistent += 1

            # Check Dimension-Wise Recall on ground-truth triggered axes
            for dim_name in RUBRIC_DIMENSIONS:
                if bool(gt_triggers.get(dim_name, False)):
                    dim_triggered_counts[dim_name] += 1
                    dim_out = dims_dict[dim_name]
                    if (
                        dim_out["detected_issue"] is True
                        and dim_out["winner"] == active_prompt.clean_slot_label
                    ):
                        dim_recalled_counts[dim_name] += 1

        macro_f1 = self._compute_macro_f1(y_true_slot, y_pred_slot)
        dimension_recall = {
            dim: round(
                dim_recalled_counts[dim] / max(1, dim_triggered_counts[dim]), 4
            )
            for dim in RUBRIC_DIMENSIONS
        }
        mean_dim_recall = round(float(np.mean(list(dimension_recall.values()))), 4)

        return {
            "model_name": self.client.model_name,
            "num_pairs_evaluated": n_pairs,
            "pairwise_preference_accuracy": round(randomized_correct / n_pairs, 4),
            "macro_f1": round(macro_f1, 4),
            "strict_swap_consistent_accuracy": round(
                strict_swap_consistent_correct / n_pairs, 4
            ),
            "position_bias_rate": round(position_biased_flips / n_pairs, 4),
            "selection_reasoning_consistency_rate": round(
                selection_reasoning_consistent / n_pairs, 4
            ),
            "dimension_wise_recall": dimension_recall,
            "mean_dimension_recall": mean_dim_recall,
        }

    @staticmethod
    def _compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
        y_t = np.asarray(y_true, dtype=np.int32)
        y_p = np.asarray(y_pred, dtype=np.int32)
        f1s: List[float] = []
        for cls in (0, 1):
            tp = int(np.sum((y_t == cls) & (y_p == cls)))
            fp = int(np.sum((y_t != cls) & (y_p == cls)))
            fn = int(np.sum((y_t == cls) & (y_p != cls)))
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1s))
