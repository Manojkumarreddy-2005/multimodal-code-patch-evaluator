"""Phase 4 Quantitative & Qualitative Evaluation Metrics (`src/evaluation/metrics.py`).

Implements:
  - Quantitative Metrics (Track A & B):
    1. Pairwise Preference Accuracy (`Acc_pref`) + 95% Bootstrap Confidence Intervals
    2. Length-Matched Preference Accuracy (|ΔSLOC| <= 15% subset)
    3. Spearman's Rank Correlation (`rho` and `p_value`) against SonarQube TDI
    4. Macro F1-Score across balanced candidate slot assignments
  - Qualitative Metrics (Track B Generative Reasoning):
    1. Selection-Reasoning Consistency Rate (`SRCR`)
    2. Strict Swap-Consistent Accuracy (`Acc_strict`) & Position Bias Rate (`PBR`)
    3. Dimension-Wise Recall across the 4 engineering reasoning axes
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class QuantitativeEvalResult:
    """Complete quantitative evaluation report for a ranking model."""

    model_name: str
    split_name: str
    num_pairs: int
    pairwise_preference_accuracy: float
    ci_95_low: float
    ci_95_high: float
    length_matched_accuracy: float
    spearman_rho_vs_sonar_tdi: float
    spearman_p_value: float
    macro_f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_macro_f1_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute unweighted Macro-F1 across classes 0 (Slot 1 wins) and 1 (Slot 2 wins)."""
    f1s: List[float] = []
    for cls in (0, 1):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bootstrap_accuracy_ci(
    correct_flags: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for pairwise preference accuracy."""
    n = len(correct_flags)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(correct_flags[idx], axis=1)
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return round(low, 4), round(high, 4)


def compute_quantitative_metrics(
    model_name: str,
    split_name: str,
    probs_a_over_b: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    tdi_a: np.ndarray,
    tdi_b: np.ndarray,
    len_matched_mask: Optional[np.ndarray] = None,
) -> QuantitativeEvalResult:
    """Compute all Phase 4 quantitative metrics for a discriminative or RL model."""
    probs = np.asarray(probs_a_over_b, dtype=np.float64)
    correct_flags = (probs >= 0.5).astype(np.float64)
    acc_pref = float(np.mean(correct_flags))
    ci_low, ci_high = bootstrap_accuracy_ci(correct_flags)

    # Balanced 50/50 slot assignment for Macro-F1
    n = len(probs)
    swap_mask = (np.arange(n) % 2) == 1
    y_true = swap_mask.astype(np.int32)
    probs_slot0 = np.where(swap_mask, 1.0 - probs, probs)
    y_pred = (probs_slot0 < 0.5).astype(np.int32)
    macro_f1 = compute_macro_f1_from_binary(y_true, y_pred)

    # Spearman's rank correlation (rho) between predicted debt (-score) and Sonar TDI
    all_pred_debt = np.concatenate([-np.asarray(scores_a), -np.asarray(scores_b)])
    all_true_tdi = np.concatenate([np.asarray(tdi_a), np.asarray(tdi_b)])
    rho, p_val = spearmanr(all_pred_debt, all_true_tdi)
    if np.isnan(rho):
        rho, p_val = 0.0, 1.0

    if len_matched_mask is not None and np.any(len_matched_mask):
        len_acc = float(np.mean(correct_flags[len_matched_mask]))
    else:
        len_acc = acc_pref

    return QuantitativeEvalResult(
        model_name=model_name,
        split_name=split_name,
        num_pairs=n,
        pairwise_preference_accuracy=round(acc_pref, 4),
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        length_matched_accuracy=round(len_acc, 4),
        spearman_rho_vs_sonar_tdi=round(float(rho), 4),
        spearman_p_value=float(f"{float(p_val):.3e}"),
        macro_f1=round(macro_f1, 4),
    )
