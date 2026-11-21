"""Unit and integration tests for Phase 4: Evaluation Metrics, Ablation Studies & Table Export."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from scripts.export_tables import generate_latex_tables, generate_markdown_report
from src.evaluation.ablations import AblationSuiteRunner
from src.evaluation.metrics import (
    bootstrap_accuracy_ci,
    compute_macro_f1_from_binary,
    compute_quantitative_metrics,
)
from tests.test_phase3 import _make_synthetic_dataset


def test_quantitative_metrics_and_bootstrap_ci() -> None:
    probs = np.array([0.9, 0.8, 0.7, 0.85, 0.95, 0.6, 0.4, 0.92], dtype=np.float32)
    scores_a = np.array([2.0, 1.8, 1.5, 2.2, 2.5, 1.2, 0.8, 2.1], dtype=np.float32)
    scores_b = np.array([-1.0, -1.2, -0.8, -1.5, -2.0, -0.5, 1.0, -1.8], dtype=np.float32)
    tdi_a = np.array([5.0, 6.0, 8.0, 4.0, 3.0, 9.0, 12.0, 5.0], dtype=np.float32)
    tdi_b = np.array([45.0, 50.0, 40.0, 60.0, 75.0, 35.0, 10.0, 55.0], dtype=np.float32)
    lm_mask = np.array([True, True, False, True, False, True, False, True], dtype=bool)

    res = compute_quantitative_metrics(
        model_name="TestModel",
        split_name="test_split",
        probs_a_over_b=probs,
        scores_a=scores_a,
        scores_b=scores_b,
        tdi_a=tdi_a,
        tdi_b=tdi_b,
        len_matched_mask=lm_mask,
    )
    assert res.pairwise_preference_accuracy == 0.875
    assert res.length_matched_accuracy == 1.0
    assert res.ci_95_low <= res.pairwise_preference_accuracy <= res.ci_95_high
    assert res.spearman_rho_vs_sonar_tdi > 0.70
    assert 0.0 <= res.macro_f1 <= 1.0


def test_modality_and_objective_ablation_runner_and_tables(tmp_path: Path) -> None:
    ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr = _make_synthetic_dataset(32)
    ca_te, sa_te, ta_te, cb_te, sb_te, tb_te = _make_synthetic_dataset(16)
    lm_te = np.ones(16, dtype=bool)

    runner = AblationSuiteRunner(seed=42)
    mod_res = runner.run_modality_ablation(
        ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
    )
    assert "Static_Metrics_Only_64D" in mod_res
    assert "GraphCodeBERT_Only_768D" in mod_res
    assert "Full_Multimodal_Fusion_832D_to_256D" in mod_res

    obj_res = runner.run_objective_ablation(
        ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
    )
    assert "Tabular_XGBoost_Delta" in obj_res
    assert "Pointwise_Regression_MSE" in obj_res
    assert "Bradley_Terry_Pairwise_Loss" in obj_res
    assert "DQN_Multi_Objective_RL" in obj_res

    mock_report = {
        "ablation_1_modality_impact": mod_res,
        "ablation_2_objective_comparison": obj_res,
        "track_b_generative_evaluation": {},
    }
    md_text = generate_markdown_report(mock_report)
    tex_text = generate_latex_tables(mock_report)
    assert "Table 1: Ablation Study 1" in md_text
    assert "Table 2: Ablation Study 2" in md_text
    assert "\\begin{table*}" in tex_text
