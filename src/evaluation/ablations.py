"""Phase 4 Ablation Studies: Modality Impact & Objective Comparison (`src/evaluation/ablations.py`).

Executes:
  1. Ablation Study 1 (Modality Impact):
     - Static Metrics Only (12-D -> 64-D)
     - GraphCodeBERT Only (768-D)
     - Full Multimodal Fusion (768-D + 64-D -> 256-D)
  2. Ablation Study 2 (Objective Comparison):
     - Tabular XGBoost on Static Metric Deltas (Δm)
     - Pointwise Regression Loss (MSE)
     - Bradley-Terry Pairwise Preference Loss
     - DQN Multi-Objective RL (Gymnasium `PatchRankingEnv`)
  3. Track B Generative CodeLLM Comparison:
     - Qwen2.5-Coder-7B-Instruct vs DeepSeek-Coder-6.7B-Instruct vs Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from src.evaluation.metrics import compute_quantitative_metrics
from src.features.extractor import FEATURE_NAMES
from src.track_a.baselines import (
    BradleyTerryPreferenceModel,
    PointwiseRegressionModel,
    XGBoostDeltaClassifier,
)
from src.track_a.dqn_agent import DQNConfig, MultiObjectiveDQNAgent
from src.track_b.llm_client import CodeLLMClient
from src.track_b.reasoner_eval import TrackBReasoningEvaluator


class AblationSuiteRunner:
    """Runs Phase 4 Modality Impact and Objective Comparison ablation experiments."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_modality_ablation(
        self,
        ca_tr: np.ndarray,
        sa_tr: np.ndarray,
        ta_tr: np.ndarray,
        cb_tr: np.ndarray,
        sb_tr: np.ndarray,
        tb_tr: np.ndarray,
        ca_te: np.ndarray,
        sa_te: np.ndarray,
        ta_te: np.ndarray,
        cb_te: np.ndarray,
        sb_te: np.ndarray,
        tb_te: np.ndarray,
        lm_te: np.ndarray,
        ca_oo: np.ndarray,
        sa_oo: np.ndarray,
        ta_oo: np.ndarray,
        cb_oo: np.ndarray,
        sb_oo: np.ndarray,
        tb_oo: np.ndarray,
        lm_oo: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare Static Metrics Only vs GraphCodeBERT Only vs Full Multimodal Fusion."""
        modality_configs = [
            ("Static_Metrics_Only_64D", "static_only"),
            ("GraphCodeBERT_Only_768D", "code_only"),
            ("Full_Multimodal_Fusion_832D_to_256D", "multimodal"),
        ]
        results: Dict[str, Any] = {}

        for label, mode in modality_configs:
            # Train Bradley-Terry Pairwise & Pointwise ensemble under the target modality mask
            bt = BradleyTerryPreferenceModel(
                modality_mode=mode, epochs=40, lr=2e-3, weight_decay=3e-4, seed=self.seed
            ).fit(ca_tr, sa_tr, cb_tr, sb_tr)
            pw = PointwiseRegressionModel(
                modality_mode=mode, epochs=35, lr=2e-3, weight_decay=2e-4, seed=self.seed
            ).fit(ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr)

            # Combine pairwise preference calibration with pointwise debt grounding
            probs_te = 0.55 * bt.predict_preference_proba(ca_te, sa_te, cb_te, sb_te) + 0.45 * pw.predict_preference_proba(ca_te, sa_te, cb_te, sb_te)
            scores_a_te = 0.5 * bt.score_snippet(ca_te, sa_te) + 0.5 * pw.score_snippet(ca_te, sa_te)
            scores_b_te = 0.5 * bt.score_snippet(cb_te, sb_te) + 0.5 * pw.score_snippet(cb_te, sb_te)

            probs_oo = 0.55 * bt.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo) + 0.45 * pw.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo)
            scores_a_oo = 0.5 * bt.score_snippet(ca_oo, sa_oo) + 0.5 * pw.score_snippet(ca_oo, sa_oo)
            scores_b_oo = 0.5 * bt.score_snippet(cb_oo, sb_oo) + 0.5 * pw.score_snippet(cb_oo, sb_oo)

            results[label] = {
                "modality_mode": mode,
                "in_domain_codenet_test": compute_quantitative_metrics(
                    label, "codenet_test", probs_te, scores_a_te, scores_b_te, ta_te, tb_te, lm_te
                ).to_dict(),
                "ood_swe_refactor_test": compute_quantitative_metrics(
                    label, "swe_refactor_ood", probs_oo, scores_a_oo, scores_b_oo, ta_oo, tb_oo, lm_oo
                ).to_dict(),
            }

        return results

    def run_objective_ablation(
        self,
        ca_tr: np.ndarray,
        sa_tr: np.ndarray,
        ta_tr: np.ndarray,
        cb_tr: np.ndarray,
        sb_tr: np.ndarray,
        tb_tr: np.ndarray,
        ca_te: np.ndarray,
        sa_te: np.ndarray,
        ta_te: np.ndarray,
        cb_te: np.ndarray,
        sb_te: np.ndarray,
        tb_te: np.ndarray,
        lm_te: np.ndarray,
        ca_oo: np.ndarray,
        sa_oo: np.ndarray,
        ta_oo: np.ndarray,
        cb_oo: np.ndarray,
        sb_oo: np.ndarray,
        tb_oo: np.ndarray,
        lm_oo: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare Tabular XGBoost vs Pointwise Regression vs Bradley-Terry Pairwise vs Multi-Objective DQN."""
        results: Dict[str, Any] = {}

        # 1. Tabular XGBoost on Δm
        xgb = XGBoostDeltaClassifier(seed=self.seed).fit(sa_tr, sb_tr, ta_tr, tb_tr)
        results["Tabular_XGBoost_Delta"] = {
            "objective_type": "Tree Boosted LogLoss on Δm",
            "feature_importances": xgb.feature_importances(FEATURE_NAMES),
            "in_domain_codenet_test": compute_quantitative_metrics(
                "Tabular_XGBoost_Delta",
                "codenet_test",
                xgb.predict_preference_proba(sa_te, sb_te),
                xgb.score_snippet(sa_te),
                xgb.score_snippet(sb_te),
                ta_te,
                tb_te,
                lm_te,
            ).to_dict(),
            "ood_swe_refactor_test": compute_quantitative_metrics(
                "Tabular_XGBoost_Delta",
                "swe_refactor_ood",
                xgb.predict_preference_proba(sa_oo, sb_oo),
                xgb.score_snippet(sa_oo),
                xgb.score_snippet(sb_oo),
                ta_oo,
                tb_oo,
                lm_oo,
            ).to_dict(),
        }

        # 2. Pointwise Regression Loss (MSE)
        pw = PointwiseRegressionModel(
            modality_mode="multimodal", epochs=35, lr=2e-3, seed=self.seed
        ).fit(ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr)
        results["Pointwise_Regression_MSE"] = {
            "objective_type": "Pointwise Continuous Debt MSE",
            "in_domain_codenet_test": compute_quantitative_metrics(
                "Pointwise_Regression_MSE",
                "codenet_test",
                pw.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
                pw.score_snippet(ca_te, sa_te),
                pw.score_snippet(cb_te, sb_te),
                ta_te,
                tb_te,
                lm_te,
            ).to_dict(),
            "ood_swe_refactor_test": compute_quantitative_metrics(
                "Pointwise_Regression_MSE",
                "swe_refactor_ood",
                pw.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
                pw.score_snippet(ca_oo, sa_oo),
                pw.score_snippet(cb_oo, sb_oo),
                ta_oo,
                tb_oo,
                lm_oo,
            ).to_dict(),
        }

        # 3. Bradley-Terry Pairwise Loss
        bt = BradleyTerryPreferenceModel(
            modality_mode="multimodal", epochs=40, lr=2e-3, weight_decay=3e-4, seed=self.seed
        ).fit(ca_tr, sa_tr, cb_tr, sb_tr)
        results["Bradley_Terry_Pairwise_Loss"] = {
            "objective_type": "Siamese Bradley-Terry -log σ(s_A - s_B)",
            "in_domain_codenet_test": compute_quantitative_metrics(
                "Bradley_Terry_Pairwise_Loss",
                "codenet_test",
                bt.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
                bt.score_snippet(ca_te, sa_te),
                bt.score_snippet(cb_te, sb_te),
                ta_te,
                tb_te,
                lm_te,
            ).to_dict(),
            "ood_swe_refactor_test": compute_quantitative_metrics(
                "Bradley_Terry_Pairwise_Loss",
                "swe_refactor_ood",
                bt.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
                bt.score_snippet(ca_oo, sa_oo),
                bt.score_snippet(cb_oo, sb_oo),
                ta_oo,
                tb_oo,
                lm_oo,
            ).to_dict(),
        }

        # 4. Multi-Objective Dueling Double-DQN RL
        dqn = MultiObjectiveDQNAgent(
            DQNConfig(modality_mode="multimodal", episodes=50, episode_horizon=48, seed=self.seed)
        ).fit(ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr)
        results["DQN_Multi_Objective_RL"] = {
            "objective_type": "Dueling Double-DQN Multi-Objective Reward (Pass + Debt + Complexity + Clone)",
            "final_episode_win_rate": dqn.episode_win_rates_[-1] if dqn.episode_win_rates_ else 0.0,
            "in_domain_codenet_test": compute_quantitative_metrics(
                "DQN_Multi_Objective_RL",
                "codenet_test",
                dqn.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
                dqn.score_snippet(ca_te, sa_te),
                dqn.score_snippet(cb_te, sb_te),
                ta_te,
                tb_te,
                lm_te,
            ).to_dict(),
            "ood_swe_refactor_test": compute_quantitative_metrics(
                "DQN_Multi_Objective_RL",
                "swe_refactor_ood",
                dqn.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
                dqn.score_snippet(ca_oo, sa_oo),
                dqn.score_snippet(cb_oo, sb_oo),
                ta_oo,
                tb_oo,
                lm_oo,
            ).to_dict(),
        }

        return results

    @staticmethod
    def run_track_b_evaluation(
        test_pairs: List[Dict[str, Any]],
        ood_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate all 3 open-weights CodeLLMs on both In-Domain and OOD splits."""
        models = [
            "Qwen2.5-Coder-7B-Instruct",
            "DeepSeek-Coder-6.7B-Instruct",
            "Meta-Llama-3-8B-Instruct",
        ]
        out: Dict[str, Any] = {}
        for m_name in models:
            client = CodeLLMClient(model_name=m_name)
            ev = TrackBReasoningEvaluator(client)
            out[m_name] = {
                "in_domain_codenet_test": ev.evaluate_pairs(test_pairs),
                "ood_swe_refactor_test": ev.evaluate_pairs(ood_pairs),
            }
        return out
