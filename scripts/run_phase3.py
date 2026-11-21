"""CLI entrypoint for Phase 3: Dual-Track System Architecture (Track A & Track B).

Trains and evaluates:
  Track A (Discriminative Quantitative Ranking & RL):
    1. Tabular XGBoost Classifier on static metric deltas (Δm)
    2. Multimodal Pointwise Regression Model
    3. Multimodal Bradley-Terry Pairwise Preference Model
    4. Gymnasium `PatchRankingEnv` + Multi-Objective Dueling Double-DQN Agent
  Track B (Generative CodeLLM Engineering Reasoning):
    1. Qwen2.5-Coder-7B-Instruct
    2. DeepSeek-Coder-6.7B-Instruct
    3. Meta-Llama-3-8B-Instruct

Usage:
    python -m scripts.run_phase3 --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.schema import PatchDatabase
from src.features.extractor import FEATURE_NAMES, StaticFeatureVector
from src.track_a.baselines import (
    BradleyTerryPreferenceModel,
    PointwiseRegressionModel,
    XGBoostDeltaClassifier,
)
from src.track_a.dqn_agent import DQNConfig, MultiObjectiveDQNAgent
from src.track_a.encoder import CodeDFGEmbedder
from src.track_b.llm_client import CodeLLMClient
from src.track_b.reasoner_eval import TrackBReasoningEvaluator


def prepare_split_tensors(
    pairs: List[Dict[str, Any]],
    embedder: CodeDFGEmbedder,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of joined database pair dicts into numpy arrays for Track A."""
    code_a_list: List[np.ndarray] = []
    code_b_list: List[np.ndarray] = []
    stat_a_list: List[np.ndarray] = []
    stat_b_list: List[np.ndarray] = []
    tdi_a_list: List[float] = []
    tdi_b_list: List[float] = []
    len_matched_list: List[bool] = []

    for p in pairs:
        ca = embedder.embed_code(p["code_a_norm"], snippet_id=p["patch_a_id"])
        cb = embedder.embed_code(p["code_b_norm"], snippet_id=p["patch_b_id"])
        va = StaticFeatureVector.from_dict(p["metrics_a"]).to_numpy()
        vb = StaticFeatureVector.from_dict(p["metrics_b"]).to_numpy()

        code_a_list.append(ca)
        code_b_list.append(cb)
        stat_a_list.append(va)
        stat_b_list.append(vb)
        tdi_a_list.append(float(p["tdi_a"]))
        tdi_b_list.append(float(p["tdi_b"]))
        len_matched_list.append(bool(p["is_length_matched"]))

    return (
        np.vstack(code_a_list).astype(np.float32),
        np.vstack(stat_a_list).astype(np.float32),
        np.array(tdi_a_list, dtype=np.float32),
        np.vstack(code_b_list).astype(np.float32),
        np.vstack(stat_b_list).astype(np.float32),
        np.array(tdi_b_list, dtype=np.float32),
        np.array(len_matched_list, dtype=bool),
    )


def evaluate_track_a_model(
    probs_a_over_b: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    tdi_a: np.ndarray,
    tdi_b: np.ndarray,
    len_matched_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute Acc_pref, Macro-F1, Spearman rho against Sonar TDI, and Length-Matched Acc."""
    acc_pref = float(np.mean(probs_a_over_b >= 0.5))

    # Evaluate on balanced 50/50 randomized slot assignment for Macro-F1
    n = len(probs_a_over_b)
    swap_mask = (np.arange(n) % 2) == 1
    y_true = swap_mask.astype(np.int32)
    # Predicted slot 0 if P(slot0 > slot1) >= 0.5 else slot 1
    probs_slot0 = np.where(swap_mask, 1.0 - probs_a_over_b, probs_a_over_b)
    y_pred = (probs_slot0 < 0.5).astype(np.int32)

    f1s = []
    for cls in (0, 1):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(f1s))

    # Spearman's rank correlation (rho) between predicted technical debt (-score) and Sonar TDI
    all_pred_debt = np.concatenate([-scores_a, -scores_b])
    all_true_tdi = np.concatenate([tdi_a, tdi_b])
    rho, _ = spearmanr(all_pred_debt, all_true_tdi)
    if np.isnan(rho):
        rho = 0.0

    len_acc = (
        float(np.mean(probs_a_over_b[len_matched_mask] >= 0.5))
        if np.any(len_matched_mask)
        else acc_pref
    )

    return {
        "pairwise_preference_accuracy": round(acc_pref, 4),
        "length_matched_accuracy": round(len_acc, 4),
        "spearman_rho_vs_sonar_tdi": round(float(rho), 4),
        "macro_f1": round(macro_f1, 4),
    }


def run_phase3_pipeline(config_path: str | Path = "configs/default_config.yaml") -> Dict[str, Any]:
    """Execute Phase 3 Track A training and Track B generative reasoning evaluation."""
    t0 = time.perf_counter()
    cfg_file = Path(config_path)
    config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    db_path = config.get("database", {}).get("sqlite_path", "data/patch_pairs.db")
    seed = int(config.get("project", {}).get("seed", 42))

    embedder = CodeDFGEmbedder(cache_path="data/code_embeddings_768.npz")

    with PatchDatabase(db_path) as db:
        conn = db.connect()
        snip_rows = conn.execute(
            "SELECT snippet_id, normalized_code FROM snippets"
        ).fetchall()
        embedder.embed_batch([(r["snippet_id"], r["normalized_code"]) for r in snip_rows])

        train_pairs = db.fetch_pairs(split="train")
        val_pairs = db.fetch_pairs(split="val")
        test_pairs = db.fetch_pairs(split="test")
        ood_pairs = db.fetch_pairs(split="ood_test")

    ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr, lm_tr = prepare_split_tensors(train_pairs, embedder)
    ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te = prepare_split_tensors(test_pairs, embedder)
    ca_oo, sa_oo, ta_oo, cb_oo, sb_oo, tb_oo, lm_oo = prepare_split_tensors(ood_pairs, embedder)

    # -------------------------------------------------------------------------
    # Track A1: Tabular XGBoost on Static Metric Deltas (Δm)
    # -------------------------------------------------------------------------
    xgb_model = XGBoostDeltaClassifier(seed=seed).fit(sa_tr, sb_tr, ta_tr, tb_tr)
    xgb_test = evaluate_track_a_model(
        xgb_model.predict_preference_proba(sa_te, sb_te),
        xgb_model.score_snippet(sa_te),
        xgb_model.score_snippet(sb_te),
        ta_te,
        tb_te,
        lm_te,
    )
    xgb_ood = evaluate_track_a_model(
        xgb_model.predict_preference_proba(sa_oo, sb_oo),
        xgb_model.score_snippet(sa_oo),
        xgb_model.score_snippet(sb_oo),
        ta_oo,
        tb_oo,
        lm_oo,
    )

    # -------------------------------------------------------------------------
    # Track A2: Multimodal Pointwise Regression
    # -------------------------------------------------------------------------
    pw_model = PointwiseRegressionModel(modality_mode="multimodal", seed=seed).fit(
        ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr
    )
    pw_test = evaluate_track_a_model(
        pw_model.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
        pw_model.score_snippet(ca_te, sa_te),
        pw_model.score_snippet(cb_te, sb_te),
        ta_te,
        tb_te,
        lm_te,
    )
    pw_ood = evaluate_track_a_model(
        pw_model.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
        pw_model.score_snippet(ca_oo, sa_oo),
        pw_model.score_snippet(cb_oo, sb_oo),
        ta_oo,
        tb_oo,
        lm_oo,
    )

    # -------------------------------------------------------------------------
    # Track A3: Multimodal Bradley-Terry Pairwise Preference Model
    # -------------------------------------------------------------------------
    bt_model = BradleyTerryPreferenceModel(modality_mode="multimodal", seed=seed).fit(
        ca_tr, sa_tr, cb_tr, sb_tr
    )
    bt_test = evaluate_track_a_model(
        bt_model.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
        bt_model.score_snippet(ca_te, sa_te),
        bt_model.score_snippet(cb_te, sb_te),
        ta_te,
        tb_te,
        lm_te,
    )
    bt_ood = evaluate_track_a_model(
        bt_model.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
        bt_model.score_snippet(ca_oo, sa_oo),
        bt_model.score_snippet(cb_oo, sb_oo),
        ta_oo,
        tb_oo,
        lm_oo,
    )

    # -------------------------------------------------------------------------
    # Track A4: Gymnasium & Multi-Objective Dueling Double-DQN Agent
    # -------------------------------------------------------------------------
    dqn_agent = MultiObjectiveDQNAgent(
        DQNConfig(modality_mode="multimodal", seed=seed)
    ).fit(ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr)
    dqn_test = evaluate_track_a_model(
        dqn_agent.predict_preference_proba(ca_te, sa_te, cb_te, sb_te),
        dqn_agent.score_snippet(ca_te, sa_te),
        dqn_agent.score_snippet(cb_te, sb_te),
        ta_te,
        tb_te,
        lm_te,
    )
    dqn_ood = evaluate_track_a_model(
        dqn_agent.predict_preference_proba(ca_oo, sa_oo, cb_oo, sb_oo),
        dqn_agent.score_snippet(ca_oo, sa_oo),
        dqn_agent.score_snippet(cb_oo, sb_oo),
        ta_oo,
        tb_oo,
        lm_oo,
    )

    # -------------------------------------------------------------------------
    # Track B: Generative CodeLLM Engineering Reasoning (3 Open-Weights Models)
    # -------------------------------------------------------------------------
    llm_models = [
        "Qwen2.5-Coder-7B-Instruct",
        "DeepSeek-Coder-6.7B-Instruct",
        "Meta-Llama-3-8B-Instruct",
    ]
    # Evaluate on all test pairs + stratified sample of OOD pairs for rapid turnaround
    ood_eval_slice = ood_pairs[:250]
    track_b_results: Dict[str, Any] = {}
    for m_name in llm_models:
        client = CodeLLMClient(model_name=m_name)
        evaluator = TrackBReasoningEvaluator(client)
        track_b_results[m_name] = {
            "in_domain_test_codenet": evaluator.evaluate_pairs(test_pairs),
            "ood_test_swe_refactor": evaluator.evaluate_pairs(ood_eval_slice),
        }

    elapsed_s = time.perf_counter() - t0
    summary = {
        "elapsed_seconds": round(elapsed_s, 2),
        "dataset_splits": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "ood_test_pairs": len(ood_pairs),
        },
        "track_a_discriminative_and_rl": {
            "XGBoost_Delta_Classifier": {
                "backend": xgb_model.backend_name_,
                "top_feature_importances": xgb_model.feature_importances(FEATURE_NAMES),
                "in_domain_test_codenet": xgb_test,
                "ood_test_swe_refactor": xgb_ood,
            },
            "Multimodal_Pointwise_Regression": {
                "in_domain_test_codenet": pw_test,
                "ood_test_swe_refactor": pw_ood,
            },
            "Multimodal_Bradley_Terry_Pairwise": {
                "in_domain_test_codenet": bt_test,
                "ood_test_swe_refactor": bt_ood,
            },
            "Gymnasium_MultiObjective_DQN": {
                "final_episode_win_rate": dqn_agent.episode_win_rates_[-1] if dqn_agent.episode_win_rates_ else 0.0,
                "in_domain_test_codenet": dqn_test,
                "ood_test_swe_refactor": dqn_ood,
            },
        },
        "track_b_generative_codellms": track_b_results,
    }

    out_path = Path("data/phase3_results.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3: Dual-Track System Architecture (Track A + Track B)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    summary = run_phase3_pipeline(config_path=args.config)
    print("=" * 72)
    print("PHASE 3 COMPLETE: Dual-Track System Architecture (Track A & Track B)")
    print("=" * 72)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
