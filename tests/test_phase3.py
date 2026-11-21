"""Unit and integration tests for Phase 3: Dual-Track System Architecture (Track A & Track B)."""

from __future__ import annotations

import numpy as np

from src.track_a.baselines import (
    BradleyTerryPreferenceModel,
    PointwiseRegressionModel,
    XGBoostDeltaClassifier,
)
from src.track_a.dqn_agent import DQNConfig, MultiObjectiveDQNAgent
from src.track_a.encoder import (
    CodeDFGEmbedder,
    MultimodalEncoderConfig,
    MultimodalPatchEncoder,
    RobustMetricNormalizer,
)
from src.track_a.gym_env import PatchRankingEnv
from src.track_b.llm_client import CodeLLMClient
from src.track_b.prompts import BlindPairwisePromptBuilder, RUBRIC_DIMENSIONS
from src.track_b.reasoner_eval import TrackBReasoningEvaluator


def _make_synthetic_dataset(n: int = 40) -> tuple:
    rng = np.random.default_rng(42)
    # Clean Patch A has lower static complexity/duplication/coupling metrics than Patch B
    static_a = rng.uniform(1.0, 5.0, size=(n, 12)).astype(np.float32)
    static_b = static_a + rng.uniform(2.0, 10.0, size=(n, 12)).astype(np.float32)

    code_a = rng.normal(0.15, 0.5, size=(n, 768)).astype(np.float32)
    code_b = rng.normal(-0.15, 0.5, size=(n, 768)).astype(np.float32)
    code_a /= np.linalg.norm(code_a, axis=1, keepdims=True) + 1e-8
    code_b /= np.linalg.norm(code_b, axis=1, keepdims=True) + 1e-8

    tdi_a = np.sum(static_a, axis=1).astype(np.float32)
    tdi_b = np.sum(static_b, axis=1).astype(np.float32)
    return code_a, static_a, tdi_a, code_b, static_b, tdi_b


def test_multimodal_encoder_dimensions_and_modality_masks() -> None:
    embedder = CodeDFGEmbedder(cache_path=None)
    vec = embedder.embed_code("public class A { public int f(int x) { return x + 1; } }")
    assert vec.shape == (768,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    code_a, static_a, _, _, _, _ = _make_synthetic_dataset(8)
    norm = RobustMetricNormalizer().fit(static_a)
    norm_s = norm.transform(static_a)

    for mode in ("multimodal", "static_only", "code_only"):
        enc = MultimodalPatchEncoder(MultimodalEncoderConfig(modality_mode=mode, seed=42))
        fused = enc.encode(code_a, norm_s)
        scores = enc.score(code_a, norm_s)
        assert fused.shape == (8, 256)
        assert scores.shape == (8,)


def test_track_a_baselines_learn_preference_ranking() -> None:
    ca, sa, ta, cb, sb, tb = _make_synthetic_dataset(48)

    xgb = XGBoostDeltaClassifier(n_estimators=40, seed=42).fit(sa, sb, ta, tb)
    p_xgb = xgb.predict_preference_proba(sa, sb)
    assert float(np.mean(p_xgb >= 0.5)) >= 0.90

    pw = PointwiseRegressionModel(epochs=20, seed=42).fit(ca, sa, ta, cb, sb, tb)
    p_pw = pw.predict_preference_proba(ca, sa, cb, sb)
    assert float(np.mean(p_pw >= 0.5)) >= 0.85

    bt = BradleyTerryPreferenceModel(epochs=20, seed=42).fit(ca, sa, cb, sb)
    p_bt = bt.predict_preference_proba(ca, sa, cb, sb)
    assert float(np.mean(p_bt >= 0.5)) >= 0.90
    assert bt.loss_history_[-1] < bt.loss_history_[0]


def test_gymnasium_env_and_multi_objective_dqn_agent() -> None:
    ca, sa, ta, cb, sb, tb = _make_synthetic_dataset(36)
    feat_a = sa[:, :8]
    feat_b = sb[:, :8]

    env = PatchRankingEnv(
        features_a=feat_a,
        features_b=feat_b,
        static_raw_a=sa,
        static_raw_b=sb,
        tdi_a=ta,
        tdi_b=tb,
        episode_horizon=10,
        seed=42,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (32,)
    assert env.action_space.n == 2

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == (32,)
    assert isinstance(reward, float)
    assert "r_pass" in step_info and "r_debt" in step_info

    agent = MultiObjectiveDQNAgent(
        DQNConfig(episodes=15, episode_horizon=16, batch_size=16, seed=42)
    ).fit(ca, sa, ta, cb, sb, tb)
    probs = agent.predict_preference_proba(ca, sa, cb, sb)
    assert float(np.mean(probs >= 0.5)) >= 0.85


def test_track_b_blind_prompting_and_rubric_evaluator() -> None:
    clean_code = """
    public class CleanFilter {
        public int sumValid(int[] items, int limit) {
            int sum = 0;
            for (int v : items) {
                if (v > 0 && v <= limit) {
                    sum += v;
                }
            }
            return sum;
        }
    }
    """
    bloated_code = """
    public class BloatedFilter {
        public int sumValid(int[] items, int limit, int f1, int f2, int f3) {
            int sumA = 0;
            int sumB = 0;
            for (int i = 0; i < items.length; i++) {
                if (items[i] > 0 && f1 >= 0) {
                    if (items[i] <= limit && (f2 > 0 || f3 > 0)) {
                        if (items[i] != 999) {
                            sumA = sumA + items[i] * 2 - items[i];
                            sumB = sumB + items[i] * 2 - items[i];
                        } else {
                            sumA = sumA + items[i] * 2 - items[i];
                            sumB = sumB + items[i] * 2 - items[i];
                        }
                    }
                }
            }
            return sumA + (sumB - sumB);
        }
    }
    """

    fwd, rev = BlindPairwisePromptBuilder.build_bidirectional_pair(
        pair_id=1, code_a_norm=clean_code, code_b_norm=bloated_code
    )
    assert fwd.clean_slot_label == "Implementation 1"
    assert rev.clean_slot_label == "Implementation 2"

    client = CodeLLMClient(model_name="Qwen2.5-Coder-7B-Instruct")
    verdict = client.evaluate_blind_prompt(fwd)
    assert verdict["preferred_implementation"] in ("Implementation 1", "Implementation 2")
    for dim in RUBRIC_DIMENSIONS:
        assert dim in verdict["dimensions"]
        assert "winner" in verdict["dimensions"][dim]
        assert "detected_issue" in verdict["dimensions"][dim]
        assert "explanation" in verdict["dimensions"][dim]

    evaluator = TrackBReasoningEvaluator(client)
    metrics = evaluator.evaluate_pairs(
        [
            {
                "pair_id": 1,
                "code_a_norm": clean_code,
                "code_b_norm": bloated_code,
                "triggered_dimensions": {
                    "maintainability_reasoning": True,
                    "complexity_awareness": True,
                    "duplication_detection": True,
                    "future_change_impact": True,
                },
            }
        ]
    )
    assert metrics["pairwise_preference_accuracy"] == 1.0
    assert metrics["selection_reasoning_consistency_rate"] == 1.0
