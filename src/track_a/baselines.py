"""Track A Baselines & Supervised Ranking Objectives.

Implements:
1. `XGBoostDeltaClassifier`: Tabular XGBoost classifier on static metric deltas (Δm = m(x2) - m(x1)).
2. `PointwiseRegressionModel`: Multimodal encoder trained with pointwise MSE on continuous debt quality.
3. `BradleyTerryPreferenceModel`: Siamese multimodal encoder trained with Bradley-Terry pairwise preference loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.track_a.encoder import (
    MultimodalEncoderConfig,
    MultimodalPatchEncoder,
    RobustMetricNormalizer,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class XGBoostDeltaClassifier:
    """Tabular XGBoost classifier operating on 12 Tree-Sitter static metric deltas (Δm)."""

    def __init__(
        self,
        n_estimators: int = 160,
        max_depth: int = 4,
        learning_rate: float = 0.06,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_ = None
        self.pointwise_reg_ = None
        self.backend_name_ = "xgboost"

    def _build_classifier(self):
        try:
            from xgboost import XGBClassifier

            self.backend_name_ = "xgboost.XGBClassifier"
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                random_state=self.seed,
                eval_metric="logloss",
                n_jobs=1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.backend_name_ = "sklearn.HistGradientBoostingClassifier"
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )

    def _build_regressor(self):
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                reg_lambda=1.5,
                random_state=self.seed,
                n_jobs=1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )

    def fit(
        self,
        static_a: np.ndarray,
        static_b: np.ndarray,
        tdi_a: Optional[np.ndarray] = None,
        tdi_b: Optional[np.ndarray] = None,
    ) -> "XGBoostDeltaClassifier":
        """Fit on symmetric pairs: (A, B) -> y=1 (first is cleaner) and (B, A) -> y=0."""
        delta_pos = static_b - static_a  # m(B) - m(A) > 0 when A is cleaner
        delta_neg = static_a - static_b  # m(A) - m(B) < 0 when B is presented first

        X_pairs = np.vstack([delta_pos, delta_neg]).astype(np.float32)
        y_pairs = np.concatenate(
            [np.ones(len(delta_pos), dtype=np.int32), np.zeros(len(delta_neg), dtype=np.int32)]
        )

        # Deterministic shuffle
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X_pairs))
        self.model_ = self._build_classifier()
        self.model_.fit(X_pairs[perm], y_pairs[perm])

        # Also fit a pointwise tree regressor on individual snippet static features for Spearman rho
        if tdi_a is not None and tdi_b is not None:
            X_single = np.vstack([static_a, static_b]).astype(np.float32)
            y_single = -np.log1p(np.maximum(0.0, np.concatenate([tdi_a, tdi_b])))
            self.pointwise_reg_ = self._build_regressor()
            self.pointwise_reg_.fit(X_single, y_single)

        return self

    def predict_preference_proba(self, static_1: np.ndarray, static_2: np.ndarray) -> np.ndarray:
        """Predict P(Patch 1 > Patch 2) from static feature vectors `static_1` and `static_2`."""
        delta = (static_2 - static_1).astype(np.float32)
        if delta.ndim == 1:
            delta = delta[None, :]
        probs = self.model_.predict_proba(delta)[:, 1]
        return probs

    def score_snippet(self, static_x: np.ndarray) -> np.ndarray:
        """Predict scalar quality score (higher = cleaner, lower debt) for Spearman correlation."""
        arr = np.asarray(static_x, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if self.pointwise_reg_ is not None:
            out = self.pointwise_reg_.predict(arr).astype(np.float32)
        else:
            zero_ref = np.zeros_like(arr)
            out = self.predict_preference_proba(arr, zero_ref).astype(np.float32)
        return out[0] if single else out

    def feature_importances(self, feature_names: Tuple[str, ...]) -> Dict[str, float]:
        if hasattr(self.model_, "feature_importances_"):
            imps = self.model_.feature_importances_
            return {
                name: round(float(val), 4)
                for name, val in zip(feature_names, imps)
            }
        return {name: round(1.0 / len(feature_names), 4) for name in feature_names}


class PointwiseRegressionModel:
    """Pointwise Regression Baseline using the Multimodal Encoder (MSE loss on quality target)."""

    def __init__(
        self,
        modality_mode: str = "multimodal",
        epochs: int = 35,
        batch_size: int = 64,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.modality_mode = modality_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.normalizer = RobustMetricNormalizer()
        self.encoder = MultimodalPatchEncoder(
            MultimodalEncoderConfig(modality_mode=modality_mode, seed=seed)
        )
        self.loss_history_: List[float] = []

    def fit(
        self,
        code_a: np.ndarray,
        static_a: np.ndarray,
        tdi_a: np.ndarray,
        code_b: np.ndarray,
        static_b: np.ndarray,
        tdi_b: np.ndarray,
    ) -> "PointwiseRegressionModel":
        all_static = np.vstack([static_a, static_b])
        self.normalizer.fit(all_static)

        X_code = np.vstack([code_a, code_b]).astype(np.float32)
        X_stat = self.normalizer.transform(all_static)

        # Quality score target: higher means lower technical debt
        raw_quality = -np.log1p(np.maximum(0.0, np.concatenate([tdi_a, tdi_b])))
        q_mean = float(np.mean(raw_quality))
        q_std = float(np.std(raw_quality)) + 1e-6
        y_target = ((raw_quality - q_mean) / q_std).astype(np.float32)

        rng = np.random.default_rng(self.seed)
        n = len(y_target)

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            epoch_losses: List[float] = []
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                c_batch = X_code[idx]
                s_batch = X_stat[idx]
                y_batch = y_target[idx]

                preds, _, cache = self.encoder.forward_with_cache(c_batch, s_batch)
                diff = preds - y_batch
                loss = float(np.mean(diff ** 2))
                epoch_losses.append(loss)

                d_scores = (2.0 / len(idx)) * diff
                grads = self.encoder.backward(d_scores, cache, weight_decay=self.weight_decay)
                self.encoder.apply_adam_grads(grads, lr=self.lr)

            self.loss_history_.append(float(np.mean(epoch_losses)))

        return self

    def score_snippet(self, code_emb: np.ndarray, static_x: np.ndarray) -> np.ndarray:
        norm_s = self.normalizer.transform(static_x)
        return self.encoder.score(code_emb, norm_s)

    def predict_preference_proba(
        self,
        code_1: np.ndarray,
        static_1: np.ndarray,
        code_2: np.ndarray,
        static_2: np.ndarray,
    ) -> np.ndarray:
        s1 = self.score_snippet(code_1, static_1)
        s2 = self.score_snippet(code_2, static_2)
        return _sigmoid(s1 - s2)


class BradleyTerryPreferenceModel:
    """Siamese Multimodal Ranking Model trained with Bradley-Terry pairwise preference loss.

    Minimizes:
        L_BT(x_A, x_B) = -log sigmoid(s(x_A) - s(x_B)) + lambda * ||theta||_2^2
    Supports `modality_mode` in {'multimodal', 'static_only', 'code_only'} for Phase 4 ablations.
    """

    def __init__(
        self,
        modality_mode: str = "multimodal",
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 2e-3,
        weight_decay: float = 2e-4,
        seed: int = 42,
    ) -> None:
        self.modality_mode = modality_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.normalizer = RobustMetricNormalizer()
        self.encoder = MultimodalPatchEncoder(
            MultimodalEncoderConfig(modality_mode=modality_mode, seed=seed)
        )
        self.loss_history_: List[float] = []

    def fit(
        self,
        code_a: np.ndarray,
        static_a: np.ndarray,
        code_b: np.ndarray,
        static_b: np.ndarray,
    ) -> "BradleyTerryPreferenceModel":
        self.normalizer.fit(np.vstack([static_a, static_b]))
        norm_stat_a = self.normalizer.transform(static_a)
        norm_stat_b = self.normalizer.transform(static_b)

        rng = np.random.default_rng(self.seed)
        n_pairs = len(code_a)

        for _ in range(self.epochs):
            perm = rng.permutation(n_pairs)
            epoch_losses: List[float] = []

            for start in range(0, n_pairs, self.batch_size):
                idx = perm[start : start + self.batch_size]
                ca, sa = code_a[idx], norm_stat_a[idx]
                cb, sb = code_b[idx], norm_stat_b[idx]

                scores_a, _, cache_a = self.encoder.forward_with_cache(ca, sa)
                scores_b, _, cache_b = self.encoder.forward_with_cache(cb, sb)

                margin = scores_a - scores_b
                probs = _sigmoid(margin)
                bt_loss = float(-np.mean(np.log(probs + 1e-8)))
                epoch_losses.append(bt_loss)

                # Gradient of -log sigmoid(s_A - s_B) w.r.t. s_A is (sigmoid(s_A - s_B) - 1)
                bsz = float(len(idx))
                d_scores_a = (probs - 1.0) / bsz
                d_scores_b = (1.0 - probs) / bsz

                grads_a = self.encoder.backward(d_scores_a, cache_a, weight_decay=self.weight_decay)
                grads_b = self.encoder.backward(d_scores_b, cache_b, weight_decay=0.0)

                combined_grads = {
                    k: grads_a[k] + grads_b[k] for k in grads_a
                }
                self.encoder.apply_adam_grads(combined_grads, lr=self.lr)

            self.loss_history_.append(float(np.mean(epoch_losses)))

        return self

    def score_snippet(self, code_emb: np.ndarray, static_x: np.ndarray) -> np.ndarray:
        norm_s = self.normalizer.transform(static_x)
        return self.encoder.score(code_emb, norm_s)

    def encode_snippet(self, code_emb: np.ndarray, static_x: np.ndarray) -> np.ndarray:
        norm_s = self.normalizer.transform(static_x)
        return self.encoder.encode(code_emb, norm_s)

    def predict_preference_proba(
        self,
        code_1: np.ndarray,
        static_1: np.ndarray,
        code_2: np.ndarray,
        static_2: np.ndarray,
    ) -> np.ndarray:
        s1 = self.score_snippet(code_1, static_1)
        s2 = self.score_snippet(code_2, static_2)
        return _sigmoid(s1 - s2)
