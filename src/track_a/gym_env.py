"""Gymnasium Sequential Candidate Patch Ranking Environment (`PatchRankingEnv`).

Structures patch evaluation as a sequential candidate triage MDP where an RL agent
observes multimodal representations of functionally passing candidate implementations
(randomized order to prevent slot bias) and selects the most maintainable patch
under a Multi-Objective Reward Policy balancing:
  1. Functional test-pass guarantee (`r_pass`)
  2. Technical debt reduction (`r_debt`)
  3. Cognitive complexity risk penalty (`r_complexity`)
  4. Code duplication / DRY violation penalty (`r_duplication`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class MultiObjectiveRewardWeights:
    """Weights for the multi-objective patch evaluation reward function."""

    w_pass: float = 1.0
    w_debt: float = 2.0
    w_complexity: float = 0.75
    w_duplication: float = 0.75


class PatchRankingEnv(gym.Env):
    """Gymnasium environment for sequential multi-objective candidate patch ranking.

    At each step in an episode (of length `episode_horizon` candidate pairs), the agent
    observes a state vector combining candidate 1 and candidate 2 multimodal features
    plus interaction deltas:
        state = [feat_1, feat_2, feat_1 - feat_2, feat_1 * feat_2]
    Action space (`Discrete(2)`):
        0 -> Select Candidate 1 as the maintainable production patch
        1 -> Select Candidate 2 as the maintainable production patch
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        static_raw_a: np.ndarray,
        static_raw_b: np.ndarray,
        tdi_a: np.ndarray,
        tdi_b: np.ndarray,
        episode_horizon: int = 32,
        reward_weights: Optional[MultiObjectiveRewardWeights] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert len(features_a) == len(features_b) == len(tdi_a) == len(tdi_b)
        self.features_a = np.asarray(features_a, dtype=np.float32)
        self.features_b = np.asarray(features_b, dtype=np.float32)
        self.static_raw_a = np.asarray(static_raw_a, dtype=np.float32)
        self.static_raw_b = np.asarray(static_raw_b, dtype=np.float32)
        self.tdi_a = np.asarray(tdi_a, dtype=np.float32)
        self.tdi_b = np.asarray(tdi_b, dtype=np.float32)

        self.n_pairs = len(self.features_a)
        self.feat_dim = self.features_a.shape[1]
        self.state_dim = self.feat_dim * 4
        self.episode_horizon = min(int(episode_horizon), self.n_pairs)
        self.weights = reward_weights or MultiObjectiveRewardWeights()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self._rng = np.random.default_rng(seed)
        self._episode_indices: np.ndarray = np.zeros(self.episode_horizon, dtype=np.int64)
        self._slot_orders: np.ndarray = np.zeros(self.episode_horizon, dtype=np.int64)
        self._step_ptr: int = 0
        self._cumulative_debt_saved: float = 0.0

    @staticmethod
    def build_pair_state(feat_1: np.ndarray, feat_2: np.ndarray) -> np.ndarray:
        """Construct symmetric pairwise interaction state `[f1, f2, f1 - f2, f1 * f2]`."""
        return np.concatenate(
            [feat_1, feat_2, feat_1 - feat_2, feat_1 * feat_2],
            axis=-1,
        ).astype(np.float32)

    def _get_current_obs(self) -> np.ndarray:
        pair_idx = int(self._episode_indices[self._step_ptr])
        flipped = bool(self._slot_orders[self._step_ptr])
        if not flipped:
            f1, f2 = self.features_a[pair_idx], self.features_b[pair_idx]
        else:
            f1, f2 = self.features_b[pair_idx], self.features_a[pair_idx]
        return self.build_pair_state(f1, f2)

    def compute_multi_objective_reward(
        self, pair_idx: int, flipped: bool, action: int
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-objective reward balancing test-pass guarantee vs debt & risk reduction."""
        # Ground-truth optimal action is 0 when not flipped (Slot 0 = Patch A), 1 when flipped (Slot 1 = Patch A)
        chose_clean = (action == 0 and not flipped) or (action == 1 and flipped)

        if chose_clean:
            tdi_chosen, tdi_alt = float(self.tdi_a[pair_idx]), float(self.tdi_b[pair_idx])
            stat_chosen, stat_alt = self.static_raw_a[pair_idx], self.static_raw_b[pair_idx]
        else:
            tdi_chosen, tdi_alt = float(self.tdi_b[pair_idx]), float(self.tdi_a[pair_idx])
            stat_chosen, stat_alt = self.static_raw_b[pair_idx], self.static_raw_a[pair_idx]

        # 1. Functional correctness guarantee (both candidates pass 100% of unit tests)
        r_pass = self.weights.w_pass * 1.0

        # 2. Technical debt reduction objective (positive when chosen patch has lower TDI than alternative)
        debt_delta = tdi_alt - tdi_chosen
        r_debt = self.weights.w_debt * float(np.tanh(debt_delta / 25.0))

        # 3. Complexity risk objective (cognitive_complexity is index 1, max_nesting_depth is index 3)
        cog_diff = (float(stat_alt[1]) - float(stat_chosen[1])) / 10.0
        nest_diff = (float(stat_alt[3]) - float(stat_chosen[3])) / 3.0
        r_comp = self.weights.w_complexity * float(np.tanh(0.6 * cog_diff + 0.4 * nest_diff))

        # 4. Duplication & bloat avoidance objective (ast_clone_ratio idx 4, token_4gram idx 5)
        clone_diff = (float(stat_alt[4]) - float(stat_chosen[4])) * 4.0
        ngram_diff = (float(stat_alt[5]) - float(stat_chosen[5])) * 4.0
        r_dup = self.weights.w_duplication * float(np.tanh(0.5 * clone_diff + 0.5 * ngram_diff))

        total_reward = r_pass + r_debt + r_comp + r_dup
        components = {
            "chose_clean_patch": 1.0 if chose_clean else 0.0,
            "r_pass": round(r_pass, 4),
            "r_debt": round(r_debt, 4),
            "r_complexity": round(r_comp, 4),
            "r_duplication": round(r_dup, 4),
            "debt_saved_minutes": round(debt_delta, 2),
        }
        return float(total_reward), components

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._episode_indices = self._rng.choice(
            self.n_pairs, size=self.episode_horizon, replace=(self.n_pairs < self.episode_horizon)
        )
        # Randomize slot order (0: [A, B], 1: [B, A]) at every step to eliminate slot bias
        self._slot_orders = self._rng.integers(0, 2, size=self.episode_horizon)
        self._step_ptr = 0
        self._cumulative_debt_saved = 0.0

        obs = self._get_current_obs()
        return obs, {"step": 0, "cumulative_debt_saved": 0.0}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = int(action)
        pair_idx = int(self._episode_indices[self._step_ptr])
        flipped = bool(self._slot_orders[self._step_ptr])

        reward, info = self.compute_multi_objective_reward(pair_idx, flipped, act)
        self._cumulative_debt_saved += info["debt_saved_minutes"]
        info["cumulative_debt_saved"] = round(self._cumulative_debt_saved, 2)

        self._step_ptr += 1
        terminated = self._step_ptr >= self.episode_horizon
        truncated = False

        if not terminated:
            next_obs = self._get_current_obs()
        else:
            next_obs = np.zeros(self.state_dim, dtype=np.float32)

        return next_obs, reward, terminated, truncated, info
