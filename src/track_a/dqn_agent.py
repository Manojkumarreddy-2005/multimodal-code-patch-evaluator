"""Dueling Double-DQN Agent for Multi-Objective Sequential Patch Ranking (`src/track_a/dqn_agent.py`).

Learns a Q-value policy over `PatchRankingEnv` that balances test-pass guarantees
against technical debt reduction, cognitive complexity risk, and DRY duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.track_a.encoder import (
    MultimodalEncoderConfig,
    MultimodalPatchEncoder,
    RobustMetricNormalizer,
    _gelu,
    _gelu_grad,
)
from src.track_a.gym_env import MultiObjectiveRewardWeights, PatchRankingEnv


@dataclass
class DQNConfig:
    """Hyperparameters for Dueling Double-DQN agent."""

    modality_mode: str = "multimodal"
    hidden_dim: int = 128
    gamma: float = 0.90
    lr: float = 2e-3
    tau: float = 0.08
    epsilon_start: float = 0.35
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.94
    buffer_capacity: int = 8192
    batch_size: int = 64
    episodes: int = 45
    episode_horizon: int = 48
    seed: int = 42


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer for Double-DQN."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 42) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self._rng.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DuelingQNetwork:
    """Dueling Q-Network decomposing Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, n_actions: int = 2, seed: int = 42) -> None:
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(0.0, np.sqrt(2.0 / state_dim), (state_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Value stream V(s): hidden_dim -> 1
        self.W_val = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), (hidden_dim, 1)).astype(np.float32)
        self.b_val = np.zeros(1, dtype=np.float32)

        # Advantage stream A(s, a): hidden_dim -> n_actions
        self.W_adv = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), (hidden_dim, n_actions)).astype(np.float32)
        self.b_adv = np.zeros(n_actions, dtype=np.float32)

        self._step = 0
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}

    def forward(self, states: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        single = states.ndim == 1
        s_in = states[None, :] if single else states
        z1 = s_in @ self.W1 + self.b1
        h1 = _gelu(z1)

        val = h1 @ self.W_val + self.b_val  # (N, 1)
        adv = h1 @ self.W_adv + self.b_adv  # (N, 2)
        adv_centered = adv - np.mean(adv, axis=-1, keepdims=True)
        q_vals = val + adv_centered

        cache = {"s_in": s_in, "z1": z1, "h1": h1}
        return (q_vals[0] if single else q_vals), cache

    def update_on_td_error(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        td_targets: np.ndarray,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
    ) -> float:
        """Compute Huber / MSE TD loss and apply Adam update to Dueling Q-Network."""
        q_vals, cache = self.forward(states)
        bsz = len(states)
        chosen_q = q_vals[np.arange(bsz), actions]
        td_diff = chosen_q - td_targets

        # Smooth L1 (Huber) gradient clipping for stable RL updates
        clipped_diff = np.clip(td_diff, -2.0, 2.0)
        loss = float(np.mean(0.5 * (clipped_diff ** 2)))

        d_q = np.zeros_like(q_vals)
        d_q[np.arange(bsz), actions] = clipped_diff / float(bsz)

        # Backprop through Dueling aggregation: Q = V + (A - mean(A))
        d_val = np.sum(d_q, axis=-1, keepdims=True)
        d_adv = d_q - np.mean(d_q, axis=-1, keepdims=True)

        h1 = cache["h1"]
        d_W_val = (h1.T @ d_val) + weight_decay * self.W_val
        d_b_val = np.sum(d_val, axis=0)

        d_W_adv = (h1.T @ d_adv) + weight_decay * self.W_adv
        d_b_adv = np.sum(d_adv, axis=0)

        d_h1 = (d_val @ self.W_val.T) + (d_adv @ self.W_adv.T)
        d_z1 = d_h1 * _gelu_grad(cache["z1"])

        d_W1 = (cache["s_in"].T @ d_z1) + weight_decay * self.W1
        d_b1 = np.sum(d_z1, axis=0)

        grads = {
            "W1": d_W1.astype(np.float32),
            "b1": d_b1.astype(np.float32),
            "W_val": d_W_val.astype(np.float32),
            "b_val": d_b_val.astype(np.float32),
            "W_adv": d_W_adv.astype(np.float32),
            "b_adv": d_b_adv.astype(np.float32),
        }
        self._apply_adam(grads, lr=lr)
        return loss

    def _apply_adam(
        self,
        grads: Dict[str, np.ndarray],
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._step += 1
        for name, grad in grads.items():
            param = getattr(self, name)
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
            self._m[name] = beta1 * self._m[name] + (1.0 - beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1.0 - beta2) * (grad ** 2)
            m_hat = self._m[name] / (1.0 - beta1 ** self._step)
            v_hat = self._v[name] / (1.0 - beta2 ** self._step)
            setattr(self, name, param - lr * m_hat / (np.sqrt(v_hat) + eps))

    def soft_update_from(self, online_net: "DuelingQNetwork", tau: float = 0.08) -> None:
        for name in ("W1", "b1", "W_val", "b_val", "W_adv", "b_adv"):
            target_p = getattr(self, name)
            online_p = getattr(online_net, name)
            setattr(self, name, (1.0 - tau) * target_p + tau * online_p)


class MultiObjectiveDQNAgent:
    """Multi-Objective Dueling Double-DQN Agent trained on `PatchRankingEnv`."""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        reward_weights: Optional[MultiObjectiveRewardWeights] = None,
    ) -> None:
        self.cfg = config or DQNConfig()
        self.reward_weights = reward_weights or MultiObjectiveRewardWeights()
        self.normalizer = RobustMetricNormalizer()
        self.encoder = MultimodalPatchEncoder(
            MultimodalEncoderConfig(
                modality_mode=self.cfg.modality_mode,
                seed=self.cfg.seed,
            )
        )
        # Compact state representation per patch: 64-dim fused features -> 256-dim pairwise state
        self.patch_feat_dim = 64
        self.state_dim = self.patch_feat_dim * 4

        self.online_q = DuelingQNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_actions=2,
            seed=self.cfg.seed,
        )
        self.target_q = DuelingQNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_actions=2,
            seed=self.cfg.seed,
        )
        self.target_q.soft_update_from(self.online_q, tau=1.0)

        self.episode_rewards_: List[float] = []
        self.episode_win_rates_: List[float] = []
        self.td_losses_: List[float] = []

    def _extract_patch_state_features(
        self, code_emb: np.ndarray, norm_static: np.ndarray
    ) -> np.ndarray:
        """Extract 64-dim multimodal patch features for the RL state representation."""
        single = code_emb.ndim == 1
        c_in = code_emb[None, :] if single else code_emb
        s_in = norm_static[None, :] if single else norm_static

        _, fused_256, cache = self.encoder.forward_with_cache(c_in, s_in)
        # Combine pooled 4-to-1 fused representation (64-dim) with masked static projection (64-dim)
        pooled_fuse = fused_256.reshape(len(c_in), 64, 4).mean(axis=-1)
        feat_64 = pooled_fuse + cache["masked_stat"]
        return feat_64[0] if single else feat_64

    def fit(
        self,
        code_a: np.ndarray,
        static_a: np.ndarray,
        tdi_a: np.ndarray,
        code_b: np.ndarray,
        static_b: np.ndarray,
        tdi_b: np.ndarray,
    ) -> "MultiObjectiveDQNAgent":
        """Warm-start multimodal representation with multi-objective reward alignment, then train Double-DQN in Gymnasium."""
        self.normalizer.fit(np.vstack([static_a, static_b]))
        norm_stat_a = self.normalizer.transform(static_a)
        norm_stat_b = self.normalizer.transform(static_b)

        # Stage 1: Representation warm-up on multi-objective utility target so state embeddings are informative
        rng = np.random.default_rng(self.cfg.seed)
        n_pairs = len(code_a)
        for _ in range(18):
            perm = rng.permutation(n_pairs)
            for start in range(0, n_pairs, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                sa_scores, _, cache_a = self.encoder.forward_with_cache(code_a[idx], norm_stat_a[idx])
                sb_scores, _, cache_b = self.encoder.forward_with_cache(code_b[idx], norm_stat_b[idx])
                margin = sa_scores - sb_scores
                probs = 1.0 / (1.0 + np.exp(-np.clip(margin, -25.0, 25.0)))
                # Weight updates by multi-objective debt & complexity magnitude
                mo_weight = 1.0 + np.tanh((tdi_b[idx] - tdi_a[idx]) / 30.0)
                bsz = float(len(idx))
                d_a = ((probs - 1.0) * mo_weight) / bsz
                d_b = ((1.0 - probs) * mo_weight) / bsz
                ga = self.encoder.backward(d_a, cache_a, weight_decay=2e-4)
                gb = self.encoder.backward(d_b, cache_b, weight_decay=0.0)
                self.encoder.apply_adam_grads({k: ga[k] + gb[k] for k in ga}, lr=self.cfg.lr)

        # Stage 2: Sequential Gymnasium RL training with Dueling Double-DQN
        feat_a = self._extract_patch_state_features(code_a, norm_stat_a)
        feat_b = self._extract_patch_state_features(code_b, norm_stat_b)

        env = PatchRankingEnv(
            features_a=feat_a,
            features_b=feat_b,
            static_raw_a=static_a,
            static_raw_b=static_b,
            tdi_a=tdi_a,
            tdi_b=tdi_b,
            episode_horizon=self.cfg.episode_horizon,
            reward_weights=self.reward_weights,
            seed=self.cfg.seed,
        )
        replay = ReplayBuffer(
            capacity=self.cfg.buffer_capacity,
            state_dim=self.state_dim,
            seed=self.cfg.seed,
        )

        epsilon = self.cfg.epsilon_start
        for ep in range(self.cfg.episodes):
            obs, _ = env.reset(seed=self.cfg.seed + ep)
            done = False
            ep_reward = 0.0
            ep_wins = 0
            steps = 0

            while not done:
                if rng.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    q_vals, _ = self.online_q.forward(obs)
                    action = int(np.argmax(q_vals))

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.push(obs, action, reward, next_obs, done)

                obs = next_obs
                ep_reward += reward
                ep_wins += int(info["chose_clean_patch"] > 0.5)
                steps += 1

                if replay.size >= self.cfg.batch_size:
                    b_s, b_a, b_r, b_ns, b_d = replay.sample(self.cfg.batch_size)
                    # Double-DQN target: select argmax action via online_q, evaluate via target_q
                    next_q_online, _ = self.online_q.forward(b_ns)
                    best_next_actions = np.argmax(next_q_online, axis=-1)
                    next_q_target, _ = self.target_q.forward(b_ns)
                    max_next_q = next_q_target[np.arange(len(b_s)), best_next_actions]

                    td_targets = b_r + (1.0 - b_d) * self.cfg.gamma * max_next_q
                    loss = self.online_q.update_on_td_error(
                        b_s, b_a, td_targets, lr=self.cfg.lr
                    )
                    self.td_losses_.append(loss)
                    self.target_q.soft_update_from(self.online_q, tau=self.cfg.tau)

            epsilon = max(self.cfg.epsilon_end, epsilon * self.cfg.epsilon_decay)
            self.episode_rewards_.append(round(ep_reward, 3))
            self.episode_win_rates_.append(round(ep_wins / max(1, steps), 4))

        return self

    def predict_preference_proba(
        self,
        code_1: np.ndarray,
        static_1: np.ndarray,
        code_2: np.ndarray,
        static_2: np.ndarray,
    ) -> np.ndarray:
        """Compute bidirectional Q-value preference probability P(Patch 1 > Patch 2)."""
        norm_s1 = self.normalizer.transform(static_1)
        norm_s2 = self.normalizer.transform(static_2)
        f1 = self._extract_patch_state_features(code_1, norm_s1)
        f2 = self._extract_patch_state_features(code_2, norm_s2)

        single = f1.ndim == 1
        if single:
            f1 = f1[None, :]
            f2 = f2[None, :]

        # Evaluate Q(s_forward, a=0) - Q(s_forward, a=1) and average with swapped state for zero slot bias
        state_fwd = PatchRankingEnv.build_pair_state(f1, f2)
        state_rev = PatchRankingEnv.build_pair_state(f2, f1)

        q_fwd, _ = self.online_q.forward(state_fwd)
        q_rev, _ = self.online_q.forward(state_rev)

        adv_1_over_2 = 0.5 * ((q_fwd[:, 0] - q_fwd[:, 1]) + (q_rev[:, 1] - q_rev[:, 0]))
        probs = 1.0 / (1.0 + np.exp(-np.clip(adv_1_over_2, -25.0, 25.0)))
        return probs[0] if single else probs

    def score_snippet(self, code_emb: np.ndarray, static_x: np.ndarray) -> np.ndarray:
        """Compute scalar Q-value utility of a single patch against the median reference state."""
        norm_s = self.normalizer.transform(static_x)
        f_x = self._extract_patch_state_features(code_emb, norm_s)
        single = f_x.ndim == 1
        if single:
            f_x = f_x[None, :]
        ref_zero = np.zeros_like(f_x)
        state = PatchRankingEnv.build_pair_state(f_x, ref_zero)
        q_vals, _ = self.online_q.forward(state)
        util = q_vals[:, 0] - q_vals[:, 1]
        return util[0] if single else util
