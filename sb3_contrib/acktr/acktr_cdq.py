"""ACKTR + on-policy Clipped Double Q (CDQ) for Stable-Baselines3.

This module is designed to sit next to an ACKTR-enabled ``sb3_contrib`` fork.
It does not modify the installed package. Usage::

    from acktr_cdq_hybrid import ACKTRCDQHybrid

    model = ACKTRCDQHybrid(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": AcktrPaperCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [],
        },
        device="mps",
        kfac_mps_only=True,
    )

Important
---------
Classical CDQ (clipped double Q-learning) is an off-policy technique with two
Q-functions and clipped targets. ACKTR is an on-policy actor-critic algorithm.
This implementation is therefore an explicit hybrid:

* Q1 shares the actor CNN; Q2 owns a fully independent CNN and head
* two action-value estimates Q1(s, a), Q2(s, a)
* clipped estimate min(Q1, Q2)
* scalar rollout bootstrap value
      V_cdq(s) = sum_a pi(a|s) min(Q1(s,a), Q2(s,a))
* both Q-heads are trained against the on-policy rollout return
* ACKTR/K-FAC optimizes the shared actor/Q1 trunk and the independent Q2 network

Only ``gymnasium.spaces.Discrete`` action spaces are supported. This covers
Atari Pong and Breakout, but not continuous-control environments.
"""

from __future__ import annotations

from functools import partial
import copy
import math
from typing import Any, Optional, Type, Union
import warnings

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

try:
    # Preferred path for the custom sb3_contrib ACKTR package used by this project.
    from sb3_contrib.acktr.acktr import ACKTR as _BaseACKTR
except (ImportError, ModuleNotFoundError):
    try:
        from sb3_contrib import ACKTR as _BaseACKTR  # type: ignore[attr-defined]
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "ACKTR was not found in sb3_contrib. Install/copy the ACKTR-enabled "
            "sb3_contrib fork before importing acktr_cdq."
        ) from exc


ACKTR_CDQ_VERSION = "acktr-cdq-hybrid-q1-shared-q2-independent-v1"


class _TwinQPolicyMixin:
    """CDQ policy with asymmetric critic representation learning.

    Q1 shares the actor feature extractor (including the Atari CNN and its
    512-unit projection). Q2 owns a fully independent copy of the feature
    extractor, critic body and output head. No Q2 parameter is shared with the
    actor or Q1.
    """

    q2_features_extractor: nn.Module
    q2_critic_net: nn.Module
    q_net_2: nn.Linear

    def __init__(
        self,
        *args: Any,
        cdq_q_init_gain: float = 0.01,
        cdq_q_clip: float = 50.0,
        cdq_logit_clip: float = 20.0,
        **kwargs: Any,
    ) -> None:
        self.cdq_q_init_gain = float(cdq_q_init_gain)
        self.cdq_q_clip = float(cdq_q_clip)
        self.cdq_logit_clip = float(cdq_logit_clip)
        if self.cdq_q_init_gain <= 0.0:
            raise ValueError("cdq_q_init_gain must be > 0.")
        if self.cdq_q_clip <= 0.0:
            raise ValueError("cdq_q_clip must be > 0.")
        if self.cdq_logit_clip <= 0.0:
            raise ValueError("cdq_logit_clip must be > 0.")

        # The requested architecture requires actor and Q1 to share one CNN.
        # Force this even if an old checkpoint/policy_kwargs contains False.
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # The base policy first builds the actor, scalar value head and optimizer.
        # Replace the scalar value head with Q1, add an independent Q2 network,
        # and finally rebuild the optimizer so all new parameters are included.
        super()._build(lr_schedule)  # type: ignore[misc]

        if not isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError(
                "ACKTR-CDQ supports only Discrete action spaces because the two "
                "critics output one Q-value per action."
            )
        if not bool(self.share_features_extractor):
            raise RuntimeError(
                "Hybrid CDQ requires share_features_extractor=True so Q1 shares "
                "the actor CNN."
            )

        latent_dim_q1 = int(self.mlp_extractor.latent_dim_vf)
        action_dim = int(self.action_space.n)

        # Q1: shared actor feature extractor -> base critic branch -> Q1 head.
        self.value_net = nn.Linear(latent_dim_q1, action_dim)

        # Q2: fully independent feature extractor, critic body and Q head.
        # deepcopy preserves the exact custom CNN configuration while creating
        # new Parameter objects. The modules are then independently initialized.
        self.q2_features_extractor = copy.deepcopy(self.features_extractor)
        self.q2_critic_net = copy.deepcopy(self.mlp_extractor.value_net)
        self.q_net_2 = nn.Linear(latent_dim_q1, action_dim)

        if self.ortho_init:
            # Match SB3's actor-critic initialization gains while consuming new
            # RNG draws, so Q2 starts independently rather than as a clone.
            self.value_net.apply(partial(self.init_weights, gain=self.cdq_q_init_gain))
            self.q2_features_extractor.apply(
                partial(self.init_weights, gain=math.sqrt(2.0))
            )
            self.q2_critic_net.apply(
                partial(self.init_weights, gain=math.sqrt(2.0))
            )
            self.q_net_2.apply(
                partial(self.init_weights, gain=self.cdq_q_init_gain)
            )

        self._assert_q2_parameter_independence()

        # The base optimizer was created before Q2 existed. Rebuild it with the
        # complete parameter set. ACKTR replaces it with K-FAC later when enabled.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _assert_q2_parameter_independence(self) -> None:
        shared_modules = (
            self.features_extractor,
            self.mlp_extractor,
            self.action_net,
            self.value_net,
        )
        shared_ids = {
            id(parameter)
            for module in shared_modules
            for parameter in module.parameters()
        }
        q2_ids = {
            id(parameter)
            for module in (
                self.q2_features_extractor,
                self.q2_critic_net,
                self.q_net_2,
            )
            for parameter in module.parameters()
        }
        overlap = shared_ids.intersection(q2_ids)
        if overlap:
            raise RuntimeError(
                "Q2 is not independent: actor/Q1 and Q2 share Parameter objects."
            )

    def _safe_action_distribution(self, latent_pi: th.Tensor) -> Any:
        logits = self.action_net(latent_pi)
        if not bool(th.isfinite(logits).all().item()):
            raise FloatingPointError(
                "ACKTR-CDQ policy logits became NaN/Inf. The preceding optimizer "
                "update was numerically unstable; use the stable CDQ defaults or "
                "lower the learning rate."
            )
        logits = logits.clamp(-self.cdq_logit_clip, self.cdq_logit_clip)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _validate_q_values(self, q: th.Tensor, *, name: str) -> th.Tensor:
        if not bool(th.isfinite(q).all().item()):
            raise FloatingPointError(
                f"ACKTR-CDQ {name} contains NaN/Inf before rollout bootstrapping."
            )
        return q

    @staticmethod
    def _categorical_probabilities(distribution: Any) -> th.Tensor:
        inner = getattr(distribution, "distribution", None)
        probs = getattr(inner, "probs", None)
        if probs is None:
            raise RuntimeError(
                "ACKTR-CDQ requires a categorical action distribution exposing "
                "distribution.probs."
            )
        if probs.ndim != 2:
            raise RuntimeError(
                f"Expected categorical probabilities with shape (batch, actions), "
                f"got {tuple(probs.shape)}."
            )
        return probs

    def _latent_actor_q1_q2(
        self,
        obs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Actor and Q1 share the complete primary feature extractor.
        shared_features = self.extract_features(obs, self.features_extractor)
        if isinstance(shared_features, tuple):
            raise RuntimeError(
                "Hybrid CDQ expected one shared feature tensor for actor and Q1."
            )
        latent_pi, latent_q1 = self.mlp_extractor(shared_features)

        # Q2 reprocesses the raw observation through its independent network.
        q2_features = self.extract_features(obs, self.q2_features_extractor)
        if isinstance(q2_features, tuple):
            raise RuntimeError("Independent Q2 extractor returned an unexpected tuple.")
        latent_q2 = self.q2_critic_net(q2_features)
        return latent_pi, latent_q1, latent_q2

    def _all_q_values_from_latents(
        self,
        latent_q1: th.Tensor,
        latent_q2: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        q1 = self._validate_q_values(self.value_net(latent_q1), name="Q1")
        q2 = self._validate_q_values(self.q_net_2(latent_q2), name="Q2")
        clipped_q = th.minimum(q1, q2).clamp(
            -self.cdq_q_clip, self.cdq_q_clip
        )
        return q1, q2, clipped_q

    def predict_q_values(
        self,
        obs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return Q1/Q2 with shape ``(batch, n_actions)``."""
        _latent_pi, latent_q1, latent_q2 = self._latent_actor_q1_q2(obs)
        q1, q2, _clipped_q = self._all_q_values_from_latents(
            latent_q1, latent_q2
        )
        return q1, q2

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Return E_pi[min(Q1,Q2)] for GAE/rollout bootstrapping."""
        latent_pi, latent_q1, latent_q2 = self._latent_actor_q1_q2(obs)
        distribution = self._safe_action_distribution(latent_pi)
        probs = self._categorical_probabilities(distribution)
        _q1, _q2, clipped_q = self._all_q_values_from_latents(
            latent_q1, latent_q2
        )
        return (probs * clipped_q).sum(dim=1, keepdim=True)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample actions and return the clipped expected rollout value."""
        latent_pi, latent_q1, latent_q2 = self._latent_actor_q1_q2(obs)
        distribution = self._safe_action_distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        probs = self._categorical_probabilities(distribution)
        _q1, _q2, clipped_q = self._all_q_values_from_latents(
            latent_q1, latent_q2
        )
        values = (probs * clipped_q).sum(dim=1, keepdim=True)

        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions_cdq(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Evaluate the actor, shared Q1 and independent Q2."""
        latent_pi, latent_q1, latent_q2 = self._latent_actor_q1_q2(obs)
        distribution = self._safe_action_distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        probs = self._categorical_probabilities(distribution)
        q1, q2, clipped_q = self._all_q_values_from_latents(
            latent_q1, latent_q2
        )
        state_values = (probs * clipped_q).sum(dim=1, keepdim=True)

        action_index = actions.long().reshape(-1, 1)
        if action_index.shape[0] != q1.shape[0]:
            raise RuntimeError(
                "Action batch size does not match the Q-value batch size: "
                f"{action_index.shape[0]} != {q1.shape[0]}."
            )
        q1_taken = q1.gather(dim=1, index=action_index)
        q2_taken = q2.gather(dim=1, index=action_index)
        return state_values, q1_taken, q2_taken, log_prob, entropy

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        state_values, _q1, _q2, log_prob, entropy = self.evaluate_actions_cdq(
            obs, actions
        )
        return state_values, log_prob, entropy


class CDQMlpPolicy(_TwinQPolicyMixin, ActorCriticPolicy):
    """MLP policy with two discrete Q-heads."""


class CDQCnnPolicy(_TwinQPolicyMixin, ActorCriticCnnPolicy):
    """CNN policy with two discrete Q-heads."""


class ACKTRCDQ(_BaseACKTR):
    """On-policy ACKTR with clipped double discrete Q critics.

    This is an experimental ACKTR/CDQ hybrid rather than the original ACKTR
    algorithm or the original off-policy TD3 CDQ algorithm.
    """

    policy_aliases = {
        "MlpPolicy": CDQMlpPolicy,
        "CnnPolicy": CDQCnnPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        *args: Any,
        cdq_value_loss_reduction: str = "mean",
        cdq_fisher_reduction: str = "mean",
        cdq_q_init_gain: float = 0.01,
        cdq_q_clip: float = 50.0,
        cdq_logit_clip: float = 20.0,
        cdq_critic_loss: str = "huber",
        cdq_huber_delta: float = 1.0,
        cdq_advantage_clip: float = 20.0,
        cdq_always_clip_grad: bool = True,
        cdq_rollback_on_nonfinite: bool = True,
        **kwargs: Any,
    ) -> None:
        self.cdq_value_loss_reduction = str(cdq_value_loss_reduction).lower()
        self.cdq_fisher_reduction = str(cdq_fisher_reduction).lower()
        self.cdq_q_clip = float(cdq_q_clip)
        self.cdq_critic_loss = str(cdq_critic_loss).lower()
        self.cdq_huber_delta = float(cdq_huber_delta)
        self.cdq_advantage_clip = float(cdq_advantage_clip)
        self.cdq_always_clip_grad = bool(cdq_always_clip_grad)
        self.cdq_rollback_on_nonfinite = bool(cdq_rollback_on_nonfinite)
        self.cdq_nonfinite_updates = 0

        if self.cdq_value_loss_reduction not in {"mean", "sum"}:
            raise ValueError("cdq_value_loss_reduction must be 'mean' or 'sum'.")
        if self.cdq_fisher_reduction not in {"mean", "sum"}:
            raise ValueError("cdq_fisher_reduction must be 'mean' or 'sum'.")
        if self.cdq_critic_loss not in {"huber", "mse"}:
            raise ValueError("cdq_critic_loss must be 'huber' or 'mse'.")
        if self.cdq_q_clip <= 0.0 or self.cdq_huber_delta <= 0.0:
            raise ValueError("cdq_q_clip and cdq_huber_delta must be > 0.")
        if self.cdq_advantage_clip <= 0.0:
            raise ValueError("cdq_advantage_clip must be > 0.")

        policy_kwargs = dict(kwargs.get("policy_kwargs") or {})
        policy_kwargs.setdefault("cdq_q_init_gain", float(cdq_q_init_gain))
        policy_kwargs.setdefault("cdq_q_clip", self.cdq_q_clip)
        policy_kwargs.setdefault("cdq_logit_clip", float(cdq_logit_clip))
        kwargs["policy_kwargs"] = policy_kwargs

        # ACKTR's historical 0.25 default is too aggressive for two sparse
        # action-value heads. Preserve an explicit user value, but use a safe
        # CDQ default when no learning rate was supplied.
        if len(args) <= 6:
            kwargs.setdefault("learning_rate", 0.01)

        super().__init__(policy, env, *args, **kwargs)

        initial_lr = float(self.lr_schedule(1.0))
        if initial_lr > 0.05:
            warnings.warn(
                f"ACKTR-CDQ learning_rate={initial_lr:g} is very aggressive for "
                "twin Q-heads. Values above 0.05 can make rollout values and GAE "
                "non-finite; 0.01 is the recommended starting point.",
                RuntimeWarning,
            )

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError(
                "ACKTRCDQ currently supports only Discrete action spaces."
            )
        if not callable(getattr(self.policy, "evaluate_actions_cdq", None)):
            raise TypeError(
                "ACKTRCDQ requires CDQMlpPolicy, CDQCnnPolicy, or another policy "
                "implementing evaluate_actions_cdq()."
            )
        if not hasattr(self.policy, "q_net_2"):
            raise TypeError("ACKTRCDQ policy is missing its second Q head q_net_2.")

    def _critic_loss(self, prediction: th.Tensor, target: th.Tensor) -> th.Tensor:
        if self.cdq_critic_loss == "huber":
            return F.smooth_l1_loss(
                prediction, target, beta=self.cdq_huber_delta
            )
        return F.mse_loss(prediction, target)

    def _safe_targets_and_advantages(
        self, returns: th.Tensor, advantages: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        if not bool(th.isfinite(returns).all().item()):
            raise FloatingPointError(
                "Rollout returns already contain NaN/Inf. Restart from the last "
                "finite checkpoint after installing the stable ACKTR-CDQ version."
            )
        if not bool(th.isfinite(advantages).all().item()):
            raise FloatingPointError(
                "Rollout advantages contain NaN/Inf because a previous rollout "
                "bootstrap value was non-finite."
            )
        safe_returns = returns.clamp(-self.cdq_q_clip, self.cdq_q_clip)
        safe_advantages = advantages.clamp(
            -self.cdq_advantage_clip, self.cdq_advantage_clip
        )
        return safe_returns, safe_advantages

    @staticmethod
    def _snapshot_parameters(parameters: list[th.nn.Parameter]) -> list[th.Tensor]:
        return [parameter.detach().clone() for parameter in parameters]

    @staticmethod
    @th.no_grad()
    def _restore_parameters(
        parameters: list[th.nn.Parameter], snapshot: list[th.Tensor]
    ) -> None:
        for parameter, saved in zip(parameters, snapshot):
            parameter.copy_(saved)

    @staticmethod
    def _parameters_are_finite(parameters: list[th.nn.Parameter]) -> bool:
        return all(bool(th.isfinite(parameter).all().item()) for parameter in parameters)

    @staticmethod
    def _reset_optimizer_numerics(optimizer: Any) -> None:
        optimizer.state.clear()
        module_states = getattr(optimizer, "module_states", None)
        if isinstance(module_states, dict):
            for state in module_states.values():
                for name in (
                    "A", "G", "A_inv", "G_inv",
                    "A_eigvals", "A_eigvecs", "G_eigvals", "G_eigvecs",
                ):
                    if hasattr(state, name):
                        setattr(state, name, None)
                if hasattr(state, "stats_count"):
                    state.stats_count = 0
        for name, value in (
            ("steps", 0),
            ("stats_steps", 0),
            ("factor_steps", 0),
            ("last_v_f_v", 0.0),
            ("last_kl_scale", 1.0),
        ):
            if hasattr(optimizer, name):
                setattr(optimizer, name, value)

    def _clip_and_validate_gradients(self) -> float:
        max_norm = float(self.max_grad_norm if self.max_grad_norm is not None else 1.0)
        total_norm = th.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm,
            error_if_nonfinite=True,
        )
        return float(total_norm.item())

    def _combine_twin_losses(self, loss1: th.Tensor, loss2: th.Tensor) -> th.Tensor:
        total = loss1 + loss2
        if self.cdq_value_loss_reduction == "mean":
            total = 0.5 * total
        return total

    def _cdq_fisher_weight(self) -> float:
        return 0.5 if self.cdq_fisher_reduction == "mean" else 1.0

    def _train_without_kfac(self) -> None:
        """A2C-like first-order update that still trains both Q heads."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        normalize_advantage = self._should_normalize_advantage()
        last_policy_loss = None
        last_q1_loss = None
        last_q2_loss = None
        last_value_loss = None
        last_entropy_loss = None
        last_gap = None

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = self._prepare_actions(rollout_data.actions)
            safe_returns, advantages = self._safe_targets_and_advantages(
                rollout_data.returns, rollout_data.advantages
            )
            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(correction=0) + 1e-8
                )

            (
                _state_values,
                q1_taken,
                q2_taken,
                log_prob,
                entropy,
            ) = self.policy.evaluate_actions_cdq(rollout_data.observations, actions)
            q1_taken = q1_taken.flatten()
            q2_taken = q2_taken.flatten()

            policy_loss = -(advantages * log_prob).mean()
            q1_loss = self._critic_loss(q1_taken, safe_returns)
            q2_loss = self._critic_loss(q2_taken, safe_returns)
            value_loss = self._combine_twin_losses(q1_loss, q2_loss)

            if entropy is None:
                entropy_loss = log_prob.mean()
            else:
                entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
            )

            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._clip_and_validate_gradients()
            parameters = [p for p in self.policy.parameters() if p.requires_grad]
            snapshot = (
                self._snapshot_parameters(parameters)
                if self.cdq_rollback_on_nonfinite
                else []
            )
            self.policy.optimizer.step()
            if not self._parameters_are_finite(parameters):
                if snapshot:
                    self._restore_parameters(parameters, snapshot)
                self._reset_optimizer_numerics(self.policy.optimizer)
                self.cdq_nonfinite_updates += 1
                warnings.warn(
                    "Skipped and rolled back a non-finite ACKTR-CDQ optimizer update.",
                    RuntimeWarning,
                )
                continue

            last_policy_loss = policy_loss.detach()
            last_q1_loss = q1_loss.detach()
            last_q2_loss = q2_loss.detach()
            last_value_loss = value_loss.detach()
            last_entropy_loss = entropy_loss.detach()
            last_gap = (q1_taken - q2_taken).abs().mean().detach()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/use_kfac", 0)
        self.logger.record("train/use_cdq", 1)
        self.logger.record("train/cdq_nonfinite_updates", self.cdq_nonfinite_updates)

        if last_policy_loss is not None:
            self.logger.record("train/policy_loss", last_policy_loss.item())
            self.logger.record("train/q1_loss", last_q1_loss.item())
            self.logger.record("train/q2_loss", last_q2_loss.item())
            self.logger.record("train/value_loss", last_value_loss.item())
            self.logger.record("train/entropy_loss", last_entropy_loss.item())
            self.logger.record("train/cdq_q_gap", last_gap.item())

    def train(self) -> None:
        if not self.use_kfac:
            self._train_without_kfac()
            return

        self.policy.set_training_mode(True)
        normalize_advantage = self._should_normalize_advantage()

        last_policy_loss = None
        last_q1_loss = None
        last_q2_loss = None
        last_value_loss = None
        last_entropy_loss = None
        last_fisher_loss = None
        last_gap = None
        last_clipped_value = None
        last_lr = None
        optimizer_steps = 0
        configured_repeats = 0

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = self._prepare_actions(rollout_data.actions)
            batch_size = int(rollout_data.observations.shape[0])
            if batch_size <= 0:
                raise RuntimeError("ACKTRCDQ received an empty rollout batch.")

            if self.optimizer_steps_per_rollout is None:
                repeats = batch_size
            else:
                repeats = min(batch_size, self.optimizer_steps_per_rollout)
            configured_repeats = repeats
            schedule_increment = float(batch_size) / float(repeats)

            safe_returns, advantages = self._safe_targets_and_advantages(
                rollout_data.returns, rollout_data.advantages
            )
            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(correction=0) + 1e-8
                )

            optimizer = self.policy.optimizer
            for _ in range(repeats):
                last_lr = self._set_sb2_learning_rate(schedule_increment)
                is_cold = optimizer.next_step_is_cold()
                collecting = not is_cold
                fisher_loss = None

                optimizer.zero_grad(set_to_none=True)
                if collecting:
                    optimizer.begin_fisher_stats_collection()

                try:
                    (
                        state_values,
                        q1_taken,
                        q2_taken,
                        log_prob,
                        entropy,
                    ) = self.policy.evaluate_actions_cdq(
                        rollout_data.observations, actions
                    )
                    state_values = state_values.flatten()
                    q1_taken = q1_taken.flatten()
                    q2_taken = q2_taken.flatten()

                    if collecting:
                        # Separate backward passes avoid cross terms in shared
                        # K-FAC factors, matching the branch-wise treatment used
                        # by the ACKTR port.
                        pg_fisher_loss = log_prob.mean()
                        pg_fisher_loss.backward(retain_graph=True)
                        optimizer.zero_grad(set_to_none=True)

                        fisher_weight = self._cdq_fisher_weight()
                        sampled_q1 = q1_taken + th.randn_like(q1_taken)
                        q1_fisher_loss = (
                            -self.vf_fisher_coef
                            * fisher_weight
                            * th.mean((q1_taken - sampled_q1.detach()).pow(2))
                        )
                        q1_fisher_loss.backward(retain_graph=True)
                        optimizer.zero_grad(set_to_none=True)

                        sampled_q2 = q2_taken + th.randn_like(q2_taken)
                        q2_fisher_loss = (
                            -self.vf_fisher_coef
                            * fisher_weight
                            * th.mean((q2_taken - sampled_q2.detach()).pow(2))
                        )
                        q2_fisher_loss.backward(retain_graph=True)

                        fisher_loss = (
                            pg_fisher_loss.detach()
                            + q1_fisher_loss.detach()
                            + q2_fisher_loss.detach()
                        )
                        optimizer.complete_fisher_stats_collection()
                        optimizer.zero_grad(set_to_none=True)

                    policy_loss = -(advantages * log_prob).mean()
                    q1_loss = self._critic_loss(q1_taken, safe_returns)
                    q2_loss = self._critic_loss(q2_taken, safe_returns)
                    value_loss = self._combine_twin_losses(q1_loss, q2_loss)

                    if entropy is None:
                        entropy_loss = log_prob.mean()
                    else:
                        entropy_loss = -entropy.mean()

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                    )
                    loss.backward()

                    if self.cdq_always_clip_grad or is_cold:
                        self._clip_and_validate_gradients()

                    parameters = [
                        p for p in self.policy.parameters() if p.requires_grad
                    ]
                    snapshot = (
                        self._snapshot_parameters(parameters)
                        if self.cdq_rollback_on_nonfinite
                        else []
                    )
                    try:
                        optimizer.step()
                    except FloatingPointError as exc:
                        if snapshot:
                            self._restore_parameters(parameters, snapshot)
                        self._reset_optimizer_numerics(optimizer)
                        self.cdq_nonfinite_updates += 1
                        warnings.warn(
                            f"Skipped unstable K-FAC update: {exc}",
                            RuntimeWarning,
                        )
                        continue

                    if not self._parameters_are_finite(parameters):
                        if snapshot:
                            self._restore_parameters(parameters, snapshot)
                        self._reset_optimizer_numerics(optimizer)
                        self.cdq_nonfinite_updates += 1
                        warnings.warn(
                            "Rolled back a K-FAC update that produced NaN/Inf "
                            "parameters.",
                            RuntimeWarning,
                        )
                        continue
                except Exception:
                    if getattr(optimizer, "collecting_fisher_stats", False):
                        optimizer.discard_fisher_stats_collection()
                    raise

                optimizer_steps += 1
                last_policy_loss = policy_loss.detach()
                last_q1_loss = q1_loss.detach()
                last_q2_loss = q2_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy_loss = entropy_loss.detach()
                last_fisher_loss = fisher_loss
                last_gap = (q1_taken - q2_taken).abs().mean().detach()
                last_clipped_value = state_values.mean().detach()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/use_kfac", 1)
        self.logger.record("train/use_cdq", 1)
        self.logger.record("train/cdq_nonfinite_updates", self.cdq_nonfinite_updates)
        self.logger.record("train/optimizer_steps", optimizer_steps)
        self.logger.record("train/repeats_per_rollout", configured_repeats)
        self.logger.record(
            "train/sb2_exact_repeats",
            int(self.optimizer_steps_per_rollout is None),
        )
        if last_lr is not None:
            self.logger.record("train/learning_rate", last_lr)

        if last_policy_loss is not None:
            self.logger.record("train/policy_loss", last_policy_loss.item())
            self.logger.record("train/q1_loss", last_q1_loss.item())
            self.logger.record("train/q2_loss", last_q2_loss.item())
            self.logger.record("train/value_loss", last_value_loss.item())
            self.logger.record("train/entropy_loss", last_entropy_loss.item())
            self.logger.record("train/cdq_q_gap", last_gap.item())
            self.logger.record(
                "train/cdq_clipped_state_value", last_clipped_value.item()
            )
        if last_fisher_loss is not None:
            self.logger.record("train/fisher_loss", last_fisher_loss.item())

        self.logger.record("train/kfac_v_f_v", optimizer.last_v_f_v)
        self.logger.record("train/kfac_kl_scale", optimizer.last_kl_scale)
        self.logger.record("train/kfac_stats_steps", optimizer.stats_steps)
        self.logger.record("train/kfac_factor_steps", optimizer.factor_steps)
        self.logger.record("train/kfac_eigh_failures", optimizer.eigh_failures)
        self.logger.record("train/kfac_pure_mps", int(optimizer.pure_mps))

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())


def audit_cdq_policy(policy: nn.Module) -> dict[str, Any]:
    """Validate the asymmetric shared-Q1/independent-Q2 policy structure."""
    action_space = getattr(policy, "action_space", None)
    if not isinstance(action_space, spaces.Discrete):
        raise RuntimeError(
            "CDQ policy audit requires a Discrete action space; got "
            f"{action_space!r}."
        )

    q1 = getattr(policy, "value_net", None)
    q2 = getattr(policy, "q_net_2", None)
    q2_extractor = getattr(policy, "q2_features_extractor", None)
    q2_body = getattr(policy, "q2_critic_net", None)
    shared_extractor = getattr(policy, "features_extractor", None)

    if not isinstance(q1, nn.Linear) or not isinstance(q2, nn.Linear):
        raise RuntimeError("CDQ policy must contain Linear Q-heads value_net and q_net_2.")
    if not isinstance(q2_extractor, nn.Module) or not isinstance(q2_body, nn.Module):
        raise RuntimeError("CDQ policy is missing the independent Q2 network.")
    if q2_extractor is shared_extractor:
        raise RuntimeError("Q2 incorrectly reuses the actor/Q1 feature extractor object.")

    action_dim = int(action_space.n)
    if int(q1.out_features) != action_dim or int(q2.out_features) != action_dim:
        raise RuntimeError(
            "CDQ Q-head output dimensions must equal the number of actions: "
            f"Q1={q1.out_features}, Q2={q2.out_features}, actions={action_dim}."
        )
    if int(q1.in_features) != int(q2.in_features):
        raise RuntimeError("CDQ Q-heads must have matching latent dimensions.")

    actor_q1_modules = (
        shared_extractor,
        getattr(policy, "mlp_extractor"),
        getattr(policy, "action_net"),
        q1,
    )
    q2_modules = (q2_extractor, q2_body, q2)
    actor_q1_ids = {
        id(parameter)
        for module in actor_q1_modules
        if isinstance(module, nn.Module)
        for parameter in module.parameters()
    }
    q2_ids = {
        id(parameter)
        for module in q2_modules
        for parameter in module.parameters()
    }
    overlap = actor_q1_ids.intersection(q2_ids)
    if overlap:
        raise RuntimeError(
            f"Q2 shares {len(overlap)} Parameter objects with actor/Q1."
        )

    optimizer = getattr(policy, "optimizer", None)
    optimizer_param_ids: set[int] = set()
    if optimizer is not None:
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group.get("params", [])
        }
        missing = []
        for module_name, module in (
            ("q1", q1),
            ("q2_extractor", q2_extractor),
            ("q2_body", q2_body),
            ("q2_head", q2),
        ):
            for parameter_name, parameter in module.named_parameters():
                if id(parameter) not in optimizer_param_ids:
                    missing.append(f"{module_name}.{parameter_name}")
        if missing:
            raise RuntimeError(
                "CDQ optimizer is missing critic parameters: " + ", ".join(missing)
            )

    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    total_parameters = int(sum(parameter.numel() for parameter in policy.parameters()))
    return {
        "version": ACKTR_CDQ_VERSION,
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "action_dim": action_dim,
        "q_input_dim": int(q1.in_features),
        "q1_output_dim": int(q1.out_features),
        "q2_output_dim": int(q2.out_features),
        "q1_shares_actor_feature_extractor": True,
        "q2_fully_independent": True,
        "shared_parameter_count_with_q2": 0,
        "q2_feature_parameters": count(q2_extractor),
        "q2_body_parameters": count(q2_body),
        "q2_head_parameters": count(q2),
        "total_parameters": total_parameters,
        "optimizer_contains_q2_network": bool(optimizer_param_ids),
        "clipped_bootstrap": "sum_a pi(a|s) * min(Q1(s,a), Q2(s,a))",
    }


# Convenient aliases for package-style imports.
ACKTRCDQHybrid = ACKTRCDQ
ACKTR_CDQ_HYBRID = ACKTRCDQ
HybridCDQMlpPolicy = CDQMlpPolicy
HybridCDQCnnPolicy = CDQCnnPolicy
ACKTR_CDQ = ACKTRCDQ
MlpPolicy = CDQMlpPolicy
CnnPolicy = CDQCnnPolicy


__all__ = [
    "ACKTRCDQ",
    "ACKTRCDQHybrid",
    "ACKTR_CDQ_HYBRID",
    "ACKTR_CDQ",
    "CDQMlpPolicy",
    "CDQCnnPolicy",
    "HybridCDQMlpPolicy",
    "HybridCDQCnnPolicy",
    "MlpPolicy",
    "CnnPolicy",
    "ACKTR_CDQ_VERSION",
    "audit_cdq_policy",
]
