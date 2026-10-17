"""ACKTR with periodic reinitialization of the policy's top layers.

The default reset keeps the convolutional/feature trunk intact and reinitializes
only the output heads:

* ``action_net`` (policy logits/mean head)
* ``value_net`` (value head)
* ``q_net_2`` when using the optional ACKTR-CDQ policy

Optimizer momentum and K-FAC statistics belonging to the reset layers are
cleared so stale curvature/momentum does not immediately overwrite the freshly
initialized parameters.

Example
-------

    from acktr_periodic_top_reset import ACKTRPeriodicTopReset

    model = ACKTRPeriodicTopReset(
        CustomCnnPolicy,
        env,
        top_reset_interval_timesteps=250_000,
        top_reset_first_timestep=250_000,
        top_reset_targets=("action_net", "value_net"),
        top_reset_kfac_scope="layers",
        device="mps",
    )

This is an experimental training intervention, not part of the original ACKTR
paper or Stable Baselines 2.
"""

from __future__ import annotations

from functools import partial
import math
from typing import Any, Iterable, Optional, Sequence, Type, Union
import warnings

import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

try:
    from sb3_contrib.acktr.acktr import ACKTR as _BaseACKTR
except (ImportError, ModuleNotFoundError):
    try:
        from sb3_contrib import ACKTR as _BaseACKTR  # type: ignore[attr-defined]
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "ACKTR was not found in sb3_contrib. Install the ACKTR-enabled "
            "sb3_contrib fork before importing this module."
        ) from exc


ACKTR_PERIODIC_RESET_VERSION = "acktr-periodic-top-reset-v2-mps-no-qr"


class PeriodicTopLayerResetMixin:
    """Mixin that periodically reinitializes selected top policy layers.

    Parameters added by this mixin
    ------------------------------
    top_reset_interval_timesteps:
        Number of environment timesteps between resets. ``0`` disables resets.
    top_reset_first_timestep:
        Timestep of the first reset. ``None`` means one full interval.
    top_reset_targets:
        Dotted module names under ``policy``. Missing optional targets such as
        ``q_net_2`` are ignored by default.
    top_reset_include_feature_projection:
        Also reset the final Linear layer of ``policy.features_extractor``.
        For the ACKTR-paper Nature CNN this is the 3136 -> 512 layer.
    top_reset_include_last_policy_mlp / top_reset_include_last_value_mlp:
        Also reset the final Linear layer of the corresponding MLP branch when
        one exists.
    top_reset_clear_optimizer_state:
        Clear momentum/RMSprop/Adam state for reset parameters.
    top_reset_kfac_scope:
        ``"layers"`` clears K-FAC factors only for reset modules, ``"all"``
        restarts all curvature statistics, and ``"none"`` leaves them intact.
    top_reset_max_count:
        Optional maximum number of resets.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        *args: Any,
        top_reset_interval_timesteps: int = 250_000,
        top_reset_first_timestep: Optional[int] = None,
        top_reset_targets: Sequence[str] = (
            "action_net",
            "value_net",
            "q_net_2",
        ),
        top_reset_include_feature_projection: bool = False,
        top_reset_include_last_policy_mlp: bool = False,
        top_reset_include_last_value_mlp: bool = False,
        top_reset_clear_optimizer_state: bool = True,
        top_reset_kfac_scope: str = "layers",
        top_reset_ignore_missing_targets: bool = True,
        top_reset_max_count: Optional[int] = None,
        top_reset_action_gain: float = 0.01,
        top_reset_value_gain: float = 1.0,
        top_reset_q_gain: float = 1.0,
        top_reset_hidden_gain: float = math.sqrt(2.0),
        top_reset_init_mode: str = "auto",
        top_reset_mps_orthogonal_max_vectors: int = 64,
        **kwargs: Any,
    ) -> None:
        interval = int(top_reset_interval_timesteps)
        if interval < 0:
            raise ValueError("top_reset_interval_timesteps must be >= 0.")

        first = interval if top_reset_first_timestep is None else int(
            top_reset_first_timestep
        )
        if interval > 0 and first <= 0:
            raise ValueError(
                "top_reset_first_timestep must be positive when resets are enabled."
            )

        scope = str(top_reset_kfac_scope).lower()
        if scope not in {"none", "layers", "all"}:
            raise ValueError(
                "top_reset_kfac_scope must be 'none', 'layers', or 'all'."
            )
        if top_reset_max_count is not None and int(top_reset_max_count) < 0:
            raise ValueError("top_reset_max_count must be None or >= 0.")

        targets = tuple(str(name).strip() for name in top_reset_targets)
        if any(not name for name in targets):
            raise ValueError("top_reset_targets cannot contain empty names.")

        self.top_reset_interval_timesteps = interval
        self.top_reset_first_timestep = first
        self.top_reset_targets = targets
        self.top_reset_include_feature_projection = bool(
            top_reset_include_feature_projection
        )
        self.top_reset_include_last_policy_mlp = bool(
            top_reset_include_last_policy_mlp
        )
        self.top_reset_include_last_value_mlp = bool(
            top_reset_include_last_value_mlp
        )
        self.top_reset_clear_optimizer_state = bool(
            top_reset_clear_optimizer_state
        )
        self.top_reset_kfac_scope = scope
        self.top_reset_ignore_missing_targets = bool(
            top_reset_ignore_missing_targets
        )
        self.top_reset_max_count = (
            None if top_reset_max_count is None else int(top_reset_max_count)
        )
        self.top_reset_action_gain = float(top_reset_action_gain)
        self.top_reset_value_gain = float(top_reset_value_gain)
        self.top_reset_q_gain = float(top_reset_q_gain)
        self.top_reset_hidden_gain = float(top_reset_hidden_gain)

        init_mode = str(top_reset_init_mode).strip().lower()
        if init_mode not in {"auto", "policy_orthogonal", "mps_native", "reset_parameters"}:
            raise ValueError(
                "top_reset_init_mode must be 'auto', 'policy_orthogonal', "
                "'mps_native', or 'reset_parameters'."
            )
        if int(top_reset_mps_orthogonal_max_vectors) < 1:
            raise ValueError("top_reset_mps_orthogonal_max_vectors must be >= 1.")
        self.top_reset_init_mode = init_mode
        self.top_reset_mps_orthogonal_max_vectors = int(
            top_reset_mps_orthogonal_max_vectors
        )

        self._top_reset_count = 0
        self._top_reset_next_timestep = first if interval > 0 else None
        self._top_reset_last_timestep: Optional[int] = None
        self._top_reset_last_modules: tuple[str, ...] = ()

        super().__init__(policy, env, *args, **kwargs)

    def learn(
        self,
        total_timesteps: int,
        *args: Any,
        reset_num_timesteps: bool = True,
        **kwargs: Any,
    ):
        # A fresh learn(reset_num_timesteps=True) starts a fresh reset schedule.
        # Continued training should use reset_num_timesteps=False, in which case
        # the saved schedule is preserved.
        if reset_num_timesteps:
            self._top_reset_count = 0
            self._top_reset_last_timestep = None
            self._top_reset_last_modules = ()
            self._top_reset_next_timestep = (
                self.top_reset_first_timestep
                if self.top_reset_interval_timesteps > 0
                else None
            )
        return super().learn(
            total_timesteps,
            *args,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs,
        )

    @staticmethod
    def _last_linear(module: Optional[nn.Module]) -> Optional[nn.Linear]:
        if module is None:
            return None
        result: Optional[nn.Linear] = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                result = child
        return result

    def _resolve_named_module(self, name: str) -> Optional[nn.Module]:
        policy = self.policy
        try:
            return policy.get_submodule(name)
        except (AttributeError, KeyError):
            # get_submodule exists in modern PyTorch. Keep a compatibility
            # fallback for older versions and simple top-level names.
            current: Any = policy
            for part in name.split("."):
                if not hasattr(current, part):
                    return None
                current = getattr(current, part)
            return current if isinstance(current, nn.Module) else None

    def _collect_reset_modules(self) -> list[tuple[str, nn.Module, float]]:
        modules: list[tuple[str, nn.Module, float]] = []
        seen: set[int] = set()

        def add(name: str, module: Optional[nn.Module], gain: float) -> None:
            if module is None or id(module) in seen:
                return
            seen.add(id(module))
            modules.append((name, module, float(gain)))

        for name in self.top_reset_targets:
            module = self._resolve_named_module(name)
            if module is None:
                if self.top_reset_ignore_missing_targets:
                    continue
                raise AttributeError(
                    f"Policy {type(self.policy).__name__} has no reset target {name!r}."
                )
            if name.endswith("action_net") or name == "action_net":
                gain = self.top_reset_action_gain
            elif name.endswith("q_net_2") or name == "q_net_2":
                gain = self.top_reset_q_gain
            elif name.endswith("value_net") or name == "value_net":
                gain = self.top_reset_value_gain
            else:
                gain = self.top_reset_hidden_gain
            add(name, module, gain)

        if self.top_reset_include_feature_projection:
            extractor = getattr(self.policy, "features_extractor", None)
            linear = self._last_linear(extractor)
            add("features_extractor.<last_linear>", linear, self.top_reset_hidden_gain)

        mlp_extractor = getattr(self.policy, "mlp_extractor", None)
        if self.top_reset_include_last_policy_mlp:
            policy_branch = getattr(mlp_extractor, "policy_net", None)
            add(
                "mlp_extractor.policy_net.<last_linear>",
                self._last_linear(policy_branch),
                self.top_reset_hidden_gain,
            )
        if self.top_reset_include_last_value_mlp:
            value_branch = getattr(mlp_extractor, "value_net", None)
            add(
                "mlp_extractor.value_net.<last_linear>",
                self._last_linear(value_branch),
                self.top_reset_hidden_gain,
            )

        if not modules:
            raise RuntimeError(
                "Periodic reset was triggered, but no resettable modules were found."
            )
        return modules

    @staticmethod
    def _module_device(module: nn.Module) -> th.device:
        for parameter in module.parameters(recurse=True):
            return parameter.device
        for buffer in module.buffers(recurse=True):
            return buffer.device
        return th.device("cpu")

    @staticmethod
    def _zero_bias(child: nn.Module) -> None:
        bias = getattr(child, "bias", None)
        if isinstance(bias, th.Tensor):
            nn.init.zeros_(bias)

    def _mps_native_orthogonal_or_xavier_(
        self,
        weight: th.Tensor,
        gain: float,
    ) -> None:
        """Initialize without QR/SVD and without moving tensors off MPS.

        For small top heads, modified Gram-Schmidt gives an orthogonal matrix
        using only basic MPS tensor operations. For large matrices (for example
        the optional 3136 -> 512 feature projection), Xavier normal is used to
        avoid hundreds of serial Gram-Schmidt iterations.
        """
        if weight.ndim < 2:
            nn.init.normal_(weight, mean=0.0, std=float(gain))
            return

        rows = int(weight.shape[0])
        cols = int(weight.numel() // max(1, rows))
        flat = weight.reshape(rows, cols)
        max_vectors = self.top_reset_mps_orthogonal_max_vectors
        eps = th.finfo(weight.dtype).eps

        if rows <= cols and rows <= max_vectors:
            vectors: list[th.Tensor] = []
            for row_index in range(rows):
                vector = th.randn_like(flat[row_index])
                # Two passes improve numerical orthogonality without QR.
                for _ in range(2):
                    for previous in vectors:
                        vector = vector - th.sum(vector * previous) * previous
                vector = vector / th.sqrt(th.sum(vector * vector)).clamp_min(eps)
                vectors.append(vector)
            flat.copy_(th.stack(vectors, dim=0) * float(gain))
            return

        if cols < rows and cols <= max_vectors:
            transposed = flat.transpose(0, 1)
            vectors = []
            for row_index in range(cols):
                vector = th.randn_like(transposed[row_index])
                for _ in range(2):
                    for previous in vectors:
                        vector = vector - th.sum(vector * previous) * previous
                vector = vector / th.sqrt(th.sum(vector * vector)).clamp_min(eps)
                vectors.append(vector)
            transposed.copy_(th.stack(vectors, dim=0) * float(gain))
            return

        # QR-free fallback for large matrices. This is MPS-native and preserves
        # the requested gain, although it is not exactly orthogonal.
        nn.init.xavier_normal_(weight, gain=float(gain))

    def _mps_native_initialize_module(self, module: nn.Module, gain: float) -> int:
        initialized = 0
        for child in module.modules():
            if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._mps_native_orthogonal_or_xavier_(child.weight, gain)
                self._zero_bias(child)
                initialized += 1
            elif isinstance(child, nn.Embedding):
                nn.init.normal_(child.weight, mean=0.0, std=float(gain))
                initialized += 1
            elif isinstance(
                child,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                ),
            ):
                weight = getattr(child, "weight", None)
                bias = getattr(child, "bias", None)
                if isinstance(weight, th.Tensor):
                    nn.init.ones_(weight)
                if isinstance(bias, th.Tensor):
                    nn.init.zeros_(bias)
                initialized += 1
        return initialized

    def _reset_parameters_initialize_module(self, module: nn.Module) -> int:
        reset_count = 0
        # Only initialize leaf modules. Calling reset_parameters() on both a
        # container and its children could initialize the same tensor twice.
        for child in module.modules():
            if any(True for _ in child.children()):
                continue
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
                reset_count += 1
        return reset_count

    def _reinitialize_module(self, module: nn.Module, gain: float) -> None:
        policy_init = getattr(self.policy, "init_weights", None)
        ortho_init = bool(getattr(self.policy, "ortho_init", False))
        device = self._module_device(module)
        mode = self.top_reset_init_mode

        if mode == "auto":
            # nn.init.orthogonal_ calls linalg_qr, which is unsupported by MPS.
            if device.type == "mps":
                mode = "mps_native"
            elif ortho_init and callable(policy_init):
                mode = "policy_orthogonal"
            else:
                mode = "reset_parameters"

        with th.no_grad():
            if mode == "policy_orthogonal":
                if device.type == "mps":
                    raise RuntimeError(
                        "top_reset_init_mode='policy_orthogonal' cannot run on MPS "
                        "because torch orthogonal initialization uses linalg_qr. "
                        "Use top_reset_init_mode='mps_native' or 'auto'."
                    )
                if not (ortho_init and callable(policy_init)):
                    raise RuntimeError(
                        "policy_orthogonal was requested, but the policy does not "
                        "provide orthogonal initialization."
                    )
                module.apply(partial(policy_init, gain=float(gain)))
                return

            if mode == "mps_native":
                initialized = self._mps_native_initialize_module(module, float(gain))
                if initialized == 0:
                    raise TypeError(
                        f"Reset target {type(module).__name__} has no supported "
                        "MPS-native resettable layers."
                    )
                return

            if mode == "reset_parameters":
                initialized = self._reset_parameters_initialize_module(module)
                if initialized == 0:
                    raise TypeError(
                        f"Reset target {type(module).__name__} has no reset_parameters()."
                    )
                return

            raise AssertionError(f"Unhandled top reset init mode: {mode!r}")

    @staticmethod
    def _parameters_from_modules(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters(recurse=True):
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    params.append(parameter)
        return params

    def _clear_optimizer_state(
        self,
        reset_modules: Sequence[nn.Module],
        reset_parameters: Sequence[nn.Parameter],
    ) -> None:
        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is None:
            return

        reset_param_ids = {id(parameter) for parameter in reset_parameters}

        if self.top_reset_clear_optimizer_state:
            for parameter in reset_parameters:
                optimizer.state.pop(parameter, None)
                parameter.grad = None

        scope = self.top_reset_kfac_scope
        if scope == "none":
            return

        kfac_modules = list(getattr(optimizer, "modules", []))
        module_states = getattr(optimizer, "module_states", None)
        pending_pairs = getattr(optimizer, "_pending_pairs", None)
        if not kfac_modules or not isinstance(module_states, dict):
            return

        if scope == "all":
            affected = kfac_modules
        else:
            affected = []
            for module in kfac_modules:
                module_param_ids = {
                    id(parameter)
                    for parameter in module.parameters(recurse=False)
                }
                if module_param_ids & reset_param_ids:
                    affected.append(module)

        for module in affected:
            old_state = module_states.get(module)
            if old_state is not None:
                try:
                    cleared_state = type(old_state)()
                except TypeError:
                    # Defensive fallback for a custom state class requiring
                    # constructor arguments.
                    cleared_state = old_state

                # Do not assume that a third-party state class initializes its
                # tensors to None. Explicitly clear every known curvature field.
                for field_name in (
                    "A",
                    "G",
                    "A_inv",
                    "G_inv",
                    "A_eigvals",
                    "A_eigvecs",
                    "G_eigvals",
                    "G_eigvecs",
                ):
                    if hasattr(cleared_state, field_name):
                        setattr(cleared_state, field_name, None)
                if hasattr(cleared_state, "stats_count"):
                    cleared_state.stats_count = 0
                module_states[module] = cleared_state
            if isinstance(pending_pairs, dict) and module in pending_pairs:
                pending_pairs[module].clear()

        if scope == "all":
            if hasattr(optimizer, "factor_steps"):
                optimizer.factor_steps = 0
            if hasattr(optimizer, "stats_steps"):
                optimizer.stats_steps = 0
            if hasattr(optimizer, "last_v_f_v"):
                optimizer.last_v_f_v = 0.0
            if hasattr(optimizer, "last_kl_scale"):
                optimizer.last_kl_scale = 1.0
            if hasattr(optimizer, "collecting_fisher_stats"):
                optimizer.collecting_fisher_stats = False

        # Some K-FAC implementations store one fused weight+bias momentum
        # buffer under the weight parameter only. Clearing every parameter in
        # reset_parameters above safely removes both standard and fused state.

    def reset_top_layers_now(self) -> dict[str, Any]:
        """Immediately reset configured layers and return an audit report."""
        selected = self._collect_reset_modules()
        modules = [module for _name, module, _gain in selected]
        parameters = self._parameters_from_modules(modules)

        # Reinitialize first, then remove stale optimizer/curvature state.
        for _name, module, gain in selected:
            self._reinitialize_module(module, gain)
        self._clear_optimizer_state(modules, parameters)

        self._top_reset_count += 1
        self._top_reset_last_timestep = int(getattr(self, "num_timesteps", 0))
        self._top_reset_last_modules = tuple(name for name, _module, _gain in selected)

        return {
            "version": ACKTR_PERIODIC_RESET_VERSION,
            "reset_count": int(self._top_reset_count),
            "timestep": int(self._top_reset_last_timestep),
            "modules": list(self._top_reset_last_modules),
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "kfac_scope": self.top_reset_kfac_scope,
            "optimizer_state_cleared": self.top_reset_clear_optimizer_state,
            "init_mode": self.top_reset_init_mode,
        }

    def _reset_limit_reached(self) -> bool:
        maximum = self.top_reset_max_count
        return maximum is not None and self._top_reset_count >= maximum

    def _maybe_reset_top_layers(self) -> Optional[dict[str, Any]]:
        interval = self.top_reset_interval_timesteps
        next_timestep = self._top_reset_next_timestep
        if interval <= 0 or next_timestep is None or self._reset_limit_reached():
            return None

        current = int(getattr(self, "num_timesteps", 0))
        if current < int(next_timestep):
            return None

        report = self.reset_top_layers_now()
        # A reset can only happen between rollouts. Schedule the next reset one
        # full interval after the actual reset rather than firing repeatedly if
        # one rollout crossed several requested thresholds.
        self._top_reset_next_timestep = current + interval
        return report

    def train(self) -> None:
        # Consume the current rollout using the policy that generated it. Reset
        # only afterwards so the next rollout is produced by the fresh heads.
        super().train()
        report = self._maybe_reset_top_layers()

        self.logger.record("train/top_reset_enabled", int(self.top_reset_interval_timesteps > 0))
        self.logger.record("train/top_reset_count", int(self._top_reset_count))
        self.logger.record("train/top_reset_happened", int(report is not None))
        if self._top_reset_next_timestep is not None:
            self.logger.record(
                "train/top_reset_next_timestep",
                int(self._top_reset_next_timestep),
            )
        if report is not None:
            self.logger.record(
                "train/top_reset_parameter_count",
                int(report["parameter_count"]),
            )


class ACKTRPeriodicTopReset(PeriodicTopLayerResetMixin, _BaseACKTR):
    """Standard ACKTR with periodic top-layer resets."""


# Optional CDQ variant. The module remains importable without acktr_cdq.py.
try:
    from acktr_cdq import ACKTRCDQ as _ACKTRCDQ
except (ImportError, ModuleNotFoundError):
    _ACKTRCDQ = None


if _ACKTRCDQ is not None:

    class ACKTRCDQPeriodicTopReset(PeriodicTopLayerResetMixin, _ACKTRCDQ):
        """ACKTR-CDQ with periodic action/Q-head resets."""

else:
    ACKTRCDQPeriodicTopReset = None  # type: ignore[assignment,misc]


ACKTRPeriodicReset = ACKTRPeriodicTopReset
ACKTRReset = ACKTRPeriodicTopReset


__all__ = [
    "ACKTRPeriodicTopReset",
    "ACKTRPeriodicReset",
    "ACKTRReset",
    "ACKTRCDQPeriodicTopReset",
    "PeriodicTopLayerResetMixin",
    "ACKTR_PERIODIC_RESET_VERSION",
]
