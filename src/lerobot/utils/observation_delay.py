"""Observation delay buffer for evaluation under simulated sensor latency.

At time step t, instead of receiving the real-time observation o_t, the policy
receives a delayed observation o_{t-δ} where δ can differ per observation modality
(e.g. images vs. proprioceptive state).

When a belief predictor is provided, the delayed state is replaced by the
predicted current state:  ŝ_t = f_θ(s_{t-δ}, a_{t-δ}, …, a_{t-1}).
"""

from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Supported delay compensation modes.
COMPENSATION_NONE = "none"
COMPENSATION_BELIEF = "belief"
_VALID_COMPENSATION_MODES = {COMPENSATION_NONE, COMPENSATION_BELIEF}


@dataclass
class ObservationDelayBuffer:
    """Maintains per-key FIFO buffers to return delayed observations and
    optionally predicts the current state using a trained belief model.

    Args:
        delay_steps: Mapping from observation key prefix to delay amount (in
            environment steps).
        default_delay: Delay applied to keys that do not match any prefix.
        delay_compensation: ``"none"`` or ``"belief"``.  When ``"belief"``,
            the ``belief_model`` must be provided.
        belief_keys: Observation key prefixes whose delayed values should be
            replaced by belief predictions.
        belief_model: A trained ``BeliefPredictor`` instance.  Required when
            ``delay_compensation == "belief"``.
    """

    delay_steps: dict[str, int] = field(default_factory=dict)
    default_delay: int = 0
    delay_compensation: str = COMPENSATION_NONE
    belief_keys: list[str] = field(default_factory=lambda: ["observation.state"])
    belief_model: Any = field(default=None, repr=False)
    # Mapping from model modality keys to runtime observation keys.
    # Auto-detected on first use if not provided.
    belief_key_mapping: dict[str, str] | None = None
    # Debug: zero-out observations matching these key prefixes (e.g. ["observation.state"]).
    mask_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for k, v in self.delay_steps.items():
            if v < 0:
                raise ValueError(f"Delay for '{k}' must be non-negative, got {v}")
        if self.default_delay < 0:
            raise ValueError(f"default_delay must be non-negative, got {self.default_delay}")
        if self.delay_compensation not in _VALID_COMPENSATION_MODES:
            raise ValueError(
                f"Unknown delay_compensation mode '{self.delay_compensation}'. "
                f"Must be one of {_VALID_COMPENSATION_MODES}."
            )
        if self.delay_compensation == COMPENSATION_BELIEF and self.belief_model is None:
            raise ValueError(
                "delay_compensation='belief' requires a belief_model to be provided."
            )
        # Sort prefixes longest-first so that more specific prefixes match first.
        self._sorted_prefixes: list[tuple[str, int]] = sorted(
            self.delay_steps.items(), key=lambda x: len(x[0]), reverse=True
        )
        self._buffers: dict[str, deque] = {}
        # Global action history – stores the last ``max_delay`` actions.
        self._action_history: deque[Tensor] = deque(maxlen=max(self.max_delay, 1))
        # Lazily resolved mapping: model_key -> obs_key.
        self._resolved_key_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal buffers.  Call at the start of each episode."""
        self._buffers.clear()
        self._action_history.clear()

    def push_action(self, action: Tensor) -> None:
        """Record the action just executed by the policy.

        Args:
            action: Tensor of shape ``(batch, action_dim)``.
        """
        self._action_history.append(action.detach().clone())

    def delay_and_augment(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Delay observations and optionally apply belief prediction.

        This is the main entry point used in the rollout loop.
        """
        delayed_obs = self._apply_delay(observation)
        if self.delay_compensation == COMPENSATION_BELIEF:
            delayed_obs = self._apply_belief(delayed_obs)
        if self.mask_keys:
            delayed_obs = self._apply_mask(delayed_obs)
        return delayed_obs

    # Keep the old name available for backward compat / simple delay-only use.
    def delay(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Accept the current step's observation and return the delayed version."""
        return self._apply_delay(observation)

    def get_action_history(self, n: int | None = None) -> list[Tensor]:
        """Return the last *n* recorded actions (oldest-first)."""
        hist = list(self._action_history)
        if n is not None and n < len(hist):
            hist = hist[-n:]
        return hist

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_delay(self) -> int:
        """The largest delay across all keys."""
        if self.delay_steps:
            return max(max(self.delay_steps.values()), self.default_delay)
        return self.default_delay

    @property
    def has_delay(self) -> bool:
        return self.max_delay > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_delay_for_key(self, key: str) -> int:
        for prefix, delay in self._sorted_prefixes:
            if key.startswith(prefix):
                return delay
        return self.default_delay

    def _is_belief_key(self, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in self.belief_keys)

    def _resolve_key_mapping(self, observation: dict[str, Any]) -> dict[str, str]:
        """Return a mapping ``{model_key: obs_key}`` for all model modalities.

        Uses ``belief_key_mapping`` if provided; otherwise auto-detects by
        matching model modality keys to observation keys.  The mapping is
        cached for subsequent calls.
        """
        if self._resolved_key_map is not None:
            return self._resolved_key_map

        from lerobot.utils.belief_predictor import SharedBackboneBeliefPredictor

        if not isinstance(self.belief_model, SharedBackboneBeliefPredictor):
            return {}

        model_keys = self.belief_model.modality_keys
        obs_keys = set(observation.keys())

        # Start from user-supplied mapping, then fill gaps.
        mapping: dict[str, str] = dict(self.belief_key_mapping or {})

        for mk in model_keys:
            if mk in mapping:
                continue
            # Exact match
            if mk in obs_keys:
                mapping[mk] = mk
                continue
            # Heuristic: match by shared prefix + suffix.
            # e.g. model="observation.images.wrist_image",
            #       obs ="observation.images.image2"
            # Both share prefix "observation.images." — pick the unmatched
            # obs key with the same prefix that isn't already claimed.
            prefix = mk.rsplit(".", 1)[0] + "."  # "observation.images."
            candidates = [
                ok for ok in obs_keys
                if ok.startswith(prefix) and ok not in mapping.values()
            ]
            if len(candidates) == 1:
                mapping[mk] = candidates[0]
                logger.info(
                    f"Belief key mapping: model key '{mk}' → obs key '{candidates[0]}'"
                )
            else:
                logger.warning(
                    f"Cannot auto-map model key '{mk}' to observation keys "
                    f"(candidates={candidates}). The key will be skipped."
                )

        self._resolved_key_map = mapping
        return mapping

    def _apply_mask(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Zero-out observation tensors whose keys match any ``mask_keys`` prefix.

        This is a debug tool to test a policy's sensitivity to individual
        observation modalities.
        """
        for key in observation:
            if any(key.startswith(prefix) for prefix in self.mask_keys):
                val = observation[key]
                if isinstance(val, Tensor):
                    observation[key] = torch.zeros_like(val)
        return observation

    def _apply_delay(self, observation: dict[str, Any]) -> dict[str, Any]:
        delayed_obs: dict[str, Any] = {}
        for key, value in observation.items():
            d = self._get_delay_for_key(key)
            if d == 0:
                delayed_obs[key] = value
                continue

            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=d + 1)

            self._buffers[key].append(deepcopy(value))
            delayed_obs[key] = deepcopy(self._buffers[key][0])

        return delayed_obs

    def _apply_belief(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Replace delayed observations with belief predictions.

        Supports ``SharedBackboneBeliefPredictor`` (batch all modalities),
        ``MultiModalBeliefPredictor`` (key-by-key routing), and legacy
        single-modality ``BaseBeliefPredictor``.
        """
        from lerobot.utils.belief_predictor import SharedBackboneBeliefPredictor

        # --- SharedBackboneBeliefPredictor: single forward for all keys ---
        if isinstance(self.belief_model, SharedBackboneBeliefPredictor):
            # Collect all belief-eligible keys with non-zero delay
            eligible_keys = [
                key for key in observation
                if self._is_belief_key(key) and self._get_delay_for_key(key) > 0
            ]
            if not eligible_keys:
                return observation

            model_device = next(self.belief_model.parameters()).device
            key_map = self._resolve_key_mapping(observation)
            rev_map = {v: k for k, v in key_map.items()}

            # Group eligible keys by delay so each group gets the correct
            # number of actions.  When all delays are equal (the common case)
            # this collapses to a single forward pass.
            delay_groups: dict[int, list[str]] = {}
            for obs_key in eligible_keys:
                d = self._get_delay_for_key(obs_key)
                delay_groups.setdefault(d, []).append(obs_key)

            for delay_d, group_keys in delay_groups.items():
                actions = self.get_action_history(delay_d)
                if not actions:
                    continue

                # Build full model input (all model keys required).
                delayed_obs = {}
                original_devices = {}
                for model_key, obs_key in key_map.items():
                    if obs_key in observation:
                        original_devices[model_key] = observation[obs_key].device
                        delayed_obs[model_key] = observation[obs_key].to(model_device)

                with torch.no_grad():
                    actions_on_model = [a.to(model_device) for a in actions]
                    predicted = self.belief_model.predict_all(
                        delayed_obs, actions_on_model,
                    )

                # Write back only the keys belonging to this delay group.
                for obs_key in group_keys:
                    model_key = rev_map.get(obs_key, obs_key)
                    if model_key in predicted:
                        observation[obs_key] = predicted[model_key].to(
                            original_devices.get(model_key, "cpu")
                        )

            return observation

        # --- Legacy per-key prediction (MultiModal / BaseBeliefPredictor) ---
        for key in list(observation.keys()):
            if not self._is_belief_key(key):
                continue

            d = self._get_delay_for_key(key)
            if d == 0:
                continue

            actions = self.get_action_history(d)
            if not actions:
                continue

            delayed_obs = observation[key]
            device = delayed_obs.device

            model_device = next(self.belief_model.parameters()).device
            delayed_on_model = delayed_obs.to(model_device)
            actions_on_model = [a.to(model_device) for a in actions]

            with torch.no_grad():
                if hasattr(self.belief_model, "match_prefix"):
                    predicted = self.belief_model.predict(
                        key, delayed_on_model, actions_on_model,
                    )
                else:
                    predicted = self.belief_model.predict(
                        delayed_on_model, actions_on_model,
                    )

            observation[key] = predicted.to(device)

        return observation
