"""Multi-modal belief prediction for observation delay compensation.

Supports proprioceptive state, images, and custom modalities via a
registry-based plugin system.

Architecture hierarchy::

    BaseBeliefPredictor (ABC)
    ├── StateBeliefPredictor  — GRU on raw state vectors
    ├── ImageBeliefPredictor  — CNN encoder → GRU → CNN decoder
    └── (register your own via @register_belief_predictor)

    SharedBackboneBeliefPredictor — shared GRU backbone with
        modality-specific encoders/decoders.  All modalities are
        processed in a **single forward pass**, sharing the temporal
        backbone and saving compute vs. per-modality models.

    MultiModalBeliefPredictor — (legacy) routes each obs key prefix
                                 to a separate sub-predictor.

Custom predictor example::

    from lerobot.utils.belief_predictor import (
        BaseBeliefPredictor, register_belief_predictor,
    )

    @register_belief_predictor("my_predictor")
    class MyPredictor(BaseBeliefPredictor):
        CONFIG_CLS = MyConfig   # dataclass with save(path) / load(path)
        def __init__(self, cfg: MyConfig): ...
        def forward(self, delayed_obs, action_seq): ...
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

_PREDICTOR_REGISTRY: dict[str, type["BaseBeliefPredictor"]] = {}


def register_belief_predictor(name: str):
    """Class decorator to register a belief predictor type.

    Usage::

        @register_belief_predictor("my_type")
        class MyPredictor(BaseBeliefPredictor):
            CONFIG_CLS = MyConfig
            ...
    """

    def _decorator(cls: type["BaseBeliefPredictor"]):
        if name in _PREDICTOR_REGISTRY:
            logger.warning(f"Overwriting belief predictor registry entry '{name}'")
        _PREDICTOR_REGISTRY[name] = cls
        cls._registry_name = name  # noqa: SLF001
        return cls

    return _decorator


def get_belief_predictor_class(name: str) -> type["BaseBeliefPredictor"]:
    """Look up a registered predictor class by name."""
    if name not in _PREDICTOR_REGISTRY:
        raise KeyError(
            f"Unknown belief predictor type '{name}'. "
            f"Registered types: {list(_PREDICTOR_REGISTRY)}"
        )
    return _PREDICTOR_REGISTRY[name]


def list_belief_predictors() -> list[str]:
    """Return the names of all registered predictor types."""
    return list(_PREDICTOR_REGISTRY)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base
# ═══════════════════════════════════════════════════════════════════════════

class BaseBeliefPredictor(nn.Module, ABC):
    """Abstract base for modality-specific belief predictors.

    Subclass contract:
        1. Set ``CONFIG_CLS`` to a dataclass with ``save(path)`` and
           class-method ``load(path)``.
        2. Store the config as ``self.cfg`` in ``__init__``.
        3. Implement ``forward(delayed_obs, action_seq) -> predicted_obs``
           where the output has the **same shape** as ``delayed_obs``.
    """

    CONFIG_CLS: type = None  # override in subclass
    _registry_name: str = ""

    @abstractmethod
    def forward(self, delayed_obs: Tensor, action_seq: Tensor) -> Tensor:
        """
        Args:
            delayed_obs: Delayed observation (batch, ...).
            action_seq: ``(batch, seq_len, action_dim)`` actions during delay.
        Returns:
            Predicted current observation, **same shape** as *delayed_obs*.
        """
        ...

    def predict(self, delayed_obs: Tensor, actions: list[Tensor]) -> Tensor:
        """Convenience: predict from a list of action tensors (oldest-first)."""
        if not actions:
            return delayed_obs.clone()
        action_seq = torch.stack(actions, dim=1)
        return self.forward(delayed_obs, action_seq)


# ═══════════════════════════════════════════════════════════════════════════
# State Belief Predictor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StateBeliefConfig:
    """Hyperparameters for the state (vector) belief predictor."""

    type: str = "state"
    state_dim: int = 8       # LIBERO: eef_pos(3) + axisangle(3) + gripper_qpos(2)
    action_dim: int = 7      # LIBERO: Δpos(3) + Δaxisangle(3) + gripper_cmd(1)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StateBeliefConfig":
        with open(path) as f:
            d = json.load(f)
        d.pop("type", None)
        return cls(**d)


@register_belief_predictor("state")
class StateBeliefPredictor(BaseBeliefPredictor):
    """GRU-based state predictor with residual connection.

    ``predicted_state = delayed_state + decoder(GRU(actions, h0=encode(delayed_state)))``
    """

    CONFIG_CLS = StateBeliefConfig

    def __init__(self, cfg: StateBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.num_layers),
        )
        self.gru = nn.GRU(
            input_size=cfg.action_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )

    def forward(self, delayed_state: Tensor, action_seq: Tensor) -> Tensor:
        B = delayed_state.shape[0]
        h0 = self.state_encoder(delayed_state)
        h0 = h0.view(B, self.cfg.num_layers, self.cfg.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()

        if action_seq.shape[1] > 0:
            _, h_final = self.gru(action_seq, h0)
        else:
            h_final = h0

        correction = self.decoder(h_final[-1])
        return delayed_state + correction


# ═══════════════════════════════════════════════════════════════════════════
# Image Belief Predictor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageBeliefConfig:
    """Hyperparameters for the image belief predictor."""

    type: str = "image"
    image_channels: int = 3
    image_height: int = 224
    image_width: int = 224
    action_dim: int = 7
    latent_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ImageBeliefConfig":
        with open(path) as f:
            d = json.load(f)
        d.pop("type", None)
        return cls(**d)


@register_belief_predictor("image")
class ImageBeliefPredictor(BaseBeliefPredictor):
    """CNN + GRU image predictor with residual connection.

    Architecture::

        delayed_image → CNN encoder → latent → project → GRU h0
                                                           │
        action_sequence ──────────── GRU ──────────────────┘
                                      │
                               h_final[-1] → project → latent'
                                                           │
        predicted_image = delayed_image + CNN decoder(latent')
    """

    CONFIG_CLS = ImageBeliefConfig

    def __init__(self, cfg: ImageBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Encoder --------------------------------------------------
        enc_layers: list[nn.Module] = []
        ch_in = cfg.image_channels
        for ch_out in cfg.encoder_channels:
            enc_layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder_fc = nn.Linear(ch_in, cfg.latent_dim)

        # --- Temporal GRU --------------------------------------------
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.num_layers),
        )
        self.gru = nn.GRU(
            input_size=cfg.action_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.hidden_to_latent = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        # --- Decoder --------------------------------------------------
        n_stages = len(cfg.encoder_channels)
        self._dec_init_h = max(1, cfg.image_height // (2 ** n_stages))
        self._dec_init_w = max(1, cfg.image_width // (2 ** n_stages))
        dec_channels = list(reversed(cfg.encoder_channels))
        self._dec_init_ch = dec_channels[0]
        self.decoder_fc = nn.Linear(
            cfg.latent_dim,
            dec_channels[0] * self._dec_init_h * self._dec_init_w,
        )

        dec_layers: list[nn.Module] = []
        for i in range(len(dec_channels) - 1):
            dec_layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dec_channels[i], dec_channels[i + 1], 3, 1, 1),
                nn.BatchNorm2d(dec_channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        dec_layers += [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_channels[-1], cfg.image_channels, 3, 1, 1),
        ]
        self.decoder_conv = nn.Sequential(*dec_layers)

    def forward(self, delayed_image: Tensor, action_seq: Tensor) -> Tensor:
        """
        Args:
            delayed_image: ``(B, C, H, W)``
            action_seq:    ``(B, T, action_dim)``
        Returns:
            ``(B, C, H, W)``
        """
        B = delayed_image.shape[0]

        # Encode
        feat = self.encoder_conv(delayed_image)
        latent = self.encoder_fc(self.encoder_pool(feat).flatten(1))

        # GRU
        h0 = self.latent_to_hidden(latent)
        h0 = h0.view(B, self.cfg.num_layers, self.cfg.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()

        if action_seq.shape[1] > 0:
            _, h_final = self.gru(action_seq, h0)
        else:
            h_final = h0

        pred_latent = self.hidden_to_latent(h_final[-1])

        # Decode
        x = self.decoder_fc(pred_latent)
        x = x.view(B, self._dec_init_ch, self._dec_init_h, self._dec_init_w)
        residual = self.decoder_conv(x)
        if residual.shape[-2:] != delayed_image.shape[-2:]:
            residual = F.interpolate(
                residual, size=delayed_image.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        return delayed_image + residual


# ═══════════════════════════════════════════════════════════════════════════
# Shared-Backbone Multi-Modal Belief Predictor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModalitySpec:
    """Description of a single observation modality."""

    key: str           # e.g. "observation.state"
    type: str          # "state" or "image"
    # State modality
    state_dim: int = 0
    # Image modality
    image_channels: int = 3
    image_height: int = 256
    image_width: int = 256
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class SharedBackboneConfig:
    """Hyperparameters for the shared-backbone multi-modal predictor."""

    type: str = "shared_backbone"
    modalities: list[ModalitySpec] = field(default_factory=list)
    action_dim: int = 7
    latent_dim: int = 256      # Per-modality encoder output dim
    hidden_dim: int = 256      # Shared GRU hidden dim
    num_layers: int = 2
    dropout: float = 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": self.type,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "modalities": [asdict(m) for m in self.modalities],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SharedBackboneConfig":
        with open(path) as f:
            d = json.load(f)
        mods = [ModalitySpec(**m) for m in d.pop("modalities", [])]
        d.pop("type", None)
        return cls(modalities=mods, **d)


class _ImageEncoderHead(nn.Module):
    """CNN encoder producing a flat latent vector from an image."""

    def __init__(self, spec: ModalitySpec, latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = spec.image_channels
        for ch_out in spec.encoder_channels:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch_in, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.pool(self.conv(x)).flatten(1))


class _ImageDecoderHead(nn.Module):
    """CNN decoder producing a residual image from a latent vector."""

    def __init__(self, spec: ModalitySpec, latent_dim: int) -> None:
        super().__init__()
        n_stages = len(spec.encoder_channels)
        self._h = max(1, spec.image_height // (2 ** n_stages))
        self._w = max(1, spec.image_width // (2 ** n_stages))
        dec_ch = list(reversed(spec.encoder_channels))
        self._init_ch = dec_ch[0]
        self.fc = nn.Linear(latent_dim, dec_ch[0] * self._h * self._w)

        layers: list[nn.Module] = []
        for i in range(len(dec_ch) - 1):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dec_ch[i], dec_ch[i + 1], 3, 1, 1),
                nn.BatchNorm2d(dec_ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_ch[-1], spec.image_channels, 3, 1, 1),
        ]
        self.conv = nn.Sequential(*layers)
        self._target_h = spec.image_height
        self._target_w = spec.image_width

    def forward(self, z: Tensor) -> Tensor:
        x = self.fc(z).view(-1, self._init_ch, self._h, self._w)
        x = self.conv(x)
        if x.shape[-2:] != (self._target_h, self._target_w):
            x = F.interpolate(x, size=(self._target_h, self._target_w),
                              mode="bilinear", align_corners=False)
        return x


class SharedBackboneBeliefPredictor(nn.Module):
    """Multi-modal belief predictor with a **shared GRU backbone**.

    All modalities are processed in a single forward pass:

    1. Each modality's delayed observation is encoded to a fixed-size
       latent vector by its own encoder head.
    2. The latent vectors are concatenated and projected to the shared
       GRU's initial hidden state.
    3. The shared GRU processes the action sequence.
    4. The GRU output is projected back and split into per-modality
       latent vectors.
    5. Each modality's decoder head produces a residual that is added
       to the delayed observation.

    This is more efficient than separate per-modality GRUs because the
    temporal processing is shared, and cross-modal information can flow
    through the shared backbone.
    """

    def __init__(self, cfg: SharedBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._key_to_idx: dict[str, int] = {}
        total_latent = cfg.latent_dim * len(cfg.modalities)

        # --- Per-modality encoder/decoder heads -----------------------
        encoders: list[nn.Module] = []
        decoders: list[nn.Module] = []
        for i, spec in enumerate(cfg.modalities):
            self._key_to_idx[spec.key] = i
            if spec.type == "state":
                encoders.append(nn.Sequential(
                    nn.Linear(spec.state_dim, cfg.latent_dim),
                    nn.ReLU(),
                ))
                decoders.append(nn.Sequential(
                    nn.Linear(cfg.latent_dim, cfg.latent_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.latent_dim, spec.state_dim),
                ))
            elif spec.type == "image":
                encoders.append(_ImageEncoderHead(spec, cfg.latent_dim))
                decoders.append(_ImageDecoderHead(spec, cfg.latent_dim))
            else:
                raise ValueError(f"Unsupported modality type: {spec.type}")

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        # --- Shared GRU backbone --------------------------------------
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(total_latent, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.num_layers),
        )
        self.gru = nn.GRU(
            input_size=cfg.action_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.hidden_to_latent = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, total_latent),
        )

    @property
    def modality_keys(self) -> list[str]:
        return [s.key for s in self.cfg.modalities]

    def match_prefix(self, key: str) -> str | None:
        """Return the registered key matching *key*, or ``None``."""
        # Exact match first, then prefix match (longest first)
        if key in self._key_to_idx:
            return key
        for k in sorted(self._key_to_idx, key=len, reverse=True):
            if key.startswith(k):
                return k
        return None

    def forward_all(
        self,
        delayed_obs: dict[str, Tensor],
        action_seq: Tensor,
        *,
        lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Joint forward pass for all modalities.

        Args:
            delayed_obs: ``{key: tensor}`` for each modality.
            action_seq: ``(B, T, action_dim)`` actions during the delay window.
            lengths: Optional ``(B,)`` actual (unpadded) sequence lengths.
                When provided the GRU uses ``pack_padded_sequence`` so that
                zero-padded timesteps are ignored (important when a batch
                mixes different delay values).

        Returns:
            ``{key: predicted_tensor}`` for each modality.
        """
        B = action_seq.shape[0]

        # Encode each modality
        latents = []
        for i, spec in enumerate(self.cfg.modalities):
            latents.append(self.encoders[i](delayed_obs[spec.key]))
        concat_latent = torch.cat(latents, dim=-1)  # (B, total_latent)

        # Shared GRU
        h0 = self.latent_to_hidden(concat_latent)
        h0 = h0.view(B, self.cfg.num_layers, self.cfg.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()

        if action_seq.shape[1] > 0:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    action_seq, lengths.cpu(), batch_first=True,
                    enforce_sorted=False,
                )
                _, h_final = self.gru(packed, h0)
            else:
                _, h_final = self.gru(action_seq, h0)
        else:
            h_final = h0

        # Split back to per-modality latents
        out_latent = self.hidden_to_latent(h_final[-1])  # (B, total_latent)
        split_latents = out_latent.split(self.cfg.latent_dim, dim=-1)

        # Decode each modality (residual connection)
        results: dict[str, Tensor] = {}
        for i, spec in enumerate(self.cfg.modalities):
            residual = self.decoders[i](split_latents[i])
            results[spec.key] = delayed_obs[spec.key] + residual

        return results

    def forward(
        self,
        delayed_obs: dict[str, Tensor],
        action_seq: Tensor,
        lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Alias for :meth:`forward_all` (needed by DDP which calls ``forward()``)."""
        return self.forward_all(delayed_obs, action_seq, lengths=lengths)

    def forward_single(
        self, key: str, delayed_obs: Tensor, action_seq: Tensor,
    ) -> Tensor:
        """Single-modality forward (for inference compatibility).

        Runs the full backbone but only returns the result for *key*.
        Missing modalities are zero-filled.
        """
        B = delayed_obs.shape[0]
        device = delayed_obs.device

        obs_dict: dict[str, Tensor] = {}
        for spec in self.cfg.modalities:
            if spec.key == key:
                obs_dict[spec.key] = delayed_obs
            elif spec.type == "state":
                obs_dict[spec.key] = torch.zeros(B, spec.state_dim, device=device)
            else:
                obs_dict[spec.key] = torch.zeros(
                    B, spec.image_channels, spec.image_height, spec.image_width,
                    device=device,
                )

        results = self.forward_all(obs_dict, action_seq)
        return results[key]

    def predict_all(
        self,
        delayed_obs: dict[str, Tensor],
        actions: list[Tensor],
    ) -> dict[str, Tensor]:
        """Predict all modalities from a list of action tensors."""
        if not actions:
            return {k: v.clone() for k, v in delayed_obs.items()}
        action_seq = torch.stack(actions, dim=1)
        return self.forward_all(delayed_obs, action_seq)

    def predict(
        self, key: str, delayed_obs: Tensor, actions: list[Tensor],
    ) -> Tensor:
        """Single-key predict (backward compat with MultiModalBeliefPredictor)."""
        if not actions:
            return delayed_obs.clone()
        action_seq = torch.stack(actions, dim=1)
        return self.forward_single(key, delayed_obs, action_seq)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Modal Belief Predictor (legacy — per-modality sub-predictors)
# ═══════════════════════════════════════════════════════════════════════════

class MultiModalBeliefPredictor(nn.Module):
    """Container that routes each observation key to its sub-predictor.

    Keys are matched **longest-prefix-first** against registered prefixes.

    Example::

        predictor = MultiModalBeliefPredictor({
            "observation.state":      state_predictor,
            "observation.images.top": image_predictor,
        })
        pred = predictor.predict("observation.state", delayed, actions)
    """

    def __init__(self, predictors: dict[str, BaseBeliefPredictor]) -> None:
        super().__init__()
        # nn.ModuleDict keys cannot contain dots → escape them.
        self._key_map: dict[str, str] = {}
        self._reverse_map: dict[str, str] = {}
        safe_dict: dict[str, nn.Module] = {}
        for prefix, pred in predictors.items():
            safe = prefix.replace(".", "__DOT__")
            safe_dict[safe] = pred
            self._key_map[safe] = prefix
            self._reverse_map[prefix] = safe

        self.predictors = nn.ModuleDict(safe_dict)
        self._sorted_prefixes = sorted(self._reverse_map, key=len, reverse=True)

    @property
    def modality_prefixes(self) -> list[str]:
        return list(self._reverse_map)

    def get_predictor(self, prefix: str) -> BaseBeliefPredictor:
        return self.predictors[self._reverse_map[prefix]]

    def match_prefix(self, key: str) -> str | None:
        """Return the longest registered prefix matching *key*, or ``None``."""
        for prefix in self._sorted_prefixes:
            if key.startswith(prefix):
                return prefix
        return None

    def predict(self, key: str, delayed_obs: Tensor, actions: list[Tensor]) -> Tensor:
        """Route *key* to matching sub-predictor. Returns unchanged if no match."""
        prefix = self.match_prefix(key)
        if prefix is None:
            return delayed_obs.clone()
        return self.get_predictor(prefix).predict(delayed_obs, actions)

    def forward(self, key: str, delayed_obs: Tensor, action_seq: Tensor) -> Tensor:
        prefix = self.match_prefix(key)
        if prefix is None:
            return delayed_obs.clone()
        return self.get_predictor(prefix)(delayed_obs, action_seq)


# ═══════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════

def save_belief_model(
    model: SharedBackboneBeliefPredictor | MultiModalBeliefPredictor | BaseBeliefPredictor,
    save_dir: str | Path,
    *,
    cfg: Any = None,
) -> None:
    """Save a belief model to *save_dir*.

    Supports ``SharedBackboneBeliefPredictor``,
    ``MultiModalBeliefPredictor`` (legacy), and single ``BaseBeliefPredictor``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, SharedBackboneBeliefPredictor):
        model.cfg.save(save_dir / "config.json")
        torch.save(model.state_dict(), save_dir / "model.pt")
        manifest = {
            "type": "shared_backbone",
            "modalities": {s.key: s.type for s in model.cfg.modalities},
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"SharedBackbone belief model saved to {save_dir}")

    elif isinstance(model, MultiModalBeliefPredictor):
        manifest: dict[str, Any] = {"type": "multi_modal", "modalities": {}}
        for prefix in model.modality_prefixes:
            pred = model.get_predictor(prefix)
            reg_name = getattr(pred, "_registry_name", type(pred).__name__)
            safe_name = prefix.replace(".", "_")
            sub_dir = save_dir / safe_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            pred.cfg.save(sub_dir / "config.json")
            torch.save(pred.state_dict(), sub_dir / "model.pt")
            manifest["modalities"][prefix] = {"type": reg_name, "dir": safe_name}
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Multi-modal belief model saved to {save_dir}")

    elif isinstance(model, BaseBeliefPredictor):
        c = cfg if cfg is not None else model.cfg
        c.save(save_dir / "config.json")
        torch.save(model.state_dict(), save_dir / "model.pt")
        logger.info(f"Belief model saved to {save_dir}")

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def load_belief_model(
    load_dir: str | Path,
    device: str | torch.device = "cpu",
) -> SharedBackboneBeliefPredictor | MultiModalBeliefPredictor:
    """Load a belief model from *load_dir*.

    Returns ``SharedBackboneBeliefPredictor`` for new-style checkpoints or
    ``MultiModalBeliefPredictor`` for legacy checkpoints.  Both support
    ``.match_prefix()`` and ``.predict(key, obs, actions)``.
    """
    load_dir = Path(load_dir)
    manifest_path = load_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        # --- SharedBackboneBeliefPredictor ---
        if manifest.get("type") == "shared_backbone":
            cfg = SharedBackboneConfig.load(load_dir / "config.json")
            model = SharedBackboneBeliefPredictor(cfg)
            sd = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
            model.load_state_dict(sd)
            model.to(device)
            model.eval()
            logger.info(
                f"SharedBackbone belief model loaded from {load_dir} "
                f"(modalities: {[s.key for s in cfg.modalities]})"
            )
            return model

        # --- Legacy MultiModalBeliefPredictor ---
        predictors: dict[str, BaseBeliefPredictor] = {}
        for prefix, info in manifest["modalities"].items():
            pred_cls = get_belief_predictor_class(info["type"])
            sub_dir = load_dir / info["dir"]
            sub_cfg = pred_cls.CONFIG_CLS.load(sub_dir / "config.json")
            pred = pred_cls(sub_cfg)
            sd = torch.load(sub_dir / "model.pt", map_location=device, weights_only=True)
            pred.load_state_dict(sd)
            predictors[prefix] = pred

        model = MultiModalBeliefPredictor(predictors)
        model.to(device)
        model.eval()
        logger.info(
            f"Multi-modal belief model loaded from {load_dir} "
            f"(modalities: {list(predictors)})"
        )
        return model

    elif (load_dir / "config.json").exists():
        # Legacy single state predictor → wrap for uniform interface
        legacy_cfg = StateBeliefConfig.load(load_dir / "config.json")
        pred = StateBeliefPredictor(legacy_cfg)
        sd = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
        pred.load_state_dict(sd)
        model = MultiModalBeliefPredictor({"observation.state": pred})
        model.to(device)
        model.eval()
        logger.info(
            f"Legacy state belief model loaded from {load_dir}, "
            "wrapped as MultiModalBeliefPredictor"
        )
        return model

    else:
        raise FileNotFoundError(f"No belief model found in {load_dir}")


def ensure_multi_modal(
    model: MultiModalBeliefPredictor | BaseBeliefPredictor,
    default_prefix: str = "observation.state",
) -> MultiModalBeliefPredictor:
    """Wrap a single-modality predictor if needed."""
    if isinstance(model, MultiModalBeliefPredictor):
        return model
    if isinstance(model, BaseBeliefPredictor):
        return MultiModalBeliefPredictor({default_prefix: model})
    raise TypeError(f"Unsupported belief model type: {type(model)}")


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases
# ═══════════════════════════════════════════════════════════════════════════
BeliefPredictorConfig = StateBeliefConfig
BeliefPredictor = StateBeliefPredictor
