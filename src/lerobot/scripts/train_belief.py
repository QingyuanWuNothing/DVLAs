#!/usr/bin/env python
"""Train multi-modal belief predictors from existing LeRobot datasets.

Supports training belief predictors for **any combination of modalities**
(state vectors, images, or custom) simultaneously.  Multiple datasets can
be passed to ``--repo_id`` and their samples will be concatenated.

Usage examples::

    # Single-GPU training
    python -m lerobot.scripts.train_belief \
        --repo_id lerobot/libero_goal_image \
        --delays 1 2 4 8 16 32 \
        --epochs 100 \
        --output_dir outputs/belief

    # Multi-GPU training (4 GPUs via torchrun)
    torchrun --nproc_per_node=4 -m lerobot.scripts.train_belief \
        --repo_id lerobot/libero_spatial_image \
                  lerobot/libero_object_image \
                  lerobot/libero_goal_image \
                  lerobot/libero_10_image \
        --delays 1 2 4 8 16 32 \
        --output_dir outputs/belief_all

    # Multi-node via SLURM (see train_belief.sh)
    sbatch train_belief.sh

The trained model can be loaded during evaluation::

    lerobot-eval ... --delay_compensation=belief \
        --belief_model_path outputs/belief_all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.belief_predictor import (
    ModalitySpec,
    SharedBackboneBeliefPredictor,
    SharedBackboneConfig,
    save_belief_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Distributed training utilities
# ═══════════════════════════════════════════════════════════════════════════

def _is_dist_available() -> bool:
    """Return True when running inside a torchrun / DDP context."""
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_available() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist_available() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_ddp() -> None:
    """Initialize the process group if ``torchrun`` env vars are set."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════
# Training Dataset  (joint — all modalities in a single sample)
# ═══════════════════════════════════════════════════════════════════════════

class JointBeliefDataset(Dataset):
    """Extract ``(delayed_obs_dict, action_seq, target_obs_dict)`` tuples
    covering **all** observation modalities from a :class:`LeRobotDataset`.

    Each sample contains delayed and target observations for every modality
    key so that the shared-backbone model can process them in a single
    forward pass.

    Data access follows ``LeRobotDataset._query_hf_dataset``: fast
    column access ``hf_dataset[key][idx]`` with fallback to row access.
    Episode boundaries are handled the same way as
    :class:`~lerobot.datasets.sampler.EpisodeAwareSampler`.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        delays: list[int],
        obs_keys: list[str],
        action_key: str = "action",
    ) -> None:
        super().__init__()
        self.hf_dataset = lerobot_dataset.hf_dataset
        self.dataset = lerobot_dataset
        self.meta = lerobot_dataset.meta
        self.obs_keys = obs_keys
        self.action_key = action_key

        # Probe fast column access per key.
        self._fast: dict[str, bool] = {}
        for k in list(obs_keys) + [action_key]:
            self._fast[k] = self._check_fast_column_access(k)

        # Build sample index — same episode-boundary logic as
        # EpisodeAwareSampler (drop_n_first_frames = delay d).
        self.samples: list[tuple[int, int]] = []  # (frame_idx, delay)
        episodes = self.meta.episodes
        for ep_idx in range(self.meta.total_episodes):
            ep_start = episodes[ep_idx]["dataset_from_index"]
            ep_end = episodes[ep_idx]["dataset_to_index"]
            for d in delays:
                for t in range(ep_start + d, ep_end):
                    self.samples.append((t, d))

        logger.info(
            f"JointBeliefDataset: {len(self.samples)} samples, "
            f"obs_keys={obs_keys}"
        )

    def _check_fast_column_access(self, key: str) -> bool:
        try:
            return isinstance(self.hf_dataset[key][0], torch.Tensor)
        except (KeyError, TypeError, IndexError):
            return False

    def _query_field(self, key: str, idx: int) -> torch.Tensor:
        fast = self._fast.get(key, False)
        if fast:
            try:
                val = self.hf_dataset[key][idx]
            except (KeyError, TypeError, IndexError):
                val = self.hf_dataset[idx][key]
        else:
            val = self.dataset[idx][key]
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, dtype=torch.float32)
        if val.dim() >= 3 and val.shape[0] == 1:
            val = val.squeeze(0)
        return val.float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, d = self.samples[idx]

        sample: dict[str, torch.Tensor] = {
            "delay": torch.tensor(d, dtype=torch.long),
        }

        # Cache full-sample loads per frame index to avoid redundant
        # video decoding when multiple image keys share the same index.
        _row_cache: dict[int, dict[str, torch.Tensor]] = {}

        def _get(key: str, frame_idx: int) -> torch.Tensor:
            fast = self._fast.get(key, False)
            if fast:
                try:
                    val = self.hf_dataset[key][frame_idx]
                except (KeyError, TypeError, IndexError):
                    val = self.hf_dataset[frame_idx][key]
            else:
                if frame_idx not in _row_cache:
                    _row_cache[frame_idx] = self.dataset[frame_idx]
                val = _row_cache[frame_idx][key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            if val.dim() >= 3 and val.shape[0] == 1:
                val = val.squeeze(0)
            return val.float()

        # Actions during the delay window
        sample["action_seq"] = torch.stack(
            [_get(self.action_key, t - d + i) for i in range(d)]
        )
        # Delayed and target observations for every modality
        for key in self.obs_keys:
            sample[f"delayed.{key}"] = _get(key, t - d)
            sample[f"target.{key}"] = _get(key, t)

        return sample


# Backward-compat aliases
BeliefDataset = JointBeliefDataset
StateBeliefDataset = JointBeliefDataset
ImageBeliefDataset = JointBeliefDataset


def collate_joint(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate that pads action sequences and stacks all modality tensors.

    Returns an ``action_seq_lengths`` tensor so that the GRU can use
    ``pack_padded_sequence`` to skip zero-padded timesteps.
    """
    delays = torch.stack([s["delay"] for s in batch])

    lengths = torch.tensor([s["action_seq"].shape[0] for s in batch],
                           dtype=torch.long)
    max_len = int(lengths.max().item())
    action_dim = batch[0]["action_seq"].shape[-1]
    padded_actions = torch.zeros(len(batch), max_len, action_dim)
    for i, s in enumerate(batch):
        sl = s["action_seq"].shape[0]
        padded_actions[i, :sl] = s["action_seq"]

    result: dict[str, torch.Tensor] = {
        "delay": delays,
        "action_seq": padded_actions,
        "action_seq_lengths": lengths,
    }
    # Stack all delayed.* and target.* keys
    other_keys = [k for k in batch[0] if k not in ("delay", "action_seq")]
    for k in other_keys:
        result[k] = torch.stack([s[k] for s in batch])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Dimension helpers — read from ``meta.features`` (cf. ``meta.shapes``)
# ═══════════════════════════════════════════════════════════════════════════

def _get_feature_shape(meta: LeRobotDatasetMetadata, key: str) -> tuple[int, ...]:
    """Return shape tuple for *key* from dataset metadata.

    This reuses ``meta.features[key]["shape"]`` which is populated from
    the dataset's ``meta/info.json`` — the same source that
    :pyattr:`LeRobotDatasetMetadata.shapes` exposes.

    For image/video features the stored shape is (H, W, C) but tensors
    are (C, H, W) after ``hf_transform_to_torch``; this function returns
    the *tensor* shape.
    """
    feat = meta.features[key]
    shape = tuple(feat["shape"])
    # Image / video features are stored as (H, W, C) in metadata but
    # transposed to (C, H, W) by hf_transform_to_torch.
    if feat.get("dtype") in ("image", "video") and len(shape) == 3:
        H, W, C = shape
        return (C, H, W)
    return shape


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_belief(
    datasets: list[LeRobotDataset] | LeRobotDataset,
    delays: list[int],
    output_dir: Path,
    obs_keys: list[str] | None = None,
    obs_types: list[str] | None = None,
    action_key: str = "action",
    hidden_dim: int = 256,
    num_layers: int = 2,
    latent_dim: int = 256,
    encoder_channels: list[int] | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    debug_samples: int = 0,
) -> SharedBackboneBeliefPredictor:
    """Train a shared-backbone belief predictor for all modalities jointly.

    All modalities share a single GRU backbone. Modality-specific encoder
    and decoder heads project to/from the shared latent space. A single
    forward pass produces predictions for all modalities simultaneously.

    Args:
        datasets: One or more loaded :class:`LeRobotDataset` instances.
        delays: List of delay values (in steps) to train on.
        output_dir: Directory to save the trained model.
        obs_keys: Observation keys to train predictors for.
            Default: ``["observation.state", "observation.images.image",
            "observation.images.wrist_image"]``.
        obs_types: Type of each obs key: ``"state"`` or ``"image"``.
            Must be same length as *obs_keys*.
        action_key: Key for actions in the dataset.
        hidden_dim: Shared GRU hidden size.
        num_layers: Number of shared GRU layers.
        latent_dim: Per-modality latent dimension.
        encoder_channels: CNN channels for image encoder heads.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        epochs: Number of training epochs.
        batch_size: Batch size.
        val_split: Fraction of data used for validation.
        seed: Random seed.
        device: Training device.
        debug_samples: If > 0, subsample the dataset to this many samples
            for fast debugging.  Epochs default to 5 if not explicitly set.

    Returns:
        The trained ``SharedBackboneBeliefPredictor``.
    """
    if isinstance(datasets, LeRobotDataset):
        datasets = [datasets]

    if obs_keys is None:
        obs_keys = [
            "observation.state",
            "observation.images.image",
            "observation.images.wrist_image",
        ]
    if obs_types is None:
        obs_types = ["state", "image", "image"]
    if encoder_channels is None:
        encoder_channels = [32, 64, 128, 256]
    if len(obs_keys) != len(obs_types):
        raise ValueError(
            f"obs_keys ({len(obs_keys)}) and obs_types ({len(obs_types)}) "
            "must have the same length."
        )

    # --- Distributed setup --------------------------------------------
    _setup_ddp()
    is_distributed = _is_dist_available()
    rank = _get_rank()
    world_size = _get_world_size()
    is_main = _is_main_process()

    if is_distributed:
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        if is_main:
            logger.info(f"Distributed training: {world_size} GPUs")
    else:
        if is_main:
            logger.info("Single-process training")

    torch.manual_seed(seed + rank)  # stagger seeds across ranks
    random.seed(seed + rank)

    # --- Infer dimensions from metadata -------------------------------
    ref_meta = datasets[0].meta
    action_shape = _get_feature_shape(ref_meta, action_key)
    action_dim = action_shape[-1]
    logger.info(f"Detected action_dim={action_dim} from meta.features "
                f"({len(datasets)} dataset(s))")

    # --- Build ModalitySpec list --------------------------------------
    modality_specs: list[ModalitySpec] = []
    for obs_key, obs_type in zip(obs_keys, obs_types):
        obs_shape = _get_feature_shape(ref_meta, obs_key)
        if obs_type == "state":
            spec = ModalitySpec(
                key=obs_key, type="state", state_dim=obs_shape[-1],
            )
            logger.info(f"  [{obs_key}] state_dim={obs_shape[-1]}")
        elif obs_type == "image":
            if len(obs_shape) != 3:
                raise ValueError(f"Expected 3D image shape for '{obs_key}', got {obs_shape}")
            C, H, W = obs_shape
            spec = ModalitySpec(
                key=obs_key, type="image",
                image_channels=C, image_height=H, image_width=W,
                encoder_channels=encoder_channels,
            )
            logger.info(f"  [{obs_key}] image shape=({C}, {H}, {W})")
        else:
            raise ValueError(f"Unsupported obs_type '{obs_type}' for key '{obs_key}'")
        modality_specs.append(spec)

    # --- Build model --------------------------------------------------
    cfg = SharedBackboneConfig(
        modalities=modality_specs,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model = SharedBackboneBeliefPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"SharedBackboneBeliefPredictor: {n_params:,} params")

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    raw_model: SharedBackboneBeliefPredictor = (
        model.module if is_distributed else model
    )

    # --- Build joint dataset ------------------------------------------
    per_ds = [JointBeliefDataset(ds, delays, obs_keys=obs_keys, action_key=action_key)
              for ds in datasets]
    full_ds = ConcatDataset(per_ds) if len(per_ds) > 1 else per_ds[0]

    # Debug mode: keep only a small subset
    if debug_samples > 0:
        n_keep = min(debug_samples, len(full_ds))
        n_discard = len(full_ds) - n_keep
        full_ds, _ = torch.utils.data.random_split(
            full_ds, [n_keep, n_discard],
            generator=torch.Generator().manual_seed(seed),
        )
        if is_main:
            logger.info(f"DEBUG MODE: using {n_keep} / {n_keep + n_discard} samples")

    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),  # same split on all ranks
    )

    use_pin_memory = device != "cpu"
    nw = 4

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    ) if is_distributed else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_joint, num_workers=nw,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_joint, num_workers=nw,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if nw > 0 else None,
    )
    if is_main:
        logger.info(
            f"Dataset: train={n_train}, val={n_val}, "
            f"batch_size={batch_size} x {world_size} GPU(s)"
        )

    # --- Optimizer ----------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ------------------------------------------------
    epoch_bar = tqdm(
        range(1, epochs + 1), desc="Epochs", disable=not is_main,
        unit="ep", position=0,
    )
    for epoch in epoch_bar:
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffle across ranks

        train_loss_sum, train_count = 0.0, 0

        train_bar = tqdm(
            train_loader, desc=f"Train {epoch}/{epochs}",
            leave=False, disable=not is_main, unit="batch", position=1,
        )
        for batch in train_bar:
            actions = batch["action_seq"].to(device)
            B = actions.shape[0]

            # Build delayed_obs and target dicts
            delayed_obs = {k: batch[f"delayed.{k}"].to(device) for k in obs_keys}
            target_obs = {k: batch[f"target.{k}"].to(device) for k in obs_keys}

            lengths = batch["action_seq_lengths"].to(device)
            pred_obs = model(delayed_obs, actions, lengths=lengths)

            # Sum MSE across modalities
            loss = sum(loss_fn(pred_obs[k], target_obs[k]) for k in obs_keys)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_count += B
            avg_loss = train_loss_sum / max(train_count, 1)
            train_bar.set_postfix(loss=f"{avg_loss:.5f}")

        scheduler.step()

        # Validate
        model.eval()
        val_losses: dict[str, float] = {k: 0.0 for k in obs_keys}
        val_count = 0
        val_bar = tqdm(
            val_loader, desc=f"Val   {epoch}/{epochs}",
            leave=False, disable=not is_main, unit="batch", position=1,
        )
        with torch.no_grad():
            for batch in val_bar:
                actions = batch["action_seq"].to(device)
                B = actions.shape[0]
                delayed_obs = {k: batch[f"delayed.{k}"].to(device) for k in obs_keys}
                target_obs = {k: batch[f"target.{k}"].to(device) for k in obs_keys}
                lengths = batch["action_seq_lengths"].to(device)
                pred_obs = raw_model.forward_all(
                    delayed_obs, actions, lengths=lengths,
                )
                for k in obs_keys:
                    val_losses[k] += loss_fn(pred_obs[k], target_obs[k]).item() * B
                val_count += B

        # Aggregate val loss across ranks
        if is_distributed:
            loss_tensor = torch.tensor(
                [val_losses[k] for k in obs_keys] + [float(val_count)],
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            all_count = loss_tensor[-1].item()
            for i, k in enumerate(obs_keys):
                val_losses[k] = loss_tensor[i].item() / max(all_count, 1)
        else:
            for k in obs_keys:
                val_losses[k] /= max(val_count, 1)
        total_val = sum(val_losses.values())

        # Save best model (rank 0 only)
        saved_flag = ""
        if is_main and total_val < best_val:
            best_val = total_val
            save_belief_model(raw_model, output_dir)
            saved_flag = " ★ saved"

        # Update epoch progress bar
        avg_train = train_loss_sum / max(train_count, 1)
        epoch_bar.set_postfix(
            train=f"{avg_train:.5f}",
            val=f"{total_val:.5f}",
            best=f"{best_val:.5f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
        )

        if is_main and (epoch % 10 == 0 or epoch == 1):
            parts = [f"{k.split('.')[-1]}: {val_losses[k]:.6f}" for k in obs_keys]
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train={avg_train:.6f} | "
                f"val: {' | '.join(parts)} (total={total_val:.6f}) | "
                f"lr={scheduler.get_last_lr()[0]:.2e}{saved_flag}"
            )

    # Save training metadata (rank 0 only)
    if is_main:
        meta = {
            "repo_ids": [ds.repo_id for ds in datasets],
            "delays": delays,
            "action_key": action_key,
            "action_dim": action_dim,
            "epochs": epochs,
            "best_val_loss": best_val,
            "modalities": {k: t for k, t in zip(obs_keys, obs_types)},
            "world_size": world_size,
        }
        with open(output_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Training complete. Model saved to: {output_dir}")

    _cleanup_ddp()

    # Re-load best checkpoint (on main process)
    if is_main:
        from lerobot.utils.belief_predictor import load_belief_model
        return load_belief_model(output_dir, device=device)
    return raw_model


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train multi-modal belief predictors for observation delay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo_id", type=str, nargs="+", required=True,
                        help="One or more LeRobot dataset repo_ids "
                             "(e.g. lerobot/libero_goal_image lerobot/libero_spatial_image)")
    parser.add_argument("--root", type=str, default=None,
                        help="Local dataset root")
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Delay values to train on (in steps)")

    # Modality specification
    parser.add_argument("--obs_keys", type=str, nargs="+",
                        default=["observation.state",
                                 "observation.images.image",
                                 "observation.images.wrist_image"],
                        help="Observation keys to train belief predictors for")
    parser.add_argument("--obs_types", type=str, nargs="+",
                        default=["state", "image", "image"],
                        help='Type per obs key: "state" or "image"')
    parser.add_argument("--action_key", type=str, default="action")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Shared GRU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Per-modality latent dimension")
    parser.add_argument("--encoder_channels", type=int, nargs="+",
                        default=[32, 64, 128, 256],
                        help="CNN channel sizes for image encoder stages")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/belief_model")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", type=int, nargs="?", const=500, default=0,
                        metavar="N",
                        help="Debug mode: use only N samples (default 500) "
                             "and 5 epochs for fast iteration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets = []
    for repo_id in args.repo_id:
        logger.info(f"Loading dataset: {repo_id}")
        datasets.append(LeRobotDataset(repo_id=repo_id, root=args.root))

    # Debug overrides
    debug_samples = args.debug
    ep = args.epochs
    if debug_samples > 0 and args.epochs == 100:  # default not overridden
        ep = 5
        logger.info(f"Debug mode: epochs reduced to {ep} (override with --epochs)")

    train_belief(
        datasets=datasets,
        delays=args.delays,
        output_dir=output_dir,
        obs_keys=args.obs_keys,
        obs_types=args.obs_types,
        action_key=args.action_key,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=ep,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        debug_samples=debug_samples,
    )


if __name__ == "__main__":
    main()
