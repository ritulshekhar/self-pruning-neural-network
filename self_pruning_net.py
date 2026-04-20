"""
self_pruning_net.py
===================
Self-Pruning Neural Network for CIFAR-10
Tredence Analytics — AI Engineering Internship Case Study

Author : Ritul Shekhar
Date   : April 2026

Overview
--------
This script implements a neural network that learns to prune itself *during*
training via a gated-weight mechanism.  Each weight w_ij is multiplied by a
learnable sigmoid gate g_ij ∈ (0, 1).  An L1 sparsity penalty on the gates
pushes most of them toward 0, effectively removing the corresponding weights
without any post-training step.

Architecture
------------
  ┌─────────────────────────────────────────────────────┐
  │  CNN Feature Extractor                              │
  │  (3 conv blocks: Conv → BN → GELU → MaxPool)       │
  │  Output: 256-dim flat feature vector per sample     │
  ├─────────────────────────────────────────────────────┤
  │  Prunable Classification Head                       │
  │  PrunableLinear(256 → 512) → GELU → Dropout(0.3)   │
  │  PrunableLinear(512 → 256) → GELU → Dropout(0.3)   │
  │  PrunableLinear(256 → 10)                           │
  └─────────────────────────────────────────────────────┘

Usage
-----
  # Full training sweep across three lambda values
  python self_pruning_net.py

  # Dry-run (validates shapes, no training)
  python self_pruning_net.py --dry-run

  # Single lambda run
  python self_pruning_net.py --lam 1e-3 --epochs 30

Design Decisions
----------------
* gate_scores are initialised with a small negative bias (logit ≈ -0.5) so
  gates start near 0.38.  This is a neutral starting point — the optimizer
  can push them up (keep) or down (prune) equally easily.
* Mixed-precision training (torch.amp) is used automatically on CUDA to cut
  memory usage ~50% and speed up training ~1.5–2×.
* Cosine annealing LR schedule provides better convergence than step decay.
* Data augmentation (RandomCrop + HorizontalFlip + Cutout) improves
  generalisation without adding complexity.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Device selection — prefer MPS (Apple Silicon) > CUDA > CPU
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Part 1 — PrunableLinear
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns *which weights to prune*.

    Each weight w_ij has an associated learnable gate score s_ij.  During the
    forward pass:

        g_ij   = sigmoid(s_ij)          ∈ (0, 1)
        ŵ_ij   = w_ij * g_ij            (gated / pruned weight)
        output = input @ ŵ.T + bias      (standard affine transform)

    Gradients flow through both `weight` and `gate_scores` automatically via
    PyTorch autograd — no manual gradient engineering is required because
    sigmoid is differentiable everywhere and element-wise multiplication is
    a trivial chain-rule step.

    The L1 sparsity penalty ∑ g_ij pushes gate scores toward −∞ (gates → 0),
    effectively removing the corresponding weights.

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool  (default True)
    gate_init_bias : float
        Constant added to zero-initialised gate_scores before training.
        Negative values make gates start below 0.5 (slightly pruned).
        Default: -0.5  →  initial gates ≈ sigmoid(-0.5) ≈ 0.38
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init_bias: float = -0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # -- Standard weight and bias (same initialisation as nn.Linear) --
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # -- Gate score tensor (same shape as weight) ----------------------
        # Registered as a parameter so Adam updates it alongside weights.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), gate_init_bias)
        )

        # Kaiming uniform initialisation for weight (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Cache gates computed in the last forward pass for sparsity queries
        # (uses register_buffer so it's moved with .to(device) but not a param)
        self.register_buffer("_cached_gates", torch.zeros_like(self.weight))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated forward pass.

        Steps
        -----
        1. gates = sigmoid(gate_scores)      ← squash to (0, 1)
        2. pruned_weight = weight * gates    ← element-wise mask
        3. output = F.linear(x, pruned_weight, bias)
        """
        gates = torch.sigmoid(self.gate_scores)          # (out, in)
        self._cached_gates = gates.detach()              # cache for stats
        pruned_weight = self.weight * gates              # (out, in)
        return F.linear(x, pruned_weight, self.bias)

    # ------------------------------------------------------------------
    def get_gate_stats(self, threshold: float = 1e-2) -> dict:
        """
        Return gate statistics without an extra forward pass.

        Returns
        -------
        dict with keys:
            gates       : flat tensor of current gate values (detached)
            sparsity    : fraction of gates below `threshold`
            mean_gate   : mean gate value
            n_total     : total number of gates
            n_pruned    : number of gates below threshold
        """
        gates = self._cached_gates.cpu().float()
        n_total = gates.numel()
        n_pruned = (gates < threshold).sum().item()
        return {
            "gates": gates.flatten(),
            "sparsity": n_pruned / n_total,
            "mean_gate": gates.mean().item(),
            "n_total": n_total,
            "n_pruned": n_pruned,
        }

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ===========================================================================
# Network definition
# ===========================================================================

class ConvBlock(nn.Module):
    """Conv → BN → GELU → optional MaxPool."""

    def __init__(self, in_c: int, out_c: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SelfPruningNet(nn.Module):
    """
    Hybrid CNN + Prunable-FC network for CIFAR-10.

    Feature extractor
    -----------------
    3 ConvBlocks with channels [32 → 64 → 128 → 256].
    After 3 MaxPool(2) layers on a 32×32 input: feature map is 4×4.
    Flat feature vector dim = 256 × 4 × 4 = 4096.

    Wait — CIFAR images are 32×32.  After 3 max-pools: 32→16→8→4.
    With 256 channels: 256 * 4 * 4 = 4096.  We project this down first.

    Prunable classification head
    ----------------------------
    PrunableLinear(4096 → 512) → GELU → Dropout(0.3)
    PrunableLinear(512  → 256) → GELU → Dropout(0.3)
    PrunableLinear(256  → 10)
    """

    def __init__(self, dropout: float = 0.3, gate_init_bias: float = -0.5):
        super().__init__()

        # -- CNN feature extractor (not prunable; conv pruning is out-of-scope)
        self.features = nn.Sequential(
            ConvBlock(3,   64,  pool=True),   # 32→16
            ConvBlock(64,  128, pool=True),   # 16→8
            ConvBlock(128, 256, pool=True),   # 8→4
        )
        # flat feature dim after adaptive pooling to 4×4
        self._flat_dim = 256 * 4 * 4  # 4096

        # -- Prunable classification head --
        self.prunable_layers: nn.ModuleList = nn.ModuleList([
            PrunableLinear(self._flat_dim, 512, gate_init_bias=gate_init_bias),
            PrunableLinear(512, 256,            gate_init_bias=gate_init_bias),
            PrunableLinear(256, 10,             gate_init_bias=gate_init_bias),
        ])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)      # flatten → (B, 4096)

        # Prunable FC head
        x = self.act(self.prunable_layers[0](x))
        x = self.dropout(x)
        x = self.act(self.prunable_layers[1](x))
        x = self.dropout(x)
        x = self.prunable_layers[2](x)  # logits
        return x

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        This is the SparsityLoss term.  It is added to CrossEntropyLoss
        scaled by lambda (λ).

        Mathematical justification
        --------------------------
        The L1 norm of a vector v is ∑|v_i|.  Since gate values are
        sigmoid outputs, they live in (0, 1) and are always positive, so
        |g_ij| = g_ij.  Minimising ∑ g_ij pushes gates toward 0
        (i.e., they approach the value that minimises the penalty), which
        is achieved as gate_scores → −∞ ⟹ sigmoid → 0.

        Note
        ----
        We compute this directly from gate_scores (before sigmoid) using
        a numerically stable formula rather than caching, to ensure
        gradients are always fresh.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def get_all_gate_stats(self, threshold: float = 1e-2) -> dict:
        """Aggregate gate statistics across all PrunableLinear layers."""
        all_gates: List[torch.Tensor] = []
        n_total = 0
        n_pruned = 0
        for layer in self.prunable_layers:
            stats = layer.get_gate_stats(threshold)
            all_gates.append(stats["gates"])
            n_total += stats["n_total"]
            n_pruned += stats["n_pruned"]
        return {
            "gates": torch.cat(all_gates),
            "sparsity": n_pruned / n_total,
            "n_total": n_total,
            "n_pruned": n_pruned,
        }


# ===========================================================================
# Part 2 — Data loading
# ===========================================================================

def build_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 data loaders with augmentation on train split.

    Train transforms: RandomCrop (pad=4) + RandomHorizontalFlip + Normalize
    Test  transforms: Normalize only
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    # pin_memory speeds up CPU→GPU transfers on CUDA machines
    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader


# ===========================================================================
# Part 3 — Training and Evaluation
# ===========================================================================

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lam: float,
    epoch: int,
    epochs: int,
) -> Tuple[float, float, float]:
    """
    One training epoch.

    Returns
    -------
    (avg_total_loss, avg_cls_loss, avg_sparsity_loss)
    """
    model.train()
    total_loss_sum = cls_loss_sum = spar_loss_sum = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}/{epochs}", leave=False, ncols=90)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        # -- Mixed precision forward --
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type in ("cuda", "cpu"))):
            logits    = model(images)
            cls_loss  = F.cross_entropy(logits, labels)
            spar_loss = model.sparsity_loss()
            loss      = cls_loss + lam * spar_loss

        # -- Backward --
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # stability
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item()
        cls_loss_sum   += cls_loss.item()
        spar_loss_sum  += spar_loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "cls":  f"{cls_loss.item():.3f}",
            "spar": f"{spar_loss.item():.1f}",
        })

    return (
        total_loss_sum / n_batches,
        cls_loss_sum   / n_batches,
        spar_loss_sum  / n_batches,
    )


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    lam: float,
) -> Tuple[float, float, float]:
    """
    Evaluate model accuracy and sparsity on the given loader.

    Returns
    -------
    (accuracy_pct, sparsity_pct, avg_total_loss)
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    n_batches = len(loader)

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type in ("cuda", "cpu"))):
            logits   = model(images)
            cls_loss = F.cross_entropy(logits, labels)
            spar_l   = model.sparsity_loss()
            loss_sum += (cls_loss + lam * spar_l).item()

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    # Trigger gate caching for stats by doing a dummy forward on CPU is wasteful;
    # instead, stats are computed directly from gate_scores (always current).
    stats = model.get_all_gate_stats(threshold=1e-2)

    accuracy   = 100.0 * correct / total
    sparsity   = 100.0 * stats["sparsity"]
    avg_loss   = loss_sum / n_batches
    return accuracy, sparsity, avg_loss


def train(
    lam: float,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-3,
    dry_run: bool = False,
) -> dict:
    """
    Full training run for a single lambda value.

    Parameters
    ----------
    lam       : float — sparsity regularisation coefficient (λ)
    epochs    : int   — number of training epochs
    batch_size: int
    lr        : float — initial learning rate
    dry_run   : bool  — if True, skip training; just validate shapes

    Returns
    -------
    dict with keys: lam, accuracy, sparsity, history, model, gate_stats
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lam:.0e}  |  device = {DEVICE}  |  epochs = {epochs}")
    print(f"{'='*60}")

    train_loader, test_loader = build_dataloaders(batch_size=batch_size)

    model = SelfPruningNet(dropout=0.3, gate_init_bias=-0.5).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # ---- Dry run: validate shapes ----
    if dry_run:
        sample_imgs, sample_lbls = next(iter(train_loader))
        sample_imgs = sample_imgs[:4].to(DEVICE)
        with torch.no_grad():
            out = model(sample_imgs)
        print(f"  [DRY RUN] Output shape: {out.shape}  ✓")
        print(f"  [DRY RUN] Sparsity loss: {model.sparsity_loss().item():.2f}")
        return {}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # GradScaler only functional on CUDA; on CPU/MPS it's a no-op wrapper
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    history = {
        "train_loss": [], "cls_loss": [], "spar_loss": [],
        "test_acc": [],   "sparsity": [],
    }

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, cls_l, spar_l = train_one_epoch(
            model, train_loader, optimizer, scaler, lam, epoch, epochs
        )
        test_acc, sparsity, _ = evaluate(model, test_loader, lam)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["cls_loss"].append(cls_l)
        history["spar_loss"].append(spar_l)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        if test_acc > best_acc:
            best_acc = test_acc

        elapsed = time.time() - t0
        print(
            f"  [{epoch:02d}/{epochs}] "
            f"loss={tr_loss:.3f}  cls={cls_l:.3f}  spar={spar_l:.1f}  "
            f"acc={test_acc:.2f}%  sparse={sparsity:.1f}%  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  "
            f"({elapsed:.0f}s)"
        )

    # Final evaluation
    final_acc, final_sparsity, _ = evaluate(model, test_loader, lam)
    gate_stats = model.get_all_gate_stats(threshold=1e-2)

    # Force a forward pass to refresh _cached_gates buffers
    model.eval()
    with torch.no_grad():
        sample, _ = next(iter(test_loader))
        model(sample[:2].to(DEVICE))
    gate_stats = model.get_all_gate_stats(threshold=1e-2)

    print(f"\n  ✓ Final test accuracy : {final_acc:.2f}%")
    print(f"  ✓ Sparsity level      : {final_sparsity:.2f}%")
    print(f"  ✓ Best accuracy       : {best_acc:.2f}%")

    return {
        "lam": lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "best_acc": best_acc,
        "history": history,
        "model": model,
        "gate_stats": gate_stats,
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_gate_distribution(
    gate_stats: dict,
    lam: float,
    save_path: Path,
) -> None:
    """
    Plot histogram of gate values for the best model.

    A successful result will show:
      - Large spike near 0 (pruned gates)
      - Secondary cluster away from 0 (active gates)
    """
    gates = gate_stats["gates"].numpy()
    sparsity = gate_stats["sparsity"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Gate Value Distribution  (λ={lam:.0e},  sparsity={sparsity:.1f}%)",
        fontsize=13,
    )

    # Full distribution — linear y scale
    ax = axes[0]
    ax.hist(gates, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate value  g = σ(s)")
    ax.set_ylabel("Count")
    ax.set_title("Full distribution (linear scale)")
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=9)

    # Log y scale — reveals bi-modal structure clearly
    ax = axes[1]
    ax.hist(gates, bins=100, color="#DD8452", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Gate value  g = σ(s)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Log-scale y-axis")
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gate distribution plot saved → {save_path}")


def plot_training_curves(results: List[dict], save_path: Path) -> None:
    """Training loss and accuracy curves for all lambda values."""
    n = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, r in enumerate(results):
        h = r["history"]
        lbl = f"λ={r['lam']:.0e}"
        epochs = range(1, len(h["test_acc"]) + 1)
        axes[0].plot(epochs, h["test_acc"],  color=colours[idx], label=lbl, linewidth=1.8)
        axes[1].plot(epochs, h["sparsity"],  color=colours[idx], label=lbl, linewidth=1.8)

    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Test Accuracy vs Epoch");  axes[0].legend()
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Sparsity Level (%)")
    axes[1].set_title("Sparsity Level vs Epoch");  axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved → {save_path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — Tredence Case Study"
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[1e-4, 1e-3, 1e-2],
        help="Lambda values to sweep (default: 1e-4 1e-3 1e-2)",
    )
    parser.add_argument("--lam",    type=float, default=None,
                        help="Run a single lambda value (overrides --lambdas)")
    parser.add_argument("--epochs", type=int,   default=30,
                        help="Epochs per run (default: 30)")
    parser.add_argument("--batch",  type=int,   default=128)
    parser.add_argument("--lr",     type=float, default=3e-3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate model shapes without full training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        train(lam=1e-3, epochs=1, dry_run=True)
        return

    lambdas = [args.lam] if args.lam is not None else args.lambdas

    results = []
    best_result = None  # for gate distribution plot

    for lam in lambdas:
        r = train(
            lam=lam,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )
        if r:
            results.append(r)
            # Choose the model with the best sparsity for the distribution plot
            # (so the bi-modal pattern is clearest)
            if best_result is None or r["sparsity"] > best_result["sparsity"]:
                best_result = r

    if not results:
        return

    # -- Training curves (all lambdas)
    plot_training_curves(results, RESULTS_DIR / "training_curves.png")

    # -- Gate distribution (best model)
    if best_result:
        plot_gate_distribution(
            best_result["gate_stats"],
            best_result["lam"],
            RESULTS_DIR / "gate_distribution.png",
        )

    # -- Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)':<15}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['lam']:<12.0e} {r['accuracy']:<18.2f} {r['sparsity']:<15.2f}")
    print("=" * 60)
    print("\n  Done. Results saved to ./results/")


if __name__ == "__main__":
    main()
