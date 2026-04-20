# Self-Pruning Neural Network
# Tredence Analytics — AI Engineering Internship Case Study

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)

A neural network that learns to **prune itself during training** — no post-training step required.

Each weight is paired with a learnable sigmoid gate. An L1 sparsity penalty drives most gates to zero, effectively removing the corresponding weights on the fly. The result: a sparse, efficient model that self-organises around its most critical connections.

---

## How It Works

```
Total Loss = CrossEntropy(logits, labels) + λ × Σ sigmoid(gate_scores)
```

- **λ = low** → light pruning, high accuracy  
- **λ = high** → aggressive pruning, some accuracy cost  

The L1 norm gives every gate a *constant* gradient pressure toward zero — unlike L2, which slows down near zero and never truly zeros out weights.

---

## Architecture

```
Input (32×32×3 CIFAR-10 image)
        │
┌───────▼────────────────────────────────────────┐
│  CNN Feature Extractor (frozen, not pruned)    │
│  Conv(3→64)  → BN → GELU → MaxPool(2)  [32→16]│
│  Conv(64→128)→ BN → GELU → MaxPool(2)  [16→8] │
│  Conv(128→256)→BN → GELU → MaxPool(2)  [8→4]  │
│  Output: flat 4096-dim feature vector          │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Prunable Classification Head                  │
│  PrunableLinear(4096 → 512) → GELU → Drop(0.3)│
│  PrunableLinear( 512 → 256) → GELU → Drop(0.3)│
│  PrunableLinear( 256 → 10 )  ← logits         │
└────────────────────────────────────────────────┘
        │
   Softmax → Predicted Class
```

Total parameters: **4,833,610**  
Prunable parameters: **2,228,234** (FC head only)

---

## Results

> Full 30-epoch training on CIFAR-10, Apple Silicon (MPS). Run `python self_pruning_net.py` to reproduce.

| Lambda (λ) | Test Accuracy (%) | Best Accuracy (%) | Gate Suppression (%) | Notes |
|:---:|:---:|:---:|:---:|:---|
| `1e-4` | 89.24 | 89.24 | 56.1 | Mild regularisation; near-dense performance |
| `1e-3` | 88.66 | 88.80 | 81.7 | Sweet spot: high accuracy + strong compression |
| `1e-2` | 83.20 | 83.41 | 96.5 | Aggressive pruning; avg gate ≈ 0.035 |

> **Gate Suppression** = `1 − (Σgᵢⱼ / N)` — measures the fraction of effective weight capacity removed by the L1 penalty. Since sigmoid gates are continuous in (0, 1), a hard binary threshold understates the real compression; gate suppression captures the true picture.

### Gate Distribution (best model)
![Gate Distribution](results/gate_distribution.png)

### Training Curves
![Training Curves](results/training_curves.png)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/ritulshekhar/self-pruning-neural-network.git
cd self-pruning-neural-network

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate (no training, just shape checks)
python self_pruning_net.py --dry-run
```

---

## Usage

```bash
# Full sweep — 3 lambda values × 30 epochs (recommended)
python self_pruning_net.py

# Single lambda run
python self_pruning_net.py --lam 1e-3 --epochs 30

# Custom sweep
python self_pruning_net.py --lambdas 1e-5 1e-4 5e-4 1e-3 --epochs 50

# Override batch size / learning rate
python self_pruning_net.py --lam 1e-3 --epochs 30 --batch 256 --lr 1e-3
```

**Outputs saved to `results/`:**
| File | Contents |
|------|----------|
| `gate_distribution.png` | Gate value histogram (linear + log scale) |
| `training_curves.png` | Accuracy & sparsity vs epoch for all λ |
| `experiment_log.json` | Full metrics, per-layer stats, loss history |

**Checkpoints saved to `checkpoints/`:**
| File | Contents |
|------|----------|
| `model_lam1e-04.pt` | Best model state dict for λ=1e-4 |
| `model_lam1e-03.pt` | Best model state dict for λ=1e-3 |
| `model_lam1e-02.pt` | Best model state dict for λ=1e-2 |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Sigmoid gates** | Differentiable everywhere; no STE hack needed |
| **L1 sparsity (not L2)** | Constant gradient → pushes all gates equally to 0 |
| **`gate_init_bias = -0.5`** | Gates start at ≈0.38; optimizer has equal room to grow or shrink |
| **CNN extractor + prunable FC head** | Realistic architecture; conv sparsity is a separate structured problem |
| **Mixed precision (torch.amp)** | ~2× faster on CUDA, ~50% less VRAM |
| **Cosine annealing LR** | Better final convergence than step decay |
| **Gradient clipping (max=5.0)** | Prevents instability when gate_scores are large magnitude |

---

## File Structure

```
self-pruning-neural-network/
├── self_pruning_net.py   # All-in-one solution (~790 lines)
├── report.md             # Theory, results, analysis
├── requirements.txt      # Pinned dependencies
├── .gitignore
├── results/
│   ├── gate_distribution.png
│   ├── training_curves.png
│   └── experiment_log.json
└── checkpoints/
    ├── model_lam1e-04.pt
    ├── model_lam1e-03.pt
    └── model_lam1e-02.pt
```

---

## Report

See [`report.md`](report.md) for:
- Mathematical explanation of why L1 encourages sparsity
- Per-layer sparsity analysis
- λ trade-off analysis
- Gate distribution interpretation
