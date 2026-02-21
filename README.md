# Replication: "Attention Is Not All You Need"

Replication of:
> **"Attention Is Not All You Need: The Importance of Feedforward Networks in Transformer Models"**
> Isaac Gerber, Johns Hopkins University, arXiv:2505.06633, May 2025.

## Paper Summary

The paper investigates the role of the Feed-Forward Network (FFN) sub-layer in
decoder-only transformer language models. By training GPT-3 medium-scale models
(d\_model=1024, 24 layers) on Booksum and WikiText-103, the paper shows:

1. **FFN depth matters**: models with more linear layers in the FFN achieve lower
   cross-entropy loss. Ordering: 3-layer > 2-layer > 1-layer > 0-layer.
2. **Three-layer FFN wins with fewer parameters**: a 3-layer FFN model with fewer
   transformer blocks outperforms the standard 2-layer baseline while using fewer
   total parameters and less compute.
3. **Dimensionality vs depth**: for 0-layer and 1-layer models, wider networks
   (larger d\_model) outperform deeper ones when parameter count is matched.

### Architecture variants studied (Figure 1)

| Variant | Architecture | FFN params (per block) |
|---------|-------------|------------------------|
| FFN0 (D) | No FFN — only attention + skip | 0 |
| FFN1 (C) | Single linear layer: d→d + GELU | d² |
| FFN2 (A) | Standard GPT FFN: d→4d→d + GELU | 8d² |
| FFN3 (B) | Extended FFN: d→4d→4d→d + GELU | 24d² |

---

## Repository Structure

```
.
├── models/
│   ├── attention.py      # Multi-head causal self-attention
│   ├── ffn.py            # FFN0, FFN1, FFN2, FFN3 implementations
│   ├── block.py          # Pre-norm transformer block
│   └── transformer.py    # Full decoder-only language model
├── data/
│   └── dataset.py        # Booksum & WikiText-103 loading utilities
├── configs.py            # Named model configurations (small/medium/paper scale)
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── run_experiments.py    # Orchestrate all paper experiments
├── plot_results.py       # Reproduce paper figures
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. An NVIDIA GPU is strongly recommended
for medium/paper-scale experiments.

---

## Quick Start

### 1. Print parameter counts for all configurations

```bash
python plot_results.py --param_table
```

### 2. Train a small model (CPU-friendly, ~5 minutes)

```bash
# 2-layer FFN baseline (architecture A)
python train.py --config small_2layer --dataset wikitext --max_steps 2000

# 3-layer FFN (architecture B)
python train.py --config small_3layer --dataset wikitext --max_steps 2000

# 0-layer (no FFN, architecture D)
python train.py --config small_0layer --dataset wikitext --max_steps 2000
```

### 3. Run all small-scale experiments

```bash
python run_experiments.py --scale small --dataset wikitext --max_steps 2000
```

### 4. Plot results

```bash
python plot_results.py --output_dir outputs --dataset wikitext
```

Figures are saved to `figures/`.

---

## Reproducing the Paper at Scale

The paper trains on an A100 GPU for ~66 hours total. To replicate at full scale:

```bash
# Primary comparison: 4 FFN variants × 2 datasets = 8 runs
python run_experiments.py --scale paper --dataset both --max_steps 100000

# Or train individual models
python train.py --config paper_2layer --dataset booksum \
    --max_steps 100000 --batch_size 32 --grad_accum_steps 4
```

### Paper-scale model configurations

| Config name | FFN layers | d\_model | n\_layers | ~Params |
|-------------|-----------|---------|---------|---------|
| `paper_0layer` | 0 | 1024 | 24 | 253M |
| `paper_1layer` | 1 | 1024 | 24 | 277M |
| `paper_2layer` | 2 | 1024 | 24 | 329M (baseline) |
| `paper_3layer` | 3 | 1024 | 24 | 726M |
| `paper_3layer_matched` | 3 | 1024 | 10 | ~329M |
| `paper_0layer_hd` | 0 | 1792 | 24 | ~329M |
| `paper_1layer_hd` | 1 | 1600 | 24 | ~329M |

---

## Architecture Details

### Pre-norm Transformer Block

```
x → LayerNorm → MultiHeadAttention → + → LayerNorm → FFN → + → output
         ↑__________________________|        ↑___________________|
              (residual connection)             (residual connection)
```

### FFN Variants

```python
# FFN0: no-op (only the residual skip connection passes information)
FFN0: x → 0

# FFN1: single projection + GELU
FFN1: x → Linear(d, d) → GELU → x

# FFN2: standard GPT MLP (baseline)
FFN2: x → Linear(d, 4d) → GELU → Linear(4d, d)

# FFN3: extended three-layer MLP
FFN3: x → Linear(d, 4d) → GELU → Linear(4d, 4d) → GELU → Linear(4d, d)
```

### Training Setup

| Hyper-parameter | Value |
|----------------|-------|
| Tokenizer | GPT-2 BPE (vocab: 50,257) |
| Context length | 1,024 tokens |
| Optimiser | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 3×10⁻⁴ with cosine decay |
| Warm-up steps | 2,000 |
| Dropout | 0.10 |
| Activation | GELU |
| Architecture | Pre-norm (LayerNorm before sub-layers) |
| Loss | Cross-entropy (next-token prediction) |

---

## Key Findings (from the paper)

1. **More FFN layers → lower loss**: consistently observed on both Booksum and
   WikiText-103. The log-loss ordering from best to worst is:
   FFN3 < FFN2 < FFN1 < FFN0.

2. **Three-layer FFN with 10 blocks ≈ better than two-layer with 24 blocks**:
   Despite having comparable (or fewer) total parameters, the 3-layer variant
   achieves lower training loss and trains faster.

3. **High-dimension 0-layer and 1-layer models do better**: when d\_model is
   increased to match the parameter count of the 2-layer baseline, the narrow
   0-layer/1-layer models improve, but still fall short of the 2-layer baseline.

---

## Citation

```bibtex
@article{gerber2025attention,
  title   = {Attention Is Not All You Need: The Importance of Feedforward
             Networks in Transformer Models},
  author  = {Gerber, Isaac},
  journal = {arXiv preprint arXiv:2505.06633},
  year    = {2025},
}
```
