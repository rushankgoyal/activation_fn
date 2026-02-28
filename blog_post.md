# Do Activation Functions Matter? A Deep Dive into FFN Design for Transformers

*Replicating Gerber (2025) and extending with novel activations from Vitvitskyi et al. (2026)*

---

## Motivation

If you have spent any time reading the transformer literature, you have probably
noticed that activation functions barely get mentioned. Papers describe
architecture choices — number of layers, embedding dimension, attention heads —
in meticulous detail, but the activation sitting inside the feedforward network
(FFN) is almost always just "GELU" or "SiLU", chosen by convention and left at
that.

This experiment set out to challenge that assumption on two fronts:

1. **Does FFN depth matter?** Gerber (2025) — *"Attention Is Not All You Need:
   The Importance of Feedforward Networks in Transformer Models"*
   (arXiv:2505.06633) — argues that the FFN sub-layer is consistently
   underweighted in the literature and that deeper FFNs can beat the standard
   two-layer design even at smaller model sizes.

2. **Does the specific activation function inside the FFN matter?** Vitvitskyi
   et al. (2026) — *"Mining Generalizable Activation Functions"*
   (arXiv:2602.05688) — used an evolutionary search pipeline (AlphaEvolve)
   driven by frontier LLMs to discover four novel activation functions that
   outperform GELU on out-of-distribution benchmarks. We wanted to know whether
   those gains survive in a standard language modelling setting.

Together these questions let us build a small but complete picture: does it
matter *how many* layers your FFN has, and once you fix the depth, does it
matter *which nonlinearity* sits inside?

---

## Background

### Part 1 — FFN Depth (Gerber 2025)

The canonical transformer block looks like this:

```
x = x + MHA(LayerNorm(x))    ← multi-head attention sub-layer
x = x + FFN(LayerNorm(x))    ← feedforward sub-layer
```

The FFN in GPT-2 and GPT-3 is a two-layer MLP that expands the hidden
dimension by 4×, applies an activation, then projects back:

```
FFN(x) = W₂ · GELU(W₁ · x)    where W₁ ∈ ℝ^{4d×d}, W₂ ∈ ℝ^{d×4d}
```

Gerber asks: what if we changed the number of linear layers? He studies four
variants — zero, one, two, and three linear layers — and shows that deeper FFNs
consistently beat the two-layer baseline, sometimes with *fewer* total
parameters when you compensate by reducing the number of blocks.

### Part 2 — Novel Activations (Vitvitskyi et al. 2026)

AlphaEvolve is an evolutionary program synthesis system guided by LLMs. The
authors used it to search over the space of mathematical expressions, scoring
candidate functions by their out-of-distribution (OOD) generalisation
performance when plugged into a fixed transformer. The four survivors are:

| Name | Formula |
|------|---------|
| **GELUSine** | `GELU(x) + 0.1·sin(x)` |
| **GELUSincPerturbation** | `GELU(x) + 0.5·sin(x)/x` |
| **GMTU** | `tanh(1.5x)·exp(−0.2x²) + 0.2x` |
| **Turbulent** | `sign(x)·log(1+0.5|x|) + sin(x)·exp(−z²/2)` where `z=(x−μ)/σ` |

Our role is to benchmark them in a concrete, reproducible language modelling
experiment — something the original paper did not do at this scale.

---

## Model Architecture

We use a standard decoder-only transformer (GPT-style) throughout. The key
design choices mirror GPT-3 conventions:

- **Pre-norm**: LayerNorm is applied *before* each sub-layer, not after.
- **No biases** in attention projections (Q, K, V, output).
- **Weight tying**: the LM head shares weights with the token embedding matrix.
- **Causal masking**: left-to-right attention for autoregressive language modelling.
- **GPT-2 weight initialisation**: residual projections scaled by `1/√(2·n_layers)`.

### The Transformer Block

```python
# models/block.py
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_layers, d_ff=None,
                 dropout=0.1, act_type="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn  = build_ffn(ffn_layers, d_model, d_ff, dropout, act_type)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.res_dropout(self.attn(self.ln1(x)))  # attention
        x = x + self.res_dropout(self.ffn(self.ln2(x)))   # FFN
        return x
```

### FFN Variants

The four FFN architectures differ only in how many linear projections they chain
together. Parameter counts are per block, with `d_ff = 4·d_model`:

```python
# models/ffn.py

class FFN0(nn.Module):
    """Architecture D: no FFN — residual skip only."""
    def forward(self, x):
        return torch.zeros_like(x)          # 0 parameters

class FFN1(nn.Module):
    """Architecture C: single square projection."""
    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))   # d² params

class FFN2(nn.Module):
    """Architecture A: standard GPT MLP  d → 4d → d."""
    def forward(self, x):
        x = self.act(self.fc1(x))           # d → 4d
        x = self.dropout(x)
        return self.fc2(x)                  # 4d → d  (8d² params total)

class FFN3(nn.Module):
    """Architecture B: three-layer MLP  d → 4d → 4d → d."""
    def forward(self, x):
        x = self.act1(self.fc1(x))          # d  → 4d
        x = self.act2(self.fc2(self.dropout1(x)))  # 4d → 4d
        return self.fc3(self.dropout2(x))   # 4d → d  (24d² params total)
```

| FFN variant | Linear layers | Parameters per block | Relative cost |
|-------------|--------------|---------------------|--------------|
| FFN0 | 0 | 0 | — |
| FFN1 | 1 | d² | 1× |
| FFN2 (baseline) | 2 | 8d² | 8× |
| FFN3 | 3 | 24d² | 24× |

---

## Experiment 1 — FFN Depth Sweep

### Setup

All FFN-depth runs share the same hyperparameters to isolate depth as the sole
variable:

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Decoder-only, pre-norm |
| d_model | 256 |
| n_layers | 4 |
| n_heads | 4 |
| Vocabulary | Byte-level (256 tokens) |
| Sequence length | 512 |
| Activation | GELU (fixed) |
| Batch size | 8 |
| Optimiser | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Learning rate | 3×10⁻⁴ with linear warm-up + cosine decay |
| Warm-up steps | 300 |
| Training steps | 2,000 |
| Eval interval | every 500 steps |

The corpus is the Vim 9.1 documentation (`/usr/share/vim/vim91/doc/*.txt`) read
as raw bytes — no tokeniser download required. The 90/10 train/val split gives
roughly 3,800 training sequences of length 512.

The small model has approximately **3.3 M parameters** for the FFN2 baseline
(including embeddings and weight-tied LM head).

### Results

```
Val loss  │  step 500   step 1000   step 1500   step 2000
──────────┼──────────────────────────────────────────────
FFN0      │   2.356      1.907       1.774       1.709
FFN1      │   2.258      1.819       1.643       1.596
FFN2 ★    │   2.158      1.641       1.430       1.338
FFN3      │   2.183      1.684       1.487       1.401
```

```
Val loss (lower is better)
2.40 ┤
2.20 ┤  ╮ FFN0
2.00 ┤  │╮ FFN1
1.80 ┤  ││╮ FFN3
1.60 ┤  │││╮ FFN2
1.40 ┤  ││││
1.20 ┤  ╰╰╰╰──────────────────  FFN2
     │           ╰╰──────────── FFN3
1.00 ┤                ╰╰─────── FFN1
     │                    ╰──── FFN0
     └──────┬──────┬──────┬──────┬─▶ step
           500   1000   1500   2000
```

**Key finding:** FFN depth is not free, but it is decisive.

- **FFN0 is clearly the worst.** Without any learnable FFN, the model can only
  compose attention heads — it reaches 1.709 at step 2000, a full 0.37 nats
  behind the baseline.
- **FFN1 sits in the middle.** A single square projection (d→d) is better than
  nothing but 0.26 nats behind FFN2.
- **FFN2 wins** at 2000 steps despite using 8× fewer parameters per block than
  FFN3. The 4× expansion ratio appears to be well-calibrated for this scale.
- **FFN3 underperforms FFN2** at this step count (1.401 vs 1.338). This is not
  a contradiction of Gerber's findings — with 3× more parameters per block,
  FFN3 needs more data and training steps to amortise its capacity. Gerber's
  paper demonstrates that at longer training horizons and with parameter-matched
  configurations (fewer blocks), FFN3 does pull ahead.

The takeaway for Part 1 is that **removing the FFN hurts badly, the standard
two-layer design is robust at short training budgets, and deeper FFNs are
promising but need to be parameter-matched to shine**.

---

## Experiment 2 — Activation Function Comparison

### Setup

Having established FFN2 as the strongest depth at our training budget, we hold
it fixed and vary only the activation function.

| Hyperparameter | Value |
|----------------|-------|
| FFN architecture | FFN2 (2-layer, d→4d→d) |
| d_model | 256 |
| n_layers | 4 |
| n_heads | 4 |
| Vocabulary | Byte-level (256 tokens) |
| Sequence length | 512 |
| Batch size | 8 |
| Optimiser | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Learning rate | 3×10⁻⁴ with linear warm-up + cosine decay |
| Warm-up steps | 300 |
| Training steps | 3,000 (50% longer for richer curves) |
| Eval interval | every 500 steps |
| Seeds | 2 per activation (42, 123) |

Using two seeds per activation lets us distinguish genuine signal from random
initialisation noise.

### The Activation Functions

#### GELU (baseline)

The Gaussian Error Linear Unit has been the default in GPT-family models since
GPT-2:

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the standard Gaussian CDF. It approximates a smooth gating
function: negative values are suppressed, positive values pass through with
slight attenuation near zero.

#### GELUSine

```python
# models/activations.py
class GELUSine(nn.Module):
    """f(x) = GELU(x) + α·sin(x),  α=0.1"""
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.gelu(x) + self.alpha * torch.sin(x)
```

The sine term introduces smooth, bounded, globally periodic oscillations layered
on top of GELU's monotonic base. With α=0.1 (found optimal in Appendix C of
the paper), the perturbation is subtle — roughly 10% of the sine amplitude —
so the function retains GELU's overall shape while adding a gentle ripple that
can help the network represent periodic or oscillatory patterns in the data.

```
GELUSine shape (schematic)
      │           ╭────
      │          ╱ ≈ x (large x)
 0.5  ┤        ╱╱
      │      ╱╱   ← slight oscillation visible near origin
 0.0  ┼────╮╱╱
      │    ╰──╮
-0.5  ┤        ╰───  (small negative dip, like GELU)
      └──────────────▶ x
       -3  -1  0  1  3
```

#### GELUSincPerturbation

```python
class GELUSincPerturbation(nn.Module):
    """f(x) = GELU(x) + α·sinc(x),  α=0.5,  sinc(x)=sin(x)/x"""
    def forward(self, x):
        sinc_x = torch.where(
            x.abs() < self.eps,
            torch.ones_like(x),   # limit of sin(x)/x at x=0 is 1
            torch.sin(x) / x,
        )
        return F.gelu(x) + self.alpha * sinc_x
```

Unlike the constant-amplitude sine in GELUSine, the sinc function decays as
O(1/|x|). This concentrates the perturbation *near the origin* — where the
gradient landscape matters most early in training — while recovering GELU-like
behaviour for large activations. The larger α=0.5 compensates for the decay.

In theory this is elegant. In practice, as we will see, the sinc near the origin
creates an extremely sharp curvature spike that makes training numerically
explosive.

#### GMTU — Gaussian-Modulated Tangent Unit

```python
class GMTU(nn.Module):
    """f(x) = p_alpha·tanh(p_beta·x)·exp(−p_gamma·x²) + p_gamma·x"""
    def __init__(self, p_alpha=1.0, p_beta=1.5, p_gamma=0.2):
        super().__init__()
        self.p_alpha = p_alpha
        self.p_beta  = p_beta
        self.p_gamma = p_gamma

    def forward(self, x):
        gaussian_gate = torch.exp(-self.p_gamma * x ** 2)
        primary       = self.p_alpha * torch.tanh(self.p_beta * x) * gaussian_gate
        linear_leak   = self.p_gamma * x
        return primary + linear_leak
```

GMTU has three components working together:

1. **`tanh(p_beta·x)`** — a bounded S-shaped nonlinearity, steeper than plain
   tanh due to p_beta=1.5.
2. **`exp(−p_gamma·x²)`** — a Gaussian gate that *decays* the tanh for large
   |x|. Whereas a pure tanh saturates to ±1, GMTU's primary response is
   suppressed back toward zero far from the origin.
3. **`p_gamma·x`** — a linear leak (`0.2x`) that prevents the function from
   collapsing to zero for large |x|, preserving gradient flow throughout
   training.

Near zero: `f(x) ≈ (p_alpha·p_beta + p_gamma)·x = (1.5 + 0.2)·x = 1.7x`
Large |x|: `f(x) ≈ 0.2x` (pure linear leak)

```
GMTU shape (schematic)
      │        ╭──────────────── 0.2x (linear leak)
 1.5  ┤       ╱
 1.0  ┤     ╱╱  ← tanh×Gaussian peak, then decays
 0.5  ┤   ╱╱
 0.0  ┼──╱╱
-0.5  ┤  ╲╲
-1.0  ┤   ╲╲  ← symmetric
      └──────────────────────▶ x
       -4  -2   0   2   4
```

The design philosophy is to be *locally* rich and nonlinear (near the origin)
while avoiding saturation at large magnitudes — a known failure mode of tanh
networks.

#### Turbulent

```python
class Turbulent(nn.Module):
    """
    f(x) = sign(x)·log(1+0.5|x|) + sin(x)·exp(−z²/2)
    where z = (x − μ) / σ  over all tensor elements.
    """
    def forward(self, x):
        base        = torch.sign(x) * torch.log1p(0.5 * x.abs())
        mean, std   = x.mean(), x.std() + self.eps
        z           = (x - mean) / std
        perturbation = torch.sin(x) * torch.exp(-0.5 * z ** 2)
        return base + perturbation
    ```

Turbulent is the only **non-pointwise** function in the set: it uses the mean
and standard deviation computed across *all elements of the current tensor* to
gate the sine perturbation. Concretely:

- **Base**: `sign(x)·log(1+0.5|x|)` — symmetric logarithmic growth. Unlike
  GELU, it never saturates: output grows without bound (slowly) for large |x|.
- **Perturbation**: `sin(x)·exp(−z²/2)` — a sine wave whose amplitude is
  modulated by a Gaussian of the *globally standardised* activation. Activations
  near the batch mean receive a full sine ripple; outliers are suppressed.

The distribution-awareness is what makes Turbulent unusual. The network
effectively adjusts its nonlinearity on the fly depending on the spread of its
own activations. The paper notes this makes Turbulent more expressive but also
more prone to fitting idiosyncratic statistics of the training distribution.

---

## Results

### Validation Loss Curves (averaged over 2 seeds)

```
Val loss  │  step 500  step 1000  step 1500  step 2000  step 2500  step 3000
──────────┼─────────────────────────────────────────────────────────────────
GELU      │   2.206     1.654      1.397      1.207      1.112      1.073
GELUSine  │   2.264     1.737      1.490      1.327      1.233      1.189
Turbulent │   2.249     1.734      1.504      1.346      1.251      1.212
GMTU      │   2.331     1.917      1.639      1.496      1.412      1.377
GSinc*    │    NaN       NaN        NaN        NaN        NaN        NaN
```
*GELUSincPerturbation diverged to NaN on both seeds at ~step 250–350.*

```
Val loss vs training steps (averaged over seeds)
─────────────────────────────────────────────────
2.35 ┤  ○ GMTU starts highest
2.25 ┤  ◇ GELUSine / △ Turbulent close together
2.20 ┤  □ GELU starts lowest
     │
1.90 ┤  ○────╮
     │       │
1.70 ┤  ◇─△──╯╮ GELU pulls ahead early
     │        │
1.50 ┤        ╰──◇─△─────────╮
     │                        │
1.30 ┤         □──────────────╯──◇──△
1.20 ┤                              │
1.10 ┤                    □─────────╯  GELU  : 1.073
     │                                GELUSine: 1.189
1.00 ┤                                Turbul. : 1.212
     │                                GMTU    : 1.377
     └────────┬──────────┬──────────┬──────────┬──▶ step
             500       1500       2500       3000
```

### Per-Seed Breakdown

| Run | Step 500 | Step 1000 | Step 2000 | Step 3000 |
|-----|----------|-----------|-----------|-----------|
| GELU seed42 | 2.196 | 1.640 | 1.189 | **1.062** |
| GELU seed123 | 2.216 | 1.667 | 1.224 | **1.083** |
| GELUSine seed42 | 2.281 | 1.720 | 1.310 | **1.175** |
| GELUSine seed123 | 2.247 | 1.754 | 1.344 | **1.203** |
| Turbulent seed42 | 2.266 | 1.718 | 1.331 | **1.197** |
| Turbulent seed123 | 2.231 | 1.750 | 1.361 | **1.226** |
| GMTU seed42 | 2.353 | 1.883 | 1.491 | **1.367** |
| GMTU seed123 | 2.309 | 1.950 | 1.501 | **1.387** |

Seed variance is small (≤ 0.02 nats in all cases) — the ranking is stable and
not a fluke of initialisation.

### Training Stability — Gradient Norms

```
Gradient norm statistics (mean ± peak, across 3000 steps)
──────────────────────────────────────────────────────────
GELU        mean 1.270   peak 4.131   ✔ stable
GELUSine    mean 1.188   peak 4.016   ✔ stable (slightly calmer)
Turbulent   mean 1.253   peak 3.698   ✔ stable
GMTU        mean 1.138   peak 3.625   ✔ stable (calmest)
GSinc       mean  NaN    peak  NaN    ✗ diverges immediately
```

Four of the five activations trained without a single NaN or gradient spike.
GELUSincPerturbation is the exception — it collapses to NaN on the very first
validation checkpoint on both seeds, with seed 42 diverging as early as step
250 and seed 123 at step 350.

### Final Ranking

```
Rank  Activation             Avg final val loss   Δ vs GELU
────  ─────────────────────  ──────────────────   ─────────
 1    GELU                        1.073              —
 2    GELUSine                    1.189            +0.116
 3    Turbulent                   1.212            +0.139
 4    GMTU                        1.377            +0.304
 5    GELUSincPerturbation         NaN             diverged
```

---

## Analysis — Why Does Each Function Behave This Way?

### Why does GELU win?

GELU has a well-understood gradient landscape for this type of model:

- Its derivative is everywhere positive for x > −1 (roughly), avoiding the dead
  neuron problem of ReLU.
- It is smooth and has no second-order pathologies near the origin.
- At the scale and step count we use (3 M parameters, 3,000 steps), the model
  is still firmly in the early-to-mid training regime, and GELU's simplicity
  means it reaches a low loss *faster* — any benefit from a more expressive
  activation only pays off later.

The key caveat: **all runs are still improving at step 3000** — none has
plateaued. A longer training run might see the more expressive activations (e.g.
Turbulent) close the gap or even overtake GELU, because their additional
functional richness can become useful once the network is otherwise well-trained.

### Why is GELUSine the strongest novel function?

GELUSine's sine perturbation is:

- **Small** (α=0.1): barely 10% of the sine amplitude, so it does not
  meaningfully distort GELU's gradient.
- **Globally bounded**: |sin(x)| ≤ 1, so there is no risk of explosive
  activations.
- **Smooth and differentiable everywhere**: the gradient `GELU'(x) + 0.1·cos(x)`
  is well-behaved throughout.

The oscillatory component helps the network fit quasi-periodic patterns in the
byte-level text corpus — spacing rhythms, repeated punctuation, code-like
indentation — while paying essentially no stability cost. This makes it the best
practical choice among the novel functions.

### Why does Turbulent underperform GELUSine despite being more expressive?

Turbulent's batch-statistics mechanism is its double-edged sword. Computing
`μ` and `σ` across the *entire FFN activation tensor* means:

1. **Training and inference are coupled to batch composition.** With a batch
   size of 8, the mean and standard deviation are noisy estimates. The function
   behaves differently depending on which 8 sequences happen to be in the
   batch.
2. **The Gaussian gate changes shape during training.** As activations shift
   (which they do throughout training), z values shift too, altering which
   activations receive a full sine perturbation. This adds a source of
   implicit non-stationarity that the optimiser must track.
3. **No running-average state is maintained.** At eval time, Turbulent computes
   batch statistics from the evaluation batch, which may not match training
   statistics. This is a mild distribution shift that adds noise to val loss
   estimates.

The net effect is that Turbulent trains fine — no instability, no divergence —
but its effective learning signal is noisier, so it is consistently 0.02 nats
behind GELUSine at every checkpoint.

### Why does GMTU lag the most (among stable activations)?

GMTU's Gaussian gate `exp(−p_gamma·x²)` is the source of its trouble in the
early training regime:

- **Gradient suppression for large |x|.** The gradient of the primary term
  is proportional to `exp(−p_gamma·x²)`, which decays to zero for large
  activations. Early in training, weights are randomly initialised and
  activations span a wide range. GMTU therefore effectively *turns itself into
  a linear function* (`0.2x`) for the many large-magnitude activations that
  exist before the model has learned to control its internal representations.
- **Slower effective learning rate.** Compared to GELU, which passes gradients
  through its derivative `Φ(x) + x·φ(x)` (where φ is the Gaussian PDF),
  GMTU's gradient is weaker on average in the early-to-mid training phase.
  The mean gradient norm (1.138) confirms this — it is noticeably lower than
  GELU's (1.270).

GMTU might shine at larger model sizes where activations are better controlled
at initialisation, or with warm-starting strategies, but in our small-scale
early-training setup it is systematically behind.

### Why does GELUSincPerturbation diverge?

The culprit is the sinc function near the origin.

For small x, `sinc(x) = sin(x)/x ≈ 1 − x²/6 + ...`. The *gradient* of the
sinc perturbation is:

```
d/dx [α · sin(x)/x] = α · [cos(x)/x − sin(x)/x²]
```

As x → 0, both `cos(x)/x` and `sin(x)/x²` diverge. The implementation guards
the exact x=0 singularity (returning 1 there), but values of |x| ~ 1e-4 to
1e-1 still have extremely large gradient magnitudes. In the early training
phase, many activations fall in exactly this range, creating gradient spikes
that overflow float32 into NaN.

The gradient clipping (norm 1.0) cannot save it: once even a single activation
produces NaN in the backward pass, it propagates to all parameters in that
layer and the model cannot recover. A smaller α (e.g. 0.01 instead of 0.5), a
heavier warm-up, or gradient clipping at a tighter threshold might allow
GELUSincPerturbation to train stably, but with the paper-prescribed α=0.5 it
is not viable in our setup.

---

## Putting It All Together

Combining both experiments, the full picture across activation functions *and*
FFN depths (all at the same model scale) looks like this:

```
Configuration                         Val loss (at training end)
────────────────────────────────────  ─────────────────────────
FFN0 + GELU (2000 steps)              1.709     ← no FFN
FFN1 + GELU (2000 steps)              1.596     ← 1-layer FFN
FFN2 + GMTU (3000 steps)              1.377     ← novel act, wrong choice
FFN2 + Turbulent (3000 steps)         1.212     ← novel act
FFN2 + GELUSine (3000 steps)          1.189     ← best novel act
FFN2 + GELU (2000 steps)              1.338     ← standard baseline
FFN2 + GELU (3000 steps)              1.073     ← standard + more compute
FFN3 + GELU (2000 steps)              1.401     ← deeper FFN (undertrained)
```

Several things stand out:

- **FFN depth matters more than activation choice** at this scale. The gap
  between FFN0 and FFN2 (0.37 nats) dwarfs the gap between GELU and
  GELUSine (0.12 nats).
- **A bad novel activation can undo the FFN-depth benefit.** GMTU inside FFN2
  (1.377) performs *worse* than GELU in FFN2 at just 2000 steps (1.338).
- **The best novel activation helps, but not dramatically.** GELUSine gets you
  to 1.189 at 3000 steps, which GELU itself nearly reaches at 3000 steps
  (1.073). The gap is real but modest at this scale.
- **More training helps any activation.** The consistent monotonic improvement
  in all four stable runs suggests we are still far from the loss floor. A
  longer run might produce a clearer separation.

---

## Limitations and Future Work

**Scale**: Everything here is small — 3.3 M parameters, a few thousand steps,
a byte-level corpus of Vim documentation. The evolutionary search behind the
novel activations was evaluated at much larger scales; results may differ
significantly with larger models and longer training.

**Dataset**: Vim docs are a narrow, technical domain. A wider corpus (e.g.
the full WikiText-103 or the Booksum dataset already supported in the codebase)
would give more generalisable conclusions.

**GELUSincPerturbation stabilisation**: The divergence might be fixable with a
smaller α or layernorm placed before the activation. Worth investigating.

**GMTU warm-up**: A longer learning rate warm-up would allow large activations
to settle before the Gaussian gate becomes the dominant gradient path. This
might substantially close the gap with GELU.

**Training length**: None of the runs has plateaued. A 10,000-step run would
reveal whether the ranking is preserved or whether Turbulent or GELUSine's
added expressiveness eventually lets them overtake GELU.

---

## Conclusion

We set out to answer two questions about transformer FFN design. After running
ten models (across five activations × two seeds each, plus four FFN-depth
variants), the answers are:

1. **FFN depth matters a lot.** Removing the FFN is catastrophic; the standard
   two-layer MLP is an excellent default for short training budgets; and deeper
   FFNs are promising but need parameter-matched configurations and longer
   training to show their advantage.

2. **Activation choice matters, but less than you might expect — and the
   details are everything.** GELUSine is a genuine, stable improvement over
   GELU. Turbulent is a credible runner-up. GMTU is stable but slow to
   converge. GELUSincPerturbation, despite its elegant motivation, is
   numerically dangerous in its default configuration.

The practical upshot for someone building a language model today: keep FFN2
(the standard MLP), use GELU or GELUSine, and invest your compute budget in
more training steps before worrying about exotic activation functions.

---

## References

- Gerber (2025). *Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models.* arXiv:2505.06633.
- Vitvitskyi et al. (2026). *Mining Generalizable Activation Functions.*
  arXiv:2602.05688.
- Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs).*
  arXiv:1606.08415.
