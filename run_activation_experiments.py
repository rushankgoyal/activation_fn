"""Compare activation functions inside a standard transformer FFN block.

This script studies how the choice of activation function affects a decoder-only
transformer language model, mirroring the methodology of the FFN-depth ablation
in Gerber (2025) but replacing the depth variable with the activation function.

Activation functions evaluated
-------------------------------
gelu                  — GELU baseline (GPT-2/GPT-3 standard)
gelusine              — GELU(x) + 0.1·sin(x)          [Vitvitskyi et al., 2026]
gelusincperturbation  — GELU(x) + 0.5·sinc(x)          [Vitvitskyi et al., 2026]
gmtu                  — Gaussian-Modulated Tangent Unit [Vitvitskyi et al., 2026]
turbulent             — Batch-stats sinusoidal ripple   [Vitvitskyi et al., 2026]

Experimental design
--------------------
* Architecture : FFN2 (the standard 2-layer baseline), d_model=256, n_layers=4
* Tokenisation : byte-level (vocab=256), no network access required
* Corpus       : /usr/share/vim/vim91/doc/*.txt  (same as local FFN-depth runs)
* Steps        : 3000  (50 % longer than the FFN-depth sweep for richer curves)
* Seeds        : 2 independent seeds per activation to quantify training noise
* Metrics      : training loss, validation loss, gradient norm (logged every 50 steps)

The same local corpus, tokeniser, model scale, and training hyper-parameters are
used across all runs so that the only variable is the activation function.

Outputs
-------
outputs/act_{name}_seed{s}/history.json   — per-run training history
outputs/experiment_summary_activations.json — aggregated results table
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from configs import ModelConfig, TrainingConfig
from data.dataset import TextDataset, make_dataloader
from models.transformer import DecoderTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local corpus loader (byte-level, no network)
# ---------------------------------------------------------------------------

def load_local_corpus(glob_pattern: str = "/usr/share/vim/vim91/doc/*.txt") -> torch.Tensor:
    """Read all matching text files and return a byte-level token tensor."""
    files = sorted(Path("/").glob(glob_pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    logger.info(f"Loading {len(files)} local text file(s) as training corpus...")
    raw: list[int] = []
    for f in files:
        raw.extend(f.read_bytes())
    logger.info(f"  Total bytes: {len(raw):,}")
    return torch.tensor(raw, dtype=torch.long)


def build_datasets(
    token_ids: torch.Tensor,
    max_seq_len: int,
    train_fraction: float = 0.90,
) -> tuple[TextDataset, TextDataset]:
    n = len(token_ids)
    split = int(n * train_fraction)
    return TextDataset(token_ids[:split], max_seq_len), TextDataset(token_ids[split:], max_seq_len)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: DecoderTransformer,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int,
) -> float:
    model.eval()
    total_loss, n_eval = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_eval += 1
    model.train()
    return total_loss / max(1, n_eval)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_ds: TextDataset,
    val_ds: TextDataset,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Train one model variant and return the full history dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Seed: {seed}")

    train_loader = make_dataloader(
        train_ds, train_cfg.batch_size, shuffle=True,
        num_workers=min(2, os.cpu_count() or 1),
    )
    val_loader = make_dataloader(
        val_ds, train_cfg.batch_size, shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
    )

    model = DecoderTransformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        ffn_layers=model_cfg.ffn_layers,
        max_seq_len=model_cfg.max_seq_len,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        act_type=model_cfg.act_type,
    ).to(device)

    logger.info(model.describe())
    logger.info(f"Total parameters: {model.count_parameters()['total']:,}")

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    scheduler = get_lr_scheduler(optimizer, train_cfg.warmup_steps, train_cfg.max_steps)

    history: dict = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "grad_norm": [],  # gradient norm at each log step for training stability analysis
    }

    model.train()
    step = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    train_iter = iter(train_loader)
    t0 = time.perf_counter()

    logger.info(f"Training for {train_cfg.max_steps} steps...")

    while step < train_cfg.max_steps:
        optimizer.zero_grad()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()

        # Compute and log gradient norm before clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item()
        running_grad_norm += grad_norm

        optimizer.step()
        scheduler.step()
        step += 1
        running_loss += loss.item()

        if step % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            avg_grad_norm = running_grad_norm / train_cfg.log_interval
            lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            tok_s = (train_cfg.log_interval * train_cfg.batch_size * model_cfg.max_seq_len / elapsed)
            logger.info(
                f"  step {step:5d} | train_loss {avg_loss:.4f} | "
                f"grad_norm {avg_grad_norm:.3f} | lr {lr:.2e} | {tok_s:.0f} tok/s"
            )
            history["step"].append(step)
            history["train_loss"].append(avg_loss)
            history["lr"].append(lr)
            history["grad_norm"].append(avg_grad_norm)
            running_loss = 0.0
            running_grad_norm = 0.0
            t0 = time.perf_counter()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            val_loss = evaluate(model, val_loader, device, train_cfg.eval_batches)
            logger.info(f"  step {step:5d} | val_loss   {val_loss:.4f}")
            history["val_loss"].append({"step": step, "loss": val_loss})

    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved → {history_path}")
    return history


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def best_val_loss(history: dict) -> float:
    entries = history.get("val_loss", [])
    if not entries:
        return float("inf")
    return min(e["loss"] for e in entries)


def final_val_loss(history: dict) -> float:
    entries = history.get("val_loss", [])
    if not entries:
        return float("inf")
    return entries[-1]["loss"]


def steps_to_threshold(history: dict, threshold: float) -> int | None:
    """Return the first step at which val_loss drops below threshold, or None."""
    for e in history.get("val_loss", []):
        if e["loss"] < threshold:
            return e["step"]
    return None


def mean_grad_norm(history: dict) -> float:
    norms = history.get("grad_norm", [])
    return sum(norms) / len(norms) if norms else float("nan")


def peak_grad_norm(history: dict) -> float:
    norms = history.get("grad_norm", [])
    return max(norms) if norms else float("nan")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256      # byte-level tokenisation
MAX_SEQ_LEN = 512     # shorter sequences for CPU training
MAX_STEPS = 3_000     # 50 % longer than FFN-depth sweep for richer learning curves
BATCH_SIZE = 8
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 50
WARMUP_STEPS = 300
OUTPUT_DIR = Path("outputs")
SEEDS = [42, 123]     # two seeds to quantify training variance

# Each entry: (act_type, display_label, description)
ACTIVATION_EXPERIMENTS = [
    (
        "gelu",
        "GELU",
        "Standard GELU — GPT-2/GPT-3 baseline activation",
    ),
    (
        "gelusine",
        "GELUSine",
        "GELU(x) + 0.1·sin(x) — additive sine perturbation (Vitvitskyi et al., 2026)",
    ),
    (
        "gelusincperturbation",
        "GELUSinc-Perturbation",
        "GELU(x) + 0.5·sinc(x) — self-decaying sinc perturbation (Vitvitskyi et al., 2026)",
    ),
    (
        "gmtu",
        "GMTU",
        "Gaussian-Modulated Tangent Unit — tanh+Gaussian gate+linear leak (Vitvitskyi et al., 2026)",
    ),
    (
        "turbulent",
        "Turbulent",
        "Batch-stats sinusoidal ripple — sign·log base + sin·Gaussian(z) (Vitvitskyi et al., 2026)",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token_ids = load_local_corpus()
    train_ds, val_ds = build_datasets(token_ids, MAX_SEQ_LEN)
    logger.info(f"Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

    # ------------------------------------------------------------------
    # GELU-baseline FFN-depth reference values from prior experiments
    # (small_ffn{0-3}_local, 2000 steps, byte-level, same corpus)
    # ------------------------------------------------------------------
    PRIOR_REFERENCE = {
        "FFN0 (no FFN)":       1.7091,
        "FFN1 (1-layer GELU)": 1.5826,
        "FFN2 (2-layer GELU)": 1.4618,
        "FFN3 (3-layer GELU)": 1.3892,
    }

    all_results = []

    for act_type, label, description in ACTIVATION_EXPERIMENTS:
        for seed in SEEDS:
            run_name = f"act_{act_type}_seed{seed}"
            logger.info(f"\n{'='*72}")
            logger.info(f"  Activation : {label}")
            logger.info(f"  Description: {description}")
            logger.info(f"  Seed       : {seed}")
            logger.info(f"  Run name   : {run_name}")
            logger.info(f"{'='*72}")

            model_cfg = ModelConfig(
                vocab_size=VOCAB_SIZE,
                d_model=256,
                n_layers=4,
                n_heads=4,
                ffn_layers=2,          # standard 2-layer FFN (FFN2)
                max_seq_len=MAX_SEQ_LEN,
                dropout=0.1,
                act_type=act_type,
            )
            train_cfg = TrainingConfig(
                learning_rate=3e-4,
                weight_decay=0.1,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                warmup_steps=WARMUP_STEPS,
                max_steps=MAX_STEPS,
                batch_size=BATCH_SIZE,
                mixed_precision=False,
                eval_interval=EVAL_INTERVAL,
                eval_batches=EVAL_BATCHES,
                log_interval=LOG_INTERVAL,
                run_name=run_name,
            )
            output_dir = OUTPUT_DIR / run_name
            history = train_one(model_cfg, train_cfg, train_ds, val_ds, output_dir, seed=seed)

            result = {
                "run": run_name,
                "act_type": act_type,
                "label": label,
                "seed": seed,
                "final_val_loss": final_val_loss(history),
                "best_val_loss": best_val_loss(history),
                "mean_grad_norm": mean_grad_norm(history),
                "peak_grad_norm": peak_grad_norm(history),
                "steps_to_1_50": steps_to_threshold(history, 1.50),
                "steps_to_1_45": steps_to_threshold(history, 1.45),
            }
            all_results.append(result)

    # ------------------------------------------------------------------
    # Aggregate: average over seeds per activation
    # ------------------------------------------------------------------
    act_summary: dict[str, dict] = {}
    for r in all_results:
        key = r["act_type"]
        if key not in act_summary:
            act_summary[key] = {
                "label": r["label"],
                "final_val_losses": [],
                "best_val_losses": [],
                "mean_grad_norms": [],
                "peak_grad_norms": [],
                "steps_to_1_50": [],
                "steps_to_1_45": [],
            }
        s = act_summary[key]
        s["final_val_losses"].append(r["final_val_loss"])
        s["best_val_losses"].append(r["best_val_loss"])
        s["mean_grad_norms"].append(r["mean_grad_norm"])
        s["peak_grad_norms"].append(r["peak_grad_norm"])
        if r["steps_to_1_50"] is not None:
            s["steps_to_1_50"].append(r["steps_to_1_50"])
        if r["steps_to_1_45"] is not None:
            s["steps_to_1_45"].append(r["steps_to_1_45"])

    def avg(lst: list) -> float | None:
        return sum(lst) / len(lst) if lst else None

    # ------------------------------------------------------------------
    # Print results tables
    # ------------------------------------------------------------------
    W = 72
    logger.info(f"\n{'='*W}")
    logger.info("ACTIVATION FUNCTION COMPARISON")
    logger.info(f"Architecture: FFN2 (2-layer), d_model=256, n_layers=4, {MAX_STEPS} steps")
    logger.info(f"Corpus: byte-level local text  |  Seeds: {SEEDS}")
    logger.info(f"{'='*W}")

    # Table 1: Validation loss
    logger.info(f"\n  {'Activation':<30s}  {'Final val (avg)':>15}  {'Best val (avg)':>14}")
    logger.info(f"  {'-'*30}  {'-'*15}  {'-'*14}")
    ranked = sorted(act_summary.items(), key=lambda kv: avg(kv[1]["final_val_losses"]) or 9999)
    for act_type, s in ranked:
        fv = avg(s["final_val_losses"])
        bv = avg(s["best_val_losses"])
        fv_str = f"{fv:.4f}" if fv is not None else "N/A"
        bv_str = f"{bv:.4f}" if bv is not None else "N/A"
        marker = " ← best" if act_type == ranked[0][0] else ""
        logger.info(f"  {s['label']:<30s}  {fv_str:>15}  {bv_str:>14}{marker}")

    # Table 2: Gradient norm (training stability)
    logger.info(f"\n  {'Activation':<30s}  {'Mean ‖∇‖':>10}  {'Peak ‖∇‖':>10}")
    logger.info(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    for act_type, s in act_summary.items():
        mg = avg(s["mean_grad_norms"])
        pg = avg(s["peak_grad_norms"])
        mg_str = f"{mg:.3f}" if mg is not None else "N/A"
        pg_str = f"{pg:.3f}" if pg is not None else "N/A"
        logger.info(f"  {s['label']:<30s}  {mg_str:>10}  {pg_str:>10}")

    # Table 3: Convergence speed
    logger.info(f"\n  {'Activation':<30s}  {'Steps → <1.50':>13}  {'Steps → <1.45':>13}")
    logger.info(f"  {'-'*30}  {'-'*13}  {'-'*13}")
    for act_type, s in act_summary.items():
        s150 = avg(s["steps_to_1_50"])
        s145 = avg(s["steps_to_1_45"])
        s150_str = f"{s150:.0f}" if s150 is not None else "not reached"
        s145_str = f"{s145:.0f}" if s145 is not None else "not reached"
        logger.info(f"  {s['label']:<30s}  {s150_str:>13}  {s145_str:>13}")

    # Table 4: Prior reference comparison
    logger.info(f"\n  Reference (FFN-depth sweep, GELU, 2000 steps):")
    for name, val in PRIOR_REFERENCE.items():
        logger.info(f"    {name:<28s} → val_loss {val:.4f}")

    logger.info(f"\n{'='*W}")

    # ------------------------------------------------------------------
    # Save aggregated summary
    # ------------------------------------------------------------------
    summary = {
        "experiment": "activation_function_comparison",
        "architecture": "FFN2 (2-layer)",
        "d_model": 256,
        "n_layers": 4,
        "max_steps": MAX_STEPS,
        "seeds": SEEDS,
        "vocab": "byte-level",
        "prior_reference_gelu_ffn2_2000steps": PRIOR_REFERENCE["FFN2 (2-layer GELU)"],
        "per_run_results": all_results,
        "aggregated": {
            act_type: {
                "label": s["label"],
                "avg_final_val_loss": avg(s["final_val_losses"]),
                "avg_best_val_loss": avg(s["best_val_losses"]),
                "avg_mean_grad_norm": avg(s["mean_grad_norms"]),
                "avg_peak_grad_norm": avg(s["peak_grad_norms"]),
                "avg_steps_to_1_50": avg(s["steps_to_1_50"]),
                "avg_steps_to_1_45": avg(s["steps_to_1_45"]),
            }
            for act_type, s in act_summary.items()
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "experiment_summary_activations.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
