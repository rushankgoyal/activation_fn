"""Run small-scale FFN-depth comparison experiments using only local resources.

Since Hugging Face Hub is unavailable in this environment, this script:
  - Reads text files from /usr/share/vim/vim91/doc/ as a local corpus
  - Tokenises at the byte level (vocab_size=256) — no tokenizer download needed
  - Trains the same four FFN-depth variants (0, 1, 2, 3 layers) that
    run_experiments.py would train at the 'small' scale
  - Saves results in the same history.json format for downstream tools

Byte-level tokenisation changes the absolute perplexity values but leaves the
*relative* ordering across FFN variants intact, which is the quantity of
interest in this ablation.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader = make_dataloader(train_ds, train_cfg.batch_size, shuffle=True,
                                   num_workers=min(2, os.cpu_count() or 1))
    val_loader = make_dataloader(val_ds, train_cfg.batch_size, shuffle=False,
                                 num_workers=min(2, os.cpu_count() or 1))

    model = DecoderTransformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        ffn_layers=model_cfg.ffn_layers,
        max_seq_len=model_cfg.max_seq_len,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
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

    history: dict = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

    model.train()
    step = 0
    running_loss = 0.0
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

        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        running_loss += loss.item()

        if step % train_cfg.log_interval == 0:
            avg = running_loss / train_cfg.log_interval
            lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            tok_s = (train_cfg.log_interval * train_cfg.batch_size * model_cfg.max_seq_len / elapsed)
            logger.info(f"  step {step:5d} | train_loss {avg:.4f} | lr {lr:.2e} | {tok_s:.0f} tok/s")
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["lr"].append(lr)
            running_loss = 0.0
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
# Main
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256   # byte-level tokenisation
MAX_SEQ_LEN = 512  # shorter than paper default to keep CPU training feasible
MAX_STEPS = 2_000
BATCH_SIZE = 8
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 50
WARMUP_STEPS = 200
OUTPUT_DIR = Path("outputs")

EXPERIMENTS = [
    (0, "small_ffn0_local", "0 FFN layers (no FFN)"),
    (1, "small_ffn1_local", "1 FFN layer"),
    (2, "small_ffn2_local", "2 FFN layers (baseline)"),
    (3, "small_ffn3_local", "3 FFN layers"),
]


def main() -> None:
    token_ids = load_local_corpus()
    train_ds, val_ds = build_datasets(token_ids, MAX_SEQ_LEN)
    logger.info(f"Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

    results = []

    for ffn_layers, run_name, description in EXPERIMENTS:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Experiment : {run_name}")
        logger.info(f"  Description: {description}")
        logger.info(f"{'='*70}")

        model_cfg = ModelConfig(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            n_layers=4,
            n_heads=4,
            ffn_layers=ffn_layers,
            max_seq_len=MAX_SEQ_LEN,
            dropout=0.1,
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
        history = train_one(model_cfg, train_cfg, train_ds, val_ds, output_dir)

        val_entries = history.get("val_loss", [])
        final_val = val_entries[-1]["loss"] if val_entries else None
        results.append({
            "run": run_name,
            "ffn_layers": ffn_layers,
            "description": description,
            "final_val_loss": final_val,
        })

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY (byte-level LM, local corpus, 2000 steps)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Run':<30s}  {'FFN layers':>10}  {'Final val loss':>14}")
    logger.info(f"  {'-'*30}  {'-'*10}  {'-'*14}")
    for r in results:
        val = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "N/A"
        logger.info(f"  {r['run']:<30s}  {r['ffn_layers']:>10}  {val:>14}")

    summary_path = OUTPUT_DIR / "experiment_summary_local.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": results, "max_steps": MAX_STEPS, "vocab": "byte-level"}, f, indent=2)
    logger.info(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
