"""Run only the missing GMTU and Turbulent activation experiments."""

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

VOCAB_SIZE = 256
MAX_SEQ_LEN = 512
MAX_STEPS = 3_000
BATCH_SIZE = 8
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 50
WARMUP_STEPS = 300
OUTPUT_DIR = Path("outputs")
SEEDS = [42, 123]

MISSING_EXPERIMENTS = [
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


def load_local_corpus(glob_pattern: str = "/usr/share/vim/vim91/doc/*.txt") -> torch.Tensor:
    files = sorted(Path("/").glob(glob_pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    logger.info(f"Loading {len(files)} local text file(s)...")
    raw: list[int] = []
    for f in files:
        raw.extend(f.read_bytes())
    logger.info(f"  Total bytes: {len(raw):,}")
    return torch.tensor(raw, dtype=torch.long)


def build_datasets(token_ids: torch.Tensor, max_seq_len: int, train_fraction: float = 0.90):
    n = len(token_ids)
    split = int(n * train_fraction)
    return TextDataset(token_ids[:split], max_seq_len), TextDataset(token_ids[split:], max_seq_len)


def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataloader, device, n_batches):
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


def train_one(model_cfg, train_cfg, train_ds, val_ds, output_dir, seed=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Seed: {seed}")

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

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}

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


def main():
    token_ids = load_local_corpus()
    train_ds, val_ds = build_datasets(token_ids, MAX_SEQ_LEN)
    logger.info(f"Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

    for act_type, label, description in MISSING_EXPERIMENTS:
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
                ffn_layers=2,
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
            train_one(model_cfg, train_cfg, train_ds, val_ds, output_dir, seed=seed)

    logger.info("\nAll missing experiments complete.")


if __name__ == "__main__":
    main()
