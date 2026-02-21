"""Training script for decoder-only transformer language models.

Replicates the pre-training procedure from:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

Training details:
  - Optimiser : AdamW (betas=(0.9, 0.95), weight_decay=0.1)
  - Scheduler : linear warm-up + cosine decay
  - Objective : cross-entropy next-token prediction
  - Datasets  : Booksum (kmfoda/booksum) or WikiText-103 (wikitext-103-raw-v1)
  - Dropout   : 10%
  - Precision : bfloat16 / float16 mixed-precision on GPU

Usage examples::

    # Train small model on WikiText-103 (quick test)
    python train.py --config small_2layer --dataset wikitext --max_steps 2000

    # Train baseline paper model on Booksum (requires A100)
    python train.py --config paper_2layer --dataset booksum --max_steps 100000

    # Train all FFN variants on WikiText-103 (small scale)
    for ffn in 0 1 2 3; do
        python train.py --ffn_layers $ffn --dataset wikitext --max_steps 5000
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from configs import ModelConfig, TrainingConfig, get_config, ALL_CONFIGS
from data.dataset import load_booksum, load_wikitext, make_dataloader
from models.transformer import DecoderTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warm-up + cosine decay
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine schedule with linear warm-up.

    LR increases linearly from 0 to peak over `warmup_steps`, then follows a
    cosine decay to `min_lr_ratio * peak_lr`.
    """

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
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Compute mean cross-entropy loss over `n_batches` batches.

    Returns:
        Mean loss (float).
    """
    model.eval()
    total_loss = 0.0
    n_evaluated = 0
    for i, (x, y) in enumerate(dataloader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_evaluated += 1
    model.train()
    return total_loss / max(1, n_evaluated)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    dataset_name: str,
    output_dir: Path,
    cache_dir: Optional[str] = None,
) -> dict:
    """Run the full training loop.

    Args:
        model_cfg: Model architecture configuration.
        train_cfg: Training hyper-parameters.
        dataset_name: 'booksum' or 'wikitext'.
        output_dir: Directory for checkpoints and logs.
        cache_dir: Optional HuggingFace dataset cache directory.

    Returns:
        Dictionary with training history (step, train_loss, val_loss).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device & precision -----------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        train_cfg.mixed_precision = False

    logger.info(f"Device: {device}  |  AMP dtype: {amp_dtype}")

    # ---- Datasets -----------------------------------------------------------
    logger.info(f"Loading {dataset_name} dataset...")
    if dataset_name == "booksum":
        train_ds = load_booksum("train", model_cfg.max_seq_len, cache_dir=cache_dir)
        val_ds = load_booksum("test", model_cfg.max_seq_len, cache_dir=cache_dir)
    elif dataset_name == "wikitext":
        train_ds = load_wikitext("train", model_cfg.max_seq_len, cache_dir=cache_dir)
        val_ds = load_wikitext("validation", model_cfg.max_seq_len, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Use 'booksum' or 'wikitext'.")

    train_loader = make_dataloader(
        train_ds,
        train_cfg.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
    )
    val_loader = make_dataloader(
        val_ds,
        train_cfg.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
    )

    # ---- Model --------------------------------------------------------------
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

    if train_cfg.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    logger.info(model.describe())
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")

    # ---- Optimiser & scheduler ----------------------------------------------
    # Separate parameters for weight decay (all 2-D tensors) vs not
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
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.mixed_precision and device.type == "cuda"))

    # ---- Training history ---------------------------------------------------
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

    # ---- Training loop ------------------------------------------------------
    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    train_loader_iter = iter(train_loader)
    t0 = time.perf_counter()

    logger.info(f"Starting training for {train_cfg.max_steps} steps...")

    while step < train_cfg.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(train_cfg.grad_accum_steps):
            try:
                x, y = next(train_loader_iter)
            except StopIteration:
                epoch += 1
                train_loader_iter = iter(train_loader)
                x, y = next(train_loader_iter)

            x, y = x.to(device), y.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=train_cfg.mixed_precision,
            ):
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        running_loss += accum_loss

        # ---- Logging --------------------------------------------------------
        if step % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0
            tokens_per_sec = (
                train_cfg.log_interval
                * train_cfg.batch_size
                * train_cfg.grad_accum_steps
                * model_cfg.max_seq_len
                / elapsed
            )
            logger.info(
                f"step {step:6d} | loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )
            history["step"].append(step)
            history["train_loss"].append(avg_loss)
            history["lr"].append(current_lr)
            running_loss = 0.0
            t0 = time.perf_counter()

        # ---- Evaluation -----------------------------------------------------
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            val_loss = evaluate(
                model, val_loader, device,
                train_cfg.eval_batches, train_cfg.mixed_precision, amp_dtype
            )
            logger.info(f"step {step:6d} | val_loss {val_loss:.4f}")
            history["val_loss"].append({"step": step, "loss": val_loss})

            # Save checkpoint
            ckpt_path = output_dir / f"ckpt_step{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": model_cfg.__dict__,
                    "train_config": train_cfg.__dict__,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved to {ckpt_path}")

    # Save training history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a decoder-only transformer (arXiv:2505.06633 replication)"
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--config",
        type=str,
        choices=list(ALL_CONFIGS.keys()),
        help="Named model configuration (see configs.py).",
    )
    model_group.add_argument(
        "--ffn_layers",
        type=int,
        choices=[0, 1, 2, 3],
        help="Number of FFN linear layers (uses default d_model=256, n_layers=4 for quick testing).",
    )

    # Model overrides
    parser.add_argument("--d_model", type=int, help="Override model dimension.")
    parser.add_argument("--n_layers", type=int, help="Override number of transformer blocks.")
    parser.add_argument("--n_heads", type=int, help="Override number of attention heads.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Context window length.")
    parser.add_argument("--dropout", type=float, help="Override dropout probability.")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["booksum", "wikitext"],
        help="Training dataset.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace dataset cache dir.")

    # Training hyper-parameters
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed-precision.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build model config
    if args.config:
        model_cfg = get_config(args.config)
    elif args.ffn_layers is not None:
        model_cfg = ModelConfig(
            d_model=args.d_model or 256,
            n_layers=args.n_layers or 4,
            n_heads=args.n_heads or 4,
            ffn_layers=args.ffn_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout or 0.1,
        )
    else:
        # Default: small 2-layer model for quick testing
        model_cfg = ModelConfig(
            d_model=256, n_layers=4, n_heads=4, ffn_layers=2,
            max_seq_len=args.max_seq_len,
        )

    # Apply overrides
    if args.d_model:
        model_cfg.d_model = args.d_model
    if args.n_layers:
        model_cfg.n_layers = args.n_layers
    if args.n_heads:
        model_cfg.n_heads = args.n_heads
    if args.dropout:
        model_cfg.dropout = args.dropout

    # Build training config
    run_name = args.run_name or f"ffn{model_cfg.ffn_layers}_{args.dataset}"
    train_cfg = TrainingConfig(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        mixed_precision=not args.no_amp,
        compile_model=args.compile,
        run_name=run_name,
    )

    output_dir = Path(args.output_dir) / run_name

    train(model_cfg, train_cfg, args.dataset, output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
