"""Evaluation script for trained decoder-only transformer language models.

Loads a checkpoint and computes cross-entropy loss on the test split of a
given dataset, replicating the evaluation protocol from:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

The paper reports mean cross-entropy loss (equivalently, log perplexity) on
the test (Booksum) and validation (WikiText-103) splits.

Usage::

    python evaluate.py --checkpoint outputs/ffn2_wikitext/ckpt_step010000.pt \\
                       --dataset wikitext

    python evaluate.py --checkpoint outputs/ffn3_booksum/ckpt_step100000.pt \\
                       --dataset booksum --n_batches 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from configs import ModelConfig
from data.dataset import load_booksum, load_wikitext, make_dataloader
from models.transformer import DecoderTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_name: str,
    n_batches: int = 100,
    batch_size: int = 8,
    cache_dir: str | None = None,
) -> dict:
    """Evaluate a checkpoint on the test split.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train.py.
        dataset_name: 'booksum' or 'wikitext'.
        n_batches: Number of batches to evaluate over.
        batch_size: Batch size for evaluation.
        cache_dir: Optional HuggingFace dataset cache directory.

    Returns:
        Dictionary with 'loss', 'perplexity', and model metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_config_dict = ckpt["model_config"]
    model_cfg = ModelConfig(**model_config_dict)

    logger.info(f"Loaded checkpoint from step {ckpt['step']}")
    logger.info(f"  ffn_layers={model_cfg.ffn_layers}, d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # Build model
    model = DecoderTransformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        ffn_layers=model_cfg.ffn_layers,
        max_seq_len=model_cfg.max_seq_len,
        d_ff=model_cfg.d_ff,
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(model.describe())

    # Load dataset
    logger.info(f"Loading {dataset_name} test split...")
    if dataset_name == "booksum":
        ds = load_booksum("test", model_cfg.max_seq_len, cache_dir=cache_dir)
    elif dataset_name == "wikitext":
        ds = load_wikitext("validation", model_cfg.max_seq_len, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    loader = make_dataloader(ds, batch_size, shuffle=False, num_workers=0)

    # Evaluate
    use_amp = device.type == "cuda"
    total_loss = 0.0
    n_evaluated = 0
    all_losses = []

    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
        all_losses.append(loss.item())
        n_evaluated += 1
        if (i + 1) % 20 == 0:
            logger.info(f"  Evaluated {i+1}/{n_batches} batches, mean loss: {total_loss / n_evaluated:.4f}")

    mean_loss = total_loss / max(1, n_evaluated)
    perplexity = math.exp(mean_loss)

    # Standard error of the mean
    if len(all_losses) > 1:
        import statistics
        std = statistics.stdev(all_losses)
        sem = std / math.sqrt(len(all_losses))
    else:
        sem = 0.0

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "step": ckpt["step"],
        "ffn_layers": model_cfg.ffn_layers,
        "d_model": model_cfg.d_model,
        "n_layers": model_cfg.n_layers,
        "n_batches_evaluated": n_evaluated,
        "mean_loss": mean_loss,
        "sem_loss": sem,
        "perplexity": perplexity,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"  Dataset      : {dataset_name}")
    logger.info(f"  FFN layers   : {model_cfg.ffn_layers}")
    logger.info(f"  d_model      : {model_cfg.d_model}")
    logger.info(f"  n_layers     : {model_cfg.n_layers}")
    logger.info(f"  Mean loss    : {mean_loss:.4f} ± {sem:.4f}")
    logger.info(f"  Perplexity   : {perplexity:.2f}")
    logger.info(f"{'='*60}\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint (arXiv:2505.06633 replication)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--dataset", type=str, default="wikitext", choices=["booksum", "wikitext"],
        help="Evaluation dataset."
    )
    parser.add_argument("--n_batches", type=int, default=100, help="Number of evaluation batches.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.dataset,
        args.n_batches,
        args.batch_size,
        args.cache_dir,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
