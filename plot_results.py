"""Generate plots replicating the figures from arXiv:2505.06633.

Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
Networks in Transformer Models."

Reproduces:
  - Figure 2: Log cross-entropy loss curves during training for the 4 FFN
    variants on the Booksum and WikiText-103 datasets.
  - A parameter-count summary table for all model configurations.

Usage::

    # Plot results after running run_experiments.py
    python plot_results.py --output_dir outputs --dataset wikitext

    # Specify which runs to plot
    python plot_results.py --runs outputs/small_ffn0_wikitext \\
                                   outputs/small_ffn1_wikitext \\
                                   outputs/small_ffn2_wikitext \\
                                   outputs/small_ffn3_wikitext
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Colour palette (distinct for 0, 1, 2, 3 FFN layers)
LAYER_COLORS = {
    0: "#d62728",   # red
    1: "#ff7f0e",   # orange
    2: "#1f77b4",   # blue   (baseline)
    3: "#2ca02c",   # green
}
LAYER_LABELS = {
    0: "0 layers (no FFN)",
    1: "1 layer",
    2: "2 layers (baseline)",
    3: "3 layers",
}


def load_history(run_dir: Path) -> Optional[dict]:
    """Load training history from a run directory.

    Args:
        run_dir: Directory containing history.json.

    Returns:
        History dictionary or None if not found.
    """
    history_file = run_dir / "history.json"
    if not history_file.exists():
        return None
    with open(history_file) as f:
        return json.load(f)


def infer_ffn_layers(run_name: str) -> int:
    """Infer the number of FFN layers from a run name."""
    for n in [3, 2, 1, 0]:
        if f"ffn{n}" in run_name:
            return n
    return -1


def plot_loss_curves(
    run_dirs: list[Path],
    title: str = "Training Loss by FFN Depth",
    log_y: bool = True,
    output_path: Optional[Path] = None,
) -> None:
    """Plot training loss curves for multiple runs.

    Replicates the style of Figure 2 in the paper: training log-loss vs step
    for models with different FFN layer counts.

    Args:
        run_dirs: List of run directories containing history.json.
        title: Plot title.
        log_y: If True, use log scale for the y-axis.
        output_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for run_dir in run_dirs:
        history = load_history(run_dir)
        if history is None:
            print(f"WARNING: No history.json in {run_dir}")
            continue

        ffn_n = infer_ffn_layers(run_dir.name)
        steps = history.get("step", [])
        losses = history.get("train_loss", [])

        if not steps or not losses:
            continue

        color = LAYER_COLORS.get(ffn_n, "gray")
        label = LAYER_LABELS.get(ffn_n, run_dir.name)

        y = [math.log(l) if log_y and l > 0 else l for l in losses]
        ax.plot(steps, y, color=color, label=label, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        print("No data to plot.")
        return

    ax.set_xlabel("Training Step", fontsize=12)
    ylabel = "Log Cross-Entropy Loss" if log_y else "Cross-Entropy Loss"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_val_loss_bar(
    run_dirs: list[Path],
    title: str = "Final Validation Loss by FFN Depth",
    output_path: Optional[Path] = None,
) -> None:
    """Bar chart of final validation loss for each FFN variant.

    Args:
        run_dirs: List of run directories containing history.json.
        title: Plot title.
        output_path: Optional path to save the figure.
    """
    data = {}
    for run_dir in run_dirs:
        history = load_history(run_dir)
        if history is None:
            continue
        ffn_n = infer_ffn_layers(run_dir.name)
        val_losses = history.get("val_loss", [])
        if not val_losses:
            continue
        final = val_losses[-1]
        if isinstance(final, dict):
            final = final["loss"]
        data[ffn_n] = final

    if not data:
        print("No validation loss data found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ordered_keys = sorted(data.keys())
    vals = [data[k] for k in ordered_keys]
    colors = [LAYER_COLORS.get(k, "gray") for k in ordered_keys]
    labels = [LAYER_LABELS.get(k, f"{k} layers") for k in ordered_keys]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def print_parameter_table() -> None:
    """Print a table of parameter counts for all model configurations.

    Replicates the parameter analysis described in Section 4 of the paper.
    """
    try:
        import torch
        from configs import ALL_CONFIGS
        from models.transformer import DecoderTransformer
    except ImportError as e:
        print(f"Cannot compute parameter counts: {e}")
        return

    print("\nParameter counts for all model configurations")
    print("=" * 90)
    print(f"{'Config':<30} {'d_model':>8} {'n_layers':>9} {'ffn_layers':>11} {'Total params':>15}")
    print("-" * 90)

    for name, cfg in sorted(ALL_CONFIGS.items()):
        try:
            model = DecoderTransformer(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                ffn_layers=cfg.ffn_layers,
                max_seq_len=cfg.max_seq_len,
                d_ff=cfg.d_ff,
                dropout=0.0,
            )
            counts = model.count_parameters()
            total = counts["total"]
            print(
                f"{name:<30} {cfg.d_model:>8} {cfg.n_layers:>9} {cfg.ffn_layers:>11} "
                f"{total:>15,}"
            )
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")

    print("=" * 90)
    print("Note: LM head is weight-tied with token embeddings and not counted separately.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot results from arXiv:2505.06633 replication"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=str,
        default=None,
        help="Specific run directories to plot (overrides --output_dir).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["booksum", "wikitext"],
        help="Filter runs by dataset name.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--param_table",
        action="store_true",
        help="Print parameter count table and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.param_table:
        print_parameter_table()
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find run directories
    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        output_dir = Path(args.output_dir)
        run_dirs = [
            d for d in sorted(output_dir.iterdir())
            if d.is_dir() and args.dataset in d.name
        ]

    if not run_dirs:
        print(f"No run directories found in {args.output_dir} for dataset '{args.dataset}'.")
        return

    print(f"Found {len(run_dirs)} run(s):")
    for r in run_dirs:
        print(f"  {r}")

    # Plot training loss curves (Figure 2 equivalent)
    dataset_label = args.dataset.capitalize()
    plot_loss_curves(
        run_dirs,
        title=f"Training Log-Loss by FFN Depth ({dataset_label})",
        log_y=True,
        output_path=save_dir / f"loss_curves_{args.dataset}.png",
    )

    # Plot final validation loss bar chart
    plot_val_loss_bar(
        run_dirs,
        title=f"Final Validation Loss by FFN Depth ({dataset_label})",
        output_path=save_dir / f"val_loss_bar_{args.dataset}.png",
    )

    # Also print parameter table
    print_parameter_table()


if __name__ == "__main__":
    main()
