"""Orchestrate the full set of experiments from arXiv:2505.06633.

Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
Networks in Transformer Models."

This script trains all model variants described in the paper and saves results
to disk.  The paper experiments were run at scale (A100, ~66 hours); this script
provides sensible defaults for smaller-scale hardware.

Experiment groups
-----------------
1. Primary FFN depth comparison (Section 5.1)
   Train models with 0, 1, 2, 3 FFN layers, same d_model and n_layers,
   on both Booksum and WikiText-103.

2. Three-layer FFN with matched parameter count (Section 5.2)
   Compare 2-layer 24-block vs 3-layer 10-block at similar total parameters.

3. Higher-dimension 0-layer and 1-layer (Section 5.3)
   Increase d_model to match the baseline parameter count for models with
   fewer FFN layers.

Usage::

    # Run all experiments at small scale (quick test, CPU-friendly)
    python run_experiments.py --scale small --dataset wikitext --max_steps 2000

    # Run primary comparison at medium scale
    python run_experiments.py --scale medium --dataset wikitext --max_steps 10000

    # Run full paper-scale experiments (requires A100)
    python run_experiments.py --scale paper --dataset booksum --max_steps 100000

    # Run only experiment group 1
    python run_experiments.py --group primary --scale small --max_steps 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ExperimentSpec(NamedTuple):
    name: str
    config: str          # Named config from configs.py
    group: str           # 'primary' | 'matched' | 'highdim'
    description: str


# All experiments defined in the paper, grouped by scale
EXPERIMENTS = {
    "small": [
        # Group 1: Primary FFN depth comparison
        ExperimentSpec("small_ffn0", "small_0layer", "primary", "0 FFN layers (no FFN)"),
        ExperimentSpec("small_ffn1", "small_1layer", "primary", "1 FFN layer"),
        ExperimentSpec("small_ffn2", "small_2layer", "primary", "2 FFN layers (baseline)"),
        ExperimentSpec("small_ffn3", "small_3layer", "primary", "3 FFN layers"),
    ],
    "medium": [
        # Group 1: Primary FFN depth comparison
        ExperimentSpec("medium_ffn0", "medium_0layer", "primary", "0 FFN layers (no FFN)"),
        ExperimentSpec("medium_ffn1", "medium_1layer", "primary", "1 FFN layer"),
        ExperimentSpec("medium_ffn2", "medium_2layer", "primary", "2 FFN layers (baseline)"),
        ExperimentSpec("medium_ffn3", "medium_3layer", "primary", "3 FFN layers"),
        # Group 2: Parameter-matched 3-layer
        ExperimentSpec("medium_ffn3_matched", "medium_3layer_matched", "matched",
                       "3 FFN layers, fewer blocks (~same params as 2-layer baseline)"),
    ],
    "paper": [
        # Group 1: Primary FFN depth comparison (paper Table 1 / Figure 2)
        ExperimentSpec("paper_ffn0", "paper_0layer", "primary", "0 FFN layers (no FFN)"),
        ExperimentSpec("paper_ffn1", "paper_1layer", "primary", "1 FFN layer"),
        ExperimentSpec("paper_ffn2", "paper_2layer", "primary", "2 FFN layers (baseline ~323M)"),
        ExperimentSpec("paper_ffn3", "paper_3layer", "primary", "3 FFN layers (~726M)"),
        # Group 2: Parameter-matched 3-layer
        ExperimentSpec("paper_ffn3_matched", "paper_3layer_matched", "matched",
                       "3 FFN layers, ~10 blocks (~same params as 2-layer 24-block)"),
        # Group 3: Higher-dimension 0-layer and 1-layer
        ExperimentSpec("paper_ffn0_hd", "paper_0layer_hd", "highdim",
                       "0 FFN layers, d_model=1792 (matched params)"),
        ExperimentSpec("paper_ffn1_hd", "paper_1layer_hd", "highdim",
                       "1 FFN layer, d_model=1600 (matched params)"),
    ],
}

# Training step counts per scale
DEFAULT_STEPS = {
    "small": 2_000,
    "medium": 20_000,
    "paper": 100_000,
}

# Batch sizes per scale
DEFAULT_BATCH = {
    "small": 8,
    "medium": 16,
    "paper": 32,
}


def run_training(
    spec: ExperimentSpec,
    dataset: str,
    max_steps: int,
    batch_size: int,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> Path:
    """Launch train.py as a subprocess for a given experiment.

    Args:
        spec: Experiment specification.
        dataset: 'booksum' or 'wikitext'.
        max_steps: Maximum training steps.
        batch_size: Batch size.
        output_dir: Root output directory.
        extra_args: Additional CLI arguments to pass to train.py.

    Returns:
        Path to the experiment output directory.
    """
    run_name = f"{spec.name}_{dataset}"
    exp_output_dir = output_dir / run_name

    cmd = [
        sys.executable, "train.py",
        "--config", spec.config,
        "--dataset", dataset,
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--output_dir", str(output_dir),
        "--run_name", run_name,
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"\n{'='*70}")
    logger.info(f"  Experiment : {spec.name}")
    logger.info(f"  Config     : {spec.config}")
    logger.info(f"  Description: {spec.description}")
    logger.info(f"  Dataset    : {dataset}")
    logger.info(f"  Max steps  : {max_steps}")
    logger.info(f"  Command    : {' '.join(cmd)}")
    logger.info(f"{'='*70}")

    result = subprocess.run(cmd, check=True)
    return exp_output_dir


def collect_results(output_dir: Path, dataset: str) -> list[dict]:
    """Gather training histories from all experiment subdirectories.

    Args:
        output_dir: Root output directory.
        dataset: Dataset name used for filtering.

    Returns:
        List of result dictionaries.
    """
    results = []
    for history_file in sorted(output_dir.rglob("history.json")):
        if dataset not in history_file.parent.name:
            continue
        with open(history_file) as f:
            history = json.load(f)
        # Extract final validation loss
        val_losses = history.get("val_loss", [])
        if val_losses:
            final_val = val_losses[-1]["loss"] if isinstance(val_losses[-1], dict) else val_losses[-1]
        else:
            final_val = None
        results.append({
            "run": history_file.parent.name,
            "final_val_loss": final_val,
            "history_file": str(history_file),
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiments from arXiv:2505.06633"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "paper"],
        help="Experiment scale. 'small' for quick testing, 'paper' for full replication.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "primary", "matched", "highdim"],
        help="Which experiment group to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["booksum", "wikitext", "both"],
        help="Dataset(s) to train on. 'both' replicates the paper which uses both.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or DEFAULT_STEPS[args.scale]
    batch_size = args.batch_size or DEFAULT_BATCH[args.scale]

    experiments = EXPERIMENTS[args.scale]
    if args.group != "all":
        experiments = [e for e in experiments if e.group == args.group]

    datasets = ["wikitext", "booksum"] if args.dataset == "both" else [args.dataset]

    extra_args = []
    if args.cache_dir:
        extra_args.extend(["--cache_dir", args.cache_dir])

    logger.info(f"Running {len(experiments)} experiment(s) x {len(datasets)} dataset(s)")
    logger.info(f"Scale: {args.scale} | Max steps: {max_steps} | Batch size: {batch_size}")

    completed = []
    failed = []

    for dataset in datasets:
        for spec in experiments:
            try:
                run_training(spec, dataset, max_steps, batch_size, output_dir, extra_args)
                completed.append(f"{spec.name}_{dataset}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Experiment {spec.name} on {dataset} failed: {e}")
                failed.append(f"{spec.name}_{dataset}")

    logger.info(f"\n{'='*70}")
    logger.info(f"Completed: {len(completed)} / {len(experiments) * len(datasets)}")
    if failed:
        logger.warning(f"Failed: {failed}")

    # Collect and print summary
    for dataset in datasets:
        results = collect_results(output_dir, dataset)
        if results:
            logger.info(f"\nResults for {dataset}:")
            for r in results:
                val = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "N/A"
                logger.info(f"  {r['run']:50s}  val_loss={val}")

    # Save summary
    summary = {
        "completed": completed,
        "failed": failed,
        "scale": args.scale,
        "datasets": datasets,
        "max_steps": max_steps,
    }
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
