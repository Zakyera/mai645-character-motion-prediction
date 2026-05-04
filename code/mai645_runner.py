"""Shared command-line scaffolding for the MAI 645 motion prediction scripts.

The assignment handout references an existing GitHub repository and BVH dataset,
but those paths are not available in this workspace yet. These entrypoints keep
the required script names stable while making the missing integration points
explicit for Colab.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPRESENTATIONS = {
    "pos": "positional",
    "euler": "Euler angle",
    "quad": "quaternion",
}

TODO_TOKENS = ("TODO", "REPLACE_ME", "UNKNOWN", "<")


def main_preprocess(representation: str) -> None:
    parser = _base_parser(representation, "preprocessing")
    parser.add_argument(
        "--assignment-repo",
        default=os.environ.get("ASSIGNMENT_REPO_PATH", "TODO_original_assignment_repo"),
        help="Path to the original assignment repository. TODO until the source repo is known.",
    )
    parser.add_argument(
        "--bvh-dir",
        default=os.environ.get("BVH_DATASET_PATH", "TODO_bvh_dataset_folder"),
        help="Folder containing source BVH files. TODO until the dataset path is known.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("PROCESSED_DIR", f"processed/{representation}"),
        help="Directory where processed numpy arrays should be written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for quick experiments. Use 0 for all files.",
    )

    args = parser.parse_args()
    _print_header("preprocessing", representation, args.dry_run)
    _print_items(
        {
            "assignment_repo": args.assignment_repo,
            "bvh_dir": args.bvh_dir,
            "output_dir": args.output_dir,
            "max_files": args.max_files or "all",
        }
    )

    if args.dry_run:
        _print_dry_run_footer("preprocessing", representation)
        return

    assignment_repo = _require_existing_dir(args.assignment_repo, "assignment repository")
    bvh_dir = _require_existing_dir(args.bvh_dir, "BVH dataset")
    output_dir = _prepare_output_dir(args.output_dir, "processed output")
    bvh_files = _find_bvh_files(bvh_dir, args.max_files)
    if not bvh_files:
        raise SystemExit(f"No .bvh files were found under {bvh_dir}.")

    raise SystemExit(
        _todo_message(
            action="preprocessing",
            representation=representation,
            details=(
                f"Found {len(bvh_files)} BVH file(s), assignment repo at {assignment_repo}, "
                f"and output directory {output_dir}.\n"
                "TODO: adapt the original repository's BVH parser/conversion code here so "
                f"{REPRESENTATIONS[representation]} arrays are written for training."
            ),
        )
    )


def main_train(representation: str) -> None:
    parser = _base_parser(representation, "training")
    parser.add_argument(
        "--assignment-repo",
        default=os.environ.get("ASSIGNMENT_REPO_PATH", "TODO_original_assignment_repo"),
        help="Path to the original assignment repository. TODO until the source repo is known.",
    )
    parser.add_argument(
        "--processed-dir",
        default=os.environ.get("PROCESSED_DIR", f"processed/{representation}"),
        help="Directory containing processed arrays for this representation.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", f"models/{representation}"),
        help="Directory where model checkpoints should be written.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    args = parser.parse_args()
    _print_header("training", representation, args.dry_run)
    _print_items(
        {
            "assignment_repo": args.assignment_repo,
            "processed_dir": args.processed_dir,
            "model_dir": args.model_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "learning_rate": args.learning_rate,
            "device": args.device,
        }
    )

    if args.dry_run:
        _print_dry_run_footer("training", representation)
        return

    assignment_repo = _require_existing_dir(args.assignment_repo, "assignment repository")
    processed_dir = _require_existing_dir(args.processed_dir, "processed data")
    model_dir = _prepare_output_dir(args.model_dir, "model output")

    raise SystemExit(
        _todo_message(
            action="training",
            representation=representation,
            details=(
                f"Using processed data at {processed_dir}, assignment repo at {assignment_repo}, "
                f"and model directory {model_dir}.\n"
                "TODO: connect the AC-LSTM model, representation-specific input channels, "
                "and the required loss function for this representation."
            ),
        )
    )


def main_synthesise(representation: str) -> None:
    parser = _base_parser(representation, "evaluation")
    parser.add_argument(
        "--assignment-repo",
        default=os.environ.get("ASSIGNMENT_REPO_PATH", "TODO_original_assignment_repo"),
        help="Path to the original assignment repository. TODO until the source repo is known.",
    )
    parser.add_argument(
        "--processed-dir",
        default=os.environ.get("PROCESSED_DIR", f"processed/{representation}"),
        help="Directory containing processed arrays for real seed/target motion.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", f"models/{representation}/TODO_model.pt"),
        help="Path to the trained checkpoint for this representation.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", f"outputs/{representation}"),
        help="Directory where generated BVH files and metrics should be written.",
    )
    parser.add_argument("--seed-frames", type=int, default=20)
    parser.add_argument("--generated-frames", type=int, default=400)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    args = parser.parse_args()
    _print_header("evaluation", representation, args.dry_run)
    _print_items(
        {
            "assignment_repo": args.assignment_repo,
            "processed_dir": args.processed_dir,
            "model_path": args.model_path,
            "output_dir": args.output_dir,
            "seed_frames": args.seed_frames,
            "generated_frames": args.generated_frames,
            "device": args.device,
        }
    )

    if args.dry_run:
        _print_dry_run_footer("evaluation", representation)
        return

    assignment_repo = _require_existing_dir(args.assignment_repo, "assignment repository")
    processed_dir = _require_existing_dir(args.processed_dir, "processed data")
    model_path = _require_existing_file(args.model_path, "trained model checkpoint")
    output_dir = _prepare_output_dir(args.output_dir, "synthesis output")

    raise SystemExit(
        _todo_message(
            action="evaluation",
            representation=representation,
            details=(
                f"Using processed data at {processed_dir}, model {model_path}, "
                f"assignment repo at {assignment_repo}, and output directory {output_dir}.\n"
                "TODO: load the trained model, compute quantitative error for the first "
                "generated frames, and decode approximately 400 generated frames to BVH."
            ),
        )
    )


def _base_parser(representation: str, stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{stage.title()} entrypoint for {REPRESENTATIONS[representation]} representation."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned command and TODO integration points without requiring data paths.",
    )
    return parser


def _looks_like_todo(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return any(token in text for token in TODO_TOKENS)


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _require_existing_dir(value: str, label: str) -> Path:
    if _looks_like_todo(value):
        raise SystemExit(f"TODO: set the {label} path. Current value: {value}")
    path = _expand_path(value)
    if not path.is_dir():
        raise SystemExit(f"Expected {label} directory to exist: {path}")
    return path


def _require_existing_file(value: str, label: str) -> Path:
    if _looks_like_todo(value):
        raise SystemExit(f"TODO: set the {label} path. Current value: {value}")
    path = _expand_path(value)
    if not path.is_file():
        raise SystemExit(f"Expected {label} file to exist: {path}")
    return path


def _prepare_output_dir(value: str, label: str) -> Path:
    if _looks_like_todo(value):
        raise SystemExit(f"TODO: set the {label} path. Current value: {value}")
    path = _expand_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_bvh_files(root: Path, max_files: int) -> list[Path]:
    files = sorted(root.rglob("*.bvh"))
    if max_files > 0:
        return files[:max_files]
    return files


def _print_header(stage: str, representation: str, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}MAI645 {stage}: {representation} ({REPRESENTATIONS[representation]})")


def _print_items(items: dict[str, object]) -> None:
    for key, value in items.items():
        print(f"  {key}: {value}")


def _print_dry_run_footer(stage: str, representation: str) -> None:
    print(
        f"[dry-run] {stage} command for {representation} is wired correctly. "
        "Replace TODO paths and fill in code/mai645_runner.py adapters for real execution."
    )


def _todo_message(action: str, representation: str, details: str) -> str:
    return (
        f"\nTODO: {action} adapter for {representation} is not implemented yet.\n"
        f"{details}\n"
    )
