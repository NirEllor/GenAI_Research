"""Experiment configuration and path management."""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExpPaths:
    """Paths for experiment outputs."""
    root: Path
    exp_dir: Path
    ckpt_dir: Path
    config_file: Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent


def get_paths(exp_name: str = None) -> ExpPaths:
    """Get experiment paths, creating directories as needed.

    Checkpoints are saved directly to: {repo_root}/checkpoints/
    """
    repo_root = get_repo_root()
    checkpoints_dir = repo_root / "checkpoints"

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = checkpoints_dir / exp_name
    ckpt_dir = exp_dir  # checkpoints saved directly in exp_dir
    config_file = exp_dir / "config.json"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return ExpPaths(
        root=checkpoints_dir,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        config_file=config_file
    )


def add_exp_arg(parser: argparse.ArgumentParser) -> None:
    """Add experiment name argument to parser."""
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (default: timestamp)"
    )


def save_config(paths: ExpPaths, section: str = None, extra: dict = None) -> None:
    """Save experiment configuration to JSON."""
    config = {}

    if paths.config_file.exists():
        with open(paths.config_file, "r") as f:
            config = json.load(f)

    if section and extra:
        config[section] = extra

    with open(paths.config_file, "w") as f:
        json.dump(config, f, indent=2)


def print_exp_summary(paths: ExpPaths, ckpt_path: Path = None) -> None:
    """Print experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Root directory:   {paths.root}")
    print(f"Experiment dir:   {paths.exp_dir}")
    print(f"Checkpoints:      {paths.ckpt_dir}")
    if ckpt_path:
        print(f"Checkpoint path:  {ckpt_path}")
    print(f"Config file:      {paths.config_file}")
    print("="*60 + "\n")
