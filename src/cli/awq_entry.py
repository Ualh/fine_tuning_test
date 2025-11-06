"""Minimal AWQ conversion CLI targeted for container execution.

Stage: post-processing — invoked by run_pipeline.bat to drive logs/run-*/convert-awq.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .awq_convert import run_convert_awq


def _build_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the AWQ entry point."""

    parser = argparse.ArgumentParser(description="Run the AWQ conversion stage using repository defaults.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the pipeline configuration file (defaults to config.yaml).",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Optional path to the merged model directory (defaults to outputs/<run>/merged).",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Optional AWQ output directory override (defaults to outputs/<run>/merged_awq).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target AWQ directory when it already exists.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Explicit run slug to reuse for logging and outputs.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Force the run counter index when selecting a run slug.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and dispatch to the shared AWQ conversion helper."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    run_convert_awq(
        config_path=args.config,
        merged_dir=args.merged_dir,
        output_dir=args.output_dir,
        force=args.force,
        run_name=args.run_name,
        run_index=args.run_index,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
