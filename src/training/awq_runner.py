#!/usr/bin/env python3
"""AWQ runner script for the awq-runner container.

This script is intended to be executed inside the `awq-runner` image where
`llmcompressor` is installed. It reads the workspace `config.yaml`, validates
the AWQ section, and runs compression either by invoking the upstream CLI (if
available) or by calling `llmcompressor.oneshot()` in-process. It writes a
`metadata.json` to the output directory with invocation details and captured
stdout/stderr tails.

Exit codes:
 0 -> success
 2 -> oneshot / compression exception
 3 -> missing merged path
 4 -> GPU requested but not available
 5 -> output exists and not forced
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _tail(s: bytes | str, max_chars: int = 10000) -> str:
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="replace")
        except Exception:
            s = str(s)
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _write_metadata(out_path: Path, metadata: Dict[str, Any]) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    meta_file = out_path / "metadata.json"
    with meta_file.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


def _build_cli_cmd(merged: Path, out: Path, options: Dict[str, Any]) -> List[str]:
    cmd = ["llmcompressor", "compress", "--model-dir", str(merged), "--output-dir", str(out)]
    if options.get("num_calibration_samples"):
        cmd += ["--num-calibration-samples", str(options["num_calibration_samples"]) ]
    if options.get("calib_text_file"):
        cmd += ["--calib-file", str(options["calib_text_file"]) ]
    if options.get("method"):
        cmd += ["--method", str(options["method"]) ]
    if options.get("scheme"):
        cmd += ["--scheme", str(options["scheme"]) ]
    extra = options.get("extra_args") or []
    if isinstance(extra, (list, tuple)):
        cmd += [str(x) for x in extra]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to workspace config YAML")
    parser.add_argument("--merged", type=Path, required=True, help="Path to merged model directory (workspace path)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for compressed model (workspace path)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output if present")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="Require GPU for this run")
    group.add_argument("--cpu", action="store_true", help="Force CPU-only run")
    args = parser.parse_args(argv)

    cfg_file = args.config
    merged = args.merged
    out = args.out

    metadata: Dict[str, Any] = {
        "invoked_at": datetime.utcnow().isoformat() + "Z",
        "merged_path": str(merged),
        "out_path": str(out),
        "options": {},
        "attempts": [],
    }

    # Load config
    try:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        metadata.update({"returncode": 2, "stderr_tail": _tail(str(exc))})
        _write_metadata(out, metadata)
        print(f"Failed to read config: {exc}", file=sys.stderr)
        return 2

    # Support legacy key name `llm_compressor` for backward compatibility
    awq_cfg = cfg.get("awq") or cfg.get("llm_compressor") or {}
    options: Dict[str, Any] = dict(awq_cfg)
    metadata["options"] = options

    enabled = bool(options.get("enabled", True))
    if not enabled and not args.force:
        print("AWQ conversion disabled in config; skipping (use --force to override)")
        metadata.update({"returncode": 0, "stdout_tail": "skipped: disabled in config"})
        _write_metadata(out, metadata)
        return 0

    # Validate merged path
    if not merged.exists():
        msg = f"Merged model path not found: {merged}"
        metadata.update({"returncode": 3, "stderr_tail": msg})
        _write_metadata(out, metadata)
        print(msg, file=sys.stderr)
        return 3

    # Check output existence
    if out.exists() and not args.force:
        msg = f"Output path already exists: {out} (use --force to overwrite)"
        metadata.update({"returncode": 5, "stderr_tail": msg})
        _write_metadata(out, metadata)
        print(msg, file=sys.stderr)
        return 5

    # GPU availability check (best-effort)
    gpu_requested = bool(args.gpu) or bool(options.get("gpu_enabled", False))
    if gpu_requested and not args.cpu:
        gpu_ok = False
        # Try torch.cuda first
        try:
            import torch

            gpu_ok = torch.cuda.is_available()
        except Exception:
            # Fallback: check nvidia-smi presence
            if shutil.which("nvidia-smi"):
                gpu_ok = True
        if not gpu_ok:
            msg = "GPU requested but not available in runner"
            metadata.update({"returncode": 4, "stderr_tail": msg})
            _write_metadata(out, metadata)
            print(msg, file=sys.stderr)
            return 4

    # Try CLI first
    exe = None
    for cand in ("llmcompressor", "llm-compressor"):
        if shutil.which(cand):
            exe = cand
            break

    if exe:
        cmd = _build_cli_cmd(merged, out, options)
        metadata["attempts"].append({"type": "cli", "cmd": cmd})
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            metadata.update({
                "returncode": int(proc.returncode),
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
            })
            _write_metadata(out, metadata)
            if proc.returncode != 0:
                print(f"CLI compression failed (rc={proc.returncode})", file=sys.stderr)
                return int(proc.returncode) or 2
            print("AWQ CLI compression finished successfully")
            return 0
        except FileNotFoundError:
            # Race: executable disappeared, fall through to python inline
            pass

    # Inline oneshot (in-process) with captured stdout/stderr
    metadata["attempts"].append({"type": "python-inline", "cmd": [sys.executable, "-c", "llmcompressor.oneshot(...)"]})
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    rc = 0
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            # Build recipe similar to previous helper logic
            recipe = []
            if options.get("use_smoothquant", True):
                try:
                    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

                    strength = float(options.get("smoothquant_strength", 0.8))
                    recipe.append(SmoothQuantModifier(smoothing_strength=strength))
                except Exception as e:
                    print("Warning: cannot import SmoothQuantModifier:", e, file=sys.stderr)

            qmethod = options.get("method", "gptq")
            try:
                if qmethod and qmethod.lower() in ("gptq", "gptqmodifier"):
                    from llmcompressor.modifiers.quantization import GPTQModifier

                    scheme = options.get("scheme", "W8A8")
                    recipe.append(GPTQModifier(scheme=scheme, targets=options.get("targets", "Linear")))
                elif qmethod and qmethod.lower() in ("rtn", "round-to-nearest"):
                    from llmcompressor.modifiers.quantization import RoundToNearestModifier

                    scheme = options.get("scheme", "W8A8")
                    recipe.append(RoundToNearestModifier(scheme=scheme))
            except Exception as e:
                print("Warning: cannot import quantization modifiers:", e, file=sys.stderr)

            from llmcompressor import oneshot

            call_kwargs = {"recipe": recipe, "output_dir": str(out)}
            if options.get("num_calibration_samples"):
                call_kwargs["num_calibration_samples"] = int(options.get("num_calibration_samples"))
            if options.get("max_seq_length"):
                call_kwargs["max_seq_length"] = int(options.get("max_seq_length"))
            dataset = options.get("calib_text_file") or options.get("dataset")
            oneshot(model=str(merged), dataset=dataset, **call_kwargs)
    except Exception:
        rc = 2
        traceback.print_exc(file=buf_err)

    stdout_val = buf_out.getvalue()
    stderr_val = buf_err.getvalue()
    metadata.update({"returncode": int(rc), "stdout_tail": _tail(stdout_val), "stderr_tail": _tail(stderr_val)})
    _write_metadata(out, metadata)

    if rc != 0:
        print("AWQ python oneshot failed; see metadata.json for details", file=sys.stderr)
    else:
        print("AWQ python oneshot completed successfully")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
