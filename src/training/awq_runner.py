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
import threading
import time
import traceback
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
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


def _resolve_workspace_path(cfg_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    return candidate


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_calibration_dataset(calib_file: Path) -> Dataset:
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    suffix = calib_file.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return Dataset.from_json(str(calib_file))
    if suffix == ".csv":
        return Dataset.from_csv(str(calib_file))

    lines: List[str] = []
    with calib_file.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            text = raw_line.strip()
            if text:
                lines.append(text)

    if not lines:
        raise ValueError(f"Calibration file {calib_file} is empty after stripping blank lines")

    return Dataset.from_dict({"text": lines})


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

    cfg_file = args.config.resolve()
    merged = args.merged.resolve()
    out = args.out.resolve()
    cfg_dir = cfg_file.parent

    print(f"AWQ runner starting (config={cfg_file}, merged={merged}, out={out})", flush=True)

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

    calib_path = _resolve_workspace_path(cfg_dir, options.get("calib_text_file"))
    if calib_path:
        options["calib_text_file"] = str(calib_path)
        metadata.setdefault("calibration_sources", {})["calib_text_file"] = str(calib_path)

    dataset_path = _resolve_workspace_path(cfg_dir, options.get("dataset_path"))
    if dataset_path:
        options["dataset_path"] = str(dataset_path)

    if merged.is_dir():
        metadata["merged_exists"] = True
    else:
        metadata["merged_exists"] = merged.exists()

    calib_dataset: Dataset | None = None

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
    if out.exists():
        if not args.force:
            msg = f"Output path already exists: {out} (use --force to overwrite)"
            metadata.update({"returncode": 5, "stderr_tail": msg})
            _write_metadata(out, metadata)
            print(msg, file=sys.stderr)
            return 5
        metadata["removed_existing_output"] = True
        try:
            if out.is_file() or out.is_symlink():
                out.unlink()
            else:
                shutil.rmtree(out)
        except Exception as exc:
            msg = f"Failed to remove existing output {out}: {exc}"
            metadata.update({"returncode": 2, "stderr_tail": _tail(msg)})
            _write_metadata(out, metadata)
            print(msg, file=sys.stderr)
            return 2

    # GPU availability check (best-effort)
    gpu_requested = bool(args.gpu) or bool(options.get("gpu_enabled", False))
    print(
        "GPU requested=%s force=%s retries=%s"
        % (gpu_requested, args.force, options.get("retries", options.get("awq_retries", 1))),
        flush=True,
    )
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

    if calib_path:
        try:
            calib_dataset = _load_calibration_dataset(calib_path)
            metadata["calibration_row_count"] = int(calib_dataset.num_rows)
        except Exception as exc:
            err = f"Failed to prepare calibration dataset {calib_path}: {exc}"
            metadata.update({"returncode": 2, "stderr_tail": _tail(err)})
            _write_metadata(out, metadata)
            print(err, file=sys.stderr)
            return 2

    # Helpers for streaming subprocess output while capturing tail
    class TailBuffer:
        def __init__(self, max_chars: int = 10000):
            self.max_chars = max_chars
            self.deq = deque()
            self.len = 0

        def write(self, s: str) -> None:
            if not s:
                return
            self.deq.append(s)
            self.len += len(s)
            # trim
            while self.len > self.max_chars and self.deq:
                removed = self.deq.popleft()
                self.len -= len(removed)

        def getvalue(self) -> str:
            return "".join(self.deq)

    def stream_subprocess(cmd: List[str], tail_out: TailBuffer, tail_err: TailBuffer) -> int:
        # Start subprocess and stream its stdout/stderr to real stdout/stderr
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def _drain(pipe, writer, tee_to=sys.stdout):
            try:
                for line in iter(pipe.readline, b""):
                    try:
                        chunk = line.decode("utf-8", errors="replace")
                    except Exception:
                        chunk = str(line)
                    writer.write(chunk)
                    writer.flush()
                    if tee_to and hasattr(tee_to, "write"):
                        try:
                            tee_to.write(chunk)
                            tee_to.flush()
                        except Exception:
                            pass
            except Exception:
                pass

        t_out = threading.Thread(target=_drain, args=(proc.stdout, tail_out, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, tail_err, sys.stderr), daemon=True)
        t_out.start()
        t_err.start()
        # heartbeat while running to reassure the user
        last = time.time()
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last >= 60:
                print(f"AWQ runner heartbeat: still running (pid={proc.pid})", flush=True)
                last = now
        t_out.join(timeout=1)
        t_err.join(timeout=1)
        return proc.returncode

    # Try CLI first (but stream output so progress is visible)
    exe = None
    for cand in ("llmcompressor", "llm-compressor"):
        if shutil.which(cand):
            exe = cand
            break

    max_retries = int(options.get("retries", options.get("awq_retries", 1)))
    attempt_index = 0
    if exe:
        print(f"Using llmcompressor CLI executable: {exe}", flush=True)
        cmd = _build_cli_cmd(merged, out, options)
        metadata["attempts"].append({"type": "cli", "cmd": cmd})
        tail_out = TailBuffer()
        tail_err = TailBuffer()
        for attempt_index in range(1, max_retries + 1):
            print(f"AWQ CLI attempt {attempt_index}/{max_retries}: {' '.join(cmd)}", flush=True)
            rc = stream_subprocess(cmd, tail_out, tail_err)
            metadata.setdefault("cli_attempts", []).append({"attempt": attempt_index, "returncode": int(rc)})
            if rc == 0:
                metadata.update({
                    "returncode": 0,
                    "stdout_tail": _tail(tail_out.getvalue()),
                    "stderr_tail": _tail(tail_err.getvalue()),
                })
                _write_metadata(out, metadata)
                print("AWQ CLI compression finished successfully", flush=True)
                return 0
            else:
                print(f"AWQ CLI compression failed (rc={rc}), attempt {attempt_index}", file=sys.stderr, flush=True)
                if attempt_index < max_retries:
                    backoff = 5 * attempt_index
                    print(f"Retrying AWQ CLI after {backoff}s...", flush=True)
                    time.sleep(backoff)
        # exhausted
        metadata.update({
            "returncode": int(rc),
            "stdout_tail": _tail(tail_out.getvalue()),
            "stderr_tail": _tail(tail_err.getvalue()),
        })
        _write_metadata(out, metadata)
        return int(rc) or 2

    # Inline oneshot (in-process) with streaming to stdout/stderr so logs are visible
    print("Falling back to llmcompressor.oneshot inline path", flush=True)
    metadata["attempts"].append({"type": "python-inline", "cmd": [sys.executable, "-c", "llmcompressor.oneshot(...)"]})
    tail_out = TailBuffer()
    tail_err = TailBuffer()
    rc = 0
    # Implement a Tee writer that writes to both original stdout and our tail buffer
    class TeeWriter:
        def __init__(self, buf: TailBuffer, orig):
            self.buf = buf
            self.orig = orig

        def write(self, s):
            try:
                self.buf.write(s)
            except Exception:
                pass
            try:
                self.orig.write(s)
            except Exception:
                pass

        def flush(self):
            try:
                self.orig.flush()
            except Exception:
                pass

    try:
        # Build recipe similar to previous helper logic
        recipe: List[Any] = []
        if options.get("use_smoothquant", True):
            try:
                from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

                strength = float(options.get("smoothquant_strength", 0.8))
                recipe.append(SmoothQuantModifier(smoothing_strength=strength))
            except Exception as exc:
                print("Warning: cannot import SmoothQuantModifier:", exc, file=sys.stderr)

        method = (options.get("method") or "gptq").lower()
        targets = _ensure_list(options.get("targets")) or ["Linear"]
        ignore = _ensure_list(options.get("ignore"))

        try:
            if method in ("awq", "activation-weighted-quantization"):
                from llmcompressor.modifiers.awq import AWQModifier

                awq_kwargs: Dict[str, Any] = {
                    "scheme": options.get("scheme", "W4A16"),
                    "targets": targets,
                }
                if ignore:
                    awq_kwargs["ignore"] = ignore
                if options.get("awq_mappings") is not None:
                    awq_kwargs["mappings"] = options["awq_mappings"]
                if options.get("awq_sequential_targets") is not None:
                    awq_kwargs["sequential_targets"] = options["awq_sequential_targets"]
                if options.get("awq_duo_scaling") is not None:
                    awq_kwargs["duo_scaling"] = bool(options["awq_duo_scaling"])
                recipe.append(AWQModifier(**awq_kwargs))
            elif method in ("gptq", "gptqmodifier"):
                from llmcompressor.modifiers.quantization import GPTQModifier

                scheme = options.get("scheme", "W8A8")
                recipe.append(GPTQModifier(scheme=scheme, targets=targets))
            elif method in ("rtn", "round-to-nearest"):
                from llmcompressor.modifiers.quantization import RoundToNearestModifier

                scheme = options.get("scheme", "W8A8")
                recipe.append(RoundToNearestModifier(scheme=scheme))
            else:
                if method:
                    print(
                        f"Warning: unsupported AWQ method '{method}', proceeding without an explicit quantization modifier",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print("Warning: cannot import quantization modifiers:", exc, file=sys.stderr)

        from llmcompressor import oneshot

        dataset_arg: Any = options.get("dataset")
        if calib_dataset is not None:
            dataset_arg = calib_dataset

        call_kwargs: Dict[str, Any] = {"recipe": recipe, "output_dir": str(out)}
        if options.get("num_calibration_samples"):
            call_kwargs["num_calibration_samples"] = int(options.get("num_calibration_samples"))
        if options.get("max_seq_length"):
            call_kwargs["max_seq_length"] = int(options.get("max_seq_length"))
        if options.get("dataset_path"):
            call_kwargs["dataset_path"] = options.get("dataset_path")
        if options.get("splits"):
            call_kwargs["splits"] = options.get("splits")
        if options.get("text_column"):
            call_kwargs["text_column"] = options.get("text_column")

        # Run oneshot with stdout/stderr teed to both real stdout and our tail buffers
        orig_out = sys.stdout
        orig_err = sys.stderr
        sys.stdout = TeeWriter(tail_out, orig_out)
        sys.stderr = TeeWriter(tail_err, orig_err)
        try:
            for attempt_index in range(1, max_retries + 1):
                print(f"AWQ inline attempt {attempt_index}/{max_retries}", flush=True)
                try:
                    oneshot(model=str(merged), dataset=dataset_arg, **call_kwargs)
                    rc = 0
                    break
                except Exception:
                    rc = 2
                    traceback.print_exc(file=sys.stderr)
                    if attempt_index < max_retries:
                        backoff = 5 * attempt_index
                        print(f"Retrying AWQ inline after {backoff}s...", flush=True)
                        time.sleep(backoff)
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
    except Exception:
        rc = 2
        traceback.print_exc(file=sys.stderr)

    metadata.update({"returncode": int(rc), "stdout_tail": _tail(tail_out.getvalue()), "stderr_tail": _tail(tail_err.getvalue())})
    _write_metadata(out, metadata)
    print(f"Metadata written to {out / 'metadata.json'}", flush=True)

    if rc != 0:
        print("AWQ python oneshot failed; see metadata.json for details", file=sys.stderr)
    else:
        print("AWQ python oneshot completed successfully")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
