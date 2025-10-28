from __future__ import annotations

from pathlib import Path

import pytest

from src.core import runtime_probe


def test_filter_env_lines_strips_banner_noise() -> None:
	banner = [
		"+-----------------------------------------------------------------------------+",
		"| NVIDIA-SMI 535.104.05    Driver Version: 535.104.05    CUDA Version: 12.2     |",
		"+-----------------------------------------------------------------------------+",
		"PROJECT_ROOT=/app",
		"DEBUG_PIPELINE=1",
		"SERVE_PORT=8080",
	]

	filtered = runtime_probe.filter_env_lines(banner)

	assert filtered == [
		"PROJECT_ROOT=/app",
		"DEBUG_PIPELINE=1",
		"SERVE_PORT=8080",
	]


def test_parse_env_preserves_debug_flag() -> None:
	lines = [
		"CUDA_VISIBLE_DEVICES=0",
		"DEBUG_PIPELINE=1",
		"COMPOSE_PROJECT=ft-test",
	]

	env = runtime_probe.parse_env(lines)

	assert env["DEBUG_PIPELINE"] == "1"
	assert env["COMPOSE_PROJECT"] == "ft-test"
	assert "CUDA_VISIBLE_DEVICES" in env


def test_main_filters_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	probe_input = tmp_path / "probe.txt"
	probe_input.write_text(
		"""\
NVSMI version something
PROJECT_ROOT=/workspace/app
DEBUG_PIPELINE=0
""",
		encoding="utf-8",
	)

	exit_code = runtime_probe.main([str(probe_input)])
	captured = capsys.readouterr()

	assert exit_code == 0
	assert captured.out.splitlines() == [
		"PROJECT_ROOT=/workspace/app",
		"DEBUG_PIPELINE=0",
	]
	assert captured.err == ""
