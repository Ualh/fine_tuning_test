from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from src.cli.main import app, _load_config


def _build_isolated_config(tmp_path: Path):
    base_config = Path(__file__).resolve().parents[1] / "config.yaml"
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    config.setdefault("paths", {})
    config["paths"]["prepared_dir"] = str(tmp_path / "prepared")
    config["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    config["paths"]["logs_dir"] = str(tmp_path / "logs")
    config["paths"]["run_metadata_file"] = str(tmp_path / "logs" / "latest.txt")

    isolated = tmp_path / "config.yaml"
    isolated.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = _load_config(isolated)
    return cfg, isolated


def test_print_runtime_creates_run_directories(tmp_path: Path):
    cfg, config_path = _build_isolated_config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["print-runtime", "--config", str(config_path), "--format", "env"])
    assert result.exit_code == 0, result.output

    env_lines = [line for line in result.output.strip().splitlines() if line]
    runtime = dict(line.split("=", 1) for line in env_lines)
    run_name = runtime["RUN_NAME"]

    outputs_run = cfg.paths.outputs_dir / run_name
    logs_run = cfg.paths.logs_dir / run_name

    assert outputs_run.is_dir(), "outputs/<run-name> folder missing"
    assert logs_run.is_dir(), "logs/<run-name> folder missing"

    # Latest pointers should point to the run roots (not stage subdirectories).
    outputs_latest = cfg.paths.outputs_dir / "latest.txt"
    assert outputs_latest.read_text().strip() == str(outputs_run.resolve())

    logs_latest = cfg.paths.run_metadata_file
    assert logs_latest.read_text().strip() == str(logs_run.resolve())

    # Container helper paths should be present in runtime metadata.
    assert runtime["RUN_OUTPUTS_DIR"].endswith(run_name)
    assert runtime["RUN_LOGS_DIR"].endswith(run_name)
    assert runtime["RUN_OUTPUTS_DIR_CONTAINER"].startswith("/app/")
    assert runtime["RUN_LOGS_DIR_CONTAINER"].startswith("/app/")

    # Subsequent preview should not create additional directories.
    before_outputs = set(p.name for p in cfg.paths.outputs_dir.iterdir())
    preview = runner.invoke(app, ["run-name-preview", "--config", str(config_path)])
    assert preview.exit_code == 0, preview.output
    after_outputs = set(p.name for p in cfg.paths.outputs_dir.iterdir())
    assert before_outputs == after_outputs
