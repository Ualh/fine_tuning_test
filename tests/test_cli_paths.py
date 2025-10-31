from pathlib import Path

from typer.testing import CliRunner
import yaml

from src.cli.main import app, _default_output_dir, _default_preprocess_dir, _load_config
from src.core.run_naming import build_run_name


def _prepare_isolated_config(tmp_path: Path):
    base_config = Path(__file__).resolve().parents[1] / "config.yaml"
    config_data = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    tmp_prepared = tmp_path / "prepared"
    tmp_outputs = tmp_path / "outputs"
    tmp_logs = tmp_path / "logs"

    config_data.setdefault("paths", {})
    config_data["paths"]["prepared_dir"] = str(tmp_prepared)
    config_data["paths"]["outputs_dir"] = str(tmp_outputs)
    config_data["paths"]["logs_dir"] = str(tmp_logs)
    config_data["paths"]["run_metadata_file"] = str(tmp_logs / "latest.txt")

    isolated_config = tmp_path / "config.yaml"
    isolated_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    cfg = _load_config(isolated_config)
    return cfg, isolated_config


def test_default_paths(tmp_path):
    cfg, _ = _prepare_isolated_config(tmp_path)

    preprocess_dir = _default_preprocess_dir(cfg)
    run_info = build_run_name(cfg, outputs_root=cfg.paths.outputs_dir, logs_root=cfg.paths.logs_dir)
    output_dir = _default_output_dir(cfg, run_info)

    assert preprocess_dir.parent == cfg.paths.prepared_dir
    expected_suffix = "full" if cfg.preprocess.sample_size is None else str(cfg.preprocess.sample_size)
    assert preprocess_dir.name.endswith(expected_suffix)
    assert output_dir.parent == cfg.paths.outputs_dir
    # Output directory name should include the sample size tag as well ("full" or n<samples>)
    expected_output_tag = "full" if cfg.preprocess.sample_size is None else f"n{cfg.preprocess.sample_size}"
    assert expected_output_tag in output_dir.name


def test_print_runtime_env_format(tmp_path):
    cfg, tmp_config = _prepare_isolated_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["print-runtime", "--config", str(tmp_config), "--format", "env"])

    assert result.exit_code == 0, result.output

    env_lines = [line for line in result.output.strip().splitlines() if line]
    runtime = dict(line.split("=", 1) for line in env_lines)

    run_name = runtime["RUN_NAME"]
    assert runtime["RUN_DIR_NAME"] == run_name
    preprocess_rel = _default_preprocess_dir(cfg).relative_to(cfg.project_root).as_posix()
    merged_rel = (
        (Path(cfg.paths.outputs_dir) / run_name / "merged").relative_to(cfg.project_root).as_posix()
    )
    served_rel = cfg.serve.served_model_relpath.replace("\\", "/")

    # Compose project should be derived from the project folder name to keep
    # container/image names stable across runs (avoid per-run proliferation).
    expected_project = cfg.project_root.name.lower()
    assert runtime["COMPOSE_PROJECT"].startswith(expected_project[:3])
    assert runtime["RUN_NAME"] == run_name
    assert runtime["PREPROCESS_DIR"] == preprocess_rel
    assert runtime["MERGED_DIR"] == merged_rel
    assert runtime["SERVED_MODEL_PATH"].endswith(served_rel)
    assert runtime["RUN_OUTPUTS_DIR"].endswith(run_name)
    assert runtime["RUN_LOGS_DIR"].endswith(run_name)
    assert runtime["RUN_OUTPUTS_DIR_CONTAINER"].startswith("/app/")
    assert runtime["RUN_LOGS_DIR_CONTAINER"].startswith("/app/")

    outputs_latest = cfg.paths.outputs_dir / "latest.txt"
    assert outputs_latest.read_text().strip().endswith(run_name)
    logs_latest = cfg.paths.run_metadata_file
    assert logs_latest.read_text().strip().endswith(run_name)


def test_run_name_preview(tmp_path):
    _, tmp_config = _prepare_isolated_config(tmp_path)
    runner = CliRunner()

    first = runner.invoke(app, ["run-name-preview", "--config", str(tmp_config)])
    assert first.exit_code == 0, first.output
    preview_name = first.output.strip()

    second = runner.invoke(app, ["run-name-preview", "--config", str(tmp_config)])
    assert second.exit_code == 0, second.output
    assert second.output.strip() == preview_name

    result = runner.invoke(
        app,
        ["print-runtime", "--config", str(tmp_config), "--format", "env"],
        env={"RUN_NAME": preview_name},
    )
    assert result.exit_code == 0, result.output
    env_lines = [line for line in result.output.strip().splitlines() if line]
    runtime = dict(line.split("=", 1) for line in env_lines)

    assert runtime["RUN_NAME"] == preview_name
    assert runtime["RUN_DIR_NAME"] == preview_name
    assert runtime["MERGED_DIR"].endswith(f"{preview_name}/merged")
