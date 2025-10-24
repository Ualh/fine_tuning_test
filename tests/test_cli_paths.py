from pathlib import Path

from typer.testing import CliRunner

from src.cli.main import app, _default_output_dir, _default_preprocess_dir, _load_config


def test_default_paths(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = _load_config(config_path)

    preprocess_dir = _default_preprocess_dir(cfg)
    output_dir = _default_output_dir(cfg)

    assert preprocess_dir.parent == cfg.paths.prepared_dir
    assert preprocess_dir.name.endswith(str(cfg.preprocess.sample_size))
    assert output_dir.parent == cfg.paths.outputs_dir


def test_print_runtime_env_format():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = _load_config(config_path)
    runner = CliRunner()
    result = runner.invoke(app, ["print-runtime", "--config", str(config_path), "--format", "env"])

    assert result.exit_code == 0, result.output

    env_lines = [line for line in result.output.strip().splitlines() if line]
    runtime = dict(line.split("=", 1) for line in env_lines)

    preprocess_rel = _default_preprocess_dir(cfg).relative_to(cfg.project_root).as_posix()
    merged_rel = (_default_output_dir(cfg) / "merged").relative_to(cfg.project_root).as_posix()
    served_rel = cfg.serve.served_model_relpath.replace("\\", "/")

    assert runtime["COMPOSE_PROJECT"].startswith("ft-")
    assert runtime["PREPROCESS_DIR"] == preprocess_rel
    assert runtime["MERGED_DIR"] == merged_rel
    assert runtime["SERVED_MODEL_PATH"].endswith(served_rel)
