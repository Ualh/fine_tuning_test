from pathlib import Path

from src.cli.main import _default_output_dir, _default_preprocess_dir, _load_config


def test_default_paths(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = _load_config(config_path)

    preprocess_dir = _default_preprocess_dir(cfg)
    output_dir = _default_output_dir(cfg)

    assert preprocess_dir.parent == cfg.paths.prepared_dir
    assert preprocess_dir.name.endswith(str(cfg.preprocess.sample_size))
    assert output_dir.parent == cfg.paths.outputs_dir
