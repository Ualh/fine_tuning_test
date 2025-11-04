from pathlib import Path

from src.core.config import ConfigLoader


def test_config_loader_reads_defaults():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    loader = ConfigLoader(config_path)
    cfg = loader.load()

    assert cfg.train.base_model == "Qwen/Qwen2-7B"
    assert cfg.preprocess.sample_size is None  # 'full' maps to None internally
    assert cfg.paths.logs_dir.name == "logs"
    assert cfg.logging.console_level == "INFO"
    assert cfg.logging.debug_pipeline is False
    assert cfg.serve.served_model_relpath is None
    assert cfg.serve.prefer_awq is True
    assert cfg.serve.model_name is None
    assert cfg.project_root == config_path.parent
