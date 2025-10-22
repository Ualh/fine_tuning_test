from pathlib import Path

from src.core.config import ConfigLoader


def test_config_loader_reads_defaults():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    loader = ConfigLoader(config_path)
    cfg = loader.load()

    assert cfg.train.base_model == "Qwen/Qwen2.5-0.5B"
    assert cfg.preprocess.sample_size == 2000
    assert cfg.paths.logs_dir.name == "logs"
    assert cfg.logging.console_level == "INFO"
