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
    assert cfg.serve.served_model_relpath == "autoif_qwen25_05b_lora/merged"
    assert cfg.project_root == config_path.parent
