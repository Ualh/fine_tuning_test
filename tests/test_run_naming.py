from pathlib import Path

import pytest

from src.core.config import ConfigLoader, PipelineConfig
from src.core.run_naming import RunNameResult, build_run_name


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    loader = ConfigLoader(Path("debug_config.yaml"))
    cfg = loader.load()
    cfg.paths.outputs_dir = tmp_path / "outputs"
    cfg.paths.logs_dir = tmp_path / "logs"
    cfg.paths.run_metadata_file = cfg.paths.logs_dir / "latest.txt"
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.naming.use_legacy = False
    cfg.naming.run_prefix = ""
    return cfg


def test_build_run_name_basic(cfg) -> None:
    result = build_run_name(cfg)
    assert isinstance(result, RunNameResult)
    assert result.name.endswith("-run1")
    assert result.index == 1
    assert result.prefix


def test_run_name_increments_when_existing(cfg) -> None:
    first = build_run_name(cfg)
    (cfg.paths.outputs_dir / first.name).mkdir(parents=True)
    second = build_run_name(cfg)
    assert second.index == first.index + 1
    assert second.name.endswith(f"run{second.index}")


def test_force_run_index(cfg, monkeypatch) -> None:
    monkeypatch.setenv("FORCE_RUN_INDEX", "7")
    result = build_run_name(cfg)
    assert result.index == 7
    assert result.name.endswith("run7")
    monkeypatch.delenv("FORCE_RUN_INDEX", raising=False)


def test_legacy_naming(cfg, monkeypatch) -> None:
    cfg.naming.use_legacy = True
    result = build_run_name(cfg)
    assert result.legacy
    assert "-run" not in result.name


def test_explicit_run_name(cfg, monkeypatch) -> None:
    monkeypatch.setenv("RUN_NAME", "qwen2-5-05b-alpaca-n2048-run3")
    result = build_run_name(cfg)
    assert result.name == "qwen2-5-05b-alpaca-n2048-run3"
    assert result.index == 3
    monkeypatch.delenv("RUN_NAME", raising=False)
