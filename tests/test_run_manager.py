from pathlib import Path

import yaml

from src.core.config import ConfigLoader
from src.core.run_manager import RunManager
from src.core.run_naming import RunNameResult, build_run_name


def _load_isolated_config(tmp_path: Path):
    base_config = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg_data = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    cfg_data.setdefault("paths", {})
    cfg_data["paths"]["prepared_dir"] = str(tmp_path / "prepared")
    cfg_data["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    cfg_data["paths"]["logs_dir"] = str(tmp_path / "logs")
    cfg_data["paths"]["run_metadata_file"] = str(tmp_path / "logs" / "latest.txt")

    isolated = tmp_path / "config.yaml"
    isolated.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")
    loader = ConfigLoader(isolated)
    return loader.load()


def test_run_manager_derives_canonical_run_name(tmp_path):
    cfg = _load_isolated_config(tmp_path)
    manager = RunManager(cfg)

    run_info, stage1 = manager.create_run_dir("stage1")
    _, stage2 = manager.create_run_dir("stage2")

    assert stage1.parent == stage2.parent
    assert stage1.name == "stage1"
    assert stage2.name == "stage2"
    assert stage1.parent.name == run_info.name

    pointer_value = cfg.paths.run_metadata_file.read_text().strip()
    assert pointer_value == str((cfg.paths.logs_dir / run_info.name).resolve())

    manager.start_new_run()
    run_info2, stage3 = manager.create_run_dir("stage3")
    assert run_info2.name != run_info.name
    assert stage3.parent.name == run_info2.name


def test_run_manager_with_explicit_run_info(tmp_path):
    cfg = _load_isolated_config(tmp_path)
    explicit: RunNameResult = build_run_name(
        cfg,
        env={"RUN_NAME": "custom-prefix-run7"},
        outputs_root=cfg.paths.outputs_dir,
        logs_root=cfg.paths.logs_dir,
    )
    manager = RunManager(cfg, run_info=explicit)

    run_info, stage_dir = manager.create_run_dir("train")

    assert run_info.name == "custom-prefix-run7"
    assert stage_dir.parent.name == "custom-prefix-run7"
    assert cfg.paths.run_metadata_file.read_text().strip() == str(stage_dir.parent.resolve())

    manager.start_new_run()
    # Explicit run names should remain stable even after start_new_run
    _, next_stage = manager.create_run_dir("eval")
    assert next_stage.parent.name == "custom-prefix-run7"
