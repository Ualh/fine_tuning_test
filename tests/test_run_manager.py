from pathlib import Path

from src.core.run_manager import RunManager


def test_run_manager_creates_incremented_directories(tmp_path):
    logs_root = tmp_path / "logs"
    pointer = tmp_path / "latest.txt"
    manager = RunManager(logs_root, pointer)

    first = manager.create_run_dir("stage1")
    second = manager.create_run_dir("stage2")

    assert first.parent == second.parent
    assert pointer.read_text().strip() == str(second.resolve())
    assert first.name == "stage1"
    assert second.name == "stage2"
    assert second.parent.name.startswith("log_v02_")
