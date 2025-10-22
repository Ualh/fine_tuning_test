from pathlib import Path

import pytest

from src.core.errors import ResumeNotFoundError
from src.core.resume import ResumeManager


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    def error(self, msg, *args):
        self.messages.append(msg % args if args else msg)


def test_resume_manager_resolves_existing(tmp_path):
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "pytorch_model.bin").write_bytes(b"test")
    logger = DummyLogger()
    manager = ResumeManager(logger)

    resolved = manager.resolve(str(target), tmp_path)

    assert resolved == target.resolve()
    assert logger.messages


def test_resume_manager_missing(tmp_path):
    logger = DummyLogger()
    manager = ResumeManager(logger)
    with pytest.raises(ResumeNotFoundError):
        manager.resolve(str(tmp_path / "missing"), tmp_path)


def test_resume_manager_normalizes_backslashes(tmp_path):
    outputs_dir = tmp_path / "outputs" / "exp"
    outputs_dir.mkdir(parents=True)
    resume_dir = outputs_dir / "trainer_state"
    resume_dir.mkdir()
    (resume_dir / "trainer_state.json").write_text("{}")
    logger = DummyLogger()
    manager = ResumeManager(logger)

    requested = f"{outputs_dir.name}\\trainer_state"

    resolved = manager.resolve(requested, outputs_dir)

    assert resolved == resume_dir.resolve()


def test_resume_manager_handles_prefixed_outputs(tmp_path):
    outputs_dir = tmp_path / "outputs" / "exp"
    outputs_dir.mkdir(parents=True)
    resume_dir = outputs_dir / "trainer_state"
    resume_dir.mkdir()
    (resume_dir / "trainer_state.json").write_text("{}")
    logger = DummyLogger()
    manager = ResumeManager(logger)

    requested = "outputs/exp/trainer_state"

    resolved = manager.resolve(requested, outputs_dir)

    assert resolved == resume_dir.resolve()
