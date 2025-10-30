import json
import sys
import types
from pathlib import Path

import pytest


from src.training import awq_runner


def write_config(path: Path, awq_block: dict):
    content = {"awq": awq_block}
    path.write_text(
        """# minimal config for awq runner
""" + "\n" + json.dumps(content), encoding="utf-8"
    )


def test_awq_missing_writes_metadata_for_missing_merged(tmp_path):
    cfg = tmp_path / "config.yaml"
    write_config(cfg, {"enabled": True})

    merged = tmp_path / "nonexistent_merged"
    out = tmp_path / "out_awq"

    rc = awq_runner.main(["--config", str(cfg), "--merged", str(merged), "--out", str(out)])
    assert rc == 3

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["returncode"] == 3
    assert "Merged model path not found" in meta["stderr_tail"]


def test_awq_success_calls_oneshot_and_writes_metadata(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    # disable smoothquant to avoid modifier imports in tests
    write_config(cfg, {"enabled": True, "use_smoothquant": False})

    merged = tmp_path / "merged"
    merged.mkdir()
    # create a dummy file to simulate a model dir
    (merged / "dummy.bin").write_text("ok")

    out = tmp_path / "out_awq"

    # Provide a fake llmcompressor module with oneshot function
    mod = types.ModuleType("llmcompressor")

    def fake_oneshot(model: str, dataset=None, **kwargs):
        # print to stdout so runner captures it
        print("ONESHOT_CALLED: model=", model)

    mod.oneshot = fake_oneshot
    monkeypatch.setitem(sys.modules, "llmcompressor", mod)

    rc = awq_runner.main(["--config", str(cfg), "--merged", str(merged), "--out", str(out)])
    assert rc == 0

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["returncode"] == 0
    # Ensure the python-inline attempt is recorded
    assert any(attempt.get("type") == "python-inline" for attempt in meta.get("attempts", []))
    # stdout tail should include our marker
    assert "ONESHOT_CALLED" in meta.get("stdout_tail", "")
