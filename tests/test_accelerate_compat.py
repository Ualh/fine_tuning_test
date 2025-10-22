import inspect

from accelerate import Accelerator

from src.training.sft_trainer import SFTTrainerRunner


def test_accelerate_keep_torch_compile_patch(monkeypatch):
    original = Accelerator.unwrap_model

    def legacy(self, model, keep_fp32_wrapper=True):
        return "legacy", keep_fp32_wrapper, model

    monkeypatch.setattr(Accelerator, "unwrap_model", legacy, raising=False)

    # Sanity check: legacy signature doesn't accept keep_torch_compile.
    legacy_sig = inspect.signature(Accelerator.unwrap_model)
    assert "keep_torch_compile" not in legacy_sig.parameters

    SFTTrainerRunner._ensure_accelerate_compat()

    patched = Accelerator.unwrap_model
    assert patched is not legacy

    accelerator = Accelerator()
    result = patched(accelerator, object(), keep_fp32_wrapper=False, keep_torch_compile=True)
    assert result[0] == "legacy"
    assert result[1] is False

    monkeypatch.setattr(Accelerator, "unwrap_model", original, raising=False)
