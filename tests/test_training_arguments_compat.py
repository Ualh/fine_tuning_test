import json
import logging
from types import SimpleNamespace
from pathlib import Path


def test_trainingarguments_compat(monkeypatch, tmp_path):
    """Ensure our runner constructs TrainingArguments when tensorboard is enabled
    without raising TypeError across Transformers versions.

    This test monkeypatches HF/TRL internals (tokenizer, model, LoraConfig,
    and SFTTrainer) so it executes quickly and deterministically up to the
    point where TrainingArguments is constructed.
    """

    # Create tiny train/val JSONL splits
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "train.jsonl").write_text(json.dumps({"text": "train example"}) + "\n")
    (prepared / "val.jsonl").write_text(json.dumps({"text": "val example"}) + "\n")

    # Minimal fake configs with fields the runner expects
    cfg = SimpleNamespace(
        base_model="dummy-model",
        bf16=False,
        fp16=False,
        lora_r=4,
        lora_alpha=16,
        lora_target_modules=["q_proj"],
        lora_dropout=0.05,
        cutoff_len=512,
        batch_size=1,
        gradient_accumulation=1,
        epochs=1,
        learning_rate=1e-5,
        lr_scheduler="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=1,
        eval_steps=1,
        max_steps=10,
        gradient_checkpointing=False,
        report_to=["tensorboard"],
        logging_dir=str(tmp_path / "tensorboard"),
    )

    preproc = SimpleNamespace(pack_sequences=False)
    logger = logging.getLogger("test")

    # Provide fake top-level modules that may not be importable in the
    # test container (peft, awq.modules, trl). We insert lightweight
    # stubs into sys.modules before importing the runner so module
    # imports in src.training.sft_trainer succeed.
    import sys
    import types

    import importlib.machinery

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = lambda *a, **k: None
    fake_peft.PeftModel = object
    # Provide a ModuleSpec so importlib.find_spec doesn't choke on the
    # injected module during downstream `find_spec` checks.
    fake_peft.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    sys.modules["peft"] = fake_peft
    tuners_mod = types.ModuleType("peft.tuners")
    tuners_mod.__spec__ = importlib.machinery.ModuleSpec("peft.tuners", None)
    sys.modules["peft.tuners"] = tuners_mod
    # AWQ shim to satisfy peft internals that try to import awq.modules
    awq_mod = types.ModuleType("awq")
    awq_mod.__spec__ = importlib.machinery.ModuleSpec("awq", None)
    sys.modules["awq"] = awq_mod
    awq_modules = types.ModuleType("awq.modules")
    awq_modules.__spec__ = importlib.machinery.ModuleSpec("awq.modules", None)
    sys.modules["awq.modules"] = awq_modules
    # Minimal TRL shim (we'll replace SFTTrainer below with a Dummy)
    fake_trl = types.ModuleType("trl")
    fake_trl.__spec__ = importlib.machinery.ModuleSpec("trl", None)
    fake_trl.SFTTrainer = lambda *a, **k: None
    sys.modules["trl"] = fake_trl

    # Monkeypatch tokenizer/model/peft/trainer to lightweight dummies
    class DummyTokenizer:
        eos_token = "<|endoftext|>"
        pad_token = "<pad>"
        def save_pretrained(self, *a, **k):
            return None

    monkeypatch.setattr("src.training.sft_trainer.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr("src.training.sft_trainer.AutoModelForCausalLM.from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr("src.training.sft_trainer.LoraConfig", lambda *a, **k: None)

    class DummyTrainer:
        def __init__(self, *a, **kw):
            pass

        def remove_callback(self, *a, **kw):
            return None

        def add_callback(self, *a, **kw):
            return None

        def train(self, *a, **kw):
            return SimpleNamespace(metrics={"train_loss": 0.1, "epoch": 1.0})

        def save_model(self, *a, **kw):
            return None

        def evaluate(self, *a, **kw):
            return {"eval_loss": 0.1}

        def save_pretrained(self, *a, **kw):
            return None

    monkeypatch.setattr("src.training.sft_trainer.SFTTrainer", DummyTrainer)

    # Import runner here (after monkeypatches) and run
    from src.training.sft_trainer import SFTTrainerRunner

    runner = SFTTrainerRunner(cfg, preproc, logger, tmp_path / "run")
    summary = runner.run(data_dir=prepared, output_dir=tmp_path / "out")

    # Basic assertion to show the run completed and returned a TrainingSummary
    assert hasattr(summary, "train_loss")
