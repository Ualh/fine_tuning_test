import os
from contextlib import contextmanager
from pathlib import Path

from src.cli.main import _build_runtime_metadata
from src.core.config import (
    AwqConversionConfig,
    ExportConfig,
    EvalConfig,
    LoggingConfig,
    NamingConfig,
    OrchestrationConfig,
    PathsConfig,
    PipelineConfig,
    PreprocessConfig,
    ServeConfig,
    TrainConfig,
)


@contextmanager
def _temp_env(**overrides):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, prior in original.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _make_cfg(
    project_root: Path,
    *,
    served_relpath: str | None,
    prefer_awq: bool = True,
    awq_suffix: str = "awq",
    awq_enabled: bool = True,
) -> PipelineConfig:
    prepared = project_root / "prepared"
    outputs = project_root / "outputs"
    logs = project_root / "logs"
    for path in (prepared, outputs, logs):
        path.mkdir(parents=True, exist_ok=True)

    paths = PathsConfig(
        data_root=project_root / "data",
        prepared_dir=prepared,
        outputs_dir=outputs,
        logs_dir=logs,
        huggingface_cache=project_root / ".cache",
        models_mount=project_root / "models_mount",
        run_metadata_file=logs / "latest.txt",
    )

    preprocess = PreprocessConfig(
        dataset_name="owner/dataset",
        sample_size=None,
        filter_langs=["en"],
        test_size=0.1,
        cutoff_len=1024,
        seed=42,
        save_splits=True,
        max_workers=2,
        pack_sequences=False,
    )

    train = TrainConfig(
        base_model="owner/model",
        cutoff_len=1024,
        batch_size=1,
        gradient_accumulation=1,
        epochs=1,
        learning_rate=1e-5,
        min_learning_rate=1e-6,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler="linear",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["q_proj"],
        gradient_checkpointing=False,
        bf16=False,
        fp16=False,
        logging_steps=10,
        eval_steps=100,
        max_steps=None,
        resume_from=None,
        report_to=["tensorboard"],
        logging_dir=None,
    )

    export = ExportConfig(include_adapter_weights=True, resume_from=None)
    evalc = EvalConfig(cutoff_len=1024, max_new_tokens=32, temperature=0.7, top_p=0.9, prompts=[], resume_from=None)

    serve = ServeConfig(
        host="0.0.0.0",
        port=8080,
        max_model_len=2048,
        served_model_name="model",
        served_model_relpath=served_relpath,
        resume_from=None,
        prefer_awq=prefer_awq,
        model_name=None,
    )

    logging = LoggingConfig(console_level="INFO", file_level="DEBUG", tqdm_refresh_rate=1.0, debug_pipeline=False)
    naming = NamingConfig()
    awq_conversion = AwqConversionConfig(
        enabled=awq_enabled,
        gpu_enabled=False,
        output_suffix=awq_suffix,
    )
    orchestration = OrchestrationConfig()

    return PipelineConfig(
        paths=paths,
        preprocess=preprocess,
        train=train,
        export=export,
        eval=evalc,
        serve=serve,
        logging=logging,
        naming=naming,
        awq_conversion=awq_conversion,
        orchestration=orchestration,
        project_root=project_root,
    )


def test_explicit_relpath_normalizes(tmp_path: Path):
    cfg = _make_cfg(tmp_path, served_relpath="\\custom\\path\\")
    target_dir = cfg.paths.outputs_dir / "custom" / "path"
    target_dir.mkdir(parents=True, exist_ok=True)

    forced_run = "override-run-run1"
    with _temp_env(RUN_NAME=forced_run):
        runtime = _build_runtime_metadata(cfg)

    assert runtime["SERVED_MODEL_RELPATH"] == "custom/path"
    assert runtime["SERVED_MODEL_PATH"] == "/models/custom/path"
    assert runtime["SERVED_MODEL_SOURCE"] == "override"


def test_prefer_awq_when_directory_exists(tmp_path: Path):
    cfg = _make_cfg(tmp_path, served_relpath=None, prefer_awq=True)
    forced_run = "run-test-run1"
    with _temp_env(RUN_NAME=forced_run):
        awq_dir = cfg.paths.outputs_dir / forced_run / "merged_awq"
        awq_dir.mkdir(parents=True, exist_ok=True)
        runtime = _build_runtime_metadata(cfg)

    assert runtime["SERVED_MODEL_RELPATH"] == f"{forced_run}/merged_awq"
    assert runtime["SERVED_MODEL_SOURCE"] == "awq"
    assert runtime["SERVED_MODEL_PATH"] == f"/models/{forced_run}/merged_awq"
    assert runtime["SERVED_MODEL_PATH_HOST"].endswith(f"{forced_run}/merged_awq")


def test_fallback_to_merged_when_awq_missing(tmp_path: Path):
    cfg = _make_cfg(tmp_path, served_relpath=None, prefer_awq=True)
    forced_run = "run-no-awq-run1"
    with _temp_env(RUN_NAME=forced_run):
        runtime = _build_runtime_metadata(cfg)

    assert runtime["SERVED_MODEL_RELPATH"] == f"{forced_run}/merged"
    assert runtime["SERVED_MODEL_SOURCE"] == "merged"


def test_prefer_awq_disabled(tmp_path: Path):
    cfg = _make_cfg(tmp_path, served_relpath=None, prefer_awq=False)
    forced_run = "run-awq-disabled-run1"
    with _temp_env(RUN_NAME=forced_run):
        awq_dir = cfg.paths.outputs_dir / forced_run / "merged_awq"
        awq_dir.mkdir(parents=True, exist_ok=True)
        runtime = _build_runtime_metadata(cfg)

    assert runtime["SERVED_MODEL_RELPATH"] == f"{forced_run}/merged"
    assert runtime["SERVED_MODEL_SOURCE"] == "merged"
