from pathlib import Path

from src.cli.main import _build_runtime_metadata
from src.core.config import (
    PathsConfig,
    PreprocessConfig,
    TrainConfig,
    ExportConfig,
    EvalConfig,
    ServeConfig,
    LoggingConfig,
    PipelineConfig,
)


def _make_cfg(project_root: Path, served_relpath, served_name="model") -> PipelineConfig:
    # Create simple directory layout inside project_root
    prepared = project_root / "prepared"
    outputs = project_root / "outputs"
    logs = project_root / "logs"
    prepared.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

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
    )

    export = ExportConfig(include_adapter_weights=True, resume_from=None)

    evalc = EvalConfig(cutoff_len=1024, max_new_tokens=32, temperature=0.7, top_p=0.9, prompts=[], resume_from=None)

    serve = ServeConfig(host="0.0.0.0", port=8080, max_model_len=2048, served_model_name=served_name, served_model_relpath=served_relpath, resume_from=None)

    logging = LoggingConfig(console_level="INFO", file_level="DEBUG", tqdm_refresh_rate=1.0, debug_pipeline=False)

    return PipelineConfig(paths=paths, preprocess=preprocess, train=train, export=export, eval=evalc, serve=serve, logging=logging, project_root=project_root)


def test_explicit_relpath_normalizes(tmp_path: Path):
    cfg = _make_cfg(tmp_path, "\\custom\\path\\")
    runtime = _build_runtime_metadata(cfg)
    assert runtime["SERVED_MODEL_RELPATH"] == "custom/path"
    assert runtime["SERVED_MODEL_PATH"] == "/models/custom/path"


def test_none_string_uses_merged(tmp_path: Path):
    cfg = _make_cfg(tmp_path, "NoNe")
    runtime = _build_runtime_metadata(cfg)
    # merged_rel is built from outputs/<dataset>_full_<model>/merged
    expected_merged = (cfg.paths.outputs_dir / f"{cfg.preprocess.dataset_name.split('/')[-1]}_full_{cfg.train.base_model.split('/')[-1]}" / "merged").relative_to(tmp_path).as_posix()
    assert runtime["SERVED_MODEL_RELPATH"] == expected_merged
    assert runtime["SERVED_MODEL_PATH"] == f"/models/{expected_merged}"


def test_none_falsy_uses_merged(tmp_path: Path):
    cfg = _make_cfg(tmp_path, None)
    runtime = _build_runtime_metadata(cfg)
    expected_merged = (cfg.paths.outputs_dir / f"{cfg.preprocess.dataset_name.split('/')[-1]}_full_{cfg.train.base_model.split('/')[-1]}" / "merged").relative_to(tmp_path).as_posix()
    assert runtime["SERVED_MODEL_RELPATH"] == expected_merged
    assert runtime["SERVED_MODEL_PATH"] == f"/models/{expected_merged}"
