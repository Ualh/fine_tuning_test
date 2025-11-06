"""Wrapper around TRL's :class:`SFTTrainer` with LoRA support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import json
import os
from pathlib import Path
from typing import Dict, Optional
import inspect
import warnings

import torch
from datasets import Dataset, DatasetDict, Features, Value
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import TrainerCallback
try:
    # Prefer torch's SummaryWriter when available (bundles tensorboard support)
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from transformers.trainer_callback import ProgressCallback, PrinterCallback
from huggingface_hub import login as hf_login
from trl import SFTTrainer


from ..core.config import PreprocessConfig, TrainConfig
from ..core.io_utils import atomic_write_json, ensure_dir
from ..core.ssl import disable_ssl_verification
from ..core.tokenizer_utils import ensure_chat_template


class StageProgressCallback(ProgressCallback):
    """Progress callback that labels tqdm bars to match pipeline stages."""

    def on_train_begin(self, args, state, control, **kwargs):
        # Call the superclass when running under a real Trainer state; tests
        # sometimes pass a minimal dummy object that lacks attributes the
        # base implementation expects (e.g. is_world_process_zero). In that
        # case fall back to initialising a progress bar via the internal
        # factory so tests can monkeypatch `_init_progress_bar`.
        bar = None
        try:
            super().on_train_begin(args, state, control, **kwargs)
            bar = getattr(self, "training_bar", None)
        except Exception:
            init = getattr(self, "_init_progress_bar", None)
            if callable(init):
                bar = init()

        if bar is not None:
            try:
                bar.set_description("TRAIN")
            except Exception:
                pass

    def on_eval_begin(self, args, state, control, **kwargs):
        bar = None
        try:
            super().on_eval_begin(args, state, control, **kwargs)
            bar = getattr(self, "evaluation_bar", None)
        except Exception:
            init = getattr(self, "_init_progress_bar", None)
            if callable(init):
                bar = init()

        if bar is not None:
            try:
                bar.set_description("EVAL")
            except Exception:
                pass

    def on_predict_begin(self, args, state, control, **kwargs):
        bar = None
        try:
            super().on_predict_begin(args, state, control, **kwargs)
            bar = getattr(self, "prediction_bar", None)
        except Exception:
            init = getattr(self, "_init_progress_bar", None)
            if callable(init):
                bar = init()

        if bar is not None:
            try:
                bar.set_description("EVAL")
            except Exception:
                pass


@dataclass
class TrainingSummary:
    output_dir: Path
    train_loss: float
    eval_loss: Optional[float]
    epochs: float
    total_tokens: Optional[int]


class SFTTrainerRunner:
    """Execute supervised fine-tuning with LoRA adapters."""

    def __init__(
        self,
        train_config: TrainConfig,
        preprocess_config: PreprocessConfig,
        logger,
        run_dir: Path,
    ) -> None:
        self.config = train_config
        self.preprocess_config = preprocess_config
        self.logger = logger
        self.run_dir = ensure_dir(run_dir)

    def run(
        self,
        data_dir: Path,
        output_dir: Path,
        resume_path: Optional[Path] = None,
    ) -> TrainingSummary:
        output_dir = ensure_dir(output_dir)
        disable_ssl_verification()
        self._configure_warning_filters()
        # Pick up HF token explicitly to support gated model downloads
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            try:
                hf_login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                self.logger.warning("HF login failed, proceeding with env vars: %s", e)

        # Prefer fast tokenizers and trust remote code to support custom architectures (e.g. Gemma)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer = ensure_chat_template(tokenizer, logger=self.logger)
        tokenizer.pad_token = tokenizer.eos_token

        # Transformers 5 renamed `tokenizer` argument to `processing_class`.
        # TRL <0.12 still forwards `tokenizer`, so patch Trainer to accept it.
        self._ensure_tokenizer_compat()
        # Some Accelerate releases (<=1.1) don't support the
        # keep_torch_compile kwarg. Patch it here so Transformers 5 Trainer
        # stays compatible with the runtime bundled in our Docker image.
        self._ensure_accelerate_compat()

        has_cuda = torch.cuda.is_available()
        torch_dtype = None
        if self.config.bf16 and has_cuda:
            torch_dtype = torch.bfloat16
        elif self.config.fp16 and has_cuda:
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch_dtype,
            device_map="auto" if has_cuda else None,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        datasets = self._load_jsonl_splits(Path(data_dir))

        enable_bf16 = self.config.bf16 and has_cuda
        enable_fp16 = self.config.fp16 and has_cuda and not enable_bf16

        report_to = list(self.config.report_to) if getattr(self.config, "report_to", None) else ["none"]
        tensorboard_dir_source = self.config.logging_dir if getattr(self.config, "logging_dir", None) else (self.run_dir / "tensorboard")
        tensorboard_dir = ensure_dir(Path(tensorboard_dir_source))
        if report_to and "tensorboard" in [item.lower() for item in report_to]:
            try:
                self.logger.info("TensorBoard logging enabled | directory=%s", tensorboard_dir)
            except Exception:
                pass

        # Build TrainingArguments kwargs defensively to remain compatible with
        # multiple Transformers versions. Some releases don't accept
        # `logging_dir` in the constructor, so inspect the signature and only
        # pass supported kwargs.
        training_kwargs = dict(
            output_dir=str(output_dir / "trainer_state"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size
            if self.config.eval_batch_size
            else max(1, self.config.batch_size // 2),
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.eval_steps,
            bf16=enable_bf16,
            fp16=enable_fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_steps=self.config.max_steps if self.config.max_steps else -1,
            report_to=report_to,
        )

        try:
            sig = inspect.signature(TrainingArguments.__init__)
            if "logging_dir" in sig.parameters:
                training_kwargs["logging_dir"] = str(tensorboard_dir)
            else:
                try:
                    self.logger.info("TrainingArguments does not accept 'logging_dir'; skipping it for compatibility.")
                except Exception:
                    pass
        except Exception:
            # If signature inspection fails for any reason, attempt to pass
            # logging_dir but catch the TypeError on construction below.
            training_kwargs["logging_dir"] = str(tensorboard_dir)

        training_args = TrainingArguments(**training_kwargs)

        # Small callback to capture Trainer/TRL logging events (metrics)

        class MetricLoggerCallback(TrainerCallback):
            """Trainer callback that writes metrics dicts to the pipeline logger at INFO level.

            The Trainer and TRL internals sometimes emit metric dicts via their
            own logging/print paths; by capturing on_log we centralise those
            events and ensure they flow through our logger (and therefore into
            the run.log file) instead of leaking to stdout.
            """

            def __init__(self, logger):
                self._logger = logger

            def on_log(self, args, state, control, **kwargs):
                logs = kwargs.get("logs") or kwargs.get("metrics") or None
                if isinstance(logs, dict):
                    try:
                        # Keep this at INFO so the console handler (WARNING)
                        # doesn't show it, but the file handler records it.
                        self._logger.info("TRAIN_METRICS: %s", json.dumps(logs, default=str))
                    except Exception:
                        self._logger.info("TRAIN_METRICS: %s", logs)

        # Optional callback that writes metrics into a TensorBoard SummaryWriter
        class TensorBoardWriterCallback(TrainerCallback):
            """Write trainer metrics into a SummaryWriter pointing at the
            canonical `tensorboard_dir`. This ensures events are written to
            our expected logs path regardless of whether TrainingArguments
            accepted the `logging_dir` kwarg.
            """

            def __init__(self, writer):
                self._writer = writer

            def on_log(self, args, state, control, **kwargs):
                logs = kwargs.get("logs") or kwargs.get("metrics") or None
                if not isinstance(logs, dict) or self._writer is None:
                    return
                step = getattr(state, "global_step", None)
                try:
                    for k, v in logs.items():
                        try:
                            # only numeric scalars
                            self._writer.add_scalar(k, float(v), step)
                        except Exception:
                            # ignore non-scalar metrics
                            pass
                    self._writer.flush()
                except Exception:
                    # Avoid raising from callbacks
                    pass

            def on_train_end(self, args, state, control, **kwargs):
                try:
                    self._writer.flush()
                    self._writer.close()
                except Exception:
                    pass

        # Build callbacks list and optionally add a TensorBoard writer so we
        # always write events to our canonical `tensorboard_dir` independent
        # of TrainingArguments constructor support.
        callbacks = [MetricLoggerCallback(self.logger)]
        tb_writer = None
        if report_to and "tensorboard" in [item.lower() for item in report_to]:
            if SummaryWriter is not None:
                try:
                    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
                    callbacks.append(TensorBoardWriterCallback(tb_writer))
                except Exception:
                    # If SummaryWriter fails, fall back to no-op but don't stop training
                    try:
                        self.logger.warning("Failed to create SummaryWriter for %s", tensorboard_dir)
                    except Exception:
                        pass

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            dataset_text_field="text",
            max_seq_length=self.config.cutoff_len,
            packing=self.preprocess_config.pack_sequences,
            peft_config=peft_config,
            callbacks=callbacks,
        )

        # Replace the default progress callback with stage-aware labels and drop printer spam.
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(StageProgressCallback())

        # Run training; the Trainer/TRL internals will call our callback with
        # intermediate metrics. We also log the returned metrics dict explicitly
        # to ensure final metrics are recorded.
        train_result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        metrics = getattr(train_result, "metrics", {}) or {}
        try:
            self.logger.info("FINAL_TRAIN_METRICS: %s", json.dumps(metrics, default=str))
        except Exception:
            self.logger.info("FINAL_TRAIN_METRICS: %s", metrics)
        adapter_dir = ensure_dir(output_dir / "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

        final_metrics = trainer.evaluate()
        try:
            self.logger.info("FINAL_EVAL_METRICS: %s", json.dumps(final_metrics, default=str))
        except Exception:
            self.logger.info("FINAL_EVAL_METRICS: %s", final_metrics)
        summary = TrainingSummary(
            output_dir=output_dir,
            train_loss=metrics.get("train_loss", float("nan")),
            eval_loss=final_metrics.get("eval_loss"),
            epochs=metrics.get("epoch", self.config.epochs),
            total_tokens=metrics.get("train_tokens"),
        )
        atomic_write_json({"train": asdict(summary)}, output_dir / "metadata.json")
        return summary

    def _configure_warning_filters(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message="Deprecated argument(s) used in '__init__': dataset_text_field, max_seq_length.*",
            category=FutureWarning,
            module="huggingface_hub.utils._deprecation",
        )
        warnings.filterwarnings(
            "ignore",
            message="You passed a `max_seq_length` argument to the SFTTrainer.*",
            category=UserWarning,
            module="trl.trainer.sft_trainer",
        )
        warnings.filterwarnings(
            "ignore",
            message="You passed a `dataset_text_field` argument to the SFTTrainer.*",
            category=UserWarning,
            module="trl.trainer.sft_trainer",
        )

    @staticmethod
    def _load_jsonl_splits(data_dir: Path) -> DatasetDict:
        """Load train/val splits from JSONL without relying on datasets' JSON loader."""

        mapping = {"train": "train.jsonl", "validation": "val.jsonl"}
        features = Features({"text": Value("string")})
        splits: Dict[str, Dataset] = {}

        for split_name, filename in mapping.items():
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Expected {filename} in {data_dir}")

            texts = []
            with path.open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, 1):
                    record_raw = raw_line.strip()
                    if not record_raw:
                        continue
                    try:
                        record = json.loads(record_raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc

                    text = record.get("text")
                    if not isinstance(text, str):
                        raise ValueError(f"Missing text field in {path} at line {line_no}")
                    texts.append(text)

            splits[split_name] = Dataset.from_dict({"text": texts}, features=features)

        return DatasetDict(splits)

    @staticmethod
    def _ensure_tokenizer_compat() -> None:
        sig = inspect.signature(Trainer.__init__)
        if "tokenizer" in sig.parameters:
            return

        if getattr(Trainer.__init__, "_tokenizer_compat_patched", False):
            return

        original_init = Trainer.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, tokenizer=None, processing_class=None, **kwargs):
            if tokenizer is not None:
                if processing_class is not None:
                    raise TypeError("Pass either tokenizer or processing_class, not both")
                processing_class = tokenizer
            return original_init(self, *args, processing_class=processing_class, **kwargs)

        patched_init._tokenizer_compat_patched = True  # type: ignore[attr-defined]
        Trainer.__init__ = patched_init  # type: ignore[assignment]

    @staticmethod
    def _ensure_accelerate_compat() -> None:
        try:
            from accelerate import Accelerator
        except Exception:
            return

        unwrap = Accelerator.unwrap_model
        sig = inspect.signature(unwrap)
        if "keep_torch_compile" in sig.parameters:
            return

        if getattr(unwrap, "_keep_torch_compile_patched", False):
            return

        original = unwrap

        @functools.wraps(original)
        def patched(self, model, keep_fp32_wrapper=True, *args, **kwargs):
            # Older Accelerate versions only accept keep_fp32_wrapper.
            return original(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

        patched._keep_torch_compile_patched = True  # type: ignore[attr-defined]
        Accelerator.unwrap_model = patched  # type: ignore[assignment]
