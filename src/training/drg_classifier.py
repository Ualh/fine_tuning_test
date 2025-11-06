# -*- coding: utf-8 -*-
"""
training.drg_classifier
Purpose: Fine-tune a sequence-classification head for DRG prediction using LoRA adapters.
Stage: training — consumes parquet splits from the DRG preprocessing pipeline and trains a classifier.
See: src/cli/main.py (`finetune-sft`) for the Typer command that dispatches to this runner when `preprocess.mode` is ``real_drg``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import ProgressCallback, PrinterCallback, TrainerCallback
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

try:  # Prefer SummaryWriter when available
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

from ..core.config import PreprocessConfig, TrainConfig
from ..core.io_utils import ensure_dir
from ..core.ssl import disable_ssl_verification
from ..training.sft_trainer import StageProgressCallback, TrainingSummary


@dataclass
class _Metrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


class _MetricLoggerCallback(TrainerCallback):
    """Mirror Trainer metrics into the pipeline logger."""

    def __init__(self, logger):
        self._logger = logger

    def on_log(self, args, state, control, **kwargs):  # pragma: no cover - exercised via integration tests
        logs = kwargs.get("logs") or kwargs.get("metrics") or None
        if isinstance(logs, dict):
            try:
                self._logger.info("TRAIN_METRICS: %s", json.dumps(logs, default=str))
            except Exception:
                self._logger.info("TRAIN_METRICS: %s", logs)


class _TensorBoardWriterCallback(TrainerCallback):
    """Write Trainer metrics to a TensorBoard SummaryWriter."""

    def __init__(self, writer: Optional[SummaryWriter]):
        self._writer = writer

    def on_log(self, args, state, control, **kwargs):  # pragma: no cover - exercised via integration tests
        if self._writer is None:
            return
        logs = kwargs.get("logs") or kwargs.get("metrics") or None
        if not isinstance(logs, dict):
            return
        step = getattr(state, "global_step", None)
        try:
            for key, value in logs.items():
                try:
                    self._writer.add_scalar(key, float(value), step)
                except Exception:
                    continue
            self._writer.flush()
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):  # pragma: no cover - exercised via integration tests
        if self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception:
            pass


class _WeightedTrainer(Trainer):
    """Custom Trainer that injects class weights into the CE loss."""

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "weighted_ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        min_learning_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._loss_type = (loss_type or "weighted_ce").lower()
        self._label_smoothing = label_smoothing
        self._min_learning_rate = float(min_learning_rate)
        alpha = class_weights if class_weights is not None else None
        self._focal = _FocalLoss(alpha=alpha, gamma=focal_gamma) if self._loss_type == "focal" else None

    def compute_loss(self, model, inputs, return_outputs=False):  # pragma: no cover - exercised via integration tests
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        if self._focal is not None:
            loss = self._focal(logits, labels)
        else:
            loss_fct = CrossEntropyLoss(
                weight=self._class_weights.to(logits.device) if self._class_weights is not None else None,
                label_smoothing=self._label_smoothing,
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer_and_scheduler(self, num_training_steps):
        super().create_optimizer_and_scheduler(num_training_steps)
        warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        base_lr = self.args.learning_rate
        min_lr = self._min_learning_rate or base_lr
        min_ratio = min_lr / base_lr if base_lr else 0.0

        def lr_lambda(current_step: int) -> float:
            if current_step < max(1, warmup_steps):
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)


class _TrainerWithSampler(_WeightedTrainer):
    """Augment the weighted trainer with a weighted random sampler for oversampling."""

    def __init__(self, *args, num_labels: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_labels = num_labels

    def get_train_dataloader(self):  # pragma: no cover - exercised via integration tests
        dataset = self.train_dataset
        labels = np.array(dataset["labels"])
        counts = np.bincount(labels, minlength=self._num_labels)
        inv = 1.0 / np.clip(counts.astype(float), a_min=1.0, a_max=None)
        weights = inv[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class _FocalLoss(torch.nn.Module):
    """Compute multi-class focal loss with optional per-class weighting."""

    def __init__(self, alpha: Optional[torch.Tensor], gamma: float) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


class DRGClassificationTrainerRunner:
    """Train a DRG classifier using LoRA adapters on top of a sequence-classification backbone."""

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

        train_parquet = Path(data_dir) / "train.parquet"
        val_parquet = Path(data_dir) / "val.parquet"
        label_map_path = Path(data_dir) / "label2id.json"

        if not train_parquet.exists() or not val_parquet.exists() or not label_map_path.exists():
            raise FileNotFoundError(
                "DRG classification expects train.parquet, val.parquet, and label2id.json in the prepared directory."
            )

        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)
        label2id = json.loads(label_map_path.read_text(encoding="utf-8"))
        id2label = {int(idx): label for label, idx in label2id.items()}
        num_labels = len(label2id)

        if "label" not in train_df.columns:
            raise ValueError("Expected 'label' column in train.parquet for DRG classification training.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            model_max_length=self.preprocess_config.cutoff_len,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "cls_token", None)
        model_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        tokenizer.padding_side = "right"

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        def _tokenize(batch: Dict[str, str]) -> Dict[str, torch.Tensor]:
            return tokenizer(batch["text"], truncation=True, max_length=self.config.cutoff_len)

        tokenized_train = train_dataset.map(_tokenize, batched=True, remove_columns=[c for c in train_df.columns if c != "label"])
        tokenized_val = val_dataset.map(_tokenize, batched=True, remove_columns=[c for c in val_df.columns if c != "label"])

        tokenized_train = tokenized_train.rename_column("label", "labels")
        tokenized_val = tokenized_val.rename_column("label", "labels")

        class_counts = np.bincount(tokenized_train["labels"], minlength=num_labels)
        inv_freq = 1.0 / np.maximum(class_counts, 1)
        scale = len(inv_freq) / inv_freq.sum() if inv_freq.sum() else 1.0
        class_weights_np = inv_freq * scale
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        has_cuda = torch.cuda.is_available()
        quant_config = None
        if self.config.load_in_4bit and has_cuda:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as exc:  # pragma: no cover - defensive fallback when bitsandbytes fails
                try:
                    self.logger.warning("Falling back to full precision; 4-bit quantisation unavailable: %s", exc)
                except Exception:
                    pass
                quant_config = None
        elif self.config.load_in_4bit and not has_cuda:
            try:
                self.logger.info("4-bit quantisation requested but no CUDA device detected; loading full precision model")
            except Exception:
                pass

        torch_dtype = None
        if quant_config is None and has_cuda:
            if self.config.bf16:
                torch_dtype = torch.bfloat16
            elif self.config.fp16:
                torch_dtype = torch.float16

        if has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:  # pragma: no cover - optional optimisation
                pass

        model_kwargs: Dict[str, Optional[object]] = {
            "num_labels": num_labels,
            "id2label": id2label,
            "label2id": {label: int(idx) for label, idx in label2id.items()},
            "trust_remote_code": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if has_cuda:
                model_kwargs["device_map"] = "auto"

        model = AutoModelForSequenceClassification.from_pretrained(self.config.base_model, **model_kwargs)
        model.config.pad_token_id = model_pad_id

        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)

        target_modules = self._resolve_lora_targets(model)
        if target_modules:
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        else:
            try:
                self.logger.warning("No LoRA target modules matched the classification backbone; proceeding without LoRA.")
            except Exception:
                pass

        report_to = list(self.config.report_to) if getattr(self.config, "report_to", None) else ["none"]
        tensorboard_dir = ensure_dir(self.run_dir / "tensorboard")

        training_kwargs = dict(
            output_dir=str(output_dir / "trainer_state"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size or max(1, self.config.batch_size),
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.eval_steps,
            bf16=self.config.bf16 and has_cuda,
            fp16=self.config.fp16 and has_cuda and not self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_steps=self.config.max_steps if self.config.max_steps else -1,
            report_to=report_to,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            save_total_limit=2,
            optim="adamw_torch_fused" if has_cuda else "adamw_torch",
            dataloader_num_workers=2,
            dataloader_pin_memory=has_cuda,
        )

        try:
            signature = TrainingArguments.__init__.__signature__  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - fallback for older Transformers
            signature = None
        if signature is not None and "logging_dir" in signature.parameters:
            training_kwargs["logging_dir"] = str(tensorboard_dir)

        training_args = TrainingArguments(**training_kwargs)

        callbacks = [_MetricLoggerCallback(self.logger)]
        tb_writer = None
        if report_to and "tensorboard" in [item.lower() for item in report_to] and SummaryWriter is not None:
            try:
                tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
                callbacks.append(_TensorBoardWriterCallback(tb_writer))
            except Exception:
                try:
                    self.logger.warning("Failed to create SummaryWriter at %s", tensorboard_dir)
                except Exception:
                    pass

        trainer_cls = _TrainerWithSampler if self.config.use_oversampling else _WeightedTrainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=self._build_metrics_fn(id2label, label2id, self.config.abstain_tau),
            class_weights=class_weights,
            loss_type=self.config.loss_type,
            focal_gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing,
            min_learning_rate=self.config.min_learning_rate,
        )
        if trainer_cls is _TrainerWithSampler:
            trainer_kwargs["num_labels"] = num_labels

        trainer = trainer_cls(**trainer_kwargs)

        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(StageProgressCallback())
        for callback in callbacks:
            trainer.add_callback(callback)

        train_result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        metrics = getattr(train_result, "metrics", {}) or {}
        try:
            self.logger.info("FINAL_TRAIN_METRICS: %s", json.dumps(metrics, default=str))
        except Exception:
            self.logger.info("FINAL_TRAIN_METRICS: %s", metrics)

        adapter_dir = ensure_dir(output_dir / "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

        eval_metrics = trainer.evaluate()
        try:
            self.logger.info("FINAL_EVAL_METRICS: %s", json.dumps(eval_metrics, default=str))
        except Exception:
            self.logger.info("FINAL_EVAL_METRICS: %s", eval_metrics)

        summary = TrainingSummary(
            output_dir=output_dir,
            train_loss=metrics.get("train_loss", float("nan")),
            eval_loss=eval_metrics.get("eval_loss"),
            epochs=metrics.get("epoch", self.config.epochs),
            total_tokens=None,
        )
        metadata_path = output_dir / "metadata.json"
        metadata_payload = {
            "train_metrics": metrics,
            "eval_metrics": eval_metrics,
            "label2id": label2id,
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _resolve_lora_targets(self, model) -> list[str]:
        available = {name for name, _ in model.named_modules()}
        resolved: list[str] = []
        for target in self.config.lora_target_modules:
            if any(name.endswith(target) or name.split(".")[-1] == target or target in name for name in available):
                resolved.append(target)
        return sorted(set(resolved))

    def _build_metrics_fn(
        self,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        abstain_tau: Optional[float],
    ):
        other_id = None
        if "__OTHER__" in label2id:
            other_id = int(label2id["__OTHER__"])

        def _compute(predictions):  # pragma: no cover - exercised via integration tests
            logits, labels = predictions
            logits_np = np.array(logits, copy=True)
            labels_np = np.array(labels)
            preds = logits_np.argmax(axis=-1)

            if abstain_tau is not None and other_id is not None:
                probs = torch.softmax(torch.tensor(logits_np), dim=-1).numpy()
                low_conf = probs.max(axis=1) < float(abstain_tau)
                if low_conf.any():
                    preds[low_conf] = other_id

            metrics = _Metrics(
                accuracy=float(accuracy_score(labels_np, preds)),
                balanced_accuracy=float(balanced_accuracy_score(labels_np, preds)),
                macro_f1=float(f1_score(labels_np, preds, average="macro", zero_division=0)),
            )
            payload: Dict[str, float] = {
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "macro_f1": metrics.macro_f1,
            }

            if other_id is not None:
                mask_pred_other = preds == other_id
                payload["abstention_rate"] = float(mask_pred_other.mean())
                mask_true_other = labels_np == other_id
                if mask_true_other.any():
                    payload["other_recall"] = float((preds[mask_true_other] == other_id).mean())
                mask_freq = labels_np != other_id
                if mask_freq.any():
                    payload["macro_f1_frequent_only"] = float(
                        f1_score(labels_np[mask_freq], preds[mask_freq], average="macro", zero_division=0)
                    )

            return payload

        return _compute
*** End of File