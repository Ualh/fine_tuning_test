"""Wrapper around TRL's :class:`SFTTrainer` with LoRA support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import os
from pathlib import Path
from typing import Dict, Optional
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import login as hf_login
from trl import SFTTrainer


from ..core.config import PreprocessConfig, TrainConfig
from ..core.io_utils import atomic_write_json, ensure_dir
from ..core.ssl import disable_ssl_verification
from ..core.tokenizer_utils import ensure_chat_template


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

        data_files = {
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        }
        datasets = load_dataset("json", data_files=data_files)

        enable_bf16 = self.config.bf16 and has_cuda
        enable_fp16 = self.config.fp16 and has_cuda and not enable_bf16

        training_args = TrainingArguments(
            output_dir=str(output_dir / "trainer_state"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=max(1, self.config.batch_size // 2),
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
            report_to=["tensorboard"],
        )

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
        )

        metrics = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None).metrics
        adapter_dir = ensure_dir(output_dir / "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

        final_metrics = trainer.evaluate()
        summary = TrainingSummary(
            output_dir=output_dir,
            train_loss=metrics.get("train_loss", float("nan")),
            eval_loss=final_metrics.get("eval_loss"),
            epochs=metrics.get("epoch", self.config.epochs),
            total_tokens=metrics.get("train_tokens"),
        )
        atomic_write_json({"train": asdict(summary)}, output_dir / "metadata.json")
        return summary

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
