"""Utilities to merge LoRA adapters back into the base model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.io_utils import atomic_write_json, ensure_dir
from ..core.ssl import disable_ssl_verification


@dataclass
class MergeSummary:
    output_dir: Path
    base_model: str
    adapter_dir: Path
    merged_parameters: int


class LoraMerger:
    """Compose base model weights with a fine-tuned LoRA adapter."""

    def __init__(self, base_model: str, logger) -> None:
        self.base_model = base_model
        self.logger = logger

    def run(self, adapter_dir: Path, output_dir: Path) -> MergeSummary:
        adapter_dir = adapter_dir.resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

        output_dir = ensure_dir(output_dir)
        disable_ssl_verification()

        self.logger.info("Loading base model %s", self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        self.logger.info("Loading LoRA adapter from %s", adapter_dir)
        peft_model = PeftModel.from_pretrained(base, adapter_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        tokenizer.save_pretrained(output_dir)

        summary = MergeSummary(
            output_dir=output_dir,
            base_model=self.base_model,
            adapter_dir=adapter_dir,
            merged_parameters=sum(p.numel() for p in merged.parameters()),
        )
        atomic_write_json({"merge": asdict(summary)}, output_dir / "metadata.json")
        return summary
