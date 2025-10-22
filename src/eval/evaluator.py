"""Evaluation helpers providing lightweight quality checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.config import EvalConfig, EvalPrompt
from ..core.io_utils import atomic_write_json, ensure_dir
from ..core.ssl import disable_ssl_verification
from ..core.tokenizer_utils import ensure_chat_template


@dataclass
class PromptResult:
    instruction: str
    generation: str
    language_pass: bool
    length_pass: bool
    tokens: int


@dataclass
class EvalSummary:
    output_dir: Path
    perplexity: Optional[float]
    prompts: List[PromptResult]


class Evaluator:
    """Run lightweight evaluations on the merged model."""

    def __init__(self, config: EvalConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def run(self, model_dir: Path, output_dir: Path, val_path: Optional[Path] = None) -> EvalSummary:
        model_dir = Path(model_dir)
        output_dir = ensure_dir(output_dir)
        disable_ssl_verification()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
        tokenizer = ensure_chat_template(tokenizer, logger=self.logger)
        tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        perplexity = self._compute_perplexity(model, tokenizer, val_path)
        prompt_results = self._run_prompts(model, tokenizer)

        summary = EvalSummary(output_dir=output_dir, perplexity=perplexity, prompts=prompt_results)
        payload = {
            "eval": {
                "perplexity": perplexity,
                "prompts": [asdict(item) for item in prompt_results],
            }
        }
        atomic_write_json(payload, output_dir / "metrics.json")
        return summary

    def _compute_perplexity(self, model, tokenizer, val_path: Optional[Path]) -> Optional[float]:
        if not val_path or not Path(val_path).exists():
            self.logger.info("Skipping perplexity computation; validation split not available.")
            return None
        dataset = load_dataset("json", data_files={"validation": str(val_path)}, split="validation")
        total_log_likelihood = 0.0
        total_tokens = 0
        for record in dataset:
            text = record.get("text")
            if not text:
                continue
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss.item() * input_ids.shape[-1]
            total_log_likelihood += neg_log_likelihood
            total_tokens += input_ids.shape[-1]
        if total_tokens == 0:
            return None
        return math.exp(total_log_likelihood / total_tokens)

    def _run_prompts(self, model, tokenizer) -> List[PromptResult]:
        results: List[PromptResult] = []
        for prompt in self.config.prompts:
            messages = self._build_messages(prompt)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            generation = tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
            language = detect(generation) if generation else "unknown"
            language_pass = language.lower().startswith(prompt.language.lower()[0]) if prompt.language else True
            length_pass = True
            if prompt.max_sentences:
                sentences = [segment for segment in generation.split(".") if segment.strip()]
                length_pass = len(sentences) <= prompt.max_sentences
            results.append(
                PromptResult(
                    instruction=prompt.instruction,
                    generation=generation,
                    language_pass=language_pass,
                    length_pass=length_pass,
                    tokens=len(output_ids[0]),
                )
            )
        return results

    @staticmethod
    def _build_messages(prompt: EvalPrompt) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.instruction},
        ]
