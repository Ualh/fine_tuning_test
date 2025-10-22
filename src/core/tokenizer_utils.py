"""Utilities for working with Hugging Face tokenizers."""

from __future__ import annotations

from typing import Optional

from transformers import PreTrainedTokenizerBase

# Simple fallback chat template that alternates user and assistant turns while
# preserving optional system prompts. The template keeps the formatting compact
# so it can be parsed by models that expect plain conversational text.
_FALLBACK_CHAT_TEMPLATE = (
    "{% set role_map = {'user': 'User', 'assistant': 'Assistant', 'system': 'System'} %}"
    "{% if bos_token %}{{ bos_token }}{% endif %}"
    "{% for message in messages %}"
    "{% set role = role_map.get(message['role'], message['role'].capitalize()) %}"
    "{{ role }}: {{ message['content'] | trim }}"
    "{% if not loop.last %}{% if eos_token %}{{ eos_token }}{% else %}\n{% endif %}{% endif %}"
    "{% endfor %}"
)

_QWEN_CHAT_TEMPLATE = (
    "{% set eos = eos_token if eos_token else '' %}"
    "{% set bos = bos_token if bos_token else '' %}"
    "{% if messages %}"
    "{% set system_prompt = '' %}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_prompt = messages[0]['content'] %}"
    "{% set messages = messages[1:] %}"
    "{% endif %}"
    "{{ bos }}"
    "{% if system_prompt %}<|im_start|>system\n{{ system_prompt | trim }}<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] | trim }}<|im_end|>{{ eos }}"
    "{% elif message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] | trim }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% else %}{{ bos }}{% endif %}"
)


def ensure_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    *,
    logger: Optional[object] = None,
) -> PreTrainedTokenizerBase:
    """Guarantee that a tokenizer exposes a usable chat template.

    Some base models (including Gemma checkpoints) ship without a
    ``chat_template`` attribute even though the pipeline expects one. This
    helper first tries to rely on ``default_chat_template`` if provided by the
    tokenizer class. As a last resort it assigns a lightweight templated format
    that mirrors a standard assistant conversation.
    """

    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    default_template: Optional[str] = getattr(tokenizer, "default_chat_template", None)
    if default_template:
        tokenizer.chat_template = default_template  # type: ignore[assignment]
        if logger:
            logger.warning("Tokenizer missing chat template; using default template from tokenizer class.")
        return tokenizer

    model_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in model_name:
        tokenizer.chat_template = _QWEN_CHAT_TEMPLATE  # type: ignore[assignment]
        if logger:
            logger.warning("Tokenizer missing chat template; applied Qwen-style fallback template.")
        return tokenizer

    tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE  # type: ignore[assignment]
    if logger:
        logger.warning("Tokenizer missing chat template; applied fallback assistant template.")
    return tokenizer
