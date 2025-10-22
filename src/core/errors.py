"""Custom exceptions used across the fine-tuning pipeline."""

from __future__ import annotations


class PipelineConfigError(Exception):
    """Raised when the configuration file or overrides are invalid."""


class ResumeNotFoundError(Exception):
    """Raised when a resume checkpoint is requested but cannot be located."""


class StageExecutionError(Exception):
    """Raised when a pipeline stage fails in a recoverable way."""
