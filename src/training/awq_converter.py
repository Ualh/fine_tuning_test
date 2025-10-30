"""Deprecated AutoAWQ converter (shim).

This module previously implemented an AutoAWQ-based converter. AutoAWQ
integration was reverted during the migration to `llm-compressor`.

The full original implementation has been archived at:
	archive/awq_autoawq/awq_converter.py

Do not import this module in new code. The active AWQ flow is implemented
in `src/training/awq_runner.py` (llm-compressor-based) and is invoked via
the `run_pipeline.bat convert-awq` wrapper.

Attempting to import this module will raise an informative ImportError so
accidental imports fail fast during tests or runtime.
"""

raise ImportError(
		"AutoAWQ converter removed — use the llm-compressor-based runner: "
		"see src/training/awq_runner.py and docs/5.awq-compression.md. "
		"Archived original: archive/awq_autoawq/awq_converter.py"
)
"""
