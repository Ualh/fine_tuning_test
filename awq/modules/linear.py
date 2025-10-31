"""Minimal placeholder for AWQ linear modules used at import time by peft.

Define `WQLinear_GEMM` so older peft/awq integration imports succeed during
pytest collection in environments where the real `awq` package is not
installed in the container.
"""

class WQLinear_GEMM:
    """Placeholder class used only to satisfy import-time references.

    The real implementation is in the AWQ project. Tests here only need a
    symbol present so imports don't fail; no runtime behavior is required.
    """

    def __init__(self, *args, **kwargs):
        # store args to avoid surprises if introspected
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise RuntimeError("awq.modules.linear.WQLinear_GEMM is a test shim and should not be called")
