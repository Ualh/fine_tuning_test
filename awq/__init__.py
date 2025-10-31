"""Lightweight shim for `awq` to satisfy optional import-time references in tests.

This package is intentionally minimal and only implements symbols that
our test environment (or older versions of `peft`) may import at module
import time. It should not be used in production — it's only for test
isolation inside the CI/container environment where the full `awq`
package isn't available.
"""

__all__ = ["modules"]
