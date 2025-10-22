import os
import sys
from pathlib import Path

# Ensure project root and src directory are importable during tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
