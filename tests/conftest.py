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
os.environ.setdefault("ORACLE_DSN", "TEST_DSN")
os.environ.setdefault("ORACLE_USER", "TEST_USER")
os.environ.setdefault("ORACLE_PASSWORD", "TEST_PASSWORD")
os.environ.setdefault("ORACLE_NETWORK_ALIAS", "TEST_ALIAS")
os.environ.setdefault("ORACLE_TNS_PATH", str(ROOT))
