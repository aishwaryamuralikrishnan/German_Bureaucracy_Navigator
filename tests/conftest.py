import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# Tests never call OpenRouter, but constructors check that a key exists.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")
