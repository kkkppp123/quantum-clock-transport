# tests/conftest.py
import sys
from pathlib import Path

# ROOT to katalog z plikiem pyproject.toml / requirements.txt (główne repo)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
