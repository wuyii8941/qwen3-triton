from pathlib import Path
import sys


# Keep the startup project lightweight (no editable package build), while
# making its top-level model package importable during pytest collection.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
