import sys
from pathlib import Path


V1_DIR = str(Path(__file__).resolve().parent.parent)
if V1_DIR not in sys.path:
    sys.path.insert(0, V1_DIR)
