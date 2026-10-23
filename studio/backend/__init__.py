"""Studio backend package. Puts the repo root on sys.path so the flat-script
modules (v7_config, generate_synthetic_v7_mp) import from any working directory."""
import os
import sys

# repo root directory, absolute path (str)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
