"""Environment helpers: load secrets from ~/.env at runtime (never from the command line)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODEL_ID = "google/gemma-3-27b-it"


def load_env(path: str = "~/.env") -> None:
    """Parse KEY=VALUE lines from ~/.env into os.environ without echoing them."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.replace("export ", "").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
