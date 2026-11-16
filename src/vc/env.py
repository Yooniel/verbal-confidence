"""Environment helpers: load secrets from ~/.env at runtime (never from the command line)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Prompt format, chosen with the environment variable VC_FMT:
#   raw : the paper's single raw-text prompt (default; data/ and results/)
#   tt  : two-turn chat format, the answer in the model's own turn and the confidence request in a new user
#         turn (data/tt/ and results/tt/)
FMT = os.environ.get("VC_FMT", "raw")
assert FMT in ("raw", "tt"), f"unknown VC_FMT {FMT!r}"
DATA_RAW, RESULTS_RAW = ROOT / "data", ROOT / "results"
DATA = DATA_RAW if FMT == "raw" else DATA_RAW / FMT
RESULTS = RESULTS_RAW if FMT == "raw" else RESULTS_RAW / FMT
FIGURES = ROOT / "figures"

MODEL_ID = "google/gemma-3-27b-it"

# Released interpretability artifacts for Gemma 3 27B IT (scripts/interp_*.py)
NLA_AV, NLA_AR, NLA_LAYER = "kitft/nla-gemma3-27b-L41-av", "kitft/nla-gemma3-27b-L41-ar", 41
SAE_REPO, SAE_DIR, SAE_LAYER = "google/gemma-scope-2-27b-it", "resid_post/layer_31_width_1m_l0_big", 31
JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt"
# The verbalizer is released in float32 (108 GB); scripts/interp_download.py keeps a bfloat16 copy here instead.
INTERP_CACHE = Path(os.environ.get("VC_INTERP_CACHE", "~/.cache/vc-interp")).expanduser()


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
