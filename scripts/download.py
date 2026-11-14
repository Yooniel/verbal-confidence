"""Download Gemma 3 27B weights and TriviaQA into the default HF cache (run as a CPU job)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.env import MODEL_ID, load_env

load_env()

from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(MODEL_ID, token=os.environ.get("HF_TOKEN"), max_workers=8)
print(f"model snapshot at {path} ({time.time() - t0:.0f}s)", flush=True)
tot = 0
for root, _, files in os.walk(path):
    for f in files:
        tot += os.path.getsize(os.path.join(root, f))
print(f"total size on disk: {tot / 1e9:.1f} GB", flush=True)

from datasets import load_dataset

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
print("trivia_qa rc.nocontext validation:", ds, flush=True)
print("done", flush=True)
