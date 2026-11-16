"""Download the released interpretability artifacts for Gemma 3 27B IT (CPU job, about 195 GB of traffic,
about 140 GB kept):

  NLA verbalizer    kitft/nla-gemma3-27b-L41-av   float32, 108 GB; cast shard by shard to bfloat16 (54 GB)
                                                  into ~/.cache/vc-interp, the float32 copy is not kept
  NLA reconstructor kitft/nla-gemma3-27b-L41-ar   38 GB, Hugging Face cache
  Jacobian lens     neuronpedia/jacobian-lens     3.5 GB, Hugging Face cache
  Sparse autoencoder google/gemma-scope-2-27b-it  resid_post/layer_31_width_1m_l0_big/params.safetensors, 45 GB

The SAE's examples.safetensors (22 GB of top-activating examples) is not downloaded; scripts/interp_sae.py
reads the rows it needs over HTTP.
"""
import json
import os
import shutil
import sys
import time

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vc.env import INTERP_CACHE, JLENS_FILE, JLENS_REPO, NLA_AR, NLA_AV, SAE_DIR, SAE_REPO, load_env

load_env()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---- small things first
p = hf_hub_download(JLENS_REPO, JLENS_FILE)
log("lens", p, f"{os.path.getsize(p) / 1e9:.1f} GB")
for f in ("config.json", "params.safetensors"):
    p = hf_hub_download(SAE_REPO, f"{SAE_DIR}/{f}")
    log("sae", p, f"{os.path.getsize(p) / 1e9:.1f} GB")
p = snapshot_download(NLA_AR)
log("reconstructor", p)

# ---- verbalizer: bfloat16 copy, one shard at a time
out = INTERP_CACHE / NLA_AV.split("/")[1]
out.mkdir(parents=True, exist_ok=True)
small = snapshot_download(NLA_AV, ignore_patterns=["*.safetensors"])
for f in os.listdir(small):
    if not (out / f).exists():
        shutil.copy(os.path.join(small, f), out / f)
index = json.load(open(out / "model.safetensors.index.json"))
shards = sorted(set(index["weight_map"].values()))
for i, f in enumerate(shards):
    if (out / f).exists():
        continue
    src = hf_hub_download(NLA_AV, f)
    sd = load_file(src)
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    save_file(sd, str(out / (f + ".tmp")), metadata={"format": "pt"})
    os.replace(out / (f + ".tmp"), out / f)
    real = os.path.realpath(src)
    os.remove(real)  # the float32 blob
    os.remove(src)   # and its snapshot link
    log(f"verbalizer shard {i + 1}/{len(shards)} -> bfloat16, {os.path.getsize(out / f) / 1e9:.1f} GB")
log("INTERP DOWNLOAD DONE", out)
