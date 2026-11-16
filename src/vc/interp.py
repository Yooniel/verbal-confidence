"""Shared pieces of the interpretability scripts: Gemma's unembedding without loading the model, the Jacobian
lens (Gurnee et al. 2026) and the sparse non-negative decomposition into J-lens vectors."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from scipy.optimize import nnls

from .env import JLENS_FILE, JLENS_REPO, MODEL_ID


def base_dir() -> str:
    d = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"))
    assert d, "base model not in the Hugging Face cache"
    return d[0]


class Unembed:
    """Final RMSNorm + unembedding of Gemma 3 27B, read straight from the checkpoint (float32)."""

    def __init__(self, device="cuda:0"):
        from safetensors import safe_open

        d = base_dir()
        wm = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
        ek, nk = "language_model.model.embed_tokens.weight", "language_model.model.norm.weight"
        with safe_open(d + wm[ek], "pt") as f:
            self.W = f.get_tensor(ek).to(device, torch.float32)  # [V, d]; the unembedding is tied to it
        with safe_open(d + wm[nk], "pt") as f:
            self.gamma = 1.0 + f.get_tensor(nk).to(device, torch.float32)
        self.device = device

    def norm(self, x):
        x = x.to(self.device, torch.float32)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gamma

    def __call__(self, x):
        return self.norm(x) @ self.W.T


class JLens:
    """J[l] maps the residual stream at the output of block l into the basis of the last block's output;
    readout = unembed(h @ J[l].T). The last block itself is read with the identity."""

    def __init__(self, device="cuda:0"):
        from huggingface_hub import hf_hub_download

        ck = torch.load(hf_hub_download(JLENS_REPO, JLENS_FILE), map_location="cpu", weights_only=True)
        self.J = ck["J"]
        self.layers = sorted(self.J)
        self.n_prompts = ck["n_prompts"]
        self.provenance = ck.get("provenance")
        self.device = device

    def matrix(self, layer):
        return self.J[layer].to(self.device, torch.float32)

    def transport(self, h, layer, use_jacobian=True):
        h = h.to(self.device, torch.float32)
        return h @ self.matrix(layer).T if (use_jacobian and layer in self.J) else h

    def dictionary(self, un: Unembed, layer):
        """J-lens vectors at `layer`: row t is J[layer]^T W_U[t], shape [V, d]."""
        return un.W @ self.matrix(layer) if layer in self.J else un.W


def sparse_decomposition(x: torch.Tensor, D: torch.Tensor, norms: torch.Tensor, k: int = 25):
    """Greedy k-sparse non-negative decomposition of x into rows of D (non-negative orthogonal matching
    pursuit, as in TransformerLens' get_sparse_decomposition): at every step the atom with the largest
    normalised correlation with the residual joins the selected set and the coefficients are re-solved by
    non-negative least squares. Returns (atom indices, coefficients, reconstruction, projection of x on the
    span of the selected atoms)."""
    x = x.to(D.device, torch.float32)
    res, sel, coef = x.clone(), [], np.zeros(0)
    tol = float(np.sqrt(torch.finfo(torch.float32).eps)) * float(x.norm())
    for _ in range(k):
        c = (D @ res) / norms
        if sel:
            c[sel] = -float("inf")
        j = int(c.argmax())
        if float(c[j]) <= tol:
            break
        sel.append(j)
        A = D[sel].T.double().cpu().numpy()
        coef, _ = nnls(A, x.double().cpu().numpy())
        res = x - torch.tensor(A @ coef, device=D.device, dtype=torch.float32)
    A = D[sel].T.double()
    recon = (A @ torch.tensor(coef, device=D.device)).float()
    proj = (A @ torch.linalg.lstsq(A, x.double()[:, None]).solution[:, 0]).float()
    return sel, coef, recon, proj


def excess_kurtosis(v: torch.Tensor) -> torch.Tensor:
    d = v.float() - v.float().mean(-1, keepdim=True)
    return (d ** 4).mean(-1) / (d ** 2).mean(-1) ** 2 - 3.0
