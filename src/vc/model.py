"""Gemma 3 27B wrapper with an explicit layer loop.

Running the decoder layers ourselves (instead of hooking the HF forward) gives three things the experiments
need: (1) capture of the residual stream after any layer, (2) *resuming* a forward pass from a cached/edited
residual stream at layer l (so an intervention at layer l only costs layers l+1..61), and (3) per-layer
attention-edge blocking via explicit boolean masks.

Conventions
-----------
* "layer l" always means the residual stream at the OUTPUT of decoder block l (after its MLP), l in 0..61.
  ``run(h, start=s)`` expects ``h`` to be the input of block ``s`` (embeddings for s=0, output of block s-1
  otherwise).
* Batches are RIGHT-padded. With causal attention, real tokens never attend to the pads on their right, so
  no padding mask is needed and results are identical to unpadded runs; logits are read at each row's own
  last real token.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence

import torch

from .env import MODEL_ID, load_env


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class VC:
    def __init__(self, max_memory: dict | None = None, dtype=torch.bfloat16, attn_implementation: str = "sdpa"):
        load_env()
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        n_gpu = torch.cuda.device_count()
        if max_memory is None and n_gpu >= 2:
            # leave head-room on both cards for cached activations
            max_memory = {0: "27GiB", 1: "40GiB"}
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=dtype, device_map="auto", max_memory=max_memory, attn_implementation=attn_implementation
        )
        self.model.eval()
        self.lm = self.model.model.language_model
        self.layers = list(self.lm.layers)
        self.L = len(self.layers)
        self.cfg = self.lm.config
        self.d = self.cfg.hidden_size
        self.dtype = dtype
        self.layer_dev = [next(l.parameters()).device for l in self.layers]
        self.embed_dev = self.lm.embed_tokens.weight.device
        self.norm_dev = self.lm.norm.weight.device
        self.head_dev = self.model.lm_head.weight.device
        self.pad_id = self.tok.pad_token_id
        assert self.cfg.sliding_window >= 1024
        assert getattr(self.cfg, "final_logit_softcapping", None) is None
        self._pos_cache: dict = {}
        self._sel_cache: dict = {}
        log(
            f"model loaded in {time.time() - t0:.0f}s | layers={self.L} d={self.d} | "
            f"layer devices: {sorted({str(d) for d in self.layer_dev})} | split at layer "
            f"{next((i for i, d in enumerate(self.layer_dev) if d != self.layer_dev[0]), None)} | "
            f"embed={self.embed_dev} head={self.head_dev} norm={self.norm_dev}"
        )

    # ------------------------------------------------------------------------------------------------
    # batching helpers
    # ------------------------------------------------------------------------------------------------
    def pad(self, seqs: Sequence[Sequence[int]]):
        """Right-pad token-id lists. Returns (ids [B,T] on the embedding device, last-token index [B])."""
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        last = torch.tensor([len(s) - 1 for s in seqs], dtype=torch.long)
        return ids.to(self.embed_dev), last

    @torch.no_grad()
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Scaled input embeddings (the residual stream entering block 0)."""
        return self.lm.embed_tokens(ids.to(self.embed_dev))

    def _pos_emb(self, T: int, device):
        key = (T, str(device))
        if key not in self._pos_cache:
            pid = torch.arange(T, device=device)[None]
            dummy = torch.zeros(1, 1, 1, device=device, dtype=self.dtype)
            self._pos_cache[key] = (
                {lt: self.lm.rotary_emb(dummy, pid, lt) for lt in set(self.cfg.layer_types)},
                pid,
            )
        return self._pos_cache[key]

    # ------------------------------------------------------------------------------------------------
    # core forward
    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        h: torch.Tensor,
        start: int = 0,
        last_idx: torch.Tensor | None = None,
        capture: Iterable[int] | None = None,
        capture_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        edit: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        block: dict | None = None,
        stop: int | None = None,
        sel: Sequence[int] | None = None,
    ):
        """Run decoder blocks ``start..stop-1`` (default: to the end) on residual stream ``h`` [B,T,d].

        capture     : layer indices whose output is returned in ``cache[l]`` (optionally reduced by
                      ``capture_fn(l, h)``, e.g. to keep only a few positions).
        edit        : ``edit(l, h) -> h`` applied to the output of every block l that is run.
        block       : {"layers": set[int], "mask": bool [B,T,T]} with mask[b,q,k]=True meaning the attention
                      edge q<-k is removed in those layers (pre-softmax knockout, Geva et al. 2023).
        last_idx    : [B] index of each row's last real token; if given, logits are read out there.
        sel         : token ids of interest (e.g. the 10 class first-tokens). If given, the returned logits are
                      [B, len(sel)], computed with a float32 final norm + unembedding (bf16 logits are
                      quantised to ~0.125-0.25 at these magnitudes), and the third return value is the
                      full-vocabulary argmax token [B] (from the stock bf16 head).
        Returns (logits or None, cache dict, extra) where extra is the full-vocab argmax (``sel`` given), the
        final residual stream (``stop`` given), or None.
        """
        B, T, _ = h.shape
        assert T <= 1024, "prompts must fit inside the local-attention window"
        stop = self.L if stop is None else stop
        capture = set(capture) if capture is not None else set()
        cache = {}
        blk_layers, blk_masks = set(), {}
        if block is not None:
            blk_layers = set(block["layers"])
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            allowed = (causal[None] & ~block["mask"].cpu())[:, None]  # [B,1,T,T], True = may attend
            assert allowed.any(-1).all(), "every query needs at least one key"
            blk_masks = {str(d): allowed.to(d) for d in {self.layer_dev[i] for i in blk_layers if start <= i < stop}}
        for i in range(start, stop):
            layer, dev = self.layers[i], self.layer_dev[i]
            if h.device != dev:
                h = h.to(dev)
            pos, pid = self._pos_emb(T, dev)
            h = layer(
                h,
                position_embeddings=pos[layer.attention_type],
                attention_mask=blk_masks[str(dev)] if i in blk_layers else None,
                position_ids=pid,
            )
            if edit is not None:
                h = edit(i, h)
            if i in capture:
                cache[i] = capture_fn(i, h) if capture_fn is not None else h
        if stop != self.L:
            return None, cache, h
        if last_idx is None:
            return None, cache, None
        hl = h[torch.arange(B, device=h.device), last_idx.to(h.device)]
        hl = self.lm.norm(hl.to(self.norm_dev).float())  # RMSNorm computes in float32 and keeps the input dtype
        full = self.model.lm_head(hl.to(self.head_dev, self.dtype))  # stock bf16 unembedding
        if sel is None:
            return full.float(), cache, None
        key = tuple(sel)
        if key not in self._sel_cache:
            self._sel_cache[key] = self.model.lm_head.weight[list(sel)].float()
        logits = hl.to(self.head_dev) @ self._sel_cache[key].T
        return logits, cache, full.argmax(-1)

    @torch.no_grad()
    def hf_logits(self, ids: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        """Reference: logits at each row's last real token from the stock HF forward (for validation)."""
        attn = (torch.arange(ids.shape[1], device=ids.device)[None] <= last_idx.to(ids.device)[:, None]).long()
        out = self.model(input_ids=ids, attention_mask=attn)
        lg = out.logits
        return lg[torch.arange(ids.shape[0], device=lg.device), last_idx.to(lg.device)].float()

    # ------------------------------------------------------------------------------------------------
    # generation (Phase 0)
    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 40, stop_fn=None):
        """Greedy batched generation from raw-text prompts.

        Returns, per prompt: generated token ids (list) and their log-probabilities (list).
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        tok = self.tok
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.embed_dev)
        n_in = enc["input_ids"].shape[1]

        class _Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                done = [bool(stop_fn(tok.decode(row[n_in:], skip_special_tokens=False))) for row in input_ids]
                return torch.tensor(done, device=input_ids.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            output_logits=True,
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([_Stop()]) if stop_fn is not None else None,
            pad_token_id=tok.pad_token_id,
        )
        seq = out.sequences[:, n_in:].cpu()
        lp = torch.stack([torch.log_softmax(l.float(), -1) for l in out.logits], 1).cpu()  # [B, steps, V]
        tok_lp = lp.gather(-1, seq[..., None]).squeeze(-1)
        res = []
        stop_ids = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
        for b in range(seq.shape[0]):
            ids, lps = [], []
            for t in range(seq.shape[1]):
                tid = int(seq[b, t])
                if tid in stop_ids:
                    break
                ids.append(tid)
                lps.append(float(tok_lp[b, t]))
            res.append((ids, lps))
        return res


def batched(items: list, bs: int):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]
