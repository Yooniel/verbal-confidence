| Experiment | Quantity | Paper | This replication |
|---|---|---|---|
| Behaviour | valid trials | 7,858 | 9,944 |
| Behaviour | accuracy | 77.4% (GPT-4o-mini grader) | 81.8% (alias match or Gemma grader); 73.1% alias match only |
| Behaviour | ECE (categorical prompt) | 0.12 | 0.05 |
| Behaviour | AUROC of verbal confidence | 0.71 | 0.82 |
| Behaviour | share 'Almost certain' / 'Highly likely' | 69% / 21% | 49% / 32% |
| Behaviour | share of bottom-3 classes | 6.0% | 11.5% |
| Behaviour | Phase-0 confidence (same generation as the answer): ECE / AUROC / share 'Almost certain' / 'Highly likely' | (Fig. 9 values: 0.12 / 0.71 / 69% / 21%) | 0.09 / 0.72 / 76% / 19% |
| Behaviour | AUROC of mean answer log-prob | 0.75 | 0.80 |
| Behaviour | r(log-prob, phase-1 confidence) | 0.29 | 0.33 |
| Behaviour | r(log-prob, phase-0 confidence) | 0.23 | 0.26 |
| Behaviour | r(phase-0 conf, phase-1 conf) | 0.63 | 0.59 |
| Behaviour | minimal 0-9 prompt: ECE / AUROC | 0.17 / 0.68 | 0.12 / 0.55 (first token is a digit in 100% of trials) |
| Steering | baseline confidence of test trials | 0.55 | 0.52 |
| Steering | PANL, scale 5: peak Δconf, high vector | ≈ +0.17 @ L24–26 (peak range L21–25) | 0.261 @ L24 |
| Steering | PANL, scale 5: peak Δconf, low vector | ≈ −0.22 @ L24 | -0.460 @ L22 |
| Steering | CC, scale 5: peak Δconf, high vector | ≈ +0.46 @ L30–34 (peak range L30–35) | 0.434 @ L32 |
| Steering | CC, scale 5: peak Δconf, low vector | ≈ −0.42 @ L30 | -0.456 @ L50 |
| Steering | PANL, scale 2: peak Δconf high / low | ≈ +0.16 / −0.09 | 0.215 @ L22 / -0.106 @ L23 |
| Steering | CC, scale 2: peak Δconf high / low | ≈ +0.42 / −0.11 | 0.411 @ L30 / -0.071 @ L34 |
| Steering | PANL+1, scale 5: largest absolute Δconf over all layers and both vectors | below 0.01 in absolute value | 0.007 |
| Steering | FCC, scale 5: largest absolute Δconf over all layers and both vectors | below 0.01 in absolute value | 0.014 |
| Steering | AC, scale 5: largest absolute Δconf over all layers and both vectors | about 0.03 or less, not significant | 0.046 |
| Steering | last answer token, scale 5: peak Δconf high / low | effective (≈ +0.33 / −0.35) | 0.373 @ L24 / -0.392 @ L23 |
| Steering | first answer token, scale 5, multi-token answers only (n=147): peak Δconf high / low | ineffective | 0.102 @ L21 / -0.060 @ L21 (last answer token on the same trials: 0.394 @ L24 / -0.340 @ L22) |
| Patching | clean logit diff / corrupted logit diff | ≈ 11.9 / ≈ −0.5 | 14.77 / -0.53 |
| Patching | corrupted run: confidence / token change | ≈ 0.05 / 100% | 0.05 / 100% |
| Patching | PANL: peak confidence after patch | ≈ 0.38 @ L25 | 0.13 @ L22 |
| Patching | PANL: peak confidence recovery | ≈ 40% @ L25 | 9.2% @ L22 |
| Patching | PANL: peak logit-diff recovery | ≈ 20% @ L25 | 10.4% @ L22 |
| Patching | PANL: recovery of the margin (clean class − corrupted-run class), supplementary | not reported | 20.3% @ L22 (margin -14.1 → -7.7; clean 17.2) |
| Patching | PANL: token change rate at best layer | ≈ 78% @ L25 | 96% @ L25 |
| Patching | PANL+1: peak confidence recovery | ≈ 0% (−1.4% @ L25) | 0.0% @ L0 |
| Patching | CC: confidence recovery @ L35 / L40 / L61 | ≈ 22% / ≈ 95% / 100% (rises after L30) | 39% / 98% / 100% |
| Patching | CC: confidence recovery @ L25 / L30 | ≈ 0% / ≈ 0% | 0% / 0% |
| Noising | clean logit diff | ≈ 9.4 | 10.76 |
| Noising | PANL: peak token change rate | ≈ 14% @ L25 | 18.5% @ L25 |
| Noising | PANL: largest logit-diff drop | ≈ −0.9 @ L25–26 | -1.04 @ L26 |
| Noising | PANL+1: peak token change rate | ≈ 3–4% (flat) | 3.2% @ L20 |
| Noising | AC: peak token change rate | ≈ PANL+1 control | 3.5% @ L0 |
| Noising | CC: token change @ L30 / L35 / L40 / L61 | ≈ 5% / 19% / 71% / 77% | 4% / 22% / 71% / 72% |
| Noising | numerical noise floor (unedited rows vs full clean pass), max over layers | not reported | 2.8% (removed by the same-pass reference) |
| Swap (PANL) | low->high (recipient→donor): peak token change | ≈ 37% @ L26 | 47.5% @ L25 |
| Swap (PANL) | high->low (recipient→donor): peak token change | ≈ 28% @ L26–27 | 21.5% @ L26 |
| Swap (PANL) | high->high (recipient→donor): peak token change | ≈ 16% @ L25–26 | 8.5% @ L22 |
| Swap (PANL) | low->low (recipient→donor): peak token change | ≈ 13% @ L25 | 27.3% @ L25 |
| Swap (PANL) | low->high: peak confidence change | ≈ +0.21 @ L26 | 0.214 @ L25 |
| Swap (PANL) | high->low: peak confidence change | ≈ −0.065 @ L26 | -0.038 @ L26 |
| Swap (PANL) | high->high: peak confidence change | ≈ +0.02 | 0.001 @ L32 |
| Swap (PANL) | low->low: peak confidence change | ≈ −0.01 | -0.007 @ L28 |
| Swap (PANL) | low->high: largest logit-diff change | ≈ −1.15 @ L26–27 | -2.64 @ L25 |
| Swap (PANL) | high->low: largest logit-diff change | ≈ −1.85 @ L26 | -2.12 @ L26 |
| Swap (PANL) | high->high: largest logit-diff change | ≈ −0.2 | -0.27 @ L22 |
| Swap (PANL) | low->low: largest logit-diff change | ≈ −0.2 | -0.24 @ L25 |
| Swap (PANL+1) | cross-confidence peak confidence change (L→H / H→L) | ≈ 0 (no effect) | 0.006 @ L40 / -0.000 @ L0 |
| Swap (CC) | cross-confidence peak confidence change (L→H / H→L) | ≈ +0.68 / −0.65 @ L40–61 (onset after L31) | 0.786 @ L40 / -0.789 @ L61 |
| Decoding | R²_CV of log-prob baselines: mean / min / first / var / last / max / all six | 0.084 / 0.101 / 0.070 / 0.051 / 0.039 / 0.025 / 0.100 | 0.094 / 0.096 / 0.084 / 0.035 / 0.051 / 0.042 / 0.095 |
| Decoding | AUROC: verbal confidence / mean log-prob (collection set) | 0.71 / 0.75 | 0.82 / 0.78 |
| Decoding | correctness AUROC from PANL @ L20 / L28 / peak | ≈ 0.75 / ≈ 0.83 / ≈ 0.84 | 0.78 / 0.84 / 0.85 @ L40 |
| Decoding | correctness AUROC from CC @ L20 / L28 / peak | ≈ 0.71 / ≈ 0.81 / ≈ 0.84 | 0.73 / 0.83 / 0.85 @ L50 |
| Decoding | correctness AUROC from QTT (control), peak | ≈ 0.57 | 0.57 @ L28 |
| Decoding | confidence R² (ridge α=1) from PANL: first layer > 0.05 / peak | L25 / ≈ 0.48 @ L40 | L25 / 0.54 @ L50 |
| Decoding | confidence R² (ridge α=1) from CC: first layer > 0.05 / peak | L28 / ≈ 0.84 @ L50 | L28 / 0.88 @ L50 |
| Decoding | R² unique to PANL beyond all six log-prob baselines, peak | 0.38 @ L40 | 0.44 @ L50 |
| Decoding | R² unique to CC beyond all six log-prob baselines, peak | ≈ 0.76 @ L50 (vs mean log-prob) | 0.79 @ L50 |
| Decoding | confidence R² (tuned ridge): AC / PANL / PANL+1, peak | ≈ 0.2 / ≈ 0.75 / ≈ 0.70 | 0.19 @ L35 / 0.78 @ L35 / 0.74 @ L40 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | baseline: clean logit diff / control-path noise floor | n/a | 8.66 / 0.5% token change |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | CC→PANL: peak token change | ≈ 21% @ windows centred L30–36 | 5.8% @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | CC→PANL: largest logit-diff change | ≈ −0.8 @ L34 | -1.91 @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | CC→PANL+1 (control): peak token change / largest logit-diff change | ≈ 10% / ≈ −0.1 | 1.2% @ L26 / -0.17 @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | CC→Q+A: peak token change / largest logit-diff change | ≈ 10–12% (= control) / ≈ −0.25 | 2.0% @ L36 / -0.16 @ L28 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | PANL→A: peak token change / largest logit-diff change | ≈ 20% @ L24–30 / ≈ −0.5 @ L22–26 | 5.4% @ L24 / -2.22 @ L24 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n\n) | PANL→last A: peak token change / largest logit-diff change | ≈ 18% @ L24–30 / ≈ −0.5 | 4.8% @ L24 / -1.44 @ L24 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | baseline: clean logit diff / control-path noise floor | n/a | 8.44 / 0.1% token change |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | CC→PANL: peak token change | ≈ 21% @ windows centred L30–36 | 7.8% @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | CC→PANL: largest logit-diff change | ≈ −0.8 @ L34 | -1.92 @ L34 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | CC→PANL+1 (control): peak token change / largest logit-diff change | ≈ 10% / ≈ −0.1 | 2.0% @ L30 / -0.17 @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | CC→Q+A: peak token change / largest logit-diff change | ≈ 10–12% (= control) / ≈ −0.25 | 1.0% @ L38 / -0.06 @ L28 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | PANL→A: peak token change / largest logit-diff change | ≈ 20% @ L24–30 / ≈ −0.5 @ L22–26 | 6.2% @ L24 / -1.84 @ L24 |
| Attention blocking (minimal prompt; prompt ends  **:'  (digit is the next token), separator \n) | PANL→last A: peak token change / largest logit-diff change | ≈ 18% @ L24–30 / ≈ −0.5 | 6.4% @ L24 / -1.34 @ L24 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | baseline: clean logit diff / control-path noise floor | n/a | 4.36 / 4.8% token change |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | CC→PANL: peak token change | ≈ 21% @ windows centred L30–36 | 42.4% @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | CC→PANL: largest logit-diff change | ≈ −0.8 @ L34 | -0.48 @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | CC→PANL+1 (control): peak token change / largest logit-diff change | ≈ 10% / ≈ −0.1 | 16.2% @ L10 / -0.04 @ L45 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | CC→Q+A: peak token change / largest logit-diff change | ≈ 10–12% (= control) / ≈ −0.25 | 26.6% @ L10 / -0.44 @ L10 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | PANL→A: peak token change / largest logit-diff change | ≈ 20% @ L24–30 / ≈ −0.5 @ L22–26 | 37.4% @ L22 / -0.61 @ L22 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n\n) | PANL→last A: peak token change / largest logit-diff change | ≈ 18% @ L24–30 / ≈ −0.5 | 25.6% @ L22 / -0.36 @ L22 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | baseline: clean logit diff / control-path noise floor | n/a | 4.60 / 2.9% token change |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | CC→PANL: peak token change | ≈ 21% @ windows centred L30–36 | 55.2% @ L32 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | CC→PANL: largest logit-diff change | ≈ −0.8 @ L34 | -0.73 @ L30 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | CC→PANL+1 (control): peak token change / largest logit-diff change | ≈ 10% / ≈ −0.1 | 20.2% @ L10 / -0.04 @ L45 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | CC→Q+A: peak token change / largest logit-diff change | ≈ 10–12% (= control) / ≈ −0.25 | 35.4% @ L10 / -0.36 @ L10 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | PANL→A: peak token change / largest logit-diff change | ≈ 20% @ L24–30 / ≈ −0.5 @ L22–26 | 23.2% @ L22 / -0.48 @ L24 |
| Attention blocking (minimal prompt; prompt ends  **:  (digit logits read at CC), separator \n) | PANL→last A: peak token change / largest logit-diff change | ≈ 18% @ L24–30 / ≈ −0.5 | 21.6% @ L22 / -0.35 @ L24 |
| Attention blocking (categorical, all layers) | CC→NL+1: token change / Δ logit diff | ≈ 10% | 0.8% / -0.02 |
| Attention blocking (categorical, all layers) | CC→NL: token change / Δ logit diff | ≈ 9% | 2.8% / -0.43 |
| Attention blocking (categorical, all layers) | CC→A: token change / Δ logit diff | ≈ 12% | 2.0% / -0.23 |
| Attention blocking (categorical, all layers) | CC→Q: token change / Δ logit diff | ≈ 11% | 1.2% / -0.06 |
| Attention blocking (categorical, all layers) | CC→Q+A: token change / Δ logit diff | ≈ 12% | 2.8% / -0.31 |
| Attention blocking (categorical, all layers) | ALL→NL: token change / Δ logit diff | ≈ 18% | 13.6% / -0.83 |
| Attention blocking (categorical, all layers) | ALL→last_A: token change / Δ logit diff | ≈ 51% | 61.4% / -7.58 |
| Attention blocking (categorical, all layers) | ALL→NL+last_A: token change / Δ logit diff | ≈ 48% | 45.8% / -5.76 |
| Attention blocking (categorical, all layers) | ALL→last_A_keepNL: token change / Δ logit diff | ≈ 24% | 24.4% / -2.04 |
| Attention blocking (categorical, all layers) | ALL→A: token change / Δ logit diff | ≈ 68% | 64.6% / -9.22 |
| Attention blocking (categorical, all layers) | ALL→NL+A: token change / Δ logit diff | ≈ 59% | 59.6% / -8.26 |
| Attention blocking (categorical, all layers) | ALL→A_keepNL: token change / Δ logit diff | ≈ 51% | 30.8% / -2.97 |
| OOD check, layer 25, PANL (Table 1) | natural pairwise cosine [p5, p95] / norm ratio [p5, p95] | [0.998, 1.000] / [0.90, 1.10] | [0.997, 1.000] / [0.90, 1.11] |
| OOD check, layer 25, PANL (Table 1) | steering, high vector, scale 2: cosine / norm ratio | 0.999 / 0.96 | 0.999 / 0.96 |
| OOD check, layer 25, PANL (Table 1) | steering, low vector, scale 2: cosine / norm ratio | 0.999 / 1.04 | 0.999 / 1.05 |
| OOD check, layer 25, PANL (Table 1) | steering, high vector, scale 5: cosine / norm ratio | 0.992 / 0.91 | 0.994 / 0.89 |
| OOD check, layer 25, PANL (Table 1) | steering, low vector, scale 5: cosine / norm ratio | 0.994 / 1.10 | 0.996 / 1.12 |
| OOD check, layer 25, PANL (Table 1) | noising (mean replacement): cosine / norm ratio | 0.999 / 0.995 | 0.999 / 1.008 |
| OOD check, layer 25, PANL (Table 1) | patching (clean vs corruption-propagated): cosine / norm ratio | 0.999 / 0.94 | 0.998 / 1.04 (corrupted ÷ clean) |
| OOD check, layer 25, PANL+1 (Table 1) | natural pairwise cosine [p5, p95] / norm ratio [p5, p95] | [0.997, 1.000] / [0.90, 1.11] | [0.997, 1.000] / [0.89, 1.12] |
| OOD check, layer 25, PANL+1 (Table 1) | steering, high vector, scale 2: cosine / norm ratio | 0.999 / 1.04 | 0.999 / 1.05 |
| OOD check, layer 25, PANL+1 (Table 1) | steering, low vector, scale 2: cosine / norm ratio | 0.998 / 0.96 | 0.999 / 0.96 |
| OOD check, layer 25, PANL+1 (Table 1) | steering, high vector, scale 5: cosine / norm ratio | 0.995 / 1.10 | 0.996 / 1.12 |
| OOD check, layer 25, PANL+1 (Table 1) | steering, low vector, scale 5: cosine / norm ratio | 0.993 / 0.91 | 0.993 / 0.90 |
| OOD check, layer 25, PANL+1 (Table 1) | noising (mean replacement): cosine / norm ratio | 0.999 / 0.999 | 0.999 / 0.998 |
| OOD check, layer 25, PANL+1 (Table 1) | patching (clean vs corruption-propagated): cosine / norm ratio | 0.999 / 0.98 | 0.999 / 0.97 (corrupted ÷ clean) |
