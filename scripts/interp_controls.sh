#!/usr/bin/env bash
# Steering controls of the interpretability extension (single-prompt format), one of two shards:
#     bash scripts/interp_controls.sh 0      (and the same with 1)
# Needs results/interp/vectors_*.pt (scripts/interp_jlens.py and scripts/interp_sae.py).
# About 1.5 hours on two A40s per shard.
set -uo pipefail
k=$1
cd "$(dirname "$0")/.."
run() { .venv/bin/python scripts/interv.py --exp steer --shard "$k" --nshards 2 "$@"; }
# the "Incorrect" lens direction and eight unrelated lens tokens, at the size of the steering direction
LL=20,22,24,26,28,30,31,33,35
run --vec-file interp/vectors_lens_incorrect.pt --steer-pos PANL,CC --steer-scales 2,5 --layers $LL --rows 72 --tag _lensinc
run --vec-file interp/vectors_lens_control.pt --steer-pos PANL,CC --steer-scales 2,5 --layers $LL --rows 72 --tag _lensctl
# the direction split into its projection on the average activation at the position and the remainder
run --vec-file interp/vectors_meanpart.pt --steer-pos PANL,CC --steer-scales 5 --layers $LL --tag _meanpart
run --vec-file interp/vectors_meanrest.pt --steer-pos PANL,CC --steer-scales 5 --layers $LL --tag _meanrest
# the block-31 direction split by the sparse autoencoder
for n in saepart saerest saefull saeerr; do
  run --vec-file interp/vectors_$n.pt --steer-pos PANL,CC --steer-scales 5 --layers 31 --tag _$n
done
# the direction split into 25 lens vectors and the remainder (the split differs for the two signs)
LJ=22,24,26,30,31,33
run --vec-file interp/vectors_jpart.pt --steer-pos PANL,CC --steer-scales 5 --steer-dirs high --layers $LJ --tag _jpart
run --vec-file interp/vectors_jpart_neg.pt --steer-pos PANL,CC --steer-scales 5 --steer-dirs low --layers $LJ --tag _jpartneg
run --vec-file interp/vectors_jrest.pt --steer-pos PANL,CC --steer-scales 5 --steer-dirs high --layers $LJ --tag _jrest
run --vec-file interp/vectors_jrest_neg.pt --steer-pos PANL,CC --steer-scales 5 --steer-dirs low --layers $LJ --tag _jrestneg
echo "CONTROLS DONE shard $k"
