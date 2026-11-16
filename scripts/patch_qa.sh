#!/usr/bin/env bash
# Patching with the question tokens corrupted as well as (or instead of) the answer tokens, one of two shards:
#     bash scripts/patch_qa.sh raw 0      (and raw 1, tt 0, tt 1): restore one position at one layer;
#                                         about 1 h on two A40s per call (tt: 45 min, both tokens corrupted only)
#     bash scripts/patch_qa.sh raw win    (and tt win): hold one position clean over a range of layers; about 45 min
#     bash scripts/patch_qa.sh raw blk    the same with tokens after PANL cut off from question and answer; about 25 min
set -uo pipefail
fmt=$1; k=$2
cd "$(dirname "$0")/.."
if [ "$fmt" = tt ]; then
  export VC_FMT=tt
  POS=PANL,PANL+1,POST5,REQ,CC,A_last; WPOS=PANL,PANL+1,POST5,REQ,A_last; SET=(); TAG=""; SPANS="qa"; WSPANS="a qa"
else
  POS=PANL,CC,PANL+1,INSTR,A_last; WPOS=PANL,PANL+1,INSTR,A_last; SET=(--set patch_p0); TAG="_p0_cabs"; SPANS="qa q"; WSPANS="a q qa"
fi
if [ "$k" = blk ]; then
  # as win, and tokens after PANL cannot attend to the question, the answer label or the answer (single prompt)
  for span in $WSPANS; do
    .venv/bin/python scripts/interv.py --exp patchwin "${SET[@]}" --corrupt abs --corrupt-span $span --patch-pos PANL,PANL+1,A_last \
        --patch-windows 0-61,20-28 --patch-block pre --rows 72 --tag "${TAG}_${span}_blk"
  done
elif [ "$k" = win ]; then
  # a position held at its clean activations over a range of layers (one job, no shards)
  for span in $WSPANS; do
    .venv/bin/python scripts/interv.py --exp patchwin "${SET[@]}" --corrupt abs --corrupt-span $span --patch-pos $WPOS \
        --rows 72 --tag "${TAG}_$span"
  done
else
  # clean activations restored at one position and one layer
  for span in $SPANS; do
    .venv/bin/python scripts/interv.py --exp patch "${SET[@]}" --corrupt abs --corrupt-span $span --patch-pos $POS \
        --rows 72 --shard "$k" --nshards 2 --tag "${TAG}_$span"
  done
fi
echo "PATCH_QA DONE $fmt $k"
