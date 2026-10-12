#!/bin/bash
# usage: [RUNS=runs_pp] scripts/pp_final_eval.sh NAME CHECKPOINT_DIR [WORKERS] [MAX_FORWARD_TOKENS]
# The evaluation reported in the README: the released test split (128) and the held-out seeds (512), greedy and
# sampled, plus the released OOD split (128, one decision = 8 ticks), greedy. Skips files that already exist.
# Run from the repository root with the virtualenv active.
NAME=$1; CK=$2; W=${3:-12}; MFT=${4:-60000}
RUNS=${RUNS:-runs_pp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=$RUNS/final; mkdir -p "$O"
RELEASED=configs/sonic_predict_supervision_v1_cases.jsonl
HELDOUT=configs/pp_heldout512_cases.jsonl
run() { out=$O/${NAME}_$1_$2.jsonl; [ -e "${out%.jsonl}.summary.json" ] && return; rm -f "$out"
  python scripts/pp_rl.py eval --checkpoint "$CK" --cases "$3" --splits "$4" --mode "$2" --workers "$W" \
    --max-forward-tokens "$MFT" --max-concurrent 256 --output "$out" 2>&1 | grep -v -i warning | tail -1 | cut -c1-200; }
run test greedy $RELEASED test
run heldout greedy $HELDOUT heldout
run ood greedy $RELEASED ood
run test sample $RELEASED test
run heldout sample $HELDOUT heldout
echo "FINAL_EVAL_DONE $NAME"
