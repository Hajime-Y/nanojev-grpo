#!/bin/bash
# usage: [INIT=checkpoint_dir] [CASES=cases.jsonl] [RUNS=runs_pp] scripts/pp_train.sh NAME grpo|sft [extra pp_rl.py args...]
# Runs pp_rl.py and restarts it with --resume after a crash (at most 5 times). Run from the repository root
# with the virtualenv active. The run is written to $RUNS/NAME and its log to $RUNS/NAME.log.
NAME=$1; MODE=$2; shift 2
INIT=${INIT:-checkpoints/NanoJev-unified}
CASES=${CASES:-configs/sonic_predict_supervision_v1_cases.jsonl}
RUNS=${RUNS:-runs_pp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$RUNS/$NAME; LOG=$RUNS/$NAME.log
mkdir -p "$RUNS"
for attempt in 0 1 2 3 4 5; do
  RESUME=""; [ -e "$OUT/ckpt/last/state.json" ] && RESUME="--resume"
  echo "== $(date -u +%FT%TZ) launch attempt=$attempt resume=${RESUME:-no} init=$INIT cases=$CASES" >> "$LOG"
  python scripts/pp_rl.py "$MODE" --init-checkpoint "$INIT" --cases "$CASES" --output-dir "$OUT" --seed 17 --forward tree "$@" $RESUME >> "$LOG" 2>&1
  code=$?
  echo "== $(date -u +%FT%TZ) exit code=$code" >> "$LOG"
  [ $code -eq 0 ] && break
  sleep 20
done
echo "== $(date -u +%FT%TZ) runner finished code=$code" >> "$LOG"
exit $code
