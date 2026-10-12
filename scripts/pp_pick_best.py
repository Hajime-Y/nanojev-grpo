"""pp_pick_best.py EVAL_JSONL PREFIX: the snapshot with the most dev greedy successes (ties go to the later one).

EVAL_JSONL is the eval.jsonl written by `pp_rl.py grpo` or `pp_rl.py sft`; prints e.g. `update-1100`.
"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
rows = [r for r in rows if r["mode"] == "greedy" and r["step"] > 0]
best = max(rows, key=lambda r: (r["successes"], r["step"]))
print(f"{sys.argv[2]}-{best['step']:04d}")
