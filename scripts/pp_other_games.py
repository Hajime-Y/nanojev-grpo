"""How a checkpoint trained only on Predict Position does on the other games of the NanoJev release.

1. ViZDoom Basic, the 128 released test cases, greedy play (the same episode loop as `pp_rl.py eval`).
2. Agreement with the released targets on the dev rows of Maze, Snake and Basic (argmax match and cross
   entropy), as an offline measure for the games this script does not play.

    python scripts/pp_other_games.py --checkpoint runs_pp/grpo_main/ckpt/step-0050 --output runs_pp/other/grpo.json

Run from the repository root on one CUDA GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pp_rl as rl  # noqa: E402
from pp_envpool import EnvPool  # noqa: E402
from predict_toy_decisions import prepare_examples  # noqa: E402
from unified_game_pipeline import read_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basic-cases", default="configs/appo_basic_supervision_v1_cases.jsonl")
    parser.add_argument("--dev-rows", default="data/NanoJev-unified/unified/hard/dev.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scorer = rl.Scorer(rl.load(args.checkpoint, 8192, "bf16"), 30000, "tree")
    result = {"checkpoint": args.checkpoint}

    cases = [c for c in read_rows(args.basic_cases) if c["variant"] == "doom_basic" and c["split"] == "test"]
    jobs = [{"key": f"{c['id']}#r0", "case": c} for c in cases]
    with EnvPool(args.workers) as pool:
        episodes = rl.eval_episodes(scorer, pool, jobs, "greedy", 128, seed=17)
    summary = rl.summarize(episodes)
    result["basic_test_greedy"] = {k: summary[k] for k in ("episodes", "successes", "fired", "mean_model_decisions", "outcomes")}

    groups = {}
    for row in read_rows(args.dev_rows):
        meta = row["metadata"]
        if meta.get("record_role") != "policy":
            continue
        game = meta["task"] if meta["task"] != "shooting" else meta["spec"]["scenario"]
        ids = list(row["questions"]["action"]["criteria"])
        if "gold" in row and row["gold"].get("action") in ids:
            target = {a: float(a == row["gold"]["action"]) for a in ids}
        elif "teacher" in row:
            probs = row["teacher"]["native_probs"]["action"]
            total = sum(probs[a] for a in ids)
            target = {a: probs[a] / total for a in ids}
        else:
            continue
        groups.setdefault(game, []).append(({"id": row["id"], "state": row["state"], "questions": row["questions"]}, target))
    result["dev_agreement"] = {}
    for game, items in sorted(groups.items()):
        agree = ce = 0.0
        done = 0
        for offset in range(0, len(items), 64):
            chunk = items[offset:offset + 64]
            try:
                examples = prepare_examples({"states": [request for request, _ in chunk]}, scorer.tokenizer, scorer.max_length)
                outputs = scorer.score_examples(examples)
            except ValueError as exc:
                result.setdefault("skipped", []).append(f"{game}: {str(exc)[:80]}")
                continue
            for (_, target), (ids, logps) in zip(chunk, outputs):
                best_target = max(sorted(target), key=target.get)
                best_model = min(ids, key=lambda a: (-logps[ids.index(a)], a))
                agree += best_target == best_model
                ce -= sum(target[a] * logps[ids.index(a)] for a in ids if target[a] > 0)
                done += 1
        if done:
            result["dev_agreement"][game] = {"questions": done, "agreement": agree / done, "ce": ce / done}
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
