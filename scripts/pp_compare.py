"""Compare two policies on the same episodes: totals and an exact two-sided sign test on the discordant pairs.

    python scripts/pp_compare.py --a runs_pp/final/sft_test_greedy.jsonl runs_pp/final/sft_heldout_greedy.jsonl \
                                 --b runs_pp/final/grpo_test_greedy.jsonl runs_pp/final/grpo_heldout_greedy.jsonl

Each file is an output of `pp_rl.py eval` (or `pp_probe.py play`). Episodes are matched by their `key`.
"""
import argparse
import json
import math


def successes(paths):
    out = {}
    for path in paths:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = bool(row["success"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    args = parser.parse_args()
    a, b = successes(args.a), successes(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no episode appears in both sets")
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    n, m = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    print(json.dumps({"episodes": len(keys), "a_successes": sum(a[k] for k in keys), "b_successes": sum(b[k] for k in keys),
                      "only_a": only_a, "only_b": only_b, "sign_test_p": p}))


if __name__ == "__main__":
    main()
