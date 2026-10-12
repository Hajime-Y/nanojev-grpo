"""Does p(shoot) separate the states where shooting now hits from the states where it misses?

    python scripts/pp_probe_auc.py --probe runs_pp/probe/probe_policy.jsonl \
        --scores base=runs_pp/probe/scores_base_policy.jsonl sft=runs_pp/probe/scores_sft_policy.jsonl

`--probe` is the output of `pp_probe.py build` (each state carries the replay-based label `hit_now`), and each
scores file is the output of `pp_probe.py score` on it. Over all states the label is dominated by trivial cues
(no monster on screen, or too little time left), so the AUC is also reported on the states where the monster is
visible and time remains; that restricted number is the one quoted in the README. Two rule-only baselines are
printed for reference.
"""
import argparse
import json


def auc(labels, scores):
    """Rank AUC with ties counted as one half."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in order[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


SUBSETS = [("all states", lambda r: True),
           ("monster visible", lambda r: r["features"]["monster_visible"]),
           ("monster visible, 3 <= t <= 57", lambda r: r["features"]["monster_visible"] and 3 <= r["t"] <= 57)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--scores", nargs="+", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.probe) if line.strip()]
    for spec in args.scores:
        name, path = spec.split("=", 1)
        scored = {json.loads(line)["id"]: json.loads(line) for line in open(path) if line.strip()}
        for label, keep in SUBSETS:
            subset = [r for r in rows if r["id"] in scored and keep(r) and scored[r["id"]].get("policy")]
            hits = [bool(r["hit_now"]) for r in subset]
            shoot = [scored[r["id"]]["policy"]["shoot"] for r in subset]
            value = auc(hits, shoot)
            mean = lambda flag: sum(p for p, h in zip(shoot, hits) if h == flag) / max(1, sum(h == flag for h in hits))
            print(json.dumps({"name": name, "subset": label, "states": len(subset), "hits": sum(hits),
                              "auc_p_shoot": None if value is None else round(value, 3),
                              "mean_p_shoot_when_hit": round(mean(True), 3), "mean_p_shoot_when_miss": round(mean(False), 3)}))
    hits = [bool(r["hit_now"]) for r in rows]
    print(json.dumps({"reference": "monster visible (yes/no) as the only predictor, all states",
                      "auc": round(auc(hits, [float(r["features"]["monster_visible"]) for r in rows]), 3)}))
    visible = [r for r in rows if SUBSETS[2][1](r)]
    print(json.dumps({"reference": "closer to the screen centre, on the restricted states", "states": len(visible),
                      "hits": sum(bool(r["hit_now"]) for r in visible),
                      "auc": round(auc([bool(r["hit_now"]) for r in visible], [-abs(r["features"]["cx"] - 160) for r in visible]), 3)}))


if __name__ == "__main__":
    main()
