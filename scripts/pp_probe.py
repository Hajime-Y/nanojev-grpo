#!/usr/bin/env python3
"""Predict Position measurements: annotated play, probe collection, scoring, and reports."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pp_rl as rl
from pp_envpool import EnvPool
from predict_toy_decisions import prepare_examples
from unified_game_pipeline import policy_request, read_rows

BOOLEAN_QUESTIONS = {
    "side_right": "In the latest observation, is the centre of the monster's bounding box to the right of the screen centre (x = 160)? The monster is the visible label that is not DoomPlayer.",
    "moved_right": "Between the two most recent observations, did the monster's bounding box move to the right (larger x)? The monster is the visible label that is not DoomPlayer.",
    "hit_now": "If the player holds attack now and then waits, will the rocket kill the monster before the deadline?",
}


def choice_answer(logps):
    return {"logps": logps, "probabilities": {a: math.exp(p) for a, p in logps.items()}}


class CheckpointEngine:
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, requests):
        if not requests:
            return []
        # Keep the normal policy request on exactly the existing Scorer path.
        if all(list(r["questions"]) == ["action"] for r in requests):
            return [{"action": choice_answer(dict(zip(ids, logps)))}
                    for ids, logps in self.scorer(requests)]
        examples = prepare_examples({"states": requests}, self.scorer.tokenizer, self.scorer.max_length)
        answers = {r["id"]: {} for r in requests}
        for ex, (ids, logps) in zip(examples, self.scorer.score_examples(examples)):
            answers[ex["state_id"]][ex["qid"]] = (
                {"p_true": math.exp(logps[ids.index("true")])} if ex["type"] == "boolean"
                else choice_answer(dict(zip(ids, logps))))
        return [answers[r["id"]] for r in requests]

    def summary(self):
        return dict(api_calls=0, input_tokens=0, tokens=dict(self.scorer.tokens))


class JevEngine:
    def __init__(self, url="http://127.0.0.1:8791/v1/systemone", model="typesafe/jev-1.13",
                 concurrency=12, retries=3, retry_delay=1.0, timeout=120):
        if concurrency < 1 or retries < 0:
            raise ValueError("Invalid Jev concurrency/retries")
        self.url, self.model, self.concurrency = url, model, concurrency
        self.retries, self.retry_delay, self.timeout = retries, retry_delay, timeout
        self.counts, self.lock = Counter(), threading.Lock()

    def _one(self, request):
        questions = {name: dict(q, type="noul") if q["type"] == "boolean" else q
                     for name, q in request["questions"].items()}
        body = json.dumps(dict(model=self.model, state=request["state"], questions=questions)).encode()
        for attempt in range(self.retries + 1):
            with self.lock:
                self.counts["api_calls"] += 1
            try:
                with urlopen(Request(self.url, data=body, headers={"Content-Type": "application/json"}),
                             timeout=self.timeout) as response:
                    if response.status != 200:
                        raise HTTPError(self.url, response.status, "Non-200 response", response.headers, response)
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError):
                    exc.close()
                if attempt == self.retries:
                    raise RuntimeError(f"Jev request {request['id']} failed after {attempt + 1} attempts: {exc}") from exc
                time.sleep(self.retry_delay * (2 ** attempt))
        usage = payload.get("usage", {})
        with self.lock:
            self.counts["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        answers = {}
        for name, q in request["questions"].items():
            answer = payload["answers"][name]
            if q["type"] == "boolean":
                p = float(answer["noul"])
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("Invalid Jev Boolean probability")
                answers[name] = {"p_true": p}
            else:
                probs = answer["probabilities"]
                if set(probs) != set(q["criteria"]):
                    raise ValueError("Jev candidate mismatch")
                probs = {a: float(probs[a]) for a in q["criteria"]}
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
                    raise ValueError("Invalid Jev choice probabilities")
                logs = {a: math.log(max(p, 1e-6)) for a, p in probs.items()}
                normalizer = math.log(sum(math.exp(p) for p in logs.values()))
                answers[name] = choice_answer({a: p - normalizer for a, p in logs.items()})
        return answers

    def iter_answers(self, requests):
        """Yield completed rows promptly so an error does not discard earlier paid calls."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._one, r): i for i, r in enumerate(requests)}
            failure = None
            try:
                for future in as_completed(futures):
                    try:
                        answer = future.result()
                    except Exception as exc:
                        # Finish persisting successful in-flight rows before raising.
                        if failure is None:
                            failure = exc
                    else:
                        yield futures[future], answer
                if failure is not None:
                    raise failure
            finally:
                for future in futures:
                    future.cancel()

    def __call__(self, requests):
        answers = [None] * len(requests)
        for i, answer in self.iter_answers(requests):
            answers[i] = answer
        return answers

    def summary(self):
        return {name: self.counts[name] for name in ("api_calls", "input_tokens")}


class RandomEngine:
    def __call__(self, requests):
        return [{name: choice_answer({a: -math.log(len(q["criteria"])) for a in q["criteria"]})
                 for name, q in r["questions"].items()} for r in requests]

    def summary(self):
        return dict(api_calls=0, input_tokens=0)


def annotation(action, hit):
    subject = "the rocket" if action == "shoot" else "shooting right after this"
    return f" Simulated result: {subject} {'kills' if hit else 'misses'} the monster."


class OracleEngine(RandomEngine):
    def __call__(self, requests):
        results = []
        for request in requests:
            criteria = request["questions"]["action"]["criteria"]
            if any(not (text.endswith(annotation(a, True)) or text.endswith(annotation(a, False)))
                   for a, text in criteria.items()):
                raise ValueError("Oracle requires lookahead annotations")
            action = next((a for a in ("shoot", "noop", "left", "right")
                           if a in criteria and criteria[a].endswith(annotation(a, True))), "noop")
            results.append({"action": choice_answer({a: 0.0 if a == action else -1e30 for a in criteria})})
        return results


class PolicyScorer:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, requests):
        return [(list(answer["action"]["logps"]), list(answer["action"]["logps"].values()))
                for answer in self.engine(requests)]


def lookahead_hook(pool, episodes, requests):
    jobs, identities = [], []
    for request in requests:
        episode = episodes[request["id"]]
        history = [d["action"] for d in episode["decisions"]]
        for action in request["questions"]["action"]["criteria"]:
            key = len(jobs)
            identities.append((request["id"], action, key))
            jobs.append(dict(key=key, case=episode["case"], tail="noop",
                             actions=history + [action] + ([] if action == "shoot" else ["shoot"])))
    replayed = pool.replay(jobs)
    extra = {r["id"]: {"lookahead": {}} for r in requests}
    for identity, action, key in identities:
        extra[identity]["lookahead"][action] = replayed[key]["success"]
    for request in requests:
        question = request["questions"]["action"]
        question["criteria"] = {a: text + annotation(a, extra[request["id"]]["lookahead"][a])
                                for a, text in question["criteria"].items()}
    return extra


def features(state):
    state = json.loads(state) if isinstance(state, str) else state
    history = state["observed_history"]

    def center(snapshot):
        monster = next((label for label in snapshot["visible_labels"] if label["name"] != "DoomPlayer"), None)
        return None if monster is None else monster["bbox"][0] + monster["bbox"][2] / 2

    cx = center(history[-1])
    previous = center(history[-2]) if len(history) >= 2 else None
    return dict(monster_visible=cx is not None, cx=cx,
                side=None if cx is None else "left" if cx < 144 else "right" if cx > 176 else "center",
                dx=cx - previous if cx is not None and previous is not None else None,
                ammo=history[-1]["ammo"], angle_degrees=history[-1]["angle_degrees"],
                remaining_decisions=state["remaining_decisions"])


def golds(row):
    f = row["features"]
    cx, dx = f["cx"], f["dx"]
    return dict(side_right=None if not f["monster_visible"] or cx is None or abs(cx - 160) < 8 else cx > 160,
                moved_right=None if dx is None or abs(dx) < 2 else dx > 0, hit_now=bool(row["hit_now"]))


class NoShootScorer:
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, requests):
        outputs = self.scorer(requests)
        result = []
        for ids, logs in outputs:
            maximum = max(p for a, p in zip(ids, logs) if a != "shoot")
            normalizer = math.log(sum(math.exp(p - maximum) for a, p in zip(ids, logs) if a != "shoot"))
            result.append((ids, [-1e30 if a == "shoot" else p - maximum - normalizer
                                 for a, p in zip(ids, logs)]))
        return result


def build_probes(pool, cases, behaviors, scorer=None, stride=3, seed=17, max_concurrent=128):
    if stride < 1 or not behaviors or len(set(behaviors)) != len(behaviors):
        raise ValueError("Positive stride and unique behaviors required")
    rows = []
    for behavior in behaviors:
        if behavior not in {"policy_noshoot", "random_noshoot"}:
            raise ValueError(f"Unknown behavior: {behavior}")
        base = scorer if behavior == "policy_noshoot" else PolicyScorer(RandomEngine())
        if base is None:
            raise ValueError("policy_noshoot requires a checkpoint scorer")

        def collect(pool, episodes, requests):
            selected = []
            for request in requests:
                episode = episodes[request["id"]]
                t = len(episode["decisions"])
                if t % stride == 0:
                    selected.append(dict(id=f"{request['id']}#t{t}", case=episode["case"], behavior=behavior,
                                         t=t, state=request["state"], candidates=request["questions"]["action"]["criteria"],
                                         history=[d["action"] for d in episode["decisions"]],
                                         features=features(request["state"])))
            outcomes = pool.replay([dict(key=r["id"], case=r["case"], actions=r["history"] + ["shoot"], tail="noop")
                                    for r in selected])
            for row in selected:
                row["hit_now"] = outcomes[row["id"]]["success"]
                rows.append(row)
            return {}

        jobs = [dict(key=f"{c['id']}#{behavior}", case=c) for c in cases]
        rl.eval_episodes(NoShootScorer(base), pool, jobs, "sample", max_concurrent,
                         seed=seed, decision_hook=collect)
    return rows


def probe_request(row, questions):
    request = dict(id=row["id"], state=row["state"], questions={})
    if "policy" in questions:
        request["questions"].update(policy_request(row, row["id"])["questions"])
    if "boolean" in questions:
        request["questions"].update({name: dict(type="boolean", instructions=BOOLEAN_QUESTIONS[name])
                                     for name, gold in golds(row).items() if gold is not None})
    return request


def score_probes(engine, rows, output, questions=("policy", "boolean"), batch_size=128):
    if not questions or set(questions) - {"policy", "boolean"} or batch_size < 1:
        raise ValueError("Invalid questions or batch size")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Recover a torn final JSONL write, but reject malformed completed lines.
    if output.exists():
        with output.open("rb+") as handle:
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    try:
                        json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        handle.truncate(start)
                    else:
                        handle.write(b"\n")
                    break
    existing = read_rows(output) if output.exists() else []
    done = {r["id"] for r in existing}
    if len(done) != len(existing) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate score or probe ids")
    pending = [r for r in rows if r["id"] not in done]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        requests = [probe_request(r, questions) for r in batch]
        iterator = engine.iter_answers(requests) if hasattr(engine, "iter_answers") else enumerate(engine(requests))
        for i, answer in iterator:
            rl.append(output, dict(id=batch[i]["id"], policy=answer.get("action", {}).get("probabilities"),
                                   boolean={name: value["p_true"] for name, value in answer.items() if name != "action"}))
    return dict(rows=len(rows), skipped=len(done & {r["id"] for r in rows}), scored=len(pending), **engine.summary())


def auc(labels, scores):
    """Mann–Whitney AUC, using average ranks for ties; undefined for one class."""
    if len(labels) != len(scores):
        raise ValueError("AUC length mismatch")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def mean(values):
    return sum(values) / len(values) if values else None


def report_group(rows, scores):
    paired = [(r, scores[r["id"]]) for r in rows if r["id"] in scores]
    policy = [(r, s["policy"]) for r, s in paired if s.get("policy") is not None]
    labels = [r["hit_now"] for r, _ in policy]
    shoot = [p["shoot"] for _, p in policy]
    predicted = [max(sorted(p), key=p.get) == "shoot" for _, p in policy]
    tp = sum(y and pred for y, pred in zip(labels, predicted))
    toward = []
    for row, probs in policy:
        side = row["features"]["side"]
        if side in {"left", "right"}:
            diff = probs[side] - probs["right" if side == "left" else "left"]
            toward.append(1.0 if diff > 0 else 0.0 if diff < 0 else .5)
    boolean = {}
    for name in BOOLEAN_QUESTIONS:
        pairs = [(golds(r)[name], s.get("boolean", {})[name]) for r, s in paired
                 if golds(r)[name] is not None and name in s.get("boolean", {})]
        ys, ps = [y for y, _ in pairs], [p for _, p in pairs]
        boolean[name] = dict(n=len(pairs), base_rate=mean(ys),
                             accuracy=mean([(p >= .5) == y for y, p in pairs]), auc=auc(ys, ps))
    return dict(n=len(rows), scored_n=len(paired), hit_now_base_rate=mean([r["hit_now"] for r in rows]),
                scored_hit_now_base_rate=mean([r["hit_now"] for r, _ in paired]),
                policy=dict(n=len(policy), auc=auc(labels, shoot),
                            mean_p_shoot_hit=mean([p for y, p in zip(labels, shoot) if y]),
                            mean_p_shoot_miss=mean([p for y, p in zip(labels, shoot) if not y]),
                            shoot_precision=tp / sum(predicted) if any(predicted) else None,
                            shoot_recall=tp / sum(labels) if any(labels) else None,
                            explicit_n=len(toward), turn_toward_share=mean(toward)), boolean=boolean)


def report(probes, named_scores):
    if len({r["id"] for r in probes}) != len(probes):
        raise ValueError("Duplicate probe ids")
    groups = {"pooled": probes}
    groups.update({b: [r for r in probes if r["behavior"] == b] for b in sorted({r["behavior"] for r in probes})})
    result = {}
    for name, rows in named_scores.items():
        scores = {r["id"]: r for r in rows}
        if len(scores) != len(rows) or set(scores) - {r["id"] for r in probes}:
            raise ValueError("Duplicate or unknown score ids")
        result[name] = {b: report_group(group, scores) for b, group in groups.items()}
    return result


def make_engine(args):
    if args.engine == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for the checkpoint engine")
        return CheckpointEngine(rl.Scorer(rl.load(args.checkpoint, args.max_length, args.precision),
                                          args.max_forward_tokens, args.forward))
    if args.engine == "jev":
        return JevEngine(args.jev_url, args.jev_model, args.jev_concurrency)
    return OracleEngine() if args.engine == "oracle" else RandomEngine()


def fresh_output(output):
    path = Path(output)
    if path.exists() or path.with_suffix(".summary.json").exists():
        raise ValueError("Output or summary already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path, rows):
    with Path(path).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play")
    build = commands.add_parser("build")
    score = commands.add_parser("score")
    reporting = commands.add_parser("report")
    for command in (play, build, score, reporting):
        command.add_argument("--output", required=True)
    for command in (play, build, score):
        command.add_argument("--checkpoint", required=command is build)
        command.add_argument("--max-length", type=int, default=8192)
        command.add_argument("--max-forward-tokens", type=int, default=140000)
        command.add_argument("--forward", choices=("tree", "paths"), default="tree")
        command.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        command.add_argument("--max-concurrent", type=int, default=128)
    for command in (play, score):
        command.add_argument("--engine", choices=("checkpoint", "jev", "oracle", "random") if command is play
                             else ("checkpoint", "jev"), required=True)
        command.add_argument("--jev-url", default="http://127.0.0.1:8791/v1/systemone")
        command.add_argument("--jev-model", default="typesafe/jev-1.13")
        command.add_argument("--jev-concurrency", type=int, default=12)
    for command in (play, build):
        command.add_argument("--cases", required=True)
        command.add_argument("--splits", default="test" if command is play else "probe")
        command.add_argument("--workers", type=int, default=28)
        command.add_argument("--seed", type=int, default=17)
    play.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    play.add_argument("--annotate", choices=("none", "lookahead"), default="none")
    build.add_argument("--behaviors", default="policy_noshoot,random_noshoot")
    build.add_argument("--stride", type=int, default=3)
    for command in (score, reporting):
        command.add_argument("--probe", required=True)
    score.add_argument("--questions", default="policy,boolean")
    reporting.add_argument("--scores", nargs="+", required=True)
    args = parser.parse_args(argv)
    for name in ("workers", "max_concurrent", "stride", "jev_concurrency", "max_length", "max_forward_tokens"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "play" and args.engine == "oracle" and args.annotate != "lookahead":
        parser.error("oracle requires --annotate lookahead")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    if args.command == "report":
        named = {}
        for spec in args.scores:
            name, path = spec.split("=", 1)
            if not name or name in named:
                raise ValueError("Score names must be nonempty and unique")
            named[name] = read_rows(path)
        result = report(read_rows(args.probe), named)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        rl.dump(args.output, result)
    elif args.command == "score":
        result = score_probes(make_engine(args), read_rows(args.probe), args.output,
                              args.questions.split(","), args.max_concurrent)
        result.update(engine=args.engine, questions=args.questions.split(","), wall_seconds=time.monotonic() - started)
        result["config"] = vars(args)
        rl.dump(Path(args.output).with_suffix(".summary.json"), result)
    else:
        output = fresh_output(args.output)
        cases = rl.select_cases(args.cases, args.splits)
        if args.command == "play":
            engine = make_engine(args)
            # A uniform random controller samples even when model evaluation is greedy.
            effective_mode = "sample" if args.engine == "random" else args.mode
            with EnvPool(args.workers) as pool:
                episodes = rl.eval_episodes(PolicyScorer(engine), pool,
                                             [dict(key=f"{c['id']}#r0", case=c) for c in cases],
                                             effective_mode, args.max_concurrent, seed=args.seed,
                                             decision_hook=lookahead_hook if args.annotate == "lookahead" else None)
            write_rows(output, episodes)
            result = dict(rl.summarize(episodes), engine=args.engine, annotate=args.annotate,
                          mode=args.mode, effective_mode=effective_mode, **engine.summary(),
                          decisions_with_kill_available=sum(any(d.get("lookahead", {}).values())
                                                           for ep in episodes for d in ep["decisions"]))
        else:
            behaviors = args.behaviors.split(",")
            scorer = None
            if "policy_noshoot" in behaviors:
                args.engine = "checkpoint"
                scorer = PolicyScorer(make_engine(args))
            with EnvPool(args.workers) as pool:
                rows = build_probes(pool, cases, behaviors, scorer, args.stride, args.seed, args.max_concurrent)
            write_rows(output, rows)
            result = dict(rows=len(rows), episodes=len(cases) * len(behaviors), stride=args.stride,
                          behaviors={b: dict(n=sum(r["behavior"] == b for r in rows),
                                             hits=sum(r["hit_now"] for r in rows if r["behavior"] == b),
                                             hit_rate=mean([r["hit_now"] for r in rows if r["behavior"] == b]))
                                     for b in behaviors})
        result.update(seed=args.seed, cases=str(Path(args.cases).resolve()), splits=args.splits,
                      max_concurrent=args.max_concurrent, wall_seconds=time.monotonic() - started)
        result["config"] = vars(args)
        rl.dump(output.with_suffix(".summary.json"), result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
