#!/usr/bin/env python3
"""CPU-only Snake lookup, one-step, and existential look-ahead measurement.

All three probes sample the same rollout pool. Items are balanced separately by
kind; the resulting subsets need not contain the same states. Only observation
text and the question are sent to a model, never gold or simulator metadata.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request

# Also support importing this module with `python -m unittest scripts/...`.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from snake_game import DIRECTIONS, REVERSE, make_snake, one_step_safe, step, valid_actions
from unified_grid_envs import UnifiedSnakeEnv


KINDS = ("lookup", "step1", "ahead")


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def summary_path(path):
    return Path(path).with_suffix(".summary.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from None


def destination(body, action):
    dr, dc = DIRECTIONS[action]
    return body[0][0] + dr, body[0][1] + dc


def lookahead_step(size, body, direction, food, action):
    """Immutable (body, direction, food) transition, or None on collision.

    Food is a coordinate or None. Eating removes it permanently. A full body is
    handled as a win by the caller. This never invokes the game's food RNG.
    """
    if action not in DIRECTIONS or action == REVERSE[direction]:
        return None
    cell = destination(body, action)
    growing = cell == food
    occupied = body if growing else body[:-1]
    if not all(0 <= x < size for x in cell) or cell in occupied:
        return None
    return ((cell,) + (body if growing else body[:-1]), action,
            None if growing else food)


def survives(state, action, depth):
    """Take action, then ask whether *depth further* non-colliding moves exist."""
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    size = state["size"]

    @lru_cache(maxsize=None)
    def dfs(body, direction, food, remaining):
        if len(body) == size * size or remaining == 0:
            return True
        for move in DIRECTIONS:
            following = lookahead_step(size, body, direction, food, move)
            if following is not None and dfs(*following, remaining - 1):
                return True
        return False

    if state.get("done"):
        return state.get("outcome") == "win"
    following = lookahead_step(size, tuple(map(tuple, state["body"])), state["direction"],
                               tuple(state["food"]) if state["food"] is not None else None, action)
    return following is not None and dfs(*following, depth)


def rollout(size, seed, policy_seed, min_length):
    """Yield eligible live snapshots, including a live snapshot after move 400."""
    rng = random.Random(f"snake-probe:{policy_seed}:{size}:{seed}")
    state = make_snake(size, seed)
    for used in range(401):
        if state["done"]:
            break
        if len(state["body"]) >= min_length:
            yield state
        if used == 400:
            break
        safe = [action for action in valid_actions(state) if one_step_safe(state, action)]
        if not safe:
            break
        distances = {action: sum(abs(a - b) for a, b in
                                 zip(destination(state["body"], action), state["food"]))
                     for action in safe}
        preferred = [action for action in safe if distances[action] == min(distances.values())]
        action = rng.choice(safe if rng.random() < 0.15 else preferred)
        state = step(state, action)


def candidate_labels(state, depth):
    for action in valid_actions(state):
        cell = destination(state["body"], action)
        if all(0 <= x < state["size"] for x in cell):
            yield "lookup", action, list(cell) in state["body"]
        safe = one_step_safe(state, action)
        yield "step1", action, safe
        if safe:
            yield "ahead", action, survives(state, action, depth)


def _balanced_dp(pools, count, rng):
    """Exact fallback for crowded pools where rejection sampling is expensive.

    Each state's polynomial is 1 + (#false actions)x + (#true actions)y.
    Suffix coefficients count valid item subsets, so weighted traversal samples
    uniformly even when states offer different numbers of actions per class.
    """
    groups = {}
    for gold in (False, True):
        for item in pools[gold]:
            groups.setdefault(item[0], [[], []])[gold].append(item)
    groups = list(groups.values())
    layers = [None] * (len(groups) + 1)
    layers[-1] = {(0, 0): 1}
    for i in range(len(groups) - 1, -1, -1):
        layer = defaultdict(int)
        for (f, t), ways in layers[i + 1].items():
            layer[f, t] += ways
            if f < count:
                layer[f + 1, t] += len(groups[i][0]) * ways
            if t < count:
                layer[f, t + 1] += len(groups[i][1]) * ways
        # Discard unreachable prefixes to limit memory near full selection.
        layers[i] = {key: value for key, value in layer.items()
                     if value and 2 * count - sum(key) <= i}
    selected, f, t = [], count, count
    for i, group in enumerate(groups):
        suffix = layers[i + 1]
        skip = suffix.get((f, t), 0)
        take_false = len(group[0]) * suffix.get((f - 1, t), 0) if f else 0
        take_true = len(group[1]) * suffix.get((f, t - 1), 0) if t else 0
        pick = rng.randrange(skip + take_false + take_true)
        if pick < skip:
            continue
        gold = pick >= skip + take_false
        selected.append((*rng.choice(group[gold]), gold))
        f -= not gold
        t -= gold
    return selected


def sample_balanced(pools, count, rng, kind):
    """Uniform valid balanced subsets, with at most one item per physical state."""
    states = {gold: {item[0] for item in pools[gold]} for gold in (False, True)}
    for gold in (False, True):
        if len(states[gold]) < count:
            raise ValueError(f"{kind} gold={str(gold).lower()}: need {count} distinct states, "
                             f"found {len(states[gold])}; increase --seeds or change build parameters")
    if len(states[False] | states[True]) < 2 * count:
        raise ValueError(f"{kind}: too few distinct states for both classes without reuse "
                         f"(need {2 * count}, found {len(states[False] | states[True])})")
    # Independent uniform samples conditioned on no repeated state are uniform
    # over all valid balanced subsets. Do not greedily discard collisions: that
    # would bias states with multiple candidate actions or both gold classes.
    for _ in range(256):
        selected = [(*item, gold) for gold in (False, True)
                    for item in rng.sample(pools[gold], count)]
        if len({item[0] for item in selected}) == 2 * count:
            return selected
    return _balanced_dp(pools, count, rng)


def make_item(state, state_id, kind, action, gold, depth):
    env = UnifiedSnakeEnv({"task": "snake", "size": state["size"], "max_steps": 96,
                           "initial_state": state})
    observation, _ = env.reset(state["seed"])
    env.close()
    cell = list(destination(state["body"], action))
    questions = {
        "lookup": f"Is the cell {compact(cell)} one of the cells listed in the snake's body?",
        "step1": f"If the snake moves {action} now, does it avoid a collision on this move?",
        "ahead": (f"If the snake moves {action} now, can it then make {depth} more moves without "
                  "any collision? Assume that after the current food is eaten no new food appears."),
    }
    return {"id": f"{kind}:{state_id}:{action}" + (f":d{depth}" if kind == "ahead" else ""),
            "kind": kind, "size": state["size"], "seed": state["seed"], "step": state["steps"],
            "action": action, "state": observation["state"], "question": questions[kind], "gold": gold,
            "meta": {"state_id": state_id, "body_length": len(state["body"]), "destination": cell,
                     "depth": depth if kind == "ahead" else 0,
                     "snapshot": state}}


def quantiles(values):
    values = sorted(values)
    result = {}
    for q in (0, 0.25, 0.5, 0.75, 1):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        result[str(q)] = values[lo] + (values[hi] - values[lo]) * (index - lo) if values else None
    return result


def build(output, sizes=(8, 10), seeds=400, min_length=8, depth=5, per_class=100, seed=17):
    if (not sizes or any(type(size) is not int or size < 2 for size in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("sizes must be distinct integers >= 2")
    if min(seeds, min_length, per_class) < 1 or depth < 0:
        raise ValueError("seeds, min-length and per-class must be positive; depth must be >= 0")
    snapshots = {}
    pools = {kind: {False: [], True: []} for kind in KINDS}
    for size in sizes:
        for episode_seed in range(seeds):
            for state in rollout(size, episode_seed, seed, min_length):
                physical = [size, state["body"], state["direction"], state["food"]]
                state_id = hashlib.sha256(compact(physical).encode()).hexdigest()
                if state_id in snapshots:
                    continue
                # Serialize once; only selected items need expensive env rendering.
                snapshots[state_id] = compact(state)
                for kind, action, gold in candidate_labels(state, depth):
                    pools[kind][gold].append((state_id, action))
    rng = random.Random(seed)
    items = []
    for kind in KINDS:
        for state_id, action, gold in sample_balanced(pools[kind], per_class, rng, kind):
            items.append(make_item(json.loads(snapshots[state_id]), state_id, kind, action, gold, depth))
    rng.shuffle(items)
    counts = {kind: {str(gold).lower(): sum(x["kind"] == kind and x["gold"] == gold for x in items)
                     for gold in (False, True)} for kind in KINDS}
    summary = {"counts": counts, "items": len(items), "candidate_states": len(snapshots),
               "candidate_counts": {kind: {str(gold).lower(): len(pools[kind][gold])
                                           for gold in (False, True)} for kind in KINDS},
               "body_length_quantiles": quantiles([x["meta"]["body_length"] for x in items]),
               "body_length_quantiles_by_kind": {
                   kind: quantiles([x["meta"]["body_length"] for x in items if x["kind"] == kind])
                   for kind in KINDS},
               "parameters": {"sizes": list(sizes), "seeds": seeds, "min_length": min_length,
                              "depth": depth, "per_class": per_class, "seed": seed,
                              "rollout_max_moves": 400, "random_safe_probability": 0.15,
                              "target_food": "max(1, size // 4)", "observation_max_steps": 96},
               "sampling": "Uniform balanced item subsets without physical-state reuse within each kind; physical duplicates collapsed."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(compact(item) + "\n" for item in items))
    write_json(summary_path(path), summary)
    return summary


def parse_reply(content):
    words = re.findall(r"\b(?:yes|no)\b", content.strip().lower())
    return words[-1] == "yes" if words else None


def http_post(url, body, headers):
    request = urllib.request.Request(url, data=compact(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def request_with_retries(url, body, headers, transport, sleep):
    for attempt in range(4):  # Initial attempt plus three retries.
        try:
            return transport(url, body, headers)
        except urllib.error.HTTPError as exc:
            retry = exc.code in (408, 429) or 500 <= exc.code < 600
            code = exc.code
            if exc.fp is not None:
                exc.fp.close()
            if not retry or attempt == 3:
                raise RuntimeError(f"HTTP {code}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            if attempt == 3:
                raise RuntimeError("Connection failed after 4 attempts") from None
        sleep(2 ** attempt)


def ask_item(item, engine, *, jev_url="http://127.0.0.1:8791/v1/systemone",
             jev_model="typesafe/jev-1.13", chat_url="https://openrouter.ai/api/v1/chat/completions",
             chat_model=None, reasoning="off", max_tokens=6000, transport=http_post, sleep=time.sleep):
    start = time.monotonic()
    key = None
    try:
        headers = {"Content-Type": "application/json"}
        if engine == "jev":
            url = jev_url
            body = {"model": jev_model, "state": item["state"],
                    "questions": {"q": {"type": "noul", "instructions": item["question"]}}}
        elif engine == "chat":
            if not chat_model:
                raise ValueError("chat-model is required")
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                return {"id": item["id"], "error": "OPENROUTER_API_KEY is not set"}
            url = chat_url
            headers["Authorization"] = "Bearer " + key
            body = {"model": chat_model, "messages": [{"role": "user", "content":
                    item["state"] + "\n\n" + item["question"] +
                    "\n\nAnswer with exactly one word: yes or no."}], "temperature": 0,
                    "max_tokens": max_tokens, "reasoning": {"enabled": reasoning == "on"},
                    "provider": {"require_parameters": True}}
        else:
            raise ValueError("Unknown engine")
        response = request_with_retries(url, body, headers, transport, sleep)
        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Invalid usage")
        if engine == "jev":
            probability = response["answers"]["q"]["noul"]
            if (type(probability) not in (int, float) or not math.isfinite(probability)
                    or not 0 <= probability <= 1):
                raise ValueError("Invalid probability")
            row = {"id": item["id"], "p_yes": probability, "answer": probability >= 0.5, "usage": usage}
        else:
            content = response["choices"][0]["message"].get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Invalid content")
            row = {"id": item["id"], "answer": parse_reply(content),
                   "raw": content.replace(key, "[REDACTED]")[:200],
                   "usage": usage, "seconds": time.monotonic() - start}
        # Defend against a gateway echoing credentials in content or usage. The
        # request headers and exception messages are never persisted or printed.
        def redact(value):
            if isinstance(value, str):
                return value.replace(key, "[REDACTED]")
            if isinstance(value, dict):
                return {redact(k): redact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v) for v in value]
            return value
        row = redact(row) if key else row
        compact(row)  # Malformed/nonfinite response values become per-item errors.
        return row
    except RuntimeError as exc:
        message = str(exc)
        if not re.fullmatch(r"HTTP \d{3}|Connection failed after 4 attempts", message):
            message = "Model request failed (RuntimeError)"
        return {"id": item["id"], "error": message}
    except Exception as exc:
        return {"id": item["id"], "error": f"Model request or response failed ({type(exc).__name__})"}


def answer_rows(path):
    """Last row wins, except a failed retry cannot erase an earlier success."""
    rows = {}
    if Path(path).exists():
        for row in read_jsonl(path):
            if "error" not in row or row["id"] not in rows or "error" in rows[row["id"]]:
                rows[row["id"]] = row
    return rows


def _repair_append_tail(path):
    """Drop only a torn final JSON line; reject malformed completed lines."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        offset = 0
        for line in handle:
            end = offset + len(line)
            try:
                json.loads(line)
            except ValueError:
                if not line.endswith(b"\n") and end == path.stat().st_size:
                    handle.seek(offset)
                    handle.truncate()
                    return
                raise ValueError("Invalid completed JSON line in answers file") from None
            offset = end
        if not line.endswith(b"\n"):
            handle.write(b"\n")


def usage_totals(rows):
    totals = Counter()

    def visit(value, prefix=""):
        for key, number in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(number, dict):
                visit(number, name)
            elif type(number) in (int, float) and math.isfinite(number):
                totals[name] += number

    for row in rows:
        visit(row.get("usage") or {})
    return dict(sorted(totals.items()))


def ask(probe, output, engine, concurrency=8, transport=http_post, sleep=time.sleep, **options):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if engine not in ("jev", "chat"):
        raise ValueError("engine must be jev or chat")
    if engine == "chat" and not options.get("chat_model"):
        raise ValueError("--chat-model is required for chat")
    if Path(probe).resolve() == Path(output).resolve():
        raise ValueError("Probe and answers must use different files")
    started = time.monotonic()
    items = list(read_jsonl(probe))
    ids = {item["id"] for item in items}
    if len(ids) != len(items):
        raise ValueError("Probe contains duplicate IDs")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_append_tail(path)
    existing = answer_rows(path)
    pending = [item for item in items if item["id"] not in existing or "error" in existing[item["id"]]]
    new_rows = []
    with path.open("a") as handle, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(ask_item, item, engine, transport=transport, sleep=sleep, **options)
                   for item in pending]
        for future in as_completed(futures):
            row = future.result()
            handle.write(compact(row) + "\n")
            handle.flush()
            existing[row["id"]] = row
            new_rows.append(row)
    rows = [existing[identity] for identity in ids if identity in existing]
    summary = {"engine": engine, "model": options.get("chat_model") if engine == "chat"
               else options.get("jev_model", "typesafe/jev-1.13"),
               "reasoning": options.get("reasoning", "off") if engine == "chat" else None,
               "n": len(items), "skipped": len(items) - len(pending), "attempted": len(pending),
               "completed": sum("error" not in row for row in rows),
               "answered": sum("error" not in row and type(row.get("answer")) is bool for row in rows),
               "unparseable": sum("error" not in row and row.get("answer") is None for row in rows),
               "errors": sum("error" in row for row in rows),
               "errors_this_run": sum("error" in row for row in new_rows),
               "usage_totals": usage_totals(rows), "usage_this_run": usage_totals(new_rows),
               "wall_seconds": time.monotonic() - started}
    write_json(summary_path(output), summary)
    return summary


def rank_auc(pairs):
    """Mann–Whitney rank AUC, giving tied scores half credit."""
    pairs = sorted(pairs)
    positives = sum(gold for _, gold in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        rank_sum += ((index + 1 + end) / 2) * sum(gold for _, gold in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(items, answers):
    n = len(items)
    correct, counts = Counter(), Counter()
    pairs = []
    for item in items:
        gold = item["gold"]
        counts[gold] += 1
        row = answers.get(item["id"], {})
        answer = row.get("answer") if "error" not in row else None
        if type(answer) is bool:
            counts["answered"] += 1
            counts["yes"] += answer
            correct[gold] += answer == gold
        if "error" in row:
            counts["errors"] += 1
        elif row and answer is None:
            counts["unparseable"] += 1
        probability = row.get("p_yes")
        if ("error" not in row and type(probability) in (int, float)
                and math.isfinite(probability) and 0 <= probability <= 1):
            pairs.append((probability, gold))
    return {"n": n, "answered": counts["answered"],
            "accuracy": sum(correct.values()) / n if n else None,
            "accuracy_gold_true": correct[True] / counts[True] if counts[True] else None,
            "accuracy_gold_false": correct[False] / counts[False] if counts[False] else None,
            "share_yes": counts["yes"] / counts["answered"] if counts["answered"] else None,
            "errors": counts["errors"], "unparseable": counts["unparseable"],
            "missing": sum(item["id"] not in answers for item in items),
            "auc": rank_auc(pairs), "auc_n": len(pairs)}


def report(probe, answers, output):
    items = list(read_jsonl(probe))
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Probe contains duplicate IDs")
    result = {}
    for name, path in answers:
        if not name or name in result:
            raise ValueError("Answer names must be nonempty and unique")
        if not Path(path).is_file():
            raise ValueError(f"Answers file does not exist: {path}")
        rows = answer_rows(path)
        result[name] = {kind: metrics([item for item in items if item["kind"] == kind], rows)
                        for kind in KINDS}
    write_json(output, result)
    print(f"{'name':<20} {'kind':<7} {'n':>6} {'ans':>6} {'acc':>7} {'true':>7} {'false':>7} {'yes':>7} {'AUC':>7}")
    def number(value):
        return "-" if value is None else f"{value:.3f}"
    for name, by_kind in result.items():
        for kind, row in by_kind.items():
            values = " ".join(f"{number(row[key]):>7}" for key in
                              ("accuracy", "accuracy_gold_true", "accuracy_gold_false", "share_yes", "auc"))
            print(f"{name:<20} {kind:<7} {row['n']:>6} {row['answered']:>6} {values}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Generate balanced offline probes")
    builder.add_argument("--output", required=True)
    builder.add_argument("--sizes", default="8,10")
    builder.add_argument("--seeds", type=int, default=400)
    builder.add_argument("--min-length", type=int, default=8)
    builder.add_argument("--depth", type=int, default=5)
    builder.add_argument("--per-class", type=int, default=100)
    builder.add_argument("--seed", type=int, default=17)
    asker = commands.add_parser("ask", help="Query a hosted model (resumes automatically)")
    asker.add_argument("--probe", required=True)
    asker.add_argument("--output", required=True)
    asker.add_argument("--engine", choices=("jev", "chat"), required=True)
    asker.add_argument("--jev-url", default="http://127.0.0.1:8791/v1/systemone")
    asker.add_argument("--jev-model", default="typesafe/jev-1.13")
    asker.add_argument("--chat-url", default="https://openrouter.ai/api/v1/chat/completions")
    asker.add_argument("--chat-model")
    asker.add_argument("--reasoning", choices=("on", "off"), default="off")
    asker.add_argument("--max-tokens", type=int, default=6000)
    asker.add_argument("--concurrency", type=int, default=8)
    reporter = commands.add_parser("report", help="Compare model answers")
    reporter.add_argument("--probe", required=True)
    reporter.add_argument("--answers", nargs="+", action="extend", required=True, metavar="NAME=PATH")
    reporter.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "build":
            args["sizes"] = [int(value) for value in args["sizes"].split(",")]
            result = build(**args)
        elif command == "ask":
            if args["max_tokens"] < 1:
                raise ValueError("max-tokens must be positive")
            result = ask(**args)
        else:
            if any("=" not in value for value in args["answers"]):
                raise ValueError("--answers entries must be NAME=PATH")
            args["answers"] = [value.split("=", 1) for value in args["answers"]]
            report(**args)
            return
        print(json.dumps(result, sort_keys=True))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
