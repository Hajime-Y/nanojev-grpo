#!/usr/bin/env python3
"""Local Predict Position evaluation, GRPO, and supervised continuation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

from pp_envpool import EnvPool
from predict_toy_decisions import DecisionPredictor, prepare_examples
from unified_game_pipeline import POLICY_QUESTION, behavior_distribution, choose, policy_request, read_rows


def stable_seed(value):
    return int(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest(), 16)


def load(checkpoint_dir, max_length=8192, precision="bf16"):
    return DecisionPredictor(checkpoint_dir, max_length=max_length, precision=precision,
                             disable_native_triton=True)


def pack(examples, budget):
    if budget <= 0:
        raise ValueError("Token budget must be positive")
    sizes = [(len(ex["leaf_tokens"]), max(map(len, ex["leaf_tokens"]))) for ex in examples]
    batches, batch, paths, longest = [], [], 0, 0
    for index in sorted(range(len(examples)), key=lambda i: sizes[i][1]):
        count, length = sizes[index]
        if count * length > budget:
            raise ValueError(f"Question {examples[index]['id']} needs {count * length} padded tokens, budget {budget}")
        if batch and (paths + count) * max(longest, length) > budget:
            batches.append(batch)
            batch, paths, longest = [], 0, 0
        batch.append(index)
        paths += count
        longest = max(longest, length)
    if batch:
        batches.append(batch)
    return batches


def token_counts(examples, batches):
    return {"forward_calls": len(batches), "questions": len(examples),
            "leaf_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
            "tokens": sum(len(leaf) for ex in examples for leaf in ex["leaf_tokens"]),
            "padded_tokens": sum(sum(len(examples[i]["leaf_tokens"]) for i in batch) *
                                 max(len(leaf) for i in batch for leaf in examples[i]["leaf_tokens"])
                                 for batch in batches)}


class Scorer:
    def __init__(self, predictor, max_forward_tokens=262144):
        self.model, self.tokenizer = predictor.model, predictor.tokenizer
        self.max_length, self.precision = predictor.limit, predictor.precision
        self.max_forward_tokens = max_forward_tokens
        self.tokens = Counter()

    def __call__(self, requests):
        import torch
        if not requests:
            return []
        examples = prepare_examples({"states": requests}, self.tokenizer, self.max_length)
        if len(examples) != len(requests) or any(ex["qid"] != "action" for ex in examples):
            raise ValueError("Scorer requires one action question per request")
        batches = pack(examples, self.max_forward_tokens)
        self.tokens.update(token_counts(examples, batches))
        results = [None] * len(examples)
        self.model.eval()
        with torch.inference_mode():
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                    logits, _ = self.model([examples[i] for i in batch], self.tokenizer.pad_token_id)
                for index, values in zip(batch, logits):
                    ids = examples[index]["candidate_ids"]
                    values = values[:len(ids)].float().log_softmax(-1)
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite policy log probabilities")
                    results[index] = (ids, values.cpu().tolist())
        return results


def ammo(obs):
    value = float(json.loads(obs["state"])["observed_history"][-1]["ammo"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid observed ammo")
    return value


def action_distribution(ids, logps, mode, temperature, epsilon):
    if mode not in {"greedy", "sample"} or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid mode or temperature")
    if not ids or len(ids) != len(logps) or len(set(ids)) != len(ids) or not all(map(math.isfinite, logps)):
        raise ValueError("Invalid scorer output")
    maximum = max(logps)
    scores = {key: math.exp((value - maximum) / temperature) for key, value in zip(ids, logps)}
    return behavior_distribution(scores, mode, epsilon)


def run_episodes(scorer, pool, jobs, mode, temperature=1.0, epsilon=0.0, seed=17, progress=None):
    action_distribution(["check"], [0.0], mode, temperature, epsilon)
    if len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate episode keys")
    if progress:
        progress({"wave": 0, "operation": "reset", "live": len(jobs)})
    current = pool.reset(jobs)
    episodes = {job["key"]: dict(key=job["key"], case=job["case"], success=False, outcome=None,
                                decisions=[], model_decisions=0, env_steps=0, fired=False, fire_decision=None)
                for job in jobs}
    rngs = {key: random.Random(stable_seed([key, seed])) for key in episodes}
    live, wave = set(episodes), 0

    def accept(transitions, drained=False):
        for key, transition in transitions.items():
            if transition["truncated"]:
                raise RuntimeError(f"Truncated episode: {key}")
            episode = episodes[key]
            episode["env_steps"] += transition["drain_steps"] if drained else 1
            episode["fired"] |= ammo(transition["observation"]) == 0
            current[key] = transition
            if transition["terminated"]:
                episode.update(success=bool(transition["info"]["success"]),
                               outcome=transition["info"]["terminal_reason"])
                live.remove(key)

    while live:
        wave += 1
        drained = [key for key in sorted(live) if ammo(current[key]["observation"]) == 0]
        if drained:
            for key in drained:
                episodes[key]["fired"] = True
            if progress:
                progress({"wave": wave, "operation": "drain", "live": len(live), "episodes": len(drained)})
            accept(pool.drain(drained), drained=True)
        keys = sorted(live)
        if not keys:
            break
        if progress:
            progress({"wave": wave, "operation": "score", "live": len(live)})
        requests = [policy_request(current[key]["observation"], key) for key in keys]
        outputs = scorer(requests)
        if len(outputs) != len(keys):
            raise ValueError("Scorer result count mismatch")
        actions = {}
        for key, (ids, logps) in zip(keys, outputs):
            obs, episode = current[key]["observation"], episodes[key]
            if set(ids) != set(obs["candidates"]):
                raise ValueError(f"Scorer candidates mismatch: {key}")
            behavior = action_distribution(ids, logps, mode, temperature, epsilon)
            action = actions[key] = choose(behavior, rngs[key])
            if action == "shoot" and episode["fire_decision"] is None:
                episode["fire_decision"] = len(episode["decisions"])
            episode["decisions"].append(dict(t=obs["step"], state=obs["state"], candidates=obs["candidates"],
                                             action=action, logp=logps[ids.index(action)],
                                             probs=dict(zip(ids, map(math.exp, logps)))))
            episode["model_decisions"] += 1
        if progress:
            progress({"wave": wave, "operation": "step", "live": len(live)})
        accept(pool.step(actions))
    return [episodes[job["key"]] for job in jobs]


def group_advantages(rewards):
    if not rewards or min(rewards) == max(rewards):
        return None
    mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
    return [(reward - mean) / (std + 1e-8) for reward in rewards]


def clipped_loss(new_logp, old_logp, advantages, clip):
    import torch
    ratio = (new_logp - old_logp).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages), ratio


def schedule(cases, seed, step, size):
    if not cases or step < 1 or size < 1:
        raise ValueError("Schedule requires cases, positive step and size")
    blocks = (len(cases) + size - 1) // size
    epoch, block = divmod(step - 1, blocks)
    indices = list(range(len(cases)))
    random.Random(seed + epoch).shuffle(indices)
    return [cases[i] for i in indices[block * size:(block + 1) * size]]


def row_schedule(cases, seed, step, size):
    if not cases or step < 1 or size < 1:
        raise ValueError("Schedule requires rows, positive step and size")
    offset, result = (step - 1) * size, []
    while len(result) < size:
        epoch, start = divmod(offset, len(cases))
        indices = list(range(len(cases)))
        random.Random(seed + epoch).shuffle(indices)
        take = min(size - len(result), len(cases) - start)
        result.extend(cases[i] for i in indices[start:start + take])
        offset += take
    return result


def split_minibatches(items, count):
    if count < 1:
        raise ValueError("minibatches must be positive")
    count = min(count, len(items))
    if not count:
        return []
    quotient, extra = divmod(len(items), count)
    sizes = [quotient + (i < extra) for i in range(count)]
    result, start = [], 0
    for size in sizes:
        result.append(items[start:start + size])
        start += size
    return result


def sft_rows(path):
    result = []
    for row in read_rows(path):
        metadata = row.get("metadata", {})
        if metadata.get("record_role") != "policy" or metadata.get("spec", {}).get("scenario") != "predict_position":
            continue
        questions = row["questions"]
        if set(questions) != {"action"} or questions["action"]["type"] != "choice":
            raise ValueError(f"Invalid PP policy question: {row['id']}")
        question = questions["action"]
        if question["instructions"] != POLICY_QUESTION:
            raise ValueError(f"Policy instructions differ: {row['id']}")
        action = row["gold"]["action"]
        if action not in question["criteria"]:
            raise ValueError(f"Gold action missing from criteria: {row['id']}")
        result.append({"request": {"id": row["id"], "state": row["state"], "questions": questions}, "action": action})
    if not result:
        raise ValueError(f"No PP policy rows in {path}")
    return result


def select_cases(path, splits):
    rows = [row for row in read_rows(path) if splits == "all" or row["split"] in splits.split(",")]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Cases must be nonempty with unique ids")
    for row in rows:
        if row["spec"]["task"] != "shooting" or row["spec"]["scenario"] != "predict_position":
            raise ValueError(f"Not a Predict Position case: {row['id']}")
    return rows


def summarize(episodes):
    if not episodes:
        raise ValueError("No episodes to summarize")
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    successes = sum(episode["success"] for episode in episodes)
    return dict(episodes=len(episodes), successes=successes, success_rate=successes / len(episodes),
                fired=sum(episode["fired"] for episode in episodes),
                mean_model_decisions=statistics.mean(episode["model_decisions"] for episode in episodes),
                outcomes=dict(Counter(episode["outcome"] for episode in episodes)),
                actions=dict(Counter(decision["action"] for decision in decisions)),
                successful_case_ids=sorted({episode["case"]["id"] for episode in episodes if episode["success"]}))


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def heartbeat(out, step, phase, detail):
    value = dict(time=time.time(), step=step, phase=phase, detail=detail)
    dump(Path(out) / "heartbeat.json", value)
    print(json.dumps(value, allow_nan=False), flush=True)


def run_eval(args):
    output = Path(args.output)
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise ValueError("Evaluation output or summary already exists")
    cases = select_cases(args.cases, args.splits)
    started = time.monotonic()
    scorer = Scorer(load(args.checkpoint, args.max_length, args.precision), args.max_forward_tokens)
    jobs = [{"key": f"{case['id']}#r{j}", "case": case} for case in cases for j in range(args.rollouts)]
    with EnvPool(args.workers) as pool:
        episodes = run_episodes(scorer, pool, jobs, args.mode, args.temperature, args.epsilon, args.seed)
    summary = dict(summarize(episodes), wall_seconds=time.monotonic() - started,
                   checkpoint=str(Path(args.checkpoint).resolve()), mode=args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for episode in episodes:
            if not args.save_decisions:
                episode["decisions"] = [{k: v for k, v in decision.items() if k not in {"state", "candidates"}}
                                        for decision in episode["decisions"]]
            handle.write(json.dumps(episode, allow_nan=False) + "\n")
    dump(summary_path, summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def make_cases(args):
    excluded = read_rows(args.exclude)
    seeds = set(range(args.seed_start, args.seed_start + args.count))
    overlap = seeds & {row["seed"] for row in excluded}
    if overlap:
        raise ValueError(f"Seeds overlap excluded cases: {sorted(overlap)}")
    if min(seeds) < 0 or max(seeds) >= 2**32:
        raise ValueError("Case seeds must be in [0, 2**32)")
    specs = {json.dumps(row["spec"], sort_keys=True) for row in excluded
             if row["split"] == "train" and row["spec"].get("scenario") == "predict_position"}
    if len(specs) != 1:
        raise ValueError("Excluded file must contain exactly one PP train spec")
    spec = json.loads(specs.pop())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for seed in sorted(seeds):
            row = dict(id=f"{args.split}-sonic_predict_position-{seed}", seed=seed, spec=spec,
                       split=args.split, variant="doom_predict_position")
            handle.write(json.dumps(row) + "\n")


def saved_args(args):
    return {key: value for key, value in vars(args).items() if key != "resume"}


def resume_state(out, args):
    last = Path(out) / "ckpt" / "last"
    backup = last.with_name("last.old")
    if not last.exists() and backup.exists():
        backup.replace(last)
    state = json.loads((last / "state.json").read_text())
    allowed = "steps" if args.command == "grpo" else "updates"
    previous, current = state["args"], saved_args(args)
    changes = sorted(key for key in previous.keys() | current.keys()
                     if key != allowed and previous.get(key) != current.get(key))
    if changes:
        raise ValueError(f"Resume arguments differ: {changes}")
    if current[allowed] < state["completed_step"]:
        raise ValueError("Requested run ends before the completed step")
    return last, state


def checkpoint(destination, source, state, save_weights, save_optimizer=None):
    destination, source = Path(destination), Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    backup = destination.with_name(destination.name + ".old")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    shutil.copy2(source / "config.json", temp / "config.json")
    for name in ("tokenizer", "backbone_config"):
        shutil.copytree(source / name, temp / name)
    save_weights(temp / "best.safetensors")
    if save_optimizer:
        save_optimizer(temp / "optimizer.pt")
    dump(temp / "state.json", state)
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    temp.replace(destination)
    if backup.exists():
        shutil.rmtree(backup)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate_logs(out, completed):
    for name, field in (("steps.jsonl", "step"), ("updates.jsonl", "update"), ("eval.jsonl", "step")):
        path = out / name
        if path.exists():
            rows = [row for row in read_rows(path) if row[field] <= completed]
            temp = path.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
            temp.replace(path)


def prepare_items(items, tokenizer, max_length):
    examples = []
    for item in items:
        example, = prepare_examples({"states": [item["request"]]}, tokenizer, max_length)
        examples.append(dict(example, target=example["candidate_ids"].index(item["action"]),
                             old_logp=item.get("old_logp"), advantage=item.get("advantage")))
    return examples


def update_model(scorer, optimizer, examples, args, step, grpo):
    import torch
    parts = split_minibatches(examples, args.minibatches if grpo else 1)
    metrics = ["mean_loss"] + (["mean_abs_ratio_minus_one", "clip_fraction", "mean_old_minus_new_logp"] if grpo else [])
    totals, tokens, norms = Counter(dict.fromkeys(metrics, 0)), Counter(token_counts([], [])), []
    scorer.model.train()
    for index, part in enumerate(parts):
        heartbeat(args.output_dir, step, "update", {"index": index + 1, "total": len(parts)})
        batches = pack(part, args.max_microbatch_tokens)
        tokens.update(token_counts(part, batches))
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            group = [part[i] for i in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                logits, _ = scorer.model(group, scorer.tokenizer.pad_token_id)
            logps = torch.stack([values[:len(ex["candidate_ids"])].float().log_softmax(-1)[ex["target"]]
                                for values, ex in zip(logits, group)])
            if grpo:
                old = logps.new_tensor([ex["old_logp"] for ex in group])
                advantages = logps.new_tensor([ex["advantage"] for ex in group])
                losses, ratio = clipped_loss(logps, old, advantages, args.clip)
                totals["mean_abs_ratio_minus_one"] += float((ratio.detach() - 1).abs().sum())
                totals["clip_fraction"] += float(((ratio.detach() < 1 - args.clip) |
                                                 (ratio.detach() > 1 + args.clip)).float().sum())
                totals["mean_old_minus_new_logp"] += float((old - logps.detach()).sum())
            else:
                losses = -logps
            loss = losses.sum() / len(part)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            totals["mean_loss"] += float(losses.detach().sum())
        norm = torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), args.grad_clip, error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(skipped=not examples, **{key: value / len(examples) if examples else None for key, value in totals.items()},
                gradient_norm_before_clip=norms, tokens=dict(tokens))


def supervised_eval(scorer, items):
    total, correct, count = 0.0, 0, 0
    # Bound host tokenization memory as well as device forward memory.
    for offset in range(0, len(items), 256):
        chunk = items[offset:offset + 256]
        outputs = scorer([item["request"] for item in chunk])
        for item, (ids, logps) in zip(chunk, outputs):
            total -= logps[ids.index(item["action"])]
            best = min(ids, key=lambda key: (-logps[ids.index(key)], key))
            correct += best == item["action"]
            count += 1
    return {"ce": total / count, "agreement": correct / count, "questions": count}


def game_eval(scorer, pool, cases, args, step, dev_rows=None):
    path = Path(args.output_dir) / "eval.jsonl"
    done = {(row["step"], row["mode"]) for row in read_rows(path)} if path.exists() else set()
    modes = [mode for mode in ("greedy", "sample") if (step, mode) not in done]
    ce = None
    if modes and dev_rows is not None:
        heartbeat(args.output_dir, step, "eval", {"mode": "supervised"})
        ce = supervised_eval(scorer, dev_rows)
    for mode in modes:
        heartbeat(args.output_dir, step, "eval", {"mode": mode})
        jobs = [{"key": f"{case['id']}#eval-{mode}", "case": case} for case in cases]
        progress = lambda detail: heartbeat(args.output_dir, step, "eval", {"mode": mode, **detail})
        episodes = run_episodes(scorer, pool, jobs, mode, seed=args.seed, progress=progress)
        record = dict(step=step, mode=mode, **summarize(episodes))
        if ce is not None:
            record["supervised"] = ce
        append(path, record)


def grpo_collection(scorer, pool, cases, args, step):
    chosen = schedule(cases, args.seed, step, args.tasks_per_step)
    jobs = [{"key": f"{case['id']}#s{step}r{j}", "case": case}
            for case in chosen for j in range(args.group_size)]
    progress = lambda detail: heartbeat(args.output_dir, step, "collect", detail)
    episodes = run_episodes(scorer, pool, jobs, "sample", args.temperature, seed=(args.seed, step), progress=progress)
    items, success_counts, dropped = [], [], 0
    for offset in range(0, len(episodes), args.group_size):
        group = episodes[offset:offset + args.group_size]
        rewards = [float(episode["success"]) for episode in group]
        success_counts.append(sum(rewards))
        advantages = group_advantages(rewards)
        if advantages is None:
            dropped += 1
            continue
        for episode, advantage in zip(group, advantages):
            for decision in episode["decisions"]:
                items.append(dict(request=policy_request(decision, episode["key"]), action=decision["action"],
                                  old_logp=decision["logp"], advantage=advantage))
    random.Random(stable_seed([args.seed, step])).shuffle(items)
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    summary = summarize(episodes)
    record = dict(case_ids=[case["id"] for case in chosen], group_success_counts=success_counts,
                  sampled_success_rate=summary["success_rate"], kept_groups=len(chosen) - dropped,
                  dropped_groups=dropped, episodes=len(episodes), decisions_collected=len(decisions),
                  decisions_used=len(items), mean_model_decisions=summary["mean_model_decisions"],
                  fired_fraction=summary["fired"] / len(episodes), action_counts=summary["actions"],
                  mean_chosen_action_probability=statistics.mean(math.exp(d["logp"]) for d in decisions),
                  mean_policy_entropy=statistics.mean(-sum(p * math.log(p) for p in d["probs"].values() if p > 0)
                                                      for d in decisions))
    return items, record


def train(args):
    out = Path(args.output_dir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory or --resume")
    initial_hash = file_sha256(Path(args.init_checkpoint) / "best.safetensors")
    last, completed = None, 0
    if args.resume:
        last, state = resume_state(out, args)
        completed = state["completed_step"]
        if json.loads((out / "config.json").read_text())["init_weights_sha256"] != initial_hash:
            raise ValueError("Initialization checkpoint changed")
    grpo = args.command == "grpo"
    train_cases = select_cases(args.cases, args.train_split) if grpo else None
    eval_cases = select_cases(args.cases, args.eval_split)
    train_rows = sft_rows(Path(args.data) / "train.jsonl") if not grpo else None
    dev_rows = sft_rows(Path(args.data) / "dev.jsonl") if not grpo else None
    import torch
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    heartbeat(out, completed, "load", {})
    torch.manual_seed(args.seed)
    runtime = load(last or args.init_checkpoint, args.max_length, args.precision)
    scorer, model = Scorer(runtime, args.max_forward_tokens), runtime.model
    if not args.no_gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": args.backbone_lr},
        {"params": [param for name, param in model.named_parameters() if not name.startswith("backbone.")],
         "lr": args.head_lr}], weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(torch.load(last / "optimizer.pt", map_location="cpu", weights_only=True))
        truncate_logs(out, completed)
    else:
        try:
            git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            git_head = git.stdout.strip() if git.returncode == 0 else None
        except FileNotFoundError:
            git_head = None
        dump(out / "config.json", dict(args=vars(args), init_weights_sha256=initial_hash,
                                        git_head=git_head,
                                        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    total = args.steps if grpo else args.updates
    prefix = "step" if grpo else "update"

    def save(step, snapshot=False):
        heartbeat(out, step, "save", {"snapshot": snapshot})
        state = dict(completed_step=step, args=saved_args(args))
        checkpoint(out / "ckpt" / "last", args.init_checkpoint, state,
                   lambda path: save_file({key: value.detach().cpu().contiguous().clone()
                                           for key, value in model.state_dict().items()}, str(path)),
                   lambda path: torch.save(optimizer.state_dict(), path))
        if snapshot:
            source = out / "ckpt" / "last"
            checkpoint(out / "ckpt" / f"{prefix}-{step:04d}", source, state,
                       lambda path: shutil.copy2(source / "best.safetensors", path))

    with EnvPool(args.workers) as pool:
        if completed == 0 or completed % args.eval_every == 0 or completed == total:
            game_eval(scorer, pool, eval_cases, args, completed, dev_rows)
        if not args.resume:
            save(0)
        elif completed and (completed % args.save_every == 0 or completed == total):
            if not (out / "ckpt" / f"{prefix}-{completed:04d}").exists():
                save(completed, snapshot=True)
        for step in range(completed + 1, total + 1):
            torch.cuda.reset_peak_memory_stats()
            scorer.tokens.clear()
            torch.manual_seed(stable_seed([args.seed, step]) % (2**63))
            started = time.monotonic()
            heartbeat(out, step, "collect" if grpo else "prepare", {})
            if grpo:
                items, record = grpo_collection(scorer, pool, train_cases, args, step)
            else:
                items = row_schedule(train_rows, args.seed, step, args.batch_questions)
                record = dict(update=step, questions_seen=step * args.batch_questions,
                              epoch=((step * args.batch_questions) - 1) // len(train_rows))
            collect_seconds = time.monotonic() - started
            collect_tokens = dict(scorer.tokens)
            torch.cuda.empty_cache()
            update_started = time.monotonic()
            heartbeat(out, step, "update", {"index": 0, "questions": len(items)})
            examples = prepare_items(items, scorer.tokenizer, args.max_length)
            stats = update_model(scorer, optimizer, examples, args, step, grpo)
            torch.cuda.synchronize()
            record.update(update_stats=stats, collect_seconds=collect_seconds,
                          update_seconds=time.monotonic() - update_started,
                          tokens={"collect": collect_tokens, "update": stats["tokens"]},
                          max_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
            if grpo:
                record["step"] = step
            else:
                record.update(loss=stats["mean_loss"], grad_norm=stats["gradient_norm_before_clip"][0],
                              elapsed_seconds=time.monotonic() - started)
            append(out / ("steps.jsonl" if grpo else "updates.jsonl"), record)
            print(json.dumps(record, allow_nan=False), flush=True)
            del examples, items
            if step % args.eval_every == 0 or step == total:
                game_eval(scorer, pool, eval_cases, args, step, dev_rows)
            save(step, snapshot=step % args.save_every == 0 or step == total)
        heartbeat(out, total, "complete", {})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cases = commands.add_parser("cases")
    cases.add_argument("--output", required=True)
    cases.add_argument("--count", type=int, default=512)
    cases.add_argument("--seed-start", type=int, default=9400000)
    cases.add_argument("--split", default="heldout")
    cases.add_argument("--exclude", default="configs/sonic_predict_supervision_v1_cases.jsonl")
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--splits", default="test")
    evaluation.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    evaluation.add_argument("--temperature", type=float, default=1.0)
    evaluation.add_argument("--epsilon", type=float, default=0.0)
    evaluation.add_argument("--rollouts", type=int, default=1)
    evaluation.add_argument("--save-decisions", action="store_true")
    grpo = commands.add_parser("grpo")
    grpo.add_argument("--steps", type=int, default=50)
    grpo.add_argument("--train-split", default="train")
    grpo.add_argument("--tasks-per-step", type=int, default=16)
    grpo.add_argument("--group-size", type=int, default=16)
    grpo.add_argument("--minibatches", type=int, default=8)
    grpo.add_argument("--clip", type=float, default=.2)
    grpo.add_argument("--temperature", type=float, default=1.0)
    sft = commands.add_parser("sft")
    sft.add_argument("--data", default="data/NanoJev-unified/unified/hard")
    sft.add_argument("--updates", type=int, default=2400)
    sft.add_argument("--batch-questions", type=int, default=64)
    for command in (grpo, sft):
        command.add_argument("--init-checkpoint", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--eval-split", default="dev")
        command.add_argument("--eval-every", type=int, default=5 if command is grpo else 200)
        command.add_argument("--save-every", type=int, default=10 if command is grpo else 200)
        command.add_argument("--backbone-lr", type=float, default=1e-5)
        command.add_argument("--head-lr", type=float, default=1e-4)
        command.add_argument("--weight-decay", type=float, default=.01)
        command.add_argument("--grad-clip", type=float, default=1.0)
        command.add_argument("--max-microbatch-tokens", type=int, default=32768)
        command.add_argument("--no-gradient-checkpointing", action="store_true")
        command.add_argument("--resume", action="store_true")
    for command in (evaluation, grpo, sft):
        command.add_argument("--cases", required=True)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--workers", type=int, default=28)
        command.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
        command.add_argument("--max-length", type=int, default=8192)
        command.add_argument("--max-forward-tokens", type=int, default=262144)
    args = parser.parse_args(argv)
    for name in ("count", "workers", "max_length", "max_forward_tokens", "max_microbatch_tokens", "rollouts",
                 "steps", "updates", "tasks_per_step", "group_size", "minibatches", "batch_questions", "eval_every", "save_every"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("temperature", "backbone_lr", "head_lr", "grad_clip"):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if hasattr(args, "epsilon") and not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be in [0, 1]")
    if hasattr(args, "clip") and not 0 < args.clip < 1:
        parser.error("--clip must be in (0, 1)")
    if hasattr(args, "weight_decay") and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("--weight-decay must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "cases":
        make_cases(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
