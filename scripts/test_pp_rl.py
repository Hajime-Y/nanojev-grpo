"""CPU-only contract tests; CUDA and ViZDoom are not needed."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pp_rl as rl
from pp_envpool import EnvPool
from predict_toy_decisions import local_checkpoint_files, prepare_examples
from unified_game_pipeline import POLICY_QUESTION, behavior_distribution, policy_request

HAS_TORCH = importlib.util.find_spec("torch") is not None
SPEC = dict(frame_skip=4, history_length=4, max_steps=8, scenario="predict_position", task="shooting")
ACTIONS = dict(left="Turn left.", right="Turn right.", shoot="Hold attack.", noop="Wait.")


class FakeEnv:
    instances = 0

    def __init__(self, spec):
        FakeEnv.instances += 1
        self.instance = [os.getpid(), FakeEnv.instances]
        self.spec, self.resets = spec, 0

    def reset(self, seed):
        if seed == -1:
            raise ValueError("fake reset failure")
        self.seed, self.t, self.shot, self.hit = seed, 0, None, False
        self.resets += 1
        return self.observation(), self.info(False)

    def observation(self):
        ammo = float(self.shot is None or self.t < self.shot + 2)
        return dict(task="shooting", step=self.t, remaining_steps=self.spec["max_steps"] - self.t,
                    state=json.dumps({"observed_history": [{"ammo": ammo}], "seed": self.seed, "t": self.t}),
                    candidates={} if self.done() else dict(ACTIONS))

    def done(self):
        return self.t >= self.spec["max_steps"] or (self.shot is not None and self.t >= self.shot + 4)

    def info(self, done):
        return dict(success=done and self.hit, terminal_reason=("hit" if self.hit else "miss") if done else None,
                    instance=self.instance, resets=self.resets)

    def step(self, action):
        if action == "explode":
            raise ValueError("fake step failure")
        if action == "die":
            os._exit(9)
        if action == "sleep":
            time.sleep(10)
        if self.done():
            raise ValueError("step after terminal")
        if action == "shoot" and self.shot is None:
            self.shot, self.hit = self.t, self.t % 3 == self.seed % 3
        self.t += 1
        return self.observation(), 0.0, self.done(), action == "truncate", self.info(self.done())

    def close(self):
        pass


class FakeScorer:
    def __init__(self, probabilities=None):
        self.probabilities = probabilities or dict(left=.05, right=.05, shoot=.85, noop=.05)
        self.requests = []

    def __call__(self, requests):
        self.requests.extend(copy.deepcopy(requests))
        results = []
        for request in requests:
            state = json.loads(request["state"])
            if state["observed_history"][-1]["ammo"] == 0:
                raise AssertionError("Model asked after ammo reached zero")
            ids = list(request["questions"]["action"]["criteria"])
            results.append((ids, [math.log(self.probabilities[key]) for key in ids]))
        return results


class FakeTokenizer:
    eos_token_id, pad_token_id = 256, 257

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def case(seed=0, identity=None, split="train"):
    return dict(id=identity or f"case-{seed}", seed=seed, spec=dict(SPEC), split=split, variant="sonic_predict_position")


def jobs(count=4):
    return [{"key": f"ep-{i}", "case": case(i)} for i in range(count)]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def policy_row(identity="row-0"):
    return dict(id=identity, state='{"observed_history":[{"ammo":1.0}]}',
                questions={"action": dict(type="choice", instructions=POLICY_QUESTION, criteria=dict(ACTIONS))},
                gold={"action": "shoot"}, metadata=dict(record_role="policy", spec=dict(SPEC)))


def tiny_model():
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 4, bias=False)
            self.head = torch.nn.Parameter(torch.tensor([.1, -.2, .3, -.4]))
            self.batches = []

        def forward(self, examples, pad_token):
            self.batches.append([ex["id"] for ex in examples])
            features = torch.tensor([[len(ex["leaf_tokens"][0]) / 10] for ex in examples])
            logits = self.backbone(features) + self.head
            return logits, torch.ones_like(logits, dtype=torch.bool)

    torch.manual_seed(7)
    return TinyModel()


class PoolTests(unittest.TestCase):
    def test_workers_step_drain_and_reuse(self):
        work = jobs(6)
        with EnvPool(3, factory=FakeEnv) as pool:
            initial = pool.reset(work)
            self.assertEqual(len({row["info"]["instance"][0] for row in initial.values()}), 3)
            first = pool.step(dict.fromkeys(initial, "shoot"))
            self.assertTrue(all(not row["terminated"] for row in first.values()))
            final = pool.drain(initial)
            self.assertTrue(all(row["terminated"] and row["drain_steps"] == 3 for row in final.values()))
            self.assertEqual([final[f"ep-{i}"]["info"]["success"] for i in range(6)],
                             [True, False, False, True, False, False])
            second = pool.reset(work)
            self.assertEqual([row["observation"] for row in initial.values()],
                             [second[key]["observation"] for key in initial])
            self.assertTrue(all(row["info"]["resets"] == 2 for row in second.values()))
            self.assertEqual({tuple(row["info"]["instance"]) for row in initial.values()},
                             {tuple(row["info"]["instance"]) for row in second.values()})
            pool.step(dict.fromkeys(second, "shoot"))
            again = pool.drain(second)
            self.assertEqual({key: row["observation"] for key, row in final.items()},
                             {key: row["observation"] for key, row in again.items()})

    def test_no_reuse_and_spec_separation(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse), EnvPool(1, reuse=reuse, factory=FakeEnv) as pool:
                before = pool.reset(jobs(1))["ep-0"]["info"]["instance"]
                pool.drain(["ep-0"])
                work = jobs(1)
                if reuse:
                    work[0]["case"]["spec"]["max_steps"] = 10
                after = pool.reset(work)["ep-0"]["info"]["instance"]
                self.assertNotEqual(before, after)

    def test_exceptions_name_key_and_traceback(self):
        for action, message in (("explode", "fake step failure"), ("truncate", "truncated")):
            with self.subTest(action=action), EnvPool(2, factory=FakeEnv) as pool:
                pool.reset(jobs(2))
                with self.assertRaisesRegex(RuntimeError, rf"(?s)ep-1.*Traceback.*{message}"):
                    pool.step({"ep-1": action})
        with EnvPool(1, factory=FakeEnv) as pool:
            with self.assertRaisesRegex(RuntimeError, "(?s)broken.*Traceback.*fake reset failure"):
                pool.reset([dict(key="broken", case=case(-1))])

    def test_dead_worker_and_timeout(self):
        with EnvPool(1, factory=FakeEnv, timeout=5) as pool:
            pool.reset(jobs(1))
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "ep-0"):
                pool.step({"ep-0": "die"})
            self.assertLess(time.monotonic() - started, 3)
        with EnvPool(1, factory=FakeEnv, timeout=5) as pool:
            pool.reset(jobs(1))
            pool.timeout = .1
            with self.assertRaisesRegex(TimeoutError, "ep-0"):
                pool.step({"ep-0": "sleep"})

    def test_duplicate_active_and_closed(self):
        with EnvPool(1, factory=FakeEnv) as pool:
            pool.reset(jobs(1))
            with self.assertRaisesRegex(ValueError, "active"):
                pool.reset(jobs(1))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.step({"ep-0": "noop"})
        pool.close()


class EpisodeTests(unittest.TestCase):
    def test_delayed_ammo_drain_success_and_prompt_identity(self):
        scorer, progress = FakeScorer(), []
        with EnvPool(2, factory=FakeEnv) as pool:
            episodes = rl.run_episodes(scorer, pool, jobs(4), "greedy", progress=progress.append)
        self.assertEqual([ep["success"] for ep in episodes], [True, False, False, True])
        for episode in episodes:
            self.assertEqual(episode["model_decisions"], 2)
            self.assertEqual(episode["env_steps"], 4)
            self.assertTrue(episode["fired"])
            self.assertEqual(episode["fire_decision"], 0)
            self.assertEqual(episode["outcome"], "hit" if episode["success"] else "miss")
            requests = [r for r in scorer.requests if r["id"] == episode["key"]]
            for request, decision in zip(requests, episode["decisions"]):
                self.assertEqual(json.dumps(request), json.dumps(policy_request(decision, episode["key"])))
                self.assertAlmostEqual(decision["logp"], math.log(.85))
                self.assertAlmostEqual(sum(decision["probs"].values()), 1)
        self.assertTrue(any(row["operation"] == "drain" for row in progress))

    def test_tie_break_and_never_fires(self):
        scorer = FakeScorer(dict.fromkeys(ACTIONS, .25))
        with EnvPool(1, factory=FakeEnv) as pool:
            episode, = rl.run_episodes(scorer, pool, jobs(1), "greedy")
        self.assertEqual({decision["action"] for decision in episode["decisions"]}, {"left"})
        self.assertFalse(episode["fired"])
        self.assertIsNone(episode["fire_decision"])
        self.assertEqual(episode["env_steps"], 8)

    def test_sample_reproducibility_independent_of_calls(self):
        with EnvPool(3, factory=FakeEnv) as pool:
            together = rl.run_episodes(FakeScorer(), pool, jobs(9), "sample", .7, .3, seed=(7, 3))
            split = []
            for job in reversed(jobs(9)):
                split.extend(rl.run_episodes(FakeScorer(), pool, [job], "sample", .7, .3, seed=(7, 3)))
            again = rl.run_episodes(FakeScorer(), pool, jobs(9), "sample", .7, .3, seed=(7, 3))
        self.assertEqual(together, again)
        self.assertEqual({ep["key"]: ep for ep in together}, {ep["key"]: ep for ep in split})

    def test_epsilon_matches_existing_distribution(self):
        probabilities = dict(left=.1, right=.2, shoot=.4, noop=.3)
        ids, logs = list(probabilities), list(map(math.log, probabilities.values()))
        for mode in ("greedy", "sample"):
            for epsilon in (0, .2, 1):
                actual = rl.action_distribution(ids, logs, mode, 1, epsilon)
                expected = behavior_distribution(probabilities, mode, epsilon)
                for key in ids:
                    self.assertAlmostEqual(actual[key], expected[key])
        tempered = rl.action_distribution(ids, logs, "sample", 2, 0)
        self.assertAlmostEqual(tempered["shoot"] / tempered["left"], 2)


class LogicTests(unittest.TestCase):
    def test_advantages(self):
        advantages = rl.group_advantages([0, 0, 1, 1])
        for actual, expected in zip(advantages, [-1, -1, 1, 1]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(sum(advantages), 0)
        self.assertIsNone(rl.group_advantages([1, 1]))
        self.assertIsNone(rl.group_advantages([0, 0]))
        self.assertIsNone(rl.group_advantages([]))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_clipped_loss_and_gradient(self):
        import torch
        new = torch.tensor([math.log(1.5), math.log(.5), math.log(1.5), math.log(.5)], requires_grad=True)
        loss, ratio = rl.clipped_loss(new, torch.zeros(4), torch.tensor([2., 2., -2., -2.]), .2)
        self.assertTrue(torch.allclose(loss, torch.tensor([-2.4, -1., 3., 1.6])))
        self.assertTrue(torch.allclose(ratio, torch.tensor([1.5, .5, 1.5, .5])))
        loss.mean().backward()
        self.assertTrue(torch.allclose(new.grad, torch.tensor([0., -.25, .75, 0.])))

    def test_token_packing(self):
        examples = [dict(id=str(i), leaf_tokens=[[1] * length] * count)
                    for i, (length, count) in enumerate([(7, 4), (3, 2), (9, 4), (4, 3)])]
        batches = rl.pack(examples, 60)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(4)))
        for batch in batches:
            self.assertLessEqual(sum(len(examples[i]["leaf_tokens"]) for i in batch) *
                                 max(len(leaf) for i in batch for leaf in examples[i]["leaf_tokens"]), 60)
        counts = rl.token_counts(examples, batches)
        self.assertEqual(counts["leaf_paths"], 13)
        self.assertEqual(counts["tokens"], 82)
        self.assertGreaterEqual(counts["padded_tokens"], counts["tokens"])
        self.assertEqual(rl.pack([], 60), [])
        with self.assertRaisesRegex(ValueError, "budget"):
            rl.pack(examples, 35)

    def test_schedules(self):
        cases = list(range(11))
        for epoch in range(3):
            selected = [i for step in range(epoch * 3 + 1, epoch * 3 + 4)
                        for i in rl.schedule(cases, 17, step, 4)]
            expected = cases[:]
            random.Random(17 + epoch).shuffle(expected)
            self.assertEqual(selected, expected)
        self.assertEqual(rl.schedule(cases, 17, 7, 4), rl.schedule(cases, 17, 7, 4))
        streamed = [i for step in range(1, 12) for i in rl.row_schedule(cases, 17, step, 3)]
        for offset in range(0, 33, 11):
            self.assertEqual(sorted(streamed[offset:offset + 11]), cases)

    def test_minibatches(self):
        for length, count, sizes in ((10, 3, [4, 3, 3]), (2, 8, [1, 1]), (0, 8, [])):
            batches = rl.split_minibatches(list(range(length)), count)
            self.assertEqual(list(map(len, batches)), sizes)
            self.assertEqual([item for batch in batches for item in batch], list(range(length)))

    def test_training_prompt_repeated_ids(self):
        env = FakeEnv(SPEC)
        obs, _ = env.reset(0)
        request = policy_request(obs, "same-key")
        items = [dict(request=request, action="shoot", old_logp=-.5, advantage=1)] * 3
        examples = rl.prepare_items(items, FakeTokenizer(), 8192)
        expected, = prepare_examples({"states": [request]}, FakeTokenizer(), 8192)
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertEqual(example["leaf_tokens"], expected["leaf_tokens"])
            self.assertEqual(example["target"], 2)

    def test_grpo_groups_and_collection(self):
        args = rl.parse_args(["grpo", "--init-checkpoint", "init", "--cases", "cases", "--output-dir", "out",
                              "--tasks-per-step", "2", "--group-size", "2"])
        episodes = []
        for i, success in enumerate([True, False, True, True]):
            episodes.append(dict(key=f"ep-{i}", case=case(i // 2), success=success, outcome="hit", fired=True,
                                 model_decisions=1, decisions=[dict(t=0, state="state", candidates=ACTIONS,
                                 action="shoot", logp=math.log(.25), probs=dict.fromkeys(ACTIONS, .25))]))
        with patch.object(rl, "run_episodes", return_value=episodes), patch.object(rl, "heartbeat"):
            items, record = rl.grpo_collection(None, None, [case(0), case(1)], args, 1)
        self.assertEqual(record["group_success_counts"], [1, 2])
        self.assertEqual(record["dropped_groups"], 1)
        self.assertEqual(record["decisions_used"], 2)
        self.assertEqual(record["sampled_success_rate"], .75)
        self.assertEqual({item["request"]["id"] for item in items}, {"ep-0", "ep-1"})
        self.assertAlmostEqual(sum(item["advantage"] for item in items), 0)
        self.assertAlmostEqual(record["mean_policy_entropy"], math.log(4))


class FileTests(unittest.TestCase):
    def test_checkpoint_layout_resume_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "init", root / "out"
            source.mkdir()
            (source / "config.json").write_text('{"set_head":"none"}')
            for name in ("tokenizer", "backbone_config"):
                (source / name).mkdir()
                (source / name / "config.json").write_text("{}")
            args = rl.parse_args(["grpo", "--init-checkpoint", str(source), "--cases", "cases", "--output-dir", str(out)])
            state = dict(completed_step=3, args=rl.saved_args(args))
            last = out / "ckpt" / "last"
            for step in (2, 3):
                state["completed_step"] = step
                rl.checkpoint(last, source, state, lambda path: path.write_bytes(b"fake weights"),
                              lambda path: path.write_bytes(b"fake optimizer"))
            _, files = local_checkpoint_files(last)
            self.assertEqual(set(files), {"run_config", "body_config", "tokenizer", "weights"})
            self.assertTrue((last / "optimizer.pt").exists())
            args.resume, args.steps = True, 100
            self.assertEqual(rl.resume_state(out, args)[1], state)
            args.clip = .3
            with self.assertRaisesRegex(ValueError, "clip"):
                rl.resume_state(out, args)
            args.clip = .2
            args.steps = 2
            with self.assertRaisesRegex(ValueError, "before"):
                rl.resume_state(out, args)
            args.steps = 100
            last.replace(last.with_name("last.old"))
            self.assertEqual(rl.resume_state(out, args)[1]["completed_step"], 3)
            write_rows(out / "steps.jsonl", [dict(step=2), dict(step=3), dict(step=4)])
            write_rows(out / "eval.jsonl", [dict(step=0), dict(step=4)])
            rl.truncate_logs(out, 3)
            self.assertEqual(rl.read_rows(out / "steps.jsonl"), [dict(step=2), dict(step=3)])
            self.assertEqual(rl.read_rows(out / "eval.jsonl"), [dict(step=0)])

    def test_cases_cli_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "excluded.jsonl", Path(temp) / "cases.jsonl"
            write_rows(source, [case(10), case(11, split="test")])
            argv = ["cases", "--exclude", str(source), "--output", str(output), "--count", "3", "--seed-start", "10"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                rl.main(argv)
            self.assertFalse(output.exists())
            argv[-1] = "20"
            rl.main(argv)
            rows = rl.select_cases(output, "all")
            self.assertEqual([row["seed"] for row in rows], [20, 21, 22])
            self.assertEqual(rows[0]["id"], "heldout-sonic_predict_position-20")
            self.assertEqual(rows[0]["spec"], SPEC)
            with self.assertRaises(FileExistsError):
                rl.main(argv)

    def test_sft_filter_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            row = policy_row()
            critic, other = copy.deepcopy(row), copy.deepcopy(row)
            critic["metadata"]["record_role"] = "critic"
            other["metadata"]["spec"]["scenario"] = "basic"
            write_rows(path, [row, critic, other])
            items = rl.sft_rows(path)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["request"], {key: row[key] for key in ("id", "state", "questions")})
            bad = copy.deepcopy(row)
            bad["questions"]["action"]["instructions"] += " "
            write_rows(path, [bad])
            with self.assertRaisesRegex(ValueError, "instructions"):
                rl.sft_rows(path)
            bad = copy.deepcopy(row)
            bad["gold"]["action"] = "unknown"
            write_rows(path, [bad])
            with self.assertRaisesRegex(ValueError, "Gold"):
                rl.sft_rows(path)
            write_rows(path, [critic, other])
            with self.assertRaisesRegex(ValueError, "No PP"):
                rl.sft_rows(path)

    def test_eval_output_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_rows(root / "cases.jsonl", [case(0, split="test"), case(1, split="train")])
            args = rl.parse_args(["eval", "--checkpoint", "fake", "--cases", str(root / "cases.jsonl"),
                                  "--output", str(root / "eval.jsonl"), "--workers", "1"])
            with patch.object(rl, "load"), patch.object(rl, "Scorer", return_value=FakeScorer()), \
                    patch.object(rl, "EnvPool", side_effect=lambda workers: EnvPool(workers, factory=FakeEnv)), \
                    patch("builtins.print"):
                rl.run_eval(args)
            episodes = rl.read_rows(root / "eval.jsonl")
            self.assertEqual(len(episodes), 1)
            self.assertNotIn("state", episodes[0]["decisions"][0])
            self.assertIn("probs", episodes[0]["decisions"][0])
            summary = json.loads((root / "eval.summary.json").read_text())
            self.assertEqual(summary["success_rate"], 1)
            self.assertEqual(summary["successful_case_ids"], ["case-0"])
            with self.assertRaisesRegex(ValueError, "exists"):
                rl.run_eval(args)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TorchTests(unittest.TestCase):
    def test_scorer_packs_and_restores_order(self):
        import torch
        model = tiny_model()
        predictor = SimpleNamespace(model=model, tokenizer=FakeTokenizer(), limit=8192, precision="fp32")
        requests = []
        for i, length in enumerate([100, 1, 50]):
            row = policy_row(str(i))
            row["state"] = "x" * length
            requests.append({key: row[key] for key in ("id", "state", "questions")})
        examples = prepare_examples({"states": requests}, predictor.tokenizer, predictor.limit)
        budget = max(len(leaf) for ex in examples for leaf in ex["leaf_tokens"]) * 4
        scorer = rl.Scorer(predictor, budget)
        outputs = scorer(requests)
        self.assertEqual([batch[0] for batch in model.batches], ["1:action", "2:action", "0:action"])
        self.assertFalse(model.training)
        with torch.no_grad():
            expected = model(examples, predictor.tokenizer.pad_token_id)[0].log_softmax(-1)
        for (ids, logps), values in zip(outputs, expected):
            self.assertEqual(ids, list(ACTIONS))
            self.assertTrue(torch.allclose(torch.tensor(logps), values))
        self.assertEqual(scorer.tokens["forward_calls"], 3)

    def test_update_accumulation_matches_full_batch(self):
        import torch
        for grpo in (False, True):
            with self.subTest(grpo=grpo):
                model = tiny_model()
                reference = copy.deepcopy(model)
                examples = [dict(id=str(i), candidate_ids=list(ACTIONS), leaf_tokens=[[1] * length] * 4,
                                 target=i % 4, advantage=(-1.) ** i)
                            for i, length in enumerate([1, 5, 3, 7, 2])]
                with torch.no_grad():
                    logits = model(examples, 0)[0]
                    for ex, values in zip(examples, logits):
                        ex["old_logp"] = float(values.log_softmax(-1)[ex["target"]]) + .3
                stats = []
                for current, budget in ((model, 28), (reference, 1000)):
                    args = SimpleNamespace(minibatches=1, output_dir="unused", precision="fp32", clip=.2,
                                           max_microbatch_tokens=budget, grad_clip=100)
                    scorer = SimpleNamespace(model=current, tokenizer=FakeTokenizer())
                    optimizer = torch.optim.SGD(current.parameters(), lr=.01)
                    with patch.object(rl, "heartbeat"):
                        stats.append(rl.update_model(scorer, optimizer, examples, args, 1, grpo))
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
                self.assertAlmostEqual(stats[0]["mean_loss"], stats[1]["mean_loss"], places=6)
                self.assertGreater(stats[0]["tokens"]["forward_calls"], 1)
                self.assertEqual(stats[1]["tokens"]["forward_calls"], 1)


if __name__ == "__main__":
    unittest.main()
