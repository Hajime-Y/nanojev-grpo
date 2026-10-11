"""CPU-only contract tests; CUDA and ViZDoom are not needed."""
import copy
from collections import Counter
from contextlib import contextmanager, ExitStack
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
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pp_rl as rl
from pp_envpool import EnvPool
from pp_tree import split_tree, tree_forward, tree_tokens
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


def causal_model(set_head="none"):
    """Small double-precision transformer with an independent, unpadded path reference."""
    import torch
    from torch import nn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(258, 12)
            self.positions = nn.Embedding(1024, 12)
            self.qkv = nn.ModuleList([nn.Linear(12, 36) for _ in range(2)])
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(12, 20), nn.Tanh(), nn.Linear(20, 12))
                                     for _ in range(2)])

        def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False):
            width = input_ids.shape[1]
            if position_ids is None:
                position_ids = torch.arange(width)[None]
            h = self.tokens(input_ids) + self.positions(position_ids)
            if attention_mask.ndim == 2:
                allowed = torch.ones(width, width, dtype=torch.bool).tril()[None] & attention_mask[:, None, :]
            else:
                allowed = attention_mask[:, 0]
            for qkv, ff in zip(self.qkv, self.ff):
                q, k, v = qkv(h).chunk(3, dim=-1)
                weights = (q @ k.transpose(-1, -2) / math.sqrt(12)).masked_fill(~allowed, -torch.inf).softmax(-1)
                h = h + weights @ v
                h = h + ff(h)
            return SimpleNamespace(last_hidden_state=h)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.norm, self.scalar = nn.LayerNorm(12), nn.Linear(12, 1)
            self.set_head = set_head
            if set_head == "attention":
                self.set_project = nn.Linear(13, 12)
                self.set_attention = nn.MultiheadAttention(12, 3, batch_first=True, dropout=0)
                self.set_output = nn.Linear(12, 1)

        def forward(self, examples, pad_token):
            results = []
            for ex in examples:
                ends = []
                for leaf in ex["leaf_tokens"]:
                    ids = torch.tensor([leaf])
                    ends.append(self.backbone(ids, torch.ones_like(ids, dtype=torch.bool)).last_hidden_state[0, -1])
                h = self.norm(torch.stack(ends))
                z = self.scalar(h).squeeze(-1).float()
                if ex["type"] == "choice" and self.set_head == "attention":
                    u = self.set_project(torch.cat([h, h.new_full((len(h), 1), math.log(len(h)))], dim=-1))[None]
                    mixed, _ = self.set_attention(u, u, u, need_weights=False)
                    z = z + self.set_output(torch.tanh(u + mixed))[0, :, 0].float()
                if ex["type"] == "boolean":
                    z = torch.stack([z[0] * 0, z[0]])
                results.append(z)
            logits = nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=-1e9)
            valid = torch.arange(logits.shape[1])[None] < torch.tensor([len(row) for row in results])[:, None]
            return logits, valid

    torch.manual_seed(123)
    return Model().double()


@contextmanager
def fake_training(root, command):
    source, out = root / "init", root / "out"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "best.safetensors").write_bytes(b"initial")
    for name in ("tokenizer", "backbone_config"):
        (source / name).mkdir()
    args = rl.parse_args([command, "--init-checkpoint", str(source), "--cases", "cases",
                          "--output-dir", str(out), "--save-every", "2", "--eval-every", "2",
                          "--steps" if command == "grpo" else "--updates", "5"])
    torch = MagicMock(__version__="fake")
    torch.cuda.get_device_name.return_value = "fake GPU"
    torch.cuda.max_memory_allocated.return_value = 0
    torch.save.side_effect = lambda state, path: Path(path).write_bytes(b"optimizer")
    safetensors = SimpleNamespace(save_file=lambda state, path: Path(path).write_bytes(b"weights"))
    model = MagicMock()
    model.state_dict.return_value = {}
    runtime = SimpleNamespace(model=model)
    scorer = SimpleNamespace(model=model, tokenizer=FakeTokenizer(), tokens=Counter())
    row = policy_row()
    items = [dict(request={key: row[key] for key in ("id", "state", "questions")}, action="shoot")]
    stats = dict(mean_loss=.5, gradient_norm_before_clip=[1.0], tokens={})
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"torch": torch, "safetensors": SimpleNamespace(torch=safetensors),
                                                      "safetensors.torch": safetensors}))
        for name, value in (("load", runtime), ("Scorer", scorer), ("select_cases", [case()]),
                            ("sft_rows", items), ("grpo_collection", (items, {}))):
            stack.enter_context(patch.object(rl, name, return_value=value))
        stack.enter_context(patch.object(rl, "EnvPool"))
        stack.enter_context(patch.object(rl, "game_eval"))
        stack.enter_context(patch.object(rl, "heartbeat"))
        stack.enter_context(patch("builtins.print"))
        saves = stack.enter_context(patch.object(rl, "checkpoint", wraps=rl.checkpoint))
        update = stack.enter_context(patch.object(rl, "update_model", return_value=stats))
        yield args, saves, update


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
    def test_chunked_evaluation_matches_unchunked(self):
        with EnvPool(2, factory=FakeEnv) as pool:
            for mode in ("greedy", "sample"):
                expected = rl.eval_episodes(FakeScorer(), pool, jobs(7), mode, 100,
                                            temperature=.7, epsilon=.3, seed=42)
                with patch.object(pool, "reset", wraps=pool.reset) as reset:
                    actual = rl.eval_episodes(FakeScorer(), pool, jobs(7), mode, 3,
                                              temperature=.7, epsilon=.3, seed=42)
                self.assertEqual(actual, expected)
                self.assertEqual([len(call.args[0]) for call in reset.call_args_list], [3, 3, 1])

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
        batches = rl.pack(examples, 60, "paths")
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(4)))
        for batch in batches:
            self.assertLessEqual(sum(len(examples[i]["leaf_tokens"]) for i in batch) *
                                 max(len(leaf) for i in batch for leaf in examples[i]["leaf_tokens"]), 60)
        counts = rl.token_counts(examples, batches, "paths")
        self.assertEqual(counts["leaf_paths"], 13)
        self.assertEqual(counts["tokens"], 82)
        self.assertGreaterEqual(counts["padded_tokens"], counts["tokens"])
        self.assertEqual(rl.pack([], 60), [])
        with self.assertRaisesRegex(ValueError, "budget"):
            rl.pack(examples, 35, "paths")

    def test_tree_structure(self):
        leaves = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 8, 9]]
        self.assertEqual(split_tree(leaves), ([1, 2], [[3], [3, 4], [8, 9]]))
        self.assertEqual(tree_tokens(dict(leaf_tokens=leaves)),
                         ([1, 2, 3, 3, 4, 8, 9], [0, 1, 2, 2, 3, 2, 3],
                          [0, 0, 1, 2, 2, 3, 3], [2, 4, 6]))
        self.assertEqual(tree_tokens(dict(leaf_tokens=[[4, 5, 6]])),
                         ([4, 5, 6], [0, 1, 2], [0, 0, 1], [2]))
        self.assertEqual(tree_tokens(dict(leaf_tokens=[[1], [1]])),
                         ([1, 1], [0, 0], [1, 2], [0, 1]))
        self.assertEqual(split_tree([[1, 2], [3, 4]]), ([], [[1, 2], [3, 4]]))

    def test_tree_token_packing(self):
        examples = [dict(id=str(i), leaf_tokens=leaves) for i, leaves in enumerate([
            [[1, 2, 3], [1, 2, 4]], [[1, 2, 3, 4, 5]], [[1, 3], [2, 4]],
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6]]])]
        batches = rl.pack(examples, 10, "tree")
        self.assertEqual(batches, [[0, 2], [1], [3]])
        for batch in batches:
            self.assertLessEqual(len(batch) * max(len(tree_tokens(examples[i])[0]) for i in batch), 10)
        self.assertEqual(rl.token_counts(examples, batches, "tree"),
                         dict(forward_calls=3, questions=4, leaf_paths=8, tokens=19, padded_tokens=19))
        self.assertEqual(rl.token_counts(examples[:2], [[0, 1]], "tree")["padded_tokens"], 10)
        self.assertEqual(rl.pack(examples, 10), batches)
        self.assertEqual(rl.pack([], 10), [])
        with self.assertRaisesRegex(ValueError, "budget"):
            rl.pack(examples, 5, "tree")

    def test_forward_cli_defaults_and_saved_args(self):
        for command, extra in (("eval", ["--checkpoint", "init", "--output", "out"]),
                               ("grpo", ["--init-checkpoint", "init", "--output-dir", "out"]),
                               ("sft", ["--init-checkpoint", "init", "--output-dir", "out"])):
            argv = [command, "--cases", "cases"] + extra
            args = rl.parse_args(argv)
            self.assertEqual(args.forward, "tree")
            self.assertEqual(args.max_forward_tokens, 140000)
            self.assertEqual(args.max_concurrent, 256)
            if command != "eval":
                self.assertEqual(args.max_microbatch_tokens, 34000)
            self.assertEqual(rl.saved_args(rl.parse_args(argv + ["--forward", "paths"]))["forward"], "paths")

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
    def test_game_eval_chunks_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = rl.parse_args(["sft", "--init-checkpoint", "init", "--cases", "cases",
                                  "--output-dir", temp, "--max-concurrent", "2"])
            with EnvPool(1, factory=FakeEnv) as pool, patch.object(rl, "heartbeat"), \
                    patch.object(rl, "run_episodes", wraps=rl.run_episodes) as run:
                rl.game_eval(FakeScorer(), pool, [case(i) for i in range(5)], args, 0)
            self.assertEqual([len(call.args[2]) for call in run.call_args_list], [2, 2, 1, 2, 2, 1])
            records = rl.read_rows(Path(temp) / "eval.jsonl")
            self.assertEqual([record["episodes"] for record in records], [5, 5])
            self.assertTrue(all(record["forward"] == "tree" for record in records))

    def test_training_save_cadence_and_config(self):
        for command, expected in (("sft", [0, 2, 4, 5]), ("grpo", [0, 1, 2, 3, 4, 5])):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temp:
                with fake_training(Path(temp), command) as (args, saves, _):
                    args.forward = "paths"
                    rl.train(args)
                    last = [call.args[2]["completed_step"] for call in saves.call_args_list
                            if call.args[0].name == "last"]
                    self.assertEqual(last, expected)
                    snapshots = [call.args[2]["completed_step"] for call in saves.call_args_list
                                 if call.args[0].name != "last"]
                    self.assertEqual(snapshots, [2, 4, 5])
                    config = json.loads((Path(args.output_dir) / "config.json").read_text())
                    self.assertEqual(config["args"]["forward"], "paths")
                    state = json.loads((Path(args.output_dir) / "ckpt/last/state.json").read_text())
                    self.assertEqual(state["completed_step"], 5)

    def test_sft_resume_replays_unsaved_updates(self):
        with tempfile.TemporaryDirectory() as temp, fake_training(Path(temp), "sft") as (args, saves, update):
            stats = update.return_value
            update.side_effect = [stats, stats, stats, RuntimeError("interrupted")]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                rl.train(args)
            out = Path(args.output_dir)
            self.assertEqual(rl.resume_state(out, args)[1]["completed_step"], 2)
            self.assertEqual([row["update"] for row in rl.read_rows(out / "updates.jsonl")], [1, 2, 3])
            args.resume = True
            update.side_effect = None
            update.reset_mock()
            saves.reset_mock()
            rl.train(args)
            self.assertEqual([call.args[4] for call in update.call_args_list], [3, 4, 5])
            self.assertEqual([row["update"] for row in rl.read_rows(out / "updates.jsonl")], [1, 2, 3, 4, 5])
            self.assertEqual([call.args[2]["completed_step"] for call in saves.call_args_list
                              if call.args[0].name == "last"], [4, 5])

    def test_forward_check_row_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dev.jsonl"
            rows = [policy_row(str(i)) for i in range(30)]
            rows[0]["metadata"]["record_role"] = "critic"
            for row in rows[5:]:
                row["metadata"]["spec"]["scenario"] = "basic"
            write_rows(path, rows)
            selected = rl.forward_check_rows(path, 3)
            self.assertEqual([row["id"] for row in selected], ["1", "2", "3"] + list(map(str, range(5, 21))))
            self.assertEqual(set(selected[0]), {"id", "state", "questions"})
            args = rl.parse_args(["check-forward", "--checkpoint", "init", "--data", temp])
            self.assertEqual((args.count, args.precision), (64, "bf16"))
            with patch.object(rl, "check_forward", return_value=False), self.assertRaises(SystemExit) as error:
                rl.main(["check-forward", "--checkpoint", "init", "--data", temp])
            self.assertEqual(error.exception.code, 1)

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
                                  "--output", str(root / "eval.jsonl"), "--workers", "1",
                                  "--rollouts", "3", "--max-concurrent", "2"])
            with patch.object(rl, "load"), patch.object(rl, "Scorer", return_value=FakeScorer()), \
                    patch.object(rl, "EnvPool", side_effect=lambda workers: EnvPool(workers, factory=FakeEnv)), \
                    patch("builtins.print"), patch.object(rl, "run_episodes", wraps=rl.run_episodes) as run:
                rl.run_eval(args)
            self.assertEqual([len(call.args[2]) for call in run.call_args_list], [2, 1])
            episodes = rl.read_rows(root / "eval.jsonl")
            self.assertEqual(len(episodes), 3)
            self.assertNotIn("state", episodes[0]["decisions"][0])
            self.assertIn("probs", episodes[0]["decisions"][0])
            self.assertEqual({key: episodes[0][key] for key in ("model_decisions", "fire_decision", "fired", "outcome")},
                             dict(model_decisions=2, fire_decision=0, fired=True, outcome="hit"))
            summary = json.loads((root / "eval.summary.json").read_text())
            self.assertEqual(summary["success_rate"], 1)
            self.assertEqual(summary["successful_case_ids"], ["case-0"])
            self.assertEqual({key: summary[key] for key in ("forward", "max_forward_tokens", "rollouts", "epsilon",
                                                           "temperature", "seed", "cases")},
                             dict(forward="tree", max_forward_tokens=140000, rollouts=3, epsilon=0.0,
                                  temperature=1.0, seed=17, cases=str((root / "cases.jsonl").resolve())))
            with self.assertRaisesRegex(ValueError, "exists"):
                rl.run_eval(args)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TorchTests(unittest.TestCase):
    def test_tree_selected_for_scoring_and_updates(self):
        import torch
        model = causal_model()
        runtime = SimpleNamespace(model=model, tokenizer=FakeTokenizer(), limit=8192, precision="fp32")
        row = policy_row()
        request = {key: row[key] for key in ("id", "state", "questions")}
        examples = rl.prepare_items([dict(request=request, action="shoot")], runtime.tokenizer, runtime.limit)
        budget = len(tree_tokens(examples[0])[0])
        scorer = rl.Scorer(runtime, budget)
        with patch.object(rl, "tree_forward", wraps=tree_forward) as forward:
            outputs = scorer([request])
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(scorer.tokens["padded_tokens"], budget)
            expected = model(examples, runtime.tokenizer.pad_token_id)[0][0].log_softmax(-1)
            torch.testing.assert_close(torch.tensor(outputs[0][1]), expected, atol=1e-5, rtol=0)
            before = model.scalar.weight.detach().clone()
            args = SimpleNamespace(forward="tree", max_microbatch_tokens=budget, output_dir="unused",
                                   precision="fp32", grad_clip=100)
            with patch.object(rl, "heartbeat"):
                stats = rl.update_model(scorer, torch.optim.SGD(model.parameters(), lr=.01), examples, args, 1, False)
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(stats["tokens"]["padded_tokens"], budget)
            self.assertFalse(torch.equal(before, model.scalar.weight))

    def test_tree_matches_causal_paths(self):
        import torch
        examples = [dict(type=typ, candidate_ids=list(range(count)), leaf_tokens=leaves)
                    for typ, count, leaves in [
                        ("choice", 3, [[1, 2, 3], [1, 2, 4, 5], [1, 2, 6]]),
                        ("choice", 2, [[7], [8, 9]]),
                        ("score", 3, [[10, 11], [10, 11, 12], [10, 11]]),
                        ("boolean", 2, [[13, 14, 15, 16]]),
                        ("choice", 1, [[17, 18, 19]])]]
        for head in ("none", "attention"):
            model = causal_model(head).eval()
            for group in (examples, examples[2:4], examples[3:4], examples[-1:]):
                with self.subTest(head=head, types=[ex["type"] for ex in group]), torch.inference_mode():
                    expected, valid = model(group, 0)
                    actual, tree_valid = tree_forward(model, group, 0)
                    self.assertEqual(model.scalar.weight.dtype, torch.float64)
                    self.assertTrue(torch.equal(tree_valid, valid))
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_check_forward_metrics_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            write_rows(Path(temp) / "dev.jsonl", [policy_row()])
            runtime = SimpleNamespace(model=causal_model(), tokenizer=FakeTokenizer())
            args = rl.parse_args(["check-forward", "--checkpoint", "fake", "--data", temp, "--precision", "fp32"])
            with patch.object(rl, "load", return_value=runtime), patch("builtins.print") as output:
                self.assertTrue(rl.check_forward(args))
                record = json.loads(output.call_args.args[0])
                self.assertLess(record["max_abs_dlogit"], 1e-5)
                self.assertEqual(record["argmax_agreement"], 1)
                self.assertGreater(record["tree_seconds"], 0)
                self.assertGreater(record["paths_seconds"], 0)
                def perturbed(model, examples, pad):
                    logits, valid = tree_forward(model, examples, pad)
                    logits[:, 0] += 10
                    return logits, valid
                with patch.object(rl, "tree_forward", side_effect=perturbed):
                    self.assertFalse(rl.check_forward(args))
                    self.assertGreater(json.loads(output.call_args.args[0])["max_abs_dprob"], .02)

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
        scorer = rl.Scorer(predictor, budget, "paths")
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
                                           max_microbatch_tokens=budget, grad_clip=100, forward="paths")
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
