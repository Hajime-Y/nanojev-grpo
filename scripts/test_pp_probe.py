"""Offline measurement contracts: spawned fake environments and a loopback HTTP relay."""
import copy
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pp_probe as probe
import pp_rl as rl
from pp_envpool import EnvPool
from test_pp_rl import ACTIONS, FakeEnv, FakeScorer, FakeTokenizer, case, jobs
from unified_game_pipeline import policy_request, read_rows


def snapshot(cx=None, ammo=1):
    labels = [dict(id=0, name="DoomPlayer", bbox=[150, 190, 30, 50])]
    if cx is not None:
        labels.append(dict(id=1, name="Cacodemon", bbox=[cx - 10, 70, 20, 40]))
    return dict(ammo=ammo, angle_degrees=90, visible_labels=labels)


def state(cx=180, previous=170):
    return json.dumps(dict(observed_history=[snapshot(previous), snapshot(cx)], remaining_decisions=8))


def row(identity="probe-0", hit=True, cx=180, previous=170, behavior="random_noshoot"):
    s = state(cx, previous)
    return dict(id=identity, state=s, case=case(), candidates=dict(ACTIONS), features=probe.features(s),
                history=[], t=0, behavior=behavior, hit_now=hit)


class ProbeEnv(FakeEnv):
    def observation(self):
        obs = super().observation()
        payload = json.loads(obs["state"])
        payload["observed_history"] = [snapshot(180 + self.t, payload["observed_history"][-1]["ammo"])]
        payload["remaining_decisions"] = self.spec["max_steps"] - self.t
        obs["state"] = json.dumps(payload)
        return obs

    def step(self, action):
        if self.spec.get("forbid_shoot") and action == "shoot":
            raise AssertionError("shoot forbidden")
        return super().step(action)


class ReplayAuditEnv(ProbeEnv):
    """Log construction/reset/action events across spawned processes."""
    def __init__(self, spec):
        super().__init__(spec)
        self.event("create")

    def event(self, kind, **extra):
        with open(self.spec["events"], "a") as handle:
            handle.write(json.dumps(dict(kind=kind, instance=self.instance, **extra)) + "\n")

    def reset(self, seed):
        self.event("reset", seed=seed)
        return super().reset(seed)

    def step(self, action):
        self.event("step", action=action)
        return super().step(action)


class FakeEngine:
    def __init__(self):
        self.requests = []

    def __call__(self, requests):
        self.requests.extend(copy.deepcopy(requests))
        return [{name: probe.choice_answer(dict.fromkeys(q["criteria"], math.log(.25)))
                 if q["type"] == "choice" else {"p_true": .7} for name, q in r["questions"].items()}
                for r in requests]

    def summary(self):
        return dict(api_calls=len(self.requests), input_tokens=0)


@contextmanager
def fake_jev(statuses=()):
    class Handler(BaseHTTPRequestHandler):
        seen, status_codes = [], list(statuses)
        lock = threading.Lock()

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.lock:
                self.seen.append(payload)
                status = self.status_codes.pop(0) if self.status_codes else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if status != 200:
                self.wfile.write(b'{"error":"cap or transient"}')
                return
            answers = {name: dict(type="choice", choice="shoot",
                                  probabilities=dict(left=.33, right=.33, shoot=.33, noop=0))
                       if q["type"] == "choice" else dict(type="noul", noul=.73)
                       for name, q in payload["questions"].items()}
            self.wfile.write(json.dumps(dict(answers=answers, usage=dict(input_tokens=41))).encode())

        def log_message(self, *_):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as exc:
        raise unittest.SkipTest(f"Sandbox denies loopback server binding: {exc}") from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/systemone", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ReplayTests(unittest.TestCase):
    def test_prefix_tail_and_coexistence(self):
        with EnvPool(2, factory=ProbeEnv) as pool:
            pool.reset(jobs(2))
            result = pool.replay([
                dict(key="ep-0", case=case(0), actions=["shoot"] + ["noop"] * 8, tail="explode"),
                dict(key="tail", case=case(1), actions=["noop", "shoot"], tail="noop"),
                dict(key="empty", case=case(2), actions=[], tail="noop"),
                dict(key="boundary", case=case(0), actions=["shoot"] + ["noop"] * 3, tail="explode"),
            ])
            self.assertEqual(result["ep-0"], dict(success=True, terminal_reason="hit", steps=4,
                                                 terminated_in_prefix=True))
            self.assertEqual(result["tail"], dict(success=True, terminal_reason="hit", steps=5,
                                                terminated_in_prefix=False))
            self.assertEqual(result["empty"]["steps"], 8)
            self.assertFalse(result["empty"]["terminated_in_prefix"])
            self.assertTrue(result["boundary"]["terminated_in_prefix"])
            self.assertEqual(set(pool.owners), {"ep-0", "ep-1"})
            transitions = pool.step({"ep-0": "shoot", "ep-1": "noop"})
            self.assertEqual(transitions["ep-0"]["observation"]["step"], 1)
            done = pool.drain(["ep-0", "ep-1"])
            self.assertTrue(done["ep-0"]["info"]["success"])
            self.assertFalse(done["ep-1"]["info"]["success"])

    def test_reuse_and_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            c = case()
            c["spec"]["events"] = str(events)
            with EnvPool(3, factory=ReplayAuditEnv) as pool:
                work = [dict(key=i, case=c, actions=["shoot"], tail="noop") for i in range(9)]
                first = pool.replay(work)
                self.assertEqual(first, pool.replay(work))
                initial = pool.reset([dict(key=i, case=c) for i in range(3)])
                self.assertTrue(all(r["info"]["resets"] == 7 for r in initial.values()))
            creations = [r for r in read_rows(events) if r["kind"] == "create"]
            self.assertEqual(len(creations), 3)
            self.assertEqual(len({r["instance"][0] for r in creations}), 3)

    def test_duplicates_and_truncation(self):
        with EnvPool(1, factory=FakeEnv) as pool:
            job = dict(key="r", case=case(), actions=["truncate"], tail="noop")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                pool.replay([job, job])
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                pool.replay([job])


class PlayTests(unittest.TestCase):
    def test_annotations_seen_by_model_and_recorded(self):
        scorer = FakeScorer()
        with EnvPool(2, factory=ProbeEnv) as pool:
            episodes = rl.run_episodes(scorer, pool, jobs(3), "greedy", decision_hook=probe.lookahead_hook)
        expected = [dict(left=False, right=False, shoot=True, noop=False),
                    dict(left=True, right=True, shoot=False, noop=True),
                    dict.fromkeys(ACTIONS, False)]
        for ep, lookahead in zip(episodes, expected):
            decision = ep["decisions"][0]
            self.assertEqual(decision["lookahead"], lookahead)
            request = next(r for r in scorer.requests if r["id"] == ep["key"])
            self.assertEqual(request["state"], decision["state"])
            self.assertEqual(request["questions"]["action"]["criteria"], decision["candidates"])
            for action, description in decision["candidates"].items():
                self.assertEqual(description, ACTIONS[action] + probe.annotation(action, lookahead[action]))
                self.assertEqual(description.count(" Simulated result:"), 1)
            # A delayed launch is already committed on subsequent decisions.
            self.assertEqual(set(ep["decisions"][1]["lookahead"].values()), {ep["success"]})
        self.assertEqual(probe.annotation("shoot", True), " Simulated result: the rocket kills the monster.")
        self.assertEqual(probe.annotation("shoot", False), " Simulated result: the rocket misses the monster.")
        self.assertEqual(probe.annotation("left", True), " Simulated result: shooting right after this kills the monster.")
        self.assertEqual(probe.annotation("noop", False), " Simulated result: shooting right after this misses the monster.")
        self.assertEqual(ACTIONS["shoot"], "Hold attack.")

    def test_oracle_preferences_and_timing(self):
        engine = probe.OracleEngine()
        with self.assertRaisesRegex(ValueError, "annotations"):
            engine([policy_request(row(), "x")])
        for hits, expected in [(["shoot", "noop"], "shoot"), (["right", "left", "noop"], "noop"),
                               (["right", "left"], "left"), (["right"], "right"), ([], "noop")]:
            r = row()
            r["candidates"] = {a: text + probe.annotation(a, a in hits) for a, text in ACTIONS.items()}
            answer, = engine([policy_request(r, "x")])
            self.assertEqual(max(answer["action"]["probabilities"], key=answer["action"]["probabilities"].get), expected)
        with EnvPool(2, factory=ProbeEnv) as pool:
            episodes = rl.run_episodes(probe.PolicyScorer(engine), pool, jobs(3), "greedy",
                                        decision_hook=probe.lookahead_hook)
        self.assertTrue(all(ep["success"] for ep in episodes))
        self.assertEqual([ep["fire_decision"] for ep in episodes], [0, 1, 2])

    def test_random_uniform_and_episode_seeded(self):
        scorer = probe.PolicyScorer(probe.RandomEngine())
        ids, logs = scorer([policy_request(row(), "x")])[0]
        self.assertEqual(dict(zip(ids, map(math.exp, logs))), dict.fromkeys(ACTIONS, .25))
        with EnvPool(2, factory=ProbeEnv) as pool:
            together = rl.eval_episodes(scorer, pool, jobs(6), "sample", 6, seed=19)
            chunks = rl.eval_episodes(scorer, pool, jobs(6), "sample", 1, seed=19)
        self.assertEqual(together, chunks)
        self.assertGreater(len({d["action"] for ep in together for d in ep["decisions"]}), 1)

    def test_no_shoot_build_stride_and_replay_gold(self):
        with EnvPool(2, factory=ProbeEnv) as pool:
            outputs = probe.build_probes(pool, [case(0), case(1)], ["policy_noshoot", "random_noshoot"],
                                         FakeScorer(), stride=3, max_concurrent=1)
        self.assertEqual(len(outputs), 12)
        self.assertEqual(len({r["id"] for r in outputs}), 12)
        for r in outputs:
            self.assertIn(r["t"], [0, 3, 6])
            self.assertEqual(len(r["history"]), r["t"])
            self.assertNotIn("shoot", r["history"])
            self.assertEqual(r["hit_now"], r["t"] % 3 == r["case"]["seed"] % 3)
            self.assertEqual(r["features"], probe.features(r["state"]))
        # Including the last step checks actions beyond the last saved probe.
        c = case()
        c["spec"]["forbid_shoot"] = True
        with EnvPool(1, factory=ProbeEnv) as pool:
            for base in (FakeScorer(), probe.PolicyScorer(probe.RandomEngine())):
                ep, = rl.run_episodes(probe.NoShootScorer(base), pool, [dict(key="noshoot", case=c)], "sample")
                self.assertEqual(ep["env_steps"], 8)
                self.assertTrue(all(d["action"] != "shoot" for d in ep["decisions"]))
                self.assertTrue(all(d["probs"]["shoot"] == 0 for d in ep["decisions"]))


class FeatureTests(unittest.TestCase):
    def test_features_and_thresholds(self):
        self.assertEqual(probe.features(state()), dict(monster_visible=True, cx=180, side="right", dx=10,
                                                       ammo=1, angle_degrees=90, remaining_decisions=8))
        for cx, side in [(143, "left"), (144, "center"), (176, "center"), (177, "right"), (None, None)]:
            self.assertEqual(probe.features(state(cx))["side"], side)
        self.assertIsNone(probe.features(state(None))["dx"])
        self.assertIsNone(probe.features(state(180, None))["dx"])
        single = json.loads(state())
        single["observed_history"] = single["observed_history"][-1:]
        self.assertIsNone(probe.features(single)["dx"])

    def test_gold_undefined_boundaries(self):
        for cx, expected in [(None, None), (160, None), (153, None), (167, None), (152, False), (168, True)]:
            self.assertIs(probe.golds(row(cx=cx))["side_right"], expected)
        for previous, expected in [(None, None), (179, None), (181, None), (178, True), (182, False)]:
            self.assertIs(probe.golds(row(previous=previous))["moved_right"], expected)
        request = probe.probe_request(row(cx=None, hit=False), ["policy", "boolean"])
        self.assertEqual(set(request["questions"]), {"action", "hit_now"})
        self.assertFalse(probe.golds(row(hit=False))["hit_now"])


class EngineTests(unittest.TestCase):
    def test_jev_transport_contract_without_sockets(self):
        class Response(io.BytesIO):
            status = 200

        payload = dict(answers=dict(action=dict(type="choice", choice="shoot",
                                               probabilities=dict(left=.33, right=.33, shoot=.33, noop=0)),
                                    side_right=dict(noul=.73), moved_right=dict(noul=.24), hit_now=dict(noul=.6)),
                       usage=dict(prompt_tokens=37))
        request = probe.probe_request(row(), ["policy", "boolean"])
        responses = [probe.HTTPError("http://fake", code, "test", {}, io.BytesIO()) for code in [429, 500, 503]]
        responses.append(Response(json.dumps(payload).encode()))
        engine = probe.JevEngine("http://fake", retry_delay=0)
        with patch.object(probe, "urlopen", side_effect=responses) as opening:
            answer, = engine([request])
            sent = json.loads(opening.call_args.args[0].data)
            self.assertEqual(set(sent["questions"]), {"action", "side_right", "moved_right", "hit_now"})
            self.assertEqual(sent["questions"]["hit_now"]["type"], "noul")
            self.assertEqual(sent["state"], request["state"])
            self.assertEqual(sent["model"], "typesafe/jev-1.13")
        self.assertEqual(answer["side_right"]["p_true"], .73)
        self.assertAlmostEqual(answer["action"]["probabilities"]["noop"], 1e-6 / .990001)
        self.assertAlmostEqual(sum(answer["action"]["probabilities"].values()), 1)
        self.assertEqual(engine.summary(), dict(api_calls=4, input_tokens=37))
        errors = [probe.HTTPError("http://fake", 402, "cap", {}, io.BytesIO()) for _ in range(4)]
        with patch.object(probe, "urlopen", side_effect=errors) as opening:
            with self.assertRaisesRegex(RuntimeError, "4 attempts.*402"):
                engine([request])
            self.assertEqual(opening.call_count, 4)

    def test_concurrent_successes_persist_when_peer_fails(self):
        engine = probe.JevEngine(concurrency=2)
        completed = threading.Event()

        def one(request):
            if request["id"] == "bad":
                completed.set()
                raise RuntimeError("cap hit")
            completed.wait(timeout=2)
            return {"hit_now": {"p_true": .8}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, "_one", side_effect=one):
            output = Path(tmp) / "scores.jsonl"
            with self.assertRaisesRegex(RuntimeError, "cap hit"):
                probe.score_probes(engine, [row("bad"), row("good")], output, ["boolean"])
            self.assertEqual(read_rows(output), [dict(id="good", policy=None, boolean=dict(hit_now=.8))])

    def test_checkpoint_wraps_scorer_and_native_booleans(self):
        class PreparedScorer(FakeScorer):
            tokenizer, max_length, tokens = FakeTokenizer(), 8192, {}

            def score_examples(self, examples):
                self.examples = examples
                return [(ex["candidate_ids"], [math.log(.2), math.log(.8)] if ex["type"] == "boolean"
                         else [math.log(.25)] * 4) for ex in examples]

        scorer = PreparedScorer()
        engine = probe.CheckpointEngine(scorer)
        policy = policy_request(row(), "x")
        answer, = engine([policy])
        self.assertEqual(scorer.requests, [policy])
        self.assertAlmostEqual(answer["action"]["probabilities"]["shoot"], .85)
        answer, = engine([probe.probe_request(row(), ["policy", "boolean"])])
        self.assertEqual(len(scorer.examples), 4)
        self.assertEqual([ex["type"] for ex in scorer.examples], ["choice"] + ["boolean"] * 3)
        self.assertTrue(all(ex["candidate_ids"] == ["false", "true"] for ex in scorer.examples[1:]))
        self.assertAlmostEqual(answer["hit_now"]["p_true"], .8)

    def test_jev_bundles_rounding_and_usage(self):
        with fake_jev() as (url, handler):
            engine = probe.JevEngine(url, concurrency=3, retry_delay=0)
            requests = [probe.probe_request(row(str(i)), ["policy", "boolean"]) for i in range(4)]
            answers = engine(requests)
            self.assertEqual(len(handler.seen), 4)
            for payload in handler.seen:
                self.assertEqual(set(payload), {"model", "state", "questions"})
                self.assertEqual(payload["model"], "typesafe/jev-1.13")
                self.assertEqual(set(payload["questions"]), {"action", "side_right", "moved_right", "hit_now"})
                self.assertEqual(payload["questions"]["hit_now"]["type"], "noul")
                self.assertNotIn("hit_now", json.loads(payload["state"]))
            for answer in answers:
                self.assertEqual(answer["hit_now"]["p_true"], .73)
                self.assertAlmostEqual(sum(answer["action"]["probabilities"].values()), 1)
                self.assertAlmostEqual(answer["action"]["probabilities"]["noop"], 1e-6 / .990001)
                self.assertTrue(all(math.isfinite(p) for p in answer["action"]["logps"].values()))
            self.assertEqual(engine.summary(), dict(api_calls=4, input_tokens=164))

    def test_retry_and_402_hard_error(self):
        request = probe.probe_request(row(), ["boolean"])
        with fake_jev([429, 503, 500, 200]) as (url, handler):
            engine = probe.JevEngine(url, retry_delay=0)
            self.assertEqual(engine([request])[0]["hit_now"]["p_true"], .73)
            self.assertEqual(engine.summary(), dict(api_calls=4, input_tokens=41))
        with fake_jev([402] * 4) as (url, handler):
            engine = probe.JevEngine(url, retry_delay=0)
            with self.assertRaisesRegex(RuntimeError, "4 attempts.*402"):
                engine([request])
            self.assertEqual(len(handler.seen), 4)
            self.assertEqual(engine.summary(), dict(api_calls=4, input_tokens=0))

    def test_resume_and_torn_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.jsonl"
            rows = [row(str(i)) for i in range(3)]
            engine = FakeEngine()
            probe.score_probes(engine, rows[:1], output)
            with output.open("ab") as handle:
                handle.write(b'{"id": "torn')
            summary = probe.score_probes(engine, rows, output, batch_size=1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["scored"], 2)
            self.assertEqual(len(engine.requests), 3)
            self.assertEqual([r["id"] for r in read_rows(output)], ["0", "1", "2"])
            probe.score_probes(engine, rows, output)
            self.assertEqual(len(engine.requests), 3)
            # A valid final line lacking a newline must also be retained.
            output.write_text(output.read_text().rstrip("\n"))
            probe.score_probes(engine, rows + [row("3")], output)
            self.assertEqual(len(read_rows(output)), 4)

    def test_completed_http_rows_survive_later_failure(self):
        with tempfile.TemporaryDirectory() as tmp, fake_jev([200] + [402] * 4) as (url, handler):
            output = Path(tmp) / "scores.jsonl"
            engine = probe.JevEngine(url, concurrency=1, retry_delay=0)
            with self.assertRaises(RuntimeError):
                probe.score_probes(engine, [row("0"), row("1")], output, batch_size=1)
            self.assertEqual([r["id"] for r in read_rows(output)], ["0"])
            probe.score_probes(engine, [row("0"), row("1")], output)
            self.assertEqual(len(handler.seen), 6)
            self.assertEqual(len(read_rows(output)), 2)


class ReportTests(unittest.TestCase):
    def test_auc_hand_computed(self):
        self.assertAlmostEqual(probe.auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)
        self.assertAlmostEqual(probe.auc([0, 1, 0, 1], [.1, .1, .5, .9]), .625)
        self.assertEqual(probe.auc([1, 0], [.5, .5]), .5)
        self.assertEqual(probe.auc([1, 0], [0, 1]), 0)
        self.assertIsNone(probe.auc([1, 1], [0, 1]))
        self.assertIsNone(probe.auc([], []))

    def test_tiny_report(self):
        rows = [row("a", True, 180, 170, "policy_noshoot"), row("b", True, 140, 150, "policy_noshoot"),
                row("c", False, 180, 180), row("d", False, None, None)]
        scores = [dict(id="a", policy=dict(shoot=.6, left=.1, right=.2, noop=.1),
                       boolean=dict(side_right=.8, moved_right=.9, hit_now=.8)),
                  dict(id="b", policy=dict(shoot=.2, left=.3, right=.3, noop=.2),
                       boolean=dict(side_right=.6, moved_right=.1, hit_now=.3)),
                  dict(id="c", policy=dict(shoot=.4, left=.3, right=.2, noop=.1),
                       boolean=dict(side_right=.7, hit_now=.6)),
                  dict(id="d", policy=dict(shoot=.1, left=.3, right=.3, noop=.3), boolean=dict(hit_now=.2))]
        result = probe.report(rows, {"fake": scores})["fake"]
        pooled = result["pooled"]
        self.assertEqual(pooled["n"], 4)
        self.assertEqual(pooled["hit_now_base_rate"], .5)
        self.assertEqual(pooled["policy"]["auc"], .75)
        self.assertAlmostEqual(pooled["policy"]["mean_p_shoot_hit"], .4)
        self.assertAlmostEqual(pooled["policy"]["mean_p_shoot_miss"], .25)
        self.assertEqual(pooled["policy"]["shoot_precision"], .5)
        self.assertEqual(pooled["policy"]["shoot_recall"], .5)
        self.assertEqual(pooled["policy"]["turn_toward_share"], .5)
        self.assertEqual(pooled["policy"]["explicit_n"], 3)
        self.assertEqual(pooled["boolean"]["side_right"], dict(n=3, base_rate=2/3, accuracy=2/3, auc=1))
        self.assertEqual(pooled["boolean"]["moved_right"], dict(n=2, base_rate=.5, accuracy=1, auc=1))
        self.assertEqual(pooled["boolean"]["hit_now"], dict(n=4, base_rate=.5, accuracy=.5, auc=.75))
        self.assertEqual(result["policy_noshoot"]["n"], 2)
        self.assertIsNone(result["policy_noshoot"]["policy"]["auc"])
        partial = probe.report(rows, {"partial": [dict(id="a", policy=None, boolean={})]})["partial"]["pooled"]
        self.assertEqual(partial["scored_n"], 1)
        self.assertEqual(partial["policy"]["n"], 0)
        self.assertIsNone(partial["policy"]["shoot_precision"])
        json.dumps(partial, allow_nan=False)


class CLITests(unittest.TestCase):
    def test_play_build_score_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.jsonl"
            probe.write_rows(cases, [case(0, split="probe"), case(1, split="probe")])
            pool_factory = lambda workers: EnvPool(workers, factory=ProbeEnv)
            with patch.object(probe, "EnvPool", side_effect=pool_factory), patch("builtins.print"):
                play = root / "play.jsonl"
                probe.main(["play", "--engine", "oracle", "--annotate", "lookahead", "--cases", str(cases),
                            "--splits", "probe", "--workers", "2", "--output", str(play)])
                summary = json.loads(play.with_suffix(".summary.json").read_text())
                self.assertEqual(summary["success_rate"], 1)
                self.assertGreater(summary["decisions_with_kill_available"], 0)
                self.assertEqual(summary["api_calls"], 0)
                probes = root / "probes.jsonl"
                probe.main(["build", "--checkpoint", "unused", "--behaviors", "random_noshoot", "--cases", str(cases),
                            "--workers", "1", "--output", str(probes)])
                self.assertEqual(len(read_rows(probes)), 6)
                self.assertEqual(json.loads(probes.with_suffix(".summary.json").read_text())["behaviors"]["random_noshoot"]["n"], 6)
                scores = root / "scores.jsonl"
                with patch.object(probe, "make_engine", return_value=FakeEngine()):
                    probe.main(["score", "--engine", "jev", "--probe", str(probes), "--output", str(scores)])
                reporting = root / "report.json"
                probe.main(["report", "--probe", str(probes), "--scores", f"fake={scores}", "--output", str(reporting)])
                self.assertEqual(json.loads(reporting.read_text())["fake"]["pooled"]["scored_n"], 6)


if __name__ == "__main__":
    unittest.main()
