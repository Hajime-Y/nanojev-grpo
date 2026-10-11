"""Offline contracts; fake transports only, no keys, GPU, or listening sockets."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import http.client
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pp_snake_probe as probe
import snake_game
from unified_grid_envs import UnifiedSnakeEnv


def snapshot(size, body, direction, food):
    state = snake_game.make_snake(size, 7)
    state.update(body=[list(cell) for cell in body], direction=direction, food=list(food))
    snake_game.validate_state(state)
    return state


def item(identity="a", kind="lookup", gold=True):
    return {"id": identity, "kind": kind, "state": "Snake state.",
            "question": "Is this true?", "gold": gold}


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def jev_response(probability=0.7):
    return {"answers": {"q": {"noul": probability}}, "usage": {"input_tokens": 11}}


class DynamicsTests(unittest.TestCase):
    def test_corridor_closes_after_exactly_four_moves(self):
        # The top row is a four-cell corridor. Row 1 remains occupied until
        # after reaching the right wall, so the fifth move must collide.
        body = [(0, 0)] + [(1, c) for c in range(5)]
        body += [(2, c) for c in reversed(range(5))] + [(3, c) for c in range(5)]
        state = snapshot(5, body, "north", (4, 4))
        original = copy.deepcopy(state)
        with patch.object(probe, "step", side_effect=AssertionError("lookahead used real step")):
            for depth in range(4):
                self.assertTrue(probe.survives(state, "east", depth))
            for depth in (4, 5, 8):
                self.assertFalse(probe.survives(state, "east", depth))
        self.assertEqual(state, original)
        self.assertFalse(probe.survives(state, "north", 0))  # Immediate wall.
        self.assertFalse(probe.survives(state, "south", 0))  # Reverse.

    def test_current_food_growth_and_no_replacement(self):
        state = snapshot(4, [(1, 1), (1, 0), (2, 0), (2, 1)], "east", (1, 2))
        body = tuple(map(tuple, state["body"]))
        grown = probe.lookahead_step(4, body, "east", (1, 2), "east")
        self.assertEqual(grown[0], ((1, 2),) + body)
        self.assertIsNone(grown[2])
        following = probe.lookahead_step(4, *grown, "south")
        self.assertEqual(len(following[0]), len(body) + 1)
        self.assertNotIn(body[-1], following[0])
        self.assertIsNone(following[2])
        # Traversing a full loop can revisit the old food cell: it stays empty.
        for action in ("south", "east", "north", "north", "west"):
            following = probe.lookahead_step(4, *following, action)
            self.assertIsNotNone(following)
            self.assertIsNone(following[2])
            self.assertEqual(len(following[0]), 5)
        self.assertEqual(following[0][0], (1, 2))
        with patch.object(snake_game, "_food", side_effect=AssertionError("new food")):
            self.assertTrue(probe.survives(state, "east", 8))

    def test_tail_vacates_and_lookup_counts_it(self):
        state = snapshot(4, [(1, 1), (1, 0), (2, 0), (2, 1)], "east", (0, 3))
        labels = {(kind, action): gold for kind, action, gold in probe.candidate_labels(state, 3)}
        self.assertTrue(labels["lookup", "south"])
        self.assertTrue(labels["step1", "south"])
        self.assertTrue(labels["ahead", "south"])
        body = tuple(map(tuple, state["body"]))
        following = probe.lookahead_step(4, body, "east", (0, 3), "south")
        self.assertEqual(following[0], ((2, 1), (1, 1), (1, 0), (2, 0)))

    def test_win_survives_any_depth(self):
        state = snapshot(2, [(0, 0), (1, 0), (1, 1)], "north", (0, 1))
        self.assertTrue(probe.survives(state, "east", 1000))
        self.assertTrue(probe.survives(snake_game.step(state, "east"), "north", 1000))
        with self.assertRaises(ValueError):
            probe.survives(state, "east", -1)

    def test_immediate_lookahead_matches_game(self):
        for seed in range(6):
            for state in probe.rollout(4, seed, 17, 3):
                for action in snake_game.valid_actions(state):
                    self.assertEqual(probe.survives(state, action, 0),
                                     snake_game.one_step_safe(state, action))


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.path = Path(cls.temp.name) / "probe.jsonl"
        cls.options = dict(sizes=[4], seeds=12, min_length=4, depth=3, per_class=4, seed=17)
        cls.summary = probe.build(cls.path, **cls.options)
        cls.rows = list(probe.read_jsonl(cls.path))

    def test_deterministic_files_and_class_balance(self):
        other = Path(self.temp.name) / "other.jsonl"
        probe.build(other, **self.options)
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        self.assertEqual(probe.summary_path(self.path).read_bytes(), probe.summary_path(other).read_bytes())
        self.assertEqual(Counter((row["kind"], row["gold"]) for row in self.rows),
                         Counter({(kind, gold): 4 for kind in probe.KINDS for gold in (False, True)}))
        self.assertEqual(len({(row["kind"], row["state"]) for row in self.rows}), 24)
        self.assertEqual(len({row["id"] for row in self.rows}), 24)
        self.assertEqual(self.summary["items"], 24)
        self.assertGreaterEqual(self.summary["body_length_quantiles"]["0"], 4)

    def test_observation_questions_and_gold(self):
        for row in self.rows:
            state = row["meta"]["snapshot"]
            env = UnifiedSnakeEnv({"task": "snake", "size": row["size"], "max_steps": 96,
                                   "initial_state": state})
            observation, _ = env.reset(row["seed"])
            self.assertEqual(row["state"], observation["state"])
            self.assertIn("Used: 0. Remaining: 96.", row["state"])
            self.assertIn("Recent physical events: [].", row["state"])
            self.assertIn("Food collected this episode: 0.", row["state"])
            self.assertEqual(row["step"], state["steps"])
            self.assertIn(row["action"], observation["candidates"])
            cell = list(probe.destination(state["body"], row["action"]))
            if row["kind"] == "lookup":
                self.assertTrue(all(0 <= value < row["size"] for value in cell))
                self.assertEqual(row["gold"], cell in state["body"])
                self.assertEqual(row["question"],
                                 f"Is the cell {probe.compact(cell)} one of the cells listed in the snake's body?")
            elif row["kind"] == "step1":
                self.assertEqual(row["gold"], snake_game.one_step_safe(state, row["action"]))
            else:
                self.assertTrue(snake_game.one_step_safe(state, row["action"]))
                self.assertEqual(row["gold"], probe.survives(state, row["action"], 3))
                self.assertIn("3 more moves", row["question"])

    def test_insufficient_classes_fail_before_writing(self):
        path = Path(self.temp.name) / "insufficient.jsonl"
        with self.assertRaisesRegex(ValueError, "ahead gold=false.*found 0"):
            probe.build(path, sizes=[4], seeds=3, min_length=3, depth=0, per_class=1)
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "both classes without reuse"):
            probe.sample_balanced({False: [("same", "a")], True: [("same", "b")]},
                                  1, random.Random(0), "lookup")

    def test_dense_exact_sampler_handles_mixed_states(self):
        pools = {False: [("mixed", "north"), ("mixed", "east"), ("false", "south")],
                 True: [("mixed", "south"), ("true", "north")]}
        for seed in range(30):
            rows = probe._balanced_dp(pools, 1, random.Random(seed))
            self.assertEqual(Counter(row[2] for row in rows), {False: 1, True: 1})
            self.assertEqual(len({row[0] for row in rows}), 2)
        # Every state must be used. Independent rejection would be wasteful.
        full = {gold: [(str(i), str(gold)) for i in range(10)] for gold in (False, True)}
        rows = probe._balanced_dp(full, 5, random.Random(0))
        self.assertEqual(len({row[0] for row in rows}), 10)
        self.assertEqual(Counter(row[2] for row in rows), {False: 5, True: 5})

    def test_rollout_is_seeded_live_and_bounded(self):
        first = list(probe.rollout(4, 8, 17, 3))
        self.assertEqual(first, list(probe.rollout(4, 8, 17, 3)))
        self.assertLessEqual(len(first), 401)
        for state in first:
            self.assertFalse(state["done"])
            self.assertLessEqual(state["steps"], 400)
        for before, after in zip(first, first[1:]):
            self.assertTrue(snake_game.one_step_safe(before, after["direction"]))
            self.assertEqual(after, snake_game.step(before, after["direction"]))


class RequestTests(unittest.TestCase):
    def test_reply_parsing(self):
        for content, expected in (("Yes.", True), ("no", False), ("... so the answer is no", False),
                                  ("yes? NO!", False), ("**YES**", True), ("nobody yesterday", None),
                                  ("junk", None), ("", None)):
            self.assertIs(probe.parse_reply(content), expected)

    def test_jev_body_and_probability(self):
        def transport(url, body, headers):
            self.assertEqual(url, "http://fake/jev")
            self.assertEqual(headers, {"Content-Type": "application/json"})
            self.assertEqual(body, {"model": "jev-test", "state": "Snake state.",
                                  "questions": {"q": {"type": "noul", "instructions": "Is this true?"}}})
            return jev_response(0.5)
        with patch.dict("os.environ", {}, clear=True):
            row = probe.ask_item(item(), "jev", jev_url="http://fake/jev", jev_model="jev-test",
                                 transport=transport)
        self.assertEqual(row, {"id": "a", "p_yes": 0.5, "answer": True, "usage": {"input_tokens": 11}})

    def test_chat_body_reasoning_and_key_at_call_time(self):
        for mode in ("off", "on"):
            key = "fake-secret-" + mode
            def transport(url, body, headers):
                self.assertEqual(url, "http://fake/chat")
                self.assertEqual(headers["Authorization"], "Bearer " + key)
                self.assertEqual(body, {"model": "chat-test", "messages": [{"role": "user", "content":
                    "Snake state.\n\nIs this true?\n\nAnswer with exactly one word: yes or no."}],
                    "temperature": 0, "max_tokens": 123, "reasoning": {"enabled": mode == "on"},
                    "provider": {"require_parameters": True}})
                return {"choices": [{"message": {"content": "thinking " * 40 + "NO."}}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                                  "completion_tokens_details": {"reasoning_tokens": 2}}}
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": key}):
                row = probe.ask_item(item(), "chat", chat_url="http://fake/chat", chat_model="chat-test",
                                     reasoning=mode, max_tokens=123, transport=transport)
            self.assertIs(row["answer"], False)
            self.assertEqual(len(row["raw"]), 200)
            self.assertGreaterEqual(row["seconds"], 0)
            self.assertNotIn(key, json.dumps(row))
            self.assertNotIn("Authorization", json.dumps(row))

    def test_missing_content_and_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-secret"}):
            row = probe.ask_item(item(), "chat", chat_model="test", transport=lambda *args:
                                 {"choices": [{"message": {"content": None}}]})
        self.assertIsNone(row["answer"])
        with patch.dict("os.environ", {}, clear=True):
            row = probe.ask_item(item(), "chat", chat_model="test",
                                 transport=lambda *args: self.fail("unexpected call"))
        self.assertIn("OPENROUTER_API_KEY", row["error"])

    def test_retries_backoff_and_hard_error(self):
        for error in (urllib.error.HTTPError("unused", 408, "timeout", {}, None),
                      urllib.error.HTTPError("unused", 429, "rate limit", {}, None),
                      urllib.error.HTTPError("unused", 503, "down", {}, None),
                      urllib.error.URLError("connection"), ConnectionResetError(), TimeoutError(),
                      http.client.IncompleteRead(b"partial")):
            calls, delays = [], []
            def transport(*args):
                calls.append(1)
                raise error
            row = probe.ask_item(item(), "jev", transport=transport, sleep=delays.append)
            self.assertIn("error", row)
            self.assertEqual(len(calls), 4)
            self.assertEqual(delays, [1, 2, 4])
        calls = []
        def eventually(*args):
            calls.append(1)
            if len(calls) < 4:
                raise ConnectionError("transient")
            return jev_response()
        self.assertNotIn("error", probe.ask_item(item(), "jev", transport=eventually, sleep=lambda _: None))
        self.assertEqual(len(calls), 4)

    def test_no_retry_on_other_http_or_invalid_probability(self):
        for code in (400, 401, 402, 403, 404):
            def transport(*args):
                raise urllib.error.HTTPError("unused", code, "secret body", {}, None)
            row = probe.ask_item(item(), "jev", transport=transport, sleep=lambda _: self.fail("retry"))
            self.assertEqual(row["error"], f"HTTP {code}")
        for p in (-0.1, 1.1, float("nan"), float("inf"), "0.5", True):
            self.assertIn("error", probe.ask_item(item(), "jev", transport=lambda *args: jev_response(p)))

    def test_secrets_never_persist_from_exceptions_or_response(self):
        key = 'fake-"-secret'
        def failure(*args):
            raise ValueError("Authorization: Bearer " + key)
        def echo(*args):
            return {"choices": [{"message": {"content": "x" * 195 + key + " yes"}}],
                    "usage": {"echo": key, "prompt_tokens": 2}}
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": key}):
            for transport in (failure, echo):
                row = probe.ask_item(item(), "chat", chat_model="test", transport=transport)
                self.assertNotIn(key, str(row))
                self.assertNotIn("Authorization", str(row))
                self.assertNotIn('fake-', row.get("raw", ""))

    def test_invalid_usage_is_an_item_error(self):
        response = jev_response()
        response["usage"] = {"input_tokens": float("nan")}
        row = probe.ask_item(item(), "jev", transport=lambda *args: response)
        self.assertIn("error", row)


class FilesAndReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.probe = self.directory / "probe.jsonl"
        self.output = self.directory / "answers.jsonl"

    def test_resume_errors_unparseable_usage_and_threads(self):
        rows = [item("already"), item("retry"), item("bad"), item("junk")]
        for row in rows:
            row["question"] = row["id"]
        write_rows(self.probe, rows)
        write_rows(self.output, [{"id": "already", "answer": True, "usage": {"prompt_tokens": 4}},
                                 {"id": "retry", "error": "earlier failure"}])
        barrier = threading.Barrier(2)
        calls = []
        def transport(url, body, headers):
            content = body["messages"][0]["content"]
            identity = content.split("\n\n")[1]
            calls.append(identity)
            if identity in ("retry", "bad"):
                barrier.wait(timeout=5)  # Proves both requests run concurrently.
            if identity == "bad":
                raise urllib.error.HTTPError("unused", 400, "bad", {}, None)
            return {"choices": [{"message": {"content": "yes" if identity == "retry" else "junk"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}}}
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-secret"}):
            summary = probe.ask(self.probe, self.output, "chat", concurrency=2,
                                chat_model="test", transport=transport)
        self.assertCountEqual(calls, ["retry", "bad", "junk"])
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["answered"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["unparseable"], 1)
        self.assertEqual(summary["usage_totals"], {"prompt_tokens": 10,
                                                 "completion_tokens_details.reasoning_tokens": 4})
        self.assertEqual(summary["usage_this_run"]["prompt_tokens"], 6)
        self.assertEqual(len(list(probe.read_jsonl(self.output))), 5)
        self.assertNotIn("fake-secret", self.output.read_text())
        self.assertNotIn("fake-secret", probe.summary_path(self.output).read_text())
        # Only the hard error is retried; a parsed null is still a completed call.
        summary = probe.ask(self.probe, self.output, "jev", transport=lambda *args: jev_response())
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["errors"], 0)
        summary = probe.ask(self.probe, self.output, "jev", transport=lambda *args: self.fail("resume called"))
        self.assertEqual(summary["skipped"], 4)
        self.assertEqual(summary["usage_this_run"], {})

    def test_resume_repairs_torn_tail_and_preserves_valid_last_line(self):
        write_rows(self.probe, [item("a"), item("b")])
        self.output.write_text('{"id":"a","answer":true}\n{"id":')
        summary = probe.ask(self.probe, self.output, "jev", transport=lambda *args: jev_response())
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(len(list(probe.read_jsonl(self.output))), 2)
        self.output.write_text('{"id":"a","answer":true}')
        probe.ask(self.probe, self.output, "jev", transport=lambda *args: jev_response())
        self.assertEqual(len(list(probe.read_jsonl(self.output))), 2)
        self.output.write_text('{broken}\n')
        with self.assertRaisesRegex(ValueError, "Invalid completed"):
            probe.ask(self.probe, self.output, "jev", transport=lambda *args: self.fail("invalid file"))

    def test_report_tiny_set_errors_missing_null_and_auc_ties(self):
        write_rows(self.probe, [item("a", gold=True), item("b", gold=True), item("c", gold=False),
                                item("d", gold=False), item("e", gold=False), item("f", gold=True),
                                item("g", kind="ahead", gold=True)])
        write_rows(self.output, [{"id": "a", "answer": True, "p_yes": 0.8},
                                 {"id": "b", "answer": False, "p_yes": 0.2},
                                 {"id": "c", "answer": False, "p_yes": 0.2},
                                 {"id": "d", "answer": None}, {"id": "e", "error": "failed"},
                                 {"id": "g", "answer": True, "p_yes": 0.9}])
        report_path = self.directory / "report.json"
        with redirect_stdout(io.StringIO()) as out:
            result = probe.report(self.probe, [("model", self.output)], report_path)
        row = result["model"]["lookup"]
        self.assertEqual(row["n"], 6)
        self.assertEqual(row["answered"], 3)
        self.assertEqual(row["accuracy"], 2 / 6)
        self.assertEqual(row["accuracy_gold_true"], 1 / 3)
        self.assertEqual(row["accuracy_gold_false"], 1 / 3)
        self.assertEqual(row["share_yes"], 1 / 3)
        self.assertEqual(row["auc"], 0.75)
        self.assertEqual(row["auc_n"], 3)
        self.assertEqual((row["errors"], row["unparseable"], row["missing"]), (1, 1, 1))
        self.assertIsNone(result["model"]["ahead"]["auc"])
        self.assertIsNone(result["model"]["step1"]["accuracy"])
        self.assertEqual(json.loads(report_path.read_text()), result)
        self.assertIn("model", out.getvalue())

    def test_auc_orderings_and_duplicate_answer_precedence(self):
        self.assertEqual(probe.rank_auc([(0.5, True), (0.5, False)]), 0.5)
        self.assertEqual(probe.rank_auc([(1, True), (0, False)]), 1)
        self.assertEqual(probe.rank_auc([(0, True), (1, False)]), 0)
        self.assertIsNone(probe.rank_auc([]))
        write_rows(self.output, [{"id": "a", "error": "failure"}, {"id": "a", "answer": True},
                                 {"id": "a", "error": "later failure"}])
        self.assertEqual(probe.answer_rows(self.output)["a"], {"id": "a", "answer": True})


if __name__ == "__main__":
    unittest.main()
