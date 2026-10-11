"""Spawned, persistent CPU environments for Predict Position."""
from collections import defaultdict
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback

from unified_game_pipeline import factory as game_factory


def _worker(pipe, factory, reuse):
    active, free = {}, defaultdict(list)
    key = "<startup>"
    try:
        while True:
            op, payload = pipe.recv()
            if op == "close":
                for key, (_, env) in list(active.items()):
                    env.close()
                    del active[key]
                for spec, envs in free.items():
                    key = f"<free:{spec}>"
                    while envs:
                        envs[-1].close()
                        envs.pop()
                pipe.send({"result": {}})
                break
            result = {}
            for key, value in payload.items():
                if op == "reset":
                    if key in active:
                        raise ValueError("Episode is already active")
                    spec = json.dumps(value["spec"], sort_keys=True, separators=(",", ":"))
                    env = free[spec].pop() if free[spec] else factory(value["spec"])
                    active[key] = (spec, env)
                    obs, info = env.reset(value["seed"])
                    result[key] = dict(observation=obs, info=info)
                elif op in {"step", "drain"}:
                    spec, env = active[key]
                    count = 0
                    while True:
                        obs, reward, terminated, truncated, info = env.step(value)
                        count += 1
                        if truncated:
                            raise RuntimeError("Environment truncated the episode")
                        if terminated or op == "step":
                            break
                    result[key] = dict(observation=obs, reward=reward, terminated=terminated,
                                       truncated=truncated, info=info)
                    if op == "drain":
                        result[key]["drain_steps"] = count
                    if terminated:
                        if reuse:
                            free[spec].append(env)
                        else:
                            env.close()
                        del active[key]
                else:
                    raise ValueError(f"Unknown operation: {op}")
            pipe.send({"result": result})
    except EOFError:
        pass
    except BaseException:
        try:
            pipe.send({"key": key, "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for key, (_, env) in list(active.items()):
            try:
                env.close()
            except BaseException:
                traceback.print_exc()
        for envs in free.values():
            for env in envs:
                try:
                    env.close()
                except BaseException:
                    traceback.print_exc()
        pipe.close()


class EnvPool:
    def __init__(self, workers, reuse=True, factory=game_factory, timeout=300):
        if workers < 1 or timeout <= 0:
            raise ValueError("workers and timeout must be positive")
        self.timeout, self.owners, self.cursor = timeout, {}, 0
        self.processes, self.pipes, self.closed = [], [], False
        ctx = mp.get_context("spawn")
        try:
            for _ in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, factory, reuse), daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.pipes.append(parent)
        except BaseException:
            self._abort()
            raise

    def _abort(self):
        self.closed = True
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for pipe in self.pipes:
            pipe.close()
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)

    def _call(self, op, payload):
        if self.closed:
            raise RuntimeError("EnvPool is closed")
        batches = defaultdict(dict)
        for key, value in payload.items():
            batches[self.owners[key]][key] = value
        return self._exchange(op, batches)

    def _exchange(self, op, batches, timeout=None):
        pending, result = {}, {}
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        try:
            for worker, batch in batches.items():
                pipe = self.pipes[worker]
                if not self.processes[worker].is_alive():
                    raise RuntimeError(f"Dead worker for keys {list(batch)}")
                try:
                    pipe.send((op, batch))
                except (BrokenPipeError, EOFError, OSError) as exc:
                    raise RuntimeError(f"Dead worker for keys {list(batch)}") from exc
                pending[pipe] = worker
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    keys = [key for worker in pending.values() for key in batches[worker]]
                    raise TimeoutError(f"Worker timeout during {op} for keys {keys}")
                for pipe in wait(list(pending), timeout=min(remaining, .1)):
                    worker = pending.pop(pipe)
                    try:
                        reply = pipe.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f"Dead worker for keys {list(batches[worker])}") from exc
                    if "traceback" in reply:
                        raise RuntimeError(f"Worker error for key {reply['key']}:\n{reply['traceback']}")
                    result.update(reply["result"])
                for worker in pending.values():
                    if not self.processes[worker].is_alive() and not self.pipes[worker].poll():
                        raise RuntimeError(f"Dead worker for keys {list(batches[worker])}")
            for key, transition in result.items():
                if transition.get("terminated"):
                    del self.owners[key]
            return result
        except BaseException:
            self._abort()
            raise

    def reset(self, jobs):
        keys = [job["key"] for job in jobs]
        if len(set(keys)) != len(keys) or any(key in self.owners for key in keys):
            raise ValueError("Duplicate or active episode key")
        for key in keys:
            self.owners[key] = self.cursor % len(self.pipes)
            self.cursor += 1
        return self._call("reset", {job["key"]: job["case"] for job in jobs})

    def step(self, actions):
        return self._call("step", actions)

    def drain(self, keys, action="noop"):
        return self._call("drain", dict.fromkeys(keys, action))

    def close(self):
        if self.closed:
            return
        try:
            self._exchange("close", {i: {} for i in range(len(self.pipes))}, timeout=min(self.timeout, 5))
            deadline = time.monotonic() + min(self.timeout, 5)
            for process in self.processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
        finally:
            self._abort()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
