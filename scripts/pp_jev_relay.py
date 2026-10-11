#!/usr/bin/env python3
"""Loopback relay for Jev (OpenRouter System One) calls made from a remote GPU box.

The API key is read from the macOS Keychain at start and stays in this process: the remote side
reaches the relay through an SSH reverse tunnel and sends no credentials. Only allowed models are
forwarded, and the relay stops forwarding once the estimated spend passes the cap.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://openrouter.ai/api/v1/systemone"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


class Relay:
    def __init__(self, args):
        user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
        self.key = subprocess.run(
            ["security", "find-generic-password", "-a", user, "-s", "OPENROUTER_API_KEY", "-w"],
            capture_output=True, text=True, check=True).stdout.strip()
        if not self.key:
            raise SystemExit("No OpenRouter key in the Keychain")
        self.models = set(args.models.split(","))
        self.usd_per_token = args.usd_per_million / 1e6
        self.max_usd = args.max_usd
        self.lock = threading.Lock()
        self.calls = self.errors = self.tokens = 0
        self.log = open(args.log, "a", buffering=1)

    def spent(self):
        return self.tokens * self.usd_per_token

    def forward(self, body):
        request = json.loads(body)
        if request.get("model") not in self.models:
            return 403, json.dumps({"error": "model not allowed by the relay"}).encode()
        if self.spent() >= self.max_usd:
            return 402, json.dumps({"error": "relay spend cap reached"}).encode()
        status, payload = 599, b""
        for attempt in range(6):
            began = time.time()
            upstream = urllib.request.Request(UPSTREAM, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(upstream, timeout=60) as response:
                    status, payload = response.status, response.read()
            except urllib.error.HTTPError as exc:
                status, payload = exc.code, exc.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                status, payload = 599, json.dumps({"error": type(exc).__name__}).encode()
            tokens = 0
            if status == 200:
                try:
                    usage = json.loads(payload).get("usage") or {}
                    tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                except (ValueError, TypeError, AttributeError):
                    tokens = 0
                if not tokens:
                    tokens = len(body) // 3  # usage missing: charge a generous byte-based estimate
            with self.lock:
                self.calls += 1
                self.errors += status != 200
                self.tokens += tokens
                self.log.write(json.dumps({"time": round(began, 2), "status": status, "attempt": attempt,
                                           "request_bytes": len(body), "tokens": tokens,
                                           "seconds": round(time.time() - began, 3), "calls": self.calls,
                                           "errors": self.errors, "usd": round(self.spent(), 5)}) + "\n")
            if status == 200 or status not in RETRY_STATUS | {599}:
                break
            time.sleep(min(2 ** attempt, 30))
        return status, payload


def handler_for(relay):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                status, payload = relay.forward(body)
            except ValueError:
                status, payload = 400, b'{"error":"invalid JSON"}'
            self.send_response(status if status != 599 else 502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            payload = json.dumps({"calls": relay.calls, "errors": relay.errors, "tokens": relay.tokens,
                                  "usd": round(relay.spent(), 5), "max_usd": relay.max_usd}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--models", default="typesafe/jev-1.13")
    parser.add_argument("--usd-per-million", type=float, default=0.042)
    parser.add_argument("--max-usd", type=float, default=3.0)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    relay = Relay(args)
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(relay))
    print(json.dumps({"listening": args.port, "models": sorted(relay.models), "max_usd": args.max_usd}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
