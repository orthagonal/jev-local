"""Persistent HTTP bridge for vinnylarouge/jevlike's native option scorer.

Run inside an environment with the upstream jevlike package installed. Model
dependencies are imported only when starting real inference, so protocol tests
can use a small injected predictor without PyTorch or a checkpoint.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


Predictor = Callable[[str, list[str]], Sequence[float]]


def load_predictor(checkpoint: str, device_name: str) -> Predictor:
    import torch
    from jevlike.data import ChoiceExample
    from jevlike.model import load_checkpoint, select_device
    from jevlike.train import move

    device = select_device(device_name)
    model, collator, _ = load_checkpoint(checkpoint, device)
    model.eval()

    def predict(context: str, options: list[str]) -> list[float]:
        batch = move(collator([ChoiceExample(context, tuple(options), 0)]), device)
        with torch.inference_mode():
            return model(batch).softmax(-1)[0, : len(options)].cpu().tolist()

    return predict


class Bridge(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        predictor: Predictor,
        model_id: str = "jevlike",
        api_key: str | None = None,
    ) -> None:
        super().__init__(address, Handler)
        self.predictor = predictor
        self.model_id = model_id
        self.api_key = api_key
        self.model_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: Bridge

    def log_message(self, _format: str, *args: object) -> None:
        # Avoid logging request data or credentials.
        pass

    def respond(self, status: int, body: object) -> None:
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def error(self, status: int, message: str) -> None:
        self.respond(status, {"error": {"message": message}})

    def authorized(self) -> bool:
        key = self.server.api_key
        if key is None:
            return True
        actual = self.headers.get("Authorization", "").encode("utf-8")
        if hmac.compare_digest(actual, f"Bearer {key}".encode("utf-8")):
            return True
        self.error(401, "Invalid bearer token")
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            self.respond(200, {"status": "ok"})
        elif not self.authorized():
            return
        elif self.path == "/v1/models":
            self.respond(200, {"data": [{"id": self.server.model_id, "name": "jevlike",
                "description": "Native option scorer served by jev-local", "release_date": None}]})
        else:
            self.error(404, "Unknown route")

    def do_POST(self) -> None:
        if not self.authorized():
            return
        if self.path != "/predict":
            self.error(404, "Unknown route")
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.error(415, "Expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= 1024 * 1024:
            self.error(413 if length > 1024 * 1024 else 411, "Expected a JSON body of at most 1 MiB")
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError):
            self.error(400, "Invalid JSON")
            return
        if not isinstance(body, dict):
            self.error(422, "Expected an object")
            return
        context, options = body.get("context"), body.get("options")
        if not isinstance(context, str) or not isinstance(options, list) or not 2 <= len(options) <= 255 or any(not isinstance(option, str) or not option for option in options):
            self.error(422, "Expected context text and 2..255 nonempty option strings")
            return
        if body.get("model") not in (None, "jevlike", self.server.model_id):
            self.error(422, "Unknown model selector")
            return
        try:
            with self.server.model_lock:
                probabilities = list(self.server.predictor(context, options))
            if len(probabilities) != len(options) or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError("Invalid predictor output")
            total = sum(probabilities)
            if total <= 0 or abs(total - 1) > 0.02:
                raise ValueError("Invalid probability total")
            probabilities = [p / total for p in probabilities]
        except Exception:
            self.error(500, "Native model inference failed")
            return
        self.respond(200, {"model": self.server.model_id, "probabilities": probabilities})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="jevlike")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--api-key-env", default="JEVLIKE_API_KEY")
    args = parser.parse_args()
    predictor = load_predictor(args.checkpoint, args.device)
    server = Bridge((args.host, args.port), predictor, args.model_id, os.environ.get(args.api_key_env) or None)
    print(f"jevlike bridge listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
