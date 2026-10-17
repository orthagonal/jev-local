import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from serve_jevlike import Bridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def predictor(context, options):
            self.calls.append((context, options))
            return [0.25, 0.75]

        self.server = Bridge(("127.0.0.1", 0), predictor, "trained-checkpoint", "bridge-secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, body=None, token="bridge-secret"):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.base + path, data=data, headers=headers)
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    def test_reuses_one_predictor_and_preserves_option_order(self):
        for _ in range(2):
            status, body = self.request("/predict", {"context": "Pick B", "options": ["A", "B"], "model": "jevlike"})
            self.assertEqual(status, 200)
            self.assertEqual(body, {"model": "trained-checkpoint", "probabilities": [0.25, 0.75]})
        self.assertEqual(self.calls, [("Pick B", ["A", "B"])] * 2)

    def test_health_is_public_and_models_are_authenticated(self):
        self.assertEqual(self.request("/health", token=None)[0], 200)
        self.assertEqual(self.request("/v1/models", token=None)[0], 401)
        status, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "trained-checkpoint")

    def test_rejects_bad_credentials_and_payloads_before_inference(self):
        payload = {"context": "Pick", "options": ["A", "B"]}
        self.assertEqual(self.request("/predict", payload, token="wrong")[0], 401)
        for body in [[], {"context": "Pick", "options": ["A"]}, {"context": "Pick", "options": [1, 2]}, dict(payload, model="unknown")]:
            self.assertEqual(self.request("/predict", body)[0], 422)
        self.assertEqual(self.calls, [])

    def test_invalid_model_probabilities_are_rejected(self):
        self.server.predictor = lambda _context, _options: [0.8, 0.8]
        status, body = self.request("/predict", {"context": "Pick", "options": ["A", "B"]})
        self.assertEqual(status, 500)
        self.assertNotIn("bridge-secret", str(body))


if __name__ == "__main__":
    unittest.main()
