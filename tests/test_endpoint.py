"""Endpoint contract tests that need no model and no GPU.

parse_request and the HTTP handler are exercised with the real server class on
an ephemeral port and an empty decider table, so everything that answers
before a readout (validation, limits, auth, routes, healthz) is covered.
Readout results themselves are checked by the benchmarks, not here.
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decide.serve import systemone as S  # noqa: E402

CHOICE = {"type": "Choice", "instructions": "Which one?", "options": ["a", "b", "c"]}
NOUL = {"type": "Noul", "instructions": "Is it true?"}
SCORE = {"type": "Score", "instructions": "How bad?", "levels": ["low", "mid", "high"],
         "criteria": "Judge the impact."}


def parse(body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return S.parse_request(raw)


class ParseRequest(unittest.TestCase):
    def test_all_three_types_parse(self):
        status, (state, parsed) = parse({"state": "s", "questions": {
            "c": CHOICE, "n": NOUL, "s": SCORE}})
        self.assertEqual(status, 200)
        self.assertEqual(state, "s")
        self.assertEqual({k: t for k, (t, _) in parsed.items()},
                         {"c": "Choice", "n": "Noul", "s": "Score"})

    def test_object_and_array_state_are_accepted(self):
        for state in ({"ticket": 1}, ["a", "b"]):
            self.assertEqual(parse({"state": state, "questions": {"n": NOUL}})[0], 200)

    def test_bad_requests_are_400_with_a_message(self):
        cases = {
            "invalid json": b"{nope",
            "not an object": b"[1]",
            "no state": {"questions": {"n": NOUL}},
            "state is a number": {"state": 5, "questions": {"n": NOUL}},
            "no questions": {"state": "s", "questions": {}},
            "unknown type": {"state": "s", "questions": {"q": {"type": "Rank"}}},
            "one option": {"state": "s", "questions": {"q": {"type": "Choice", "options": ["a"]}}},
            "duplicate options": {"state": "s", "questions": {"q": {"type": "Choice", "options": ["a", "a"]}}},
            "27 options": {"state": "s", "questions": {"q": {"type": "Choice", "options": [f"o{i}" for i in range(27)]}}},
            "11 levels": {"state": "s", "questions": {"q": {"type": "Score", "levels": [f"l{i}" for i in range(11)]}}},
            "bad name": {"state": "s", "questions": {"bad name": NOUL}},
        }
        for name, body in cases.items():
            with self.subTest(name):
                status, payload = parse(body)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["type"], "invalid_request")
                self.assertTrue(payload["error"]["message"])

    def test_limits(self):
        # the documented contract (docs/API.md): 8 questions, 32k state chars —
        # pin the constants so a silent regression cannot pass the suite
        self.assertEqual(S.MAX_QUESTIONS, 8)
        self.assertEqual(S.MAX_STATE_CHARS, 32000)
        nine = {f"q{i}": NOUL for i in range(S.MAX_QUESTIONS + 1)}
        self.assertEqual(parse({"state": "s", "questions": nine})[0], 400)
        eight = {f"q{i}": NOUL for i in range(S.MAX_QUESTIONS)}
        self.assertEqual(parse({"state": "s", "questions": eight})[0], 200)
        long_text = {"state": "s", "questions": {"q": {**NOUL, "instructions": "i" * (S.MAX_INSTRUCTIONS + 1)}}}
        self.assertEqual(parse(long_text)[0], 400)

    def test_state_length_cap(self):
        ok = {"state": "x" * S.MAX_STATE_CHARS, "questions": {"n": NOUL}}
        over = {"state": "x" * (S.MAX_STATE_CHARS + 1), "questions": {"n": NOUL}}
        self.assertEqual(parse(ok)[0], 200)
        status, payload = parse(over)
        self.assertEqual(status, 400)
        self.assertIn(str(S.MAX_STATE_CHARS), payload["error"]["message"])


class HandlerWithoutModel(unittest.TestCase):
    """The real Handler on an ephemeral port; D is empty, so nothing that
    reaches the readout is exercised."""

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def request(self, method, path, body=None, headers=None, content_length=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest(method, path)
        for k, v in (headers or {}).items():
            c.putheader(k, v)
        if body is not None or content_length is not None:
            c.putheader("Content-Length", str(len(body) if content_length is None else content_length))
        c.endheaders()
        if body:
            c.send(body)
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        c.close()
        return r.status, data, hdrs

    def test_healthz_reports_the_limits(self):
        status, data, _ = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["limits"], {"max_questions": S.MAX_QUESTIONS,
                                          "max_state_chars": S.MAX_STATE_CHARS,
                                          "max_body_bytes": S.MAX_BODY_BYTES})

    def test_get_on_the_endpoint_is_405_with_allow(self):
        status, data, hdrs = self.request("GET", "/v1/systemone")
        self.assertEqual(status, 405)
        self.assertEqual(hdrs.get("allow"), "POST")
        self.assertEqual(data["error"]["type"], "method_not_allowed")

    def test_unknown_paths_are_404(self):
        self.assertEqual(self.request("GET", "/")[0], 404)
        self.assertEqual(self.request("POST", "/other", b"{}")[0], 404)

    def test_oversize_body_is_413_and_not_read(self):
        status, data, hdrs = self.request("POST", "/v1/systemone",
                                          content_length=S.MAX_BODY_BYTES + 1)
        self.assertEqual(status, 413)
        self.assertEqual(data["error"]["type"], "payload_too_large")
        self.assertEqual(hdrs.get("connection"), "close")

    def test_bad_content_length_is_400(self):
        for value in ("abc", "-5"):
            with self.subTest(value):
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                c.putrequest("POST", "/v1/systemone")
                c.putheader("Content-Length", value)
                c.endheaders()
                r = c.getresponse()
                self.assertEqual(r.status, 400)
                r.read()
                c.close()

    def test_validation_errors_come_back_as_json_400(self):
        status, data, _ = self.request("POST", "/v1/systemone", b'{"questions": {}}')
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "invalid_request")

    def test_bearer_auth(self):
        S.API_KEY = "secret"
        try:
            body = json.dumps({"state": "s", "questions": {}}).encode()
            self.assertEqual(self.request("POST", "/v1/systemone", body)[0], 401)
            self.assertEqual(self.request("POST", "/v1/systemone", body,
                                          {"Authorization": "Bearer wrong"})[0], 401)
            # right key passes auth and reaches validation (400 for the empty map)
            self.assertEqual(self.request("POST", "/v1/systemone", body,
                                          {"Authorization": "Bearer secret"})[0], 400)
            self.assertEqual(self.request("GET", "/healthz")[0], 200)     # healthz needs no key
        finally:
            S.API_KEY = None


if __name__ == "__main__":
    unittest.main()
