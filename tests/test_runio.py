"""The output/preflight policy shared by the scripts (decide/runio.py)."""
from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decide import runio  # noqa: E402


def parser():
    ap = argparse.ArgumentParser()
    runio.add_out(ap, "thing.json")
    return ap


def exits(*args, **kwargs):
    """Assert-friendly: the SystemExit raised by an argparse error / sys.exit,
    with its stderr noise swallowed."""
    return contextlib.redirect_stderr(io.StringIO())


class CheckOut(unittest.TestCase):
    def test_default_out_is_under_runs(self):
        args = parser().parse_args([])
        self.assertEqual(Path(args.out), runio.RUNS / "thing.json")
        self.assertFalse(args.force)

    def test_frozen_files_are_not_replaced_without_force(self):
        for frozen in (ROOT / "results" / "INDEX.md",
                       ROOT / "registry" / "mlx-community__Qwen3.5-4B-MLX-4bit" / "base" / "next_move.json",
                       ROOT / "data" / "eval" / "questions.jsonl"):
            with self.subTest(frozen.name):
                self.assertTrue(frozen.exists())
                with exits(), self.assertRaises(SystemExit) as cm:
                    runio.check_out(parser(), frozen, force=False)
                self.assertEqual(cm.exception.code, 2)
                self.assertEqual(runio.check_out(parser(), frozen, force=True), frozen)

    def test_new_files_and_other_dirs_are_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out.json"
            self.assertEqual(runio.check_out(parser(), target), target)
            self.assertTrue(target.parent.is_dir())            # parent is created
            target.write_text("{}")
            runio.check_out(parser(), target)                  # existing, but not protected
        new_in_results = ROOT / "results" / "does_not_exist_yet.json"
        self.assertFalse(new_in_results.exists())
        self.assertEqual(runio.check_out(parser(), new_in_results), new_in_results)

    def test_rel_never_leaks_the_home_directory(self):
        self.assertEqual(runio.rel(ROOT / "results" / "banking77.json"), "results/banking77.json")
        self.assertEqual(runio.rel("/somewhere/else.json"), "/somewhere/else.json")


class Preflight(unittest.TestCase):
    def test_require_key(self):
        with exits(), self.assertRaises(SystemExit):
            runio.require_key(parser(), None)
        runio.require_key(parser(), "k")

    def test_require_endpoint(self):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"ok": True, "model": "m"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            info = runio.require_endpoint(parser(), f"http://127.0.0.1:{port}")
            self.assertEqual(info["model"], "m")
        finally:
            srv.shutdown()
            srv.server_close()
        with exits(), self.assertRaises(SystemExit):           # nothing listening now
            runio.require_endpoint(parser(), f"http://127.0.0.1:{port}", timeout=1)

    def test_abort_helpers(self):
        runio.abort_if_all_failed(1, 5, "call")                # some succeeded: fine
        runio.abort_if_all_failed(0, 0, "call")                # nothing attempted: fine
        with exits(), self.assertRaises(SystemExit) as cm:
            runio.abort_if_all_failed(0, 5, "call")
        self.assertIn("nothing written", str(cm.exception))
        runio.abort_early(2, 3, "call")                        # not every call failed
        runio.abort_early(3, 4, "call")                        # past the check point
        with exits(), self.assertRaises(SystemExit):
            runio.abort_early(3, 3, "call")


if __name__ == "__main__":
    unittest.main()
