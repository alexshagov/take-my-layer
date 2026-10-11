"""Smoke tests for every script: no model, no GPU, no network.

- every script answers --help from another working directory (so the repo root
  is found without being the cwd) and writes nothing;
- the two authoring scripts regenerate the shipped eval sets bit for bit;
- pool_evals regenerates results/jev_compare.pooled.json byte for byte from
  the frozen inputs;
- shadow_stats reproduces the shadow-log numbers quoted in the docs.
"""
from __future__ import annotations

import compileall
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for d in ("adapt", "authoring", "bench", "research", "serve")
                 for p in (ROOT / "decide" / d).glob("*.py"))
WATCHED = ("results", "registry", "data", "docs")


def _have_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:                                   # not installed / not importable here
        return False


HAVE_MLX = _have_mlx()


def tree_state():
    return {str(p): p.stat().st_mtime_ns
            for d in WATCHED for p in (ROOT / d).rglob("*") if p.is_file()}


def run(args, cwd):
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


class Compile(unittest.TestCase):
    def test_decide_compiles(self):
        self.assertTrue(compileall.compile_dir(str(ROOT / "decide"), quiet=1))


class HelpWritesNothing(unittest.TestCase):
    def test_help_from_another_directory(self):
        before = tree_state()
        with tempfile.TemporaryDirectory() as elsewhere:
            for script in SCRIPTS:
                with self.subTest(script.name):
                    r = run([script, "--help"], elsewhere)
                    if r.returncode != 0 and not HAVE_MLX and "mlx" in r.stderr.lower():
                        self.skipTest("imports MLX at module level; MLX is not installed here")
                    self.assertEqual(r.returncode, 0, r.stderr[-400:])
                    self.assertIn("usage", r.stdout.lower())
        self.assertEqual(tree_state(), before, "a --help run changed a tracked directory")


class Regeneration(unittest.TestCase):
    def test_eval_sets_regenerate_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            for name in ("author_v2", "author_long"):
                with self.subTest(name):
                    r = run([ROOT / "decide" / "authoring" / f"{name}.py", "--check"], elsewhere)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("bit-for-bit", r.stdout)

    def test_pooled_report_regenerates_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pooled.json"
            r = run([ROOT / "decide" / "bench" / "pool_evals.py", "--out", out], tmp)
            self.assertEqual(r.returncode, 0, r.stderr[-400:])
            self.assertEqual(out.read_bytes(), (ROOT / "results" / "jev_compare.pooled.json").read_bytes())

    def test_shadow_stats_reproduces_the_quoted_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run([ROOT / "decide" / "bench" / "shadow_stats.py"], tmp)
            self.assertEqual(r.returncode, 0, r.stderr[-400:])
            self.assertIn("rows 81 (labeled 73)", r.stdout)
            self.assertIn("ECE 0.238", r.stdout)
            self.assertIn("coverage 0.59 (43/73), routed error 0.372", r.stdout)


class FrozenEvidenceGuard(unittest.TestCase):
    def test_pool_evals_refuses_to_replace_a_frozen_file(self):
        frozen = ROOT / "results" / "jev_compare.pooled.json"
        before = frozen.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            r = run([ROOT / "decide" / "bench" / "pool_evals.py", "--out", frozen], tmp)
        self.assertEqual(r.returncode, 2)
        self.assertIn("shipped artifact", r.stderr)
        self.assertEqual(frozen.read_bytes(), before)

    def test_http_benchmarks_fail_fast_without_an_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, extra in (("bench_banking77", ["--no-jev"]),
                                ("bench_typed_decisions", []),
                                ("eval_vs_jev", ["--no-jev"])):
                with self.subTest(name):
                    r = run([ROOT / "decide" / "bench" / f"{name}.py", "--local-url",
                             "http://127.0.0.1:1", *extra, "--out", Path(tmp) / "x.json"], tmp)
                    self.assertEqual(r.returncode, 2)
                    self.assertIn("not reachable", r.stderr)
                    self.assertFalse((Path(tmp) / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
