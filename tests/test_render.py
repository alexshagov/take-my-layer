"""The state text the head reads must not depend on where the proxy was started."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SNIPPET = """
import sys
sys.path.insert(0, {root!r})
from decide.core.state_feed import render
calls = [{{"tool": "read", "tool_call_id": "c1",
          "input": {{"filePath": {path!r}}}, "output": "x = 1\\n"}}]
print(render("Fix it", "qjev-sandbox", calls))
"""


class RendererIsCwdIndependent(unittest.TestCase):
    def render_from(self, cwd):
        path = str(ROOT / "data" / "repos" / "qjev-sandbox" / "greeter" / "__init__.py")
        out = subprocess.run([sys.executable, "-c", SNIPPET.format(root=str(ROOT), path=path)],
                             cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        return out.stdout

    def test_repo_paths_are_stripped_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            from_elsewhere = self.render_from(elsewhere)
        from_root = self.render_from(ROOT)
        self.assertIn("1. read greeter/__init__.py (1 lines)", from_elsewhere)
        self.assertEqual(from_elsewhere, from_root)


if __name__ == "__main__":
    unittest.main()
