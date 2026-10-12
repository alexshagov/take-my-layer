"""The README stays a short front page, and the docs stay navigable.

- every relative markdown link in the repo's own docs resolves to a file, and
  every `#anchor` resolves to a heading in the target file;
- the README links every file in docs/ (so the index cannot silently rot);
- the README stays short: the long text lives in docs/;
- docs/RUNNING.md mentions every script, so the command list cannot drift.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(p for p in
              [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "AGENTS.md",
               ROOT / "THIRD_PARTY.md", ROOT / "registry" / "README.md",
               ROOT / "results" / "INDEX.md", *(ROOT / "docs").glob("*.md")]
              if p.exists())
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
README_MAX_LINES = 150
SCRIPT_DIRS = ("adapt", "authoring", "bench", "research", "serve")


def strip_code(text: str) -> str:
    return re.sub(r"(?s)```.*?```", "", text)


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation and symbols, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors_of(md: Path) -> set[str]:
    seen: dict[str, int] = {}
    out: set[str] = set()
    for m in re.finditer(r"(?m)^#{1,6} (.+?)\s*#*$", strip_code(md.read_text())):
        base = slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def relative_links(md: Path):
    """Yield (file, anchor) for every non-external link: file is None for an in-page anchor."""
    for target in LINK.findall(strip_code(md.read_text())):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue                                                # external
        file_part, _, anchor = target.partition("#")
        yield (file_part or None), anchor


class Links(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self):
        for md in DOCS:
            for file_part, anchor in relative_links(md):
                target = (md.parent / file_part) if file_part else md
                with self.subTest(f"{md.relative_to(ROOT)} -> {file_part or ''}#{anchor}"):
                    self.assertTrue(target.exists())
                    if anchor and target.suffix == ".md":
                        self.assertIn(anchor, anchors_of(target))

    def test_readme_links_every_doc(self):
        linked = {(ROOT / f).resolve() for f, _ in relative_links(ROOT / "README.md") if f}
        for doc in sorted((ROOT / "docs").glob("*.md")):
            with self.subTest(doc.name):
                self.assertIn(doc.resolve(), linked, f"{doc.name} is missing from the README index")

    def test_the_results_contents_list_is_there(self):
        # RESULTS.md is 700 lines: the contents block must link every H2 section
        text = (ROOT / "docs" / "RESULTS.md").read_text()
        h2 = [slug(m.group(1)) for m in re.finditer(r"(?m)^## (.+)$", strip_code(text))]
        linked = {a for f, a in relative_links(ROOT / "docs" / "RESULTS.md") if f is None}
        for anchor in h2:
            with self.subTest(anchor[:50]):
                self.assertIn(anchor, linked)


URL = re.compile(r"https?://\S+")
MD_NAME = re.compile(r"(?<![*\w./-])[\w./-]*[\w-]\.md\b")   # not the tail of a glob like *.sheet.md
FOREIGN = ("mlx_lm/",)                                          # files that live in other packages
SKIP_DIRS = {".venv", ".hf", ".direnv", ".cache", ".uv-cache", "archive", "runs", "logs", "__pycache__"}
# wording that points a stranger at material they cannot reach
LEAKS = re.compile(
    r"/Users/\w+|/home/\w+"
    r"|\bprivate (?:single-domain |production |reference )?(?:prototype|decision-server|server|seed)\b"
    r"|\bperused in full\b|\bresearch_log\b", re.I)


class Hygiene(unittest.TestCase):
    """The docs may not cite files that do not exist, or leak where they came from."""

    def test_named_markdown_files_exist(self):
        existing = {p.name for p in ROOT.rglob("*.md") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)}
        for md in DOCS:
            text = URL.sub("", strip_code(md.read_text()))
            cited = {m.group(0) for m in MD_NAME.finditer(text) if not m.group(0).startswith(FOREIGN)}
            for name in sorted({c.rsplit("/", 1)[-1] for c in cited}):
                with self.subTest(f"{md.relative_to(ROOT)} names {name}"):
                    self.assertIn(name, existing, f"{md.name} cites {name}, which is not in the repository")

    def test_no_private_provenance_or_local_paths(self):
        for md in DOCS:
            for m in LEAKS.finditer(md.read_text()):
                with self.subTest(f"{md.relative_to(ROOT)}: {m.group(0)}"):
                    self.fail(f"{md.name} contains {m.group(0)!r}")


class Readme(unittest.TestCase):
    def test_readme_is_short(self):
        n = len((ROOT / "README.md").read_text().splitlines())
        self.assertLessEqual(n, README_MAX_LINES, f"README is {n} lines; move the long text to docs/")


class RunningGuide(unittest.TestCase):
    def test_every_script_is_mentioned(self):
        text = (ROOT / "docs" / "RUNNING.md").read_text()
        for d in SCRIPT_DIRS:
            for script in sorted((ROOT / "decide" / d).glob("*.py")):
                with self.subTest(script.name):
                    self.assertIn(script.name, text, f"docs/RUNNING.md never mentions {script.name}")


if __name__ == "__main__":
    unittest.main()
