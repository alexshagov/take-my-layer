"""Traces -> decision points + compact state rendering.

See docs/DESIGN.md, 'Feasibility on the prototype's data'.

A "decision point" is the moment just before a tool call: the state the agent
saw (task + action history so far) and the tool it actually called next (the
label). Reused later by L2 head fitting — keep rendering deterministic.

Fidelity notes (a documented lower bound, same DESIGN.md section):
- Full tool outputs existed in the agent's context (file bodies, spec runs);
  we render compact summaries: paths, commands, exit codes, tails, diffs.
  A production renderer with real file access would only have MORE signal.
- Assistant `text` events are deliberately EXCLUDED: at router time the
  decision replaces the generation that would produce that preamble, so the
  state must not contain it.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Anchored to the repo root, not the cwd: _repo_path_abspath() decides which
# path prefix the renderer strips from tool paths, so a cwd-relative DATA made
# the state text (and so the head's input) depend on where the proxy started.
DATA = str(Path(__file__).resolve().parents[2] / "data")
TOOLS = ["read", "bash", "grep", "edit", "glob", "write"]  # webfetch (n=2) excluded

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean(s: str) -> str:
    return _ANSI.sub("", s).replace("\r", "")


def _rel(repo_path: str, p: str) -> str:
    return os.path.relpath(p, repo_path) if p.startswith(repo_path) else p


@dataclass
class DecisionPoint:
    task_id: str
    repo: str
    task_text: str
    history: list          # completed tool events (dicts) before the decision
    label: str             # the tool actually called next
    i: int                 # index of the decision (0-based) within the trace
    n: int                 # total calls in the trace

    @property
    def third(self) -> str:
        if self.i == 0:
            return "early"
        if self.i == self.n - 1:
            return "late"
        return "mid"


def _render_event(ev: dict, repo_path: str) -> str:
    tool = ev["tool"]
    st = ev.get("state") or {}
    inp = st.get("input") or {}
    out = _clean(str(st.get("output") or ""))
    if tool == "read":
        n = out.count("\n")
        return f"read {_rel(repo_path, inp.get('filePath', '?'))} ({n} lines)"
    if tool == "bash":
        cmd = inp.get("command", "?")
        exit_code = (st.get("metadata") or {}).get("exit")
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        tail_s = " | ".join(l.strip()[:90] for l in tail)
        return f"bash `{cmd}` -> exit {exit_code}; {tail_s}"[:400]
    if tool == "edit":
        old = str(inp.get("oldString", ""))[:60].replace("\n", "\\n")
        new = str(inp.get("newString", ""))[:60].replace("\n", "\\n")
        return f"edit {_rel(repo_path, inp.get('filePath', '?'))}: `{old}` -> `{new}` (applied)"
    if tool == "grep":
        pat = inp.get("pattern", "?")
        paths = [l.split(":")[0] for l in out.splitlines() if re.match(r"^\S+:\d+", l)][:4]
        hits = ", ".join(_rel(repo_path, p) for p in paths) if paths else "no matches"
        return f"grep `{pat}` -> {hits}"[:300]
    if tool == "glob":
        pat = inp.get("pattern", "?")
        paths = [l.strip() for l in out.splitlines() if l.strip().endswith(
            (".rb", ".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java",
             ".c", ".h", ".cpp", ".tf", ".yaml", ".yml", ".toml", ".json"))][:4]
        hits = ", ".join(_rel(repo_path, p) for p in paths) if paths else "no matches"
        return f"glob `{pat}` -> {hits}"[:300]
    if tool == "write":
        return f"write {_rel(repo_path, inp.get('filePath', '?'))}"
    return f"{tool} {json.dumps(inp)[:120]}"


def render_state(dp: DecisionPoint, max_actions: int = 12) -> str:
    hist = dp.history[-max_actions:]
    lines = [_render_event(ev, _repo_path(dp)) for ev in hist]
    body = "\n".join(f"{k+1}. {l}" for k, l in enumerate(lines))
    if len(dp.history) > max_actions:
        body = "(earlier actions omitted)\n" + body
    hist_s = f"Actions so far this session:\n{body}" if lines else \
        "No actions taken yet this session."
    return f"Repo: {dp.repo}. Task: {dp.task_text}\n{hist_s}"


def _repo_path(dp: DecisionPoint) -> str:
    return _repo_path_abspath(dp.repo)


def _repo_path_abspath(repo: str) -> str:
    return os.path.abspath(f"{DATA}/repos/{repo}")


def load_decision_points() -> list[DecisionPoint]:
    if not os.path.exists(f"{DATA}/tasks.json"):
        raise FileNotFoundError(
            "the gold-seed dataset (data/traces/ + data/tasks.json, 595 "
            "recorded work sessions) is NOT shipped in this repo — "
            "removed for the GitHub release. The "
            "registry heads fit from it ship as artifacts; to refit, "
            "regenerate a corpus via decide/adapt/synth_states.py and/or the "
            "observe() telemetry loop (see docs/POLICY.md §3).")
    tasks = {t["id"]: t for t in json.load(open(f"{DATA}/tasks.json"))}
    points: list[DecisionPoint] = []
    for f in sorted(glob.glob(f"{DATA}/traces/*.jsonl")):
        tid = os.path.basename(f)[:-6]
        task = tasks.get(tid)
        if not task:
            continue
        calls = []
        for line in open(f):
            ev = json.loads(line)
            if ev["type"] != "tool_use":
                continue
            st = ev["part"].get("state") or {}
            if not st.get("input"):        # pending/incomplete events
                continue
            calls.append({"tool": ev["part"]["tool"], "state": st})
        n = len(calls)
        for i, ev in enumerate(calls):
            label = ev["tool"]
            if label not in TOOLS:
                continue
            points.append(DecisionPoint(
                task_id=tid, repo=task["repo"], task_text=task["task"],
                history=calls[:i], label=label, i=i, n=n))
    return points


if __name__ == "__main__":
    pts = load_decision_points()
    from collections import Counter
    print(f"{len(pts)} decision points")
    print("by third:", dict(Counter(p.third for p in pts)))
    print("labels:", dict(Counter(p.label for p in pts)))
    print("\n--- sample late state ---")
    late = [p for p in pts if p.third == "late" and 4 <= p.n <= 8][0]
    print(render_state(late)[:900])
    print(f"\n[label={late.label}]")
