"""Renderer v2 / the one-pipeline bridge (docs/DESIGN.md, 'Renderer and head registry').

Both worlds converge on the same canonical call list before rendering:
  live proxy:  OpenAI messages[] -> calls_from_messages()
  calibration: DecisionPoint     -> calls_from_point()
then render() applies the SAME rules as the trace renderer v1
(decide/core/trace_states.py) — assistant prose is dropped, tool outcomes are
compacted to paths/commands/exits/tails/diffs. Heads fit on states from this
pipeline; live traffic renders through it too, so calibration and production
share one code path.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from decide.core.trace_states import _render_event, _repo_path_abspath


def calls_from_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """OpenAI messages -> (task_text, calls). Tool results are matched back to
    their calls by tool_call_id. Assistant prose is dropped by design (at
    router time the decision replaces the generation that would produce it)."""
    task = None
    pending: dict[str, dict] = {}
    calls: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "user" and task is None:
            c = m.get("content")
            task = c if isinstance(c, str) else \
                " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {})
                args = fn.get("arguments")
                try:
                    inp = json.loads(args) if isinstance(args, str) else (args or {})
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                call = {"tool": fn.get("name") or "", "tool_call_id": tc.get("id"),
                        "input": inp, "output": None}
                calls.append(call)
                if tc.get("id"):
                    pending[tc["id"]] = call
        elif role == "tool":
            c = m.get("content")
            out = c if isinstance(c, str) else \
                " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
            call = pending.get(m.get("tool_call_id"))
            if call is not None:
                call["output"] = out
    return task, calls


def calls_from_point(dp) -> list[dict]:
    """Bridge: a trace DecisionPoint's history -> canonical calls."""
    return [{"tool": c["tool"], "tool_call_id": None,
             "input": (c["state"].get("input") or {}),
             "output": c["state"].get("output")}
            for c in dp.history]


def infer_repo(messages: list[dict], calls: list[dict]) -> Optional[str]:
    """Repo name from file paths in the conversation (proxy sees no cwd)."""
    blob = json.dumps(messages) + json.dumps(calls)
    m = re.search(r"data/repos/([\w.\-]+)/", blob)
    return m.group(1) if m else None


def render(task_text: str, repo: str, calls: list[dict], max_actions: int = 12) -> str:
    """Compact state text (renderer v3: repo-kind tag dropped, glob hits not
    filtered by extension). Identical rules to the trace renderer
    (decide/core/trace_states.py); heads served on v2 states are refit on v3
    in the same change (docs/POLICY.md: ONE atomic change, never a cleanup
    edit)."""
    events = [{"tool": c["tool"], "state": {"input": c["input"], "output": c["output"]}}
              for c in calls]
    hist = events[-max_actions:]
    lines = [_render_event(ev, _repo_path_abspath(repo)) for ev in hist]
    body = "\n".join(f"{k + 1}. {l}" for k, l in enumerate(lines))
    if len(events) > max_actions:
        body = "(earlier actions omitted)\n" + body
    hist_s = f"Actions so far this session:\n{body}" if lines else \
        "No actions taken yet this session."
    return f"Repo: {repo}. Task: {task_text}\n{hist_s}"
