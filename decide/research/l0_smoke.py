"""L0 zero-label smoke (docs/DECIDE.md §5.2).

Runs the AnyJev pipeline (raw + L0) through the MLX backend against the local
Qwen3.5-4B on hand-written coding-agent states, prints distributions, and
measures prefill throughput (docs/DECIDE.md §6 assumption: ~500–1000 tok/s local
prefill — first local number).

Run:  nix develop -c bash -c 'python decide/research/l0_smoke.py'
(prints only; writes no files)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anyjev import Decider, Question

from decide.core.mlx_backend import MLXBackend

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

TOOLS = ["read", "bash", "grep", "edit", "glob", "write"]

STATES = {
    "fresh-task-inspect": (
        "Repo: demo-lib. Task: fix the failing test in "
        "tests/test_validators.py:35 — expected [1, 2] to match [2, 1] "
        "for the `order` option. No files inspected yet this session. "
        "The CI log points at src/demo_lib/core.py."
    ),
    "fault-located": (
        "Repo: demo-lib. Task: fix the failing test in "
        "tests/test_validators.py:35. Already inspected: read "
        "tests/test_validators.py (the test asserts first=1, second=2), "
        "grep for 'assign_order' found the handling in "
        "src/demo_lib/core.py:88 (second assigned before first — "
        "order is the bug). Next: apply the one-line swap in "
        "src/demo_lib/core.py."
    ),
    "edit-applied": (
        "Repo: demo-lib. Task: fix the failing test in "
        "tests/test_validators.py:35. Already done: read the test, located "
        "the reversed assignment in src/demo_lib/core.py:88, applied the "
        "swap edit. The working tree has exactly one modified file."
    ),
}

QUESTION = Question.choice(
    "Which tool should the agent call next?",
    TOOLS,
    name="next_tool",
)


class TimedMLXBackend(MLXBackend):
    """Counts prompts, prompt tokens and wall time through the scoring path."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_prompts = 0
        self.n_tokens = 0
        self.seconds = 0.0

    def next_token_logprobs(self, prompts, token_ids):
        for p in prompts:
            self.n_prompts += 1
            self.n_tokens += len(self.tokenizer.encode(p, add_special_tokens=False))
        t0 = time.perf_counter()
        out = super().next_token_logprobs(prompts, token_ids)
        self.seconds += time.perf_counter() - t0
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    print(f"loading {args.model} ...")
    t0 = time.perf_counter()
    backend = TimedMLXBackend(args.model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s "
          f"({backend.n_layers} blocks, hidden {backend.hidden_size})")

    for level in ("raw", "L0"):
        be = TimedMLXBackend.__new__(TimedMLXBackend)  # reuse loaded weights
        be.__dict__.update(backend.__dict__)
        be.n_prompts = be.n_tokens = 0
        be.seconds = 0.0
        d = Decider(be, level=level)
        print(f"\n=== level {level} ===")
        for name, state in STATES.items():
            res = d.decide(state, [QUESTION])
            dec = res["next_tool"]
            dist = "  ".join(f"{k}={v:.2f}" for k, v in
                             sorted(dec.distribution.items(), key=lambda kv: -kv[1]))
            print(f"[{name}] argmax={dec.argmax!r} conf={dec.confidence:.2f}")
            print(f"    {dist}")
        if be.n_prompts:
            print(f"    -- {be.n_prompts} prompts, {be.n_tokens} tokens, "
                  f"{be.seconds:.2f}s -> {be.n_tokens / be.seconds:.0f} tok/s prefill, "
                  f"{be.seconds / be.n_prompts * 1000:.0f} ms/prompt")


if __name__ == "__main__":
    main()
