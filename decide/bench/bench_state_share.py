"""State-share bench: the long-context latency fix, measured + parity-checked.

Same questions through BOTH paths (in-process, one MLX instance):
  - old: Decider.decide(state, qs) — the served pipeline; each question
    re-prefills the whole state once per rotation (prefix = state+question);
  - new: decide/core/state_shared.decide_state_shared — ONE state forward, all
    (question, rotation) suffixes off copies of that cache.

Parity: argmax flips and max |conf delta| across every question (expected:
the bf16 cache-incremental boundary wiggle, 0 flips / conf <= 0.02 class).
Speed: per-request wall ms, old vs new. Datasets: questions_long.jsonl (the
regime the fix targets; 10 requests x 5 questions, states 0.9-4.8k tok) and
questions.jsonl (short, sanity).

Run (memory discipline: no other MLX instance):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/bench/bench_state_share.py'
-> runs/state_share.json (results/ is frozen; see decide/runio.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev import Decider
from anyjev.state import render_state

from decide.runio import add_out, check_out
from decide.core.mlx_backend import MLXBackend
from decide.core.state_shared import decide_state_shared

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


def build_q(spec: dict, qid: str):
    from anyjev import Question
    t = spec["type"]
    text = str(spec.get("instructions") or "").strip()
    if t == "Choice":
        return Question.choice(text or "Which option applies?", spec["options"], name=qid)
    if t == "Noul":
        return Question.noul(text or "Is this true?", name=qid)
    crit = str(spec.get("criteria") or "").strip()
    return Question.score((text + ("\nCriteria: " + crit if crit else "")).strip()
                          or "Rate this.", levels=spec["levels"], name=qid)


def run_file(dec, be, path: str, label: str):
    reqs = [json.loads(l) for l in open(path) if l.strip()]
    # warmup both paths
    q0 = build_q(list(reqs[0]["questions"].values())[0], "warmup")
    dec.decide(render_state(reqs[0]["state"]), [q0])
    decide_state_shared(dec, be, render_state(reqs[0]["state"]), [q0])

    per_req, flips, conf_deltas, n_q = [], 0, [], 0
    for i, body in enumerate(reqs, 1):
        state = render_state(body["state"])
        qs = [build_q(spec, qid) for qid, spec in body["questions"].items()]
        t0 = time.perf_counter()
        old = {d.question.id: d for d in dec.decide(state, qs)}
        ms_old = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        new = decide_state_shared(dec, be, state, qs)
        ms_new = (time.perf_counter() - t0) * 1000
        for q in qs:
            a, b = old[q.id], new[q.id]
            n_q += 1
            flips += int(a.argmax != b.argmax)
            conf_deltas.append(abs(float(a.confidence) - float(b.confidence)))
        per_req.append(dict(i=i, n_q=len(qs), ms_old=round(ms_old),
                            ms_new=round(ms_new),
                            speedup=round(ms_old / ms_new, 2)))
        print(f"  [{label}] req {i}/{len(reqs)}: {ms_old:.0f} -> {ms_new:.0f} ms"
              f" ({ms_old / ms_new:.1f}x)", flush=True)
    return dict(
        label=label, n_requests=len(per_req), n_questions=n_q,
        ms_old_mean=round(float(np.mean([r["ms_old"] for r in per_req]))),
        ms_new_mean=round(float(np.mean([r["ms_new"] for r in per_req]))),
        speedup_mean=round(float(np.mean([r["ms_old"] / r["ms_new"] for r in per_req])), 2),
        parity=dict(argmax_flips=f"{flips}/{n_q}",
                    max_abs_conf_delta=round(max(conf_deltas), 4),
                    mean_abs_conf_delta=round(float(np.mean(conf_deltas)), 5)),
        per_request=per_req,
    )


def main():
    global MODEL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    add_out(ap, "state_share.json")
    args = ap.parse_args()
    MODEL = args.model
    out = check_out(ap, args.out, args.force)
    be = MLXBackend(MODEL)
    dec = Decider(be, level="auto", shared_prefix=True)   # endpoint config

    report = dict(
        long=run_file(dec, be, str(ROOT / "data/eval/questions_long.jsonl"), "long"),
        short=run_file(dec, be, str(ROOT / "data/eval/questions.jsonl"), "short"),
        note="old = served Decider path (state re-prefilled per question per "
             "rotation); new = one state forward + suffix fan-out "
             "(score_shared_pairs). Parity expectation: the documented "
             "bf16 cache-boundary class (0 flips, conf delta <= 0.02).",
    )
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_request"}
                          if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
