"""Mass-read removal bench: before/after measurement of skipping the
full-depth answer-mass read in the proxy worker's L2 decide.

The worker's guarded() reads the head hidden state at block `layer` AND the
label log-probs at FULL depth in one hidden_states_to call — the mass leg
forces the forward through all blocks (~0.3 s/decision). MASS_TAU=0.0 is
log-only (guard measured non-separating, docs/RESULTS.md), so skipping it is
behavior-safe. This bench replicates guarded() exactly, both modes, on REAL
rendered states (observe_log.jsonl — the surface the worker serves), and
measures wall ms + GPU (ResourceSampler, pure stdlib ioreg) + numeric parity.

Note: the SystemOne endpoint is unaffected by this flag — its L2 path already
early-exits (_run_heads reads one layer, no token_ids) and its L0 mass comes
out of the same forward for free. The full-forward mass read exists only in
the proxy worker.

Run (memory discipline: no other MLX instance running):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/bench/bench_mass_removal.py'
-> runs/mass_removal.json (results/ is frozen; see decide/runio.py).
States come from the frozen results/observe_log.jsonl (--states).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev import Question
from anyjev.heads import LinearHead
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels

from decide.bench.eval_vs_jev import ResourceSampler
from decide.runio import add_out, check_out
from decide.core.mlx_backend import MLXBackend

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
REG = ROOT / "registry" / MODEL.replace("/", "__") / "base" / "next_move.json"
STATES = ROOT / "results/observe_log.jsonl"
N_STATES = 30
REPS = 3


def main():
    ap = argparse.ArgumentParser(
        description="Before/after of skipping the full-depth answer-mass read "
                    "in the proxy worker's L2 decide (2.05x, parity). Loads "
                    "the 4B model; no other MLX instance may be running.")
    ap.add_argument("--states", default=str(STATES),
                    help="observe log whose rendered states are replayed "
                         "(default: the frozen results/ copy)")
    add_out(ap, "mass_removal.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    reg = json.loads(REG.read_text())
    q = Question.choice(reg["question"]["text"], reg["question"]["options"],
                        name=reg["question"]["name"])
    layer = int(reg["artifact"]["layer_abs"])
    be = MLXBackend(MODEL)
    head = LinearHead.from_dict(reg["artifact"])
    labels, label_ids = resolve_labels(be.tokenizer, q)
    n_blocks = be.n_layers

    states = [json.loads(l)["state"] for l in open(args.states) if l.strip()][:N_STATES]
    prompts = []
    for st in states:
        spec = build_prompt(st, q, list(range(q.k)), labels=labels)
        pre, suf = render_chat_parts(be.tokenizer, spec)
        prompts.append(pre + suf)

    def decide_with_mass(prompt):
        feats, lps, _, _ = be.hidden_states_to([prompt], [layer, n_blocks],
                                               token_ids=[label_ids],
                                               max_layer=n_blocks)
        mass = float(np.exp(lps[0]).sum())
        probs = head.probs(feats[:, 0])[0]
        return probs, mass

    def decide_no_mass(prompt):
        feats, _, _, _ = be.hidden_states_to([prompt], [layer], token_ids=None,
                                             max_layer=layer)
        return head.probs(feats[:, 0])[0], None

    # warmup both paths (compile etc.)
    decide_with_mass(prompts[0])
    decide_no_mass(prompts[0])

    # numeric parity first: same prompts, both modes (bf16 kernel paths differ
    # only in how far the forward runs — the head reads the SAME block)
    flips = 0
    conf_deltas = []
    for p in prompts:
        pm, _ = decide_with_mass(p)
        pn, _ = decide_no_mass(p)
        flips += int(int(np.argmax(pm)) != int(np.argmax(pn)))
        conf_deltas.append(float(np.max(pm) - np.max(pn)))
    parity = dict(n=len(prompts), argmax_flips=flips,
                  max_abs_conf_delta=round(max(abs(x) for x in conf_deltas), 6))

    # timed: interleave modes over all states x REPS, GPU sampled throughout
    ms_mass, ms_nomass = [], []
    sampler = ResourceSampler(os.getpid())
    with sampler:
        for rep in range(REPS):
            for p in prompts:
                t0 = time.perf_counter()
                decide_with_mass(p)
                ms_mass.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                decide_no_mass(p)
                ms_nomass.append((time.perf_counter() - t0) * 1000)
            print(f"  rep {rep + 1}/{REPS}: mass {np.mean(ms_mass):.0f}ms "
                  f"no-mass {np.mean(ms_nomass):.0f}ms", flush=True)

    def stat(v):
        v = sorted(v)
        return dict(mean=round(float(np.mean(v)), 1), p50=round(v[len(v) // 2], 1))

    report = dict(
        n_states=len(prompts), reps=REPS,
        head_layer=layer, n_blocks=n_blocks,
        with_mass=dict(ms_per_decision=stat(ms_mass), prefills=1,
                       forward="full depth (blocks 0..%d)" % n_blocks),
        no_mass=dict(ms_per_decision=stat(ms_nomass), prefills=1,
                     forward="early exit at block %d" % layer),
        speedup=round(float(np.mean(ms_mass)) / float(np.mean(ms_nomass)), 2),
        parity=parity,
        gpu=dict(
            with_mass=None,  # sampler is process-wide; see resources below
            note="GPU sampled across the interleaved run (both modes share the "
                 "process); per-mode GPU delta = ms ratio on a saturated device"),
        resources=sampler.summary(),
        flag="--no-mass on decide/serve/proxy.py (or DECIDE_NO_MASS=1); mass logs as "
             "None; guard code kept, gated. SystemOne endpoint unaffected "
             "(L2 already early-exits; L0 mass is free).",
    )
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
