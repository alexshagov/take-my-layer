"""Rotation-budget bench on BANKING77 (the Clef one-pass question, our way).

Cloudflare's Clef trained order-robustness in (permuted synthetic data) so
it scores choices in ONE prefill. Our label-free substitute is permutation
marginalization — but it pays K prefills (one per option rotation). This
bench measures what capping the rotation budget costs on a real gold-label
task: the same 400 BANKING77 items / 10 options as bench_banking77.py, read
at m=1/2/3 rotations (plus anyjev's certified adaptive stopping), accuracy
+ macro-F1 + ms per config. The full-K row is the stored endpoint run
(results/banking77.json): ours 87.5% / 1.55s p50.

Also answers whether the batch prior (active here: one question layout x
400 states, min_prior_n=8) shifts the tradeoff.

Run (memory discipline: no other MLX instance):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/bench/rotation_budget.py'
-> runs/rotation_budget.json (results/ is frozen; see decide/runio.py).
Reads the stored served run (--stored, default results/banking77.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anyjev import Decider, Question

from decide.bench.bench_banking77 import load_items, macro_f1
from decide.runio import add_out, check_out
from decide.core.mlx_backend import MLXBackend

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
STORED = ROOT / "results/banking77.json"


def run_config(be, items, options, *, max_permutations=None, adaptive=False):
    dec = Decider(be, level="L0", shared_prefix=True,
                  **({"max_permutations": max_permutations}
                     if max_permutations is not None else {}),
                  **({"adaptive_shifts": True} if adaptive else {}))
    q = Question.choice("Which customer intent does this message express?",
                        options, name="intent")
    preds, golds, ms = [], [], []
    prior_methods = {}
    # warmup one call (compile)
    dec.decide(items[0][0], [q])
    for text, gold in items:
        t0 = time.perf_counter()
        d = dec.decide(text, [q])[q.name]
        ms.append((time.perf_counter() - t0) * 1000)
        preds.append(d.argmax)
        golds.append(gold)
        pm = str(d.diagnostics.get("prior_method"))
        prior_methods[pm] = prior_methods.get(pm, 0) + 1
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1, _ = macro_f1(preds, golds)
    n_perms = float(dec.stats["adaptive_shifts_total"] / len(golds)) if adaptive \
        else (max_permutations or len(options))
    return dict(preds=preds,
                summary=dict(accuracy=round(acc, 4), macro_f1=round(f1, 4),
                             ms_mean=round(sum(ms) / len(ms), 1),
                             ms_p50=round(sorted(ms)[len(ms) // 2], 1),
                             prefills_per_decision=round(n_perms, 2),
                             prior_methods=prior_methods))


def main():
    global MODEL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stored", default=str(STORED),
                    help="stored bench_banking77 run to take the full-K and "
                         "Jev reference rows from")
    add_out(ap, "rotation_budget.json")
    args = ap.parse_args()
    MODEL = args.model
    out = check_out(ap, args.out, args.force)
    top, options, items = load_items()
    print(f"{len(items)} items, {len(options)} options", flush=True)

    from decide.core.mlx_backend import MLXBackend
    be = MLXBackend(MODEL)
    configs = {}
    for m in (1, 2, 3):
        print(f"m={m} ...", flush=True)
        configs[f"m{m}"] = run_config(be, items, options, max_permutations=m)
        print(json.dumps(configs[f"m{m}"]["summary"]), flush=True)
    print("adaptive ...", flush=True)
    configs["adaptive"] = run_config(be, items, options, adaptive=True)
    print(json.dumps(configs["adaptive"]["summary"]), flush=True)

    stored = json.loads(Path(args.stored).read_text())
    rows = dict(
        full_K_endpoint=dict(
            accuracy=stored["ours"]["accuracy"],
            macro_f1=stored["ours"]["macro_f1"],
            ms_p50=stored["latency_ms"]["ours"]["p50"],
            prefills_per_decision=len(options),
            prior_methods={"batch": "active (served run)"}),
        jev_reference=dict(accuracy=stored["jev"]["accuracy"],
                           macro_f1=stored["jev"]["macro_f1"],
                           ms_p50=stored["latency_ms"]["jev"]["p50"]),
        **{k: v["summary"] for k, v in configs.items()},
        note="same 400 BANKING77 items/options as bench_banking77.py; "
             "full-K endpoint row = the stored served run (identical config "
             "chain; l0_ablation verified 105/105 endpoint==in-process "
             "answer identity on the eval set).",
    )
    preds_out = {k: v["preds"] for k, v in configs.items()}
    out.write_text(json.dumps(dict(rows, preds=preds_out), indent=1))
    print(json.dumps(rows, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
