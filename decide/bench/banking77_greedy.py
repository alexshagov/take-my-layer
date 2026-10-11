"""BANKING77 greedy-generation leg — the accuracy-grade readout-vs-
generation test.

The readout answers on these 400 gold-labeled items are already stored
(bench_banking77.py: 4B 87.5 / 9B 91.25). This script runs ONLY the
greedy leg: same deterministic item selection, same options, same
label-only prompt construction the in-process readout uses
(build_prompt + DEFAULT_SYSTEM), temp-0 decode, typed answers parsed
and scored against GOLD labels. No Jev, no API.

Paired exact McNemar vs the stored readout answers answers the single
question: is the readout more CORRECT than generation on public gold
data (not just more aligned with the reference)?

Run (one model per process — memory discipline):
  python decide/bench/banking77_greedy.py \
    --model mlx-community/Qwen3.5-4B-MLX-4bit \
    --readout results/banking77.json
-> runs/banking77_greedy.4b.json (results/ is frozen; see decide/runio.py;
--readout may point at a fresh runs/ file instead of the frozen one).
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anyjev import Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, render_chat_parts, resolve_labels
from anyjev.state import render_state

from decide.bench.bench_banking77 import load_items, macro_f1, prettify
from decide.runio import add_out, check_out, rel
from decide.core.mlx_backend import MLXBackend

MAX_TOKENS = 24            # banking77 labels run 5-7 tokens; 12 truncates


def parse_choice(text: str, options: list[str]) -> str | None:
    t = text.strip().strip(".,:;`*#\"'")
    low = t.lower()
    letters = string.ascii_uppercase[:len(options)]
    first = low.split()[0] if low.split() else ""
    for i, (letter, opt) in enumerate(zip(letters, options)):
        if first == letter.lower():
            return options[i]
    for opt in sorted(options, key=len, reverse=True):   # longest first
        if low.startswith(opt.lower()):
            return opt
    return None


def wilson(hits: int, n: int):
    from decide.bench.eval_vs_jev import wilson as w
    return w(hits, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--readout", default=str(ROOT / "results/banking77.json"),
                    help="stored bench_banking77 artifact with the readout "
                         "'ours' answers on the same items")
    add_out(ap, "banking77_greedy.4b.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)

    top, options, items = load_items()
    readout_ans = [r["ours"] for r in json.load(open(args.readout))["rows"]]
    assert len(readout_ans) == len(items), "stored/readout item count mismatch"
    print(f"{len(items)} items, {len(options)} options, model {args.model}",
          flush=True)

    be = MLXBackend(args.model)
    q = Question.choice("Which customer intent does this message express?",
                        options, name="intent")
    labels, _ = resolve_labels(be.tokenizer, q)

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    greedy = make_sampler(temp=0.0)

    preds, golds, lat = [], [], []
    parse_fail = 0
    for i, (text, gold) in enumerate(items, 1):
        state = render_state(text)
        prompt = render_chat_parts(be.tokenizer, build_prompt(
            state, q, list(range(q.k)), DEFAULT_SYSTEM, labels))
        prompt = prompt[0] + prompt[1]
        t0 = time.perf_counter()
        text = mlx_lm.generate(be.model, be.tokenizer, prompt=prompt,
                               max_tokens=MAX_TOKENS, sampler=greedy,
                               verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
        a = parse_choice(text, options)
        if a is None:
            parse_fail += 1
        preds.append(a)
        golds.append(gold)
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  greedy_acc="
                  f"{sum(p == g for p, g in zip(preds, golds)) / i:.3f}",
                  flush=True)

    n = len(items)
    acc = sum(p == g for p, g in zip(preds, golds))
    f1, table = macro_f1(preds, golds)
    r_acc = sum(r == g for r, g in zip(readout_ans, golds))
    b = sum(1 for p, r, g in zip(preds, readout_ans, golds)
            if r == g and p != g)          # only-readout-right
    c = sum(1 for p, r, g in zip(preds, readout_ans, golds)
            if p == g and r != g)          # only-greedy-right
    nd = b + c
    p_mcn = (sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd * 2
             if nd else 1.0)

    lat_sorted = sorted(lat)
    report = dict(
        benchmark="BANKING77 10-way, 400 gold-labeled items — GREEDY leg "
                  "only (readout answers replayed from stored artifact)",
        model=args.model,
        greedy=dict(accuracy=round(acc / n, 4),
                    accuracy_wilson95=[round(x, 3) for x in wilson(acc, n)],
                    macro_f1=round(f1, 4), parse_failures=parse_fail,
                    latency_ms=dict(p50=round(lat_sorted[n // 2]),
                                    p95=round(lat_sorted[int(n * .95)]))),
        readout_stored=dict(accuracy=round(r_acc / n, 4), source=rel(args.readout)),
        paired=dict(only_readout_right=b, only_greedy_right=c,
                    mcnemar_exact_p=round(min(p_mcn, 1.0), 5)),
        max_tokens=MAX_TOKENS,
    )
    out.write_text(json.dumps(
        dict(report, rows=[dict(text=t, gold=g, greedy=p, readout=r)
                           for (t, g), p, r in zip(items, preds, readout_ans)]),
        indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
