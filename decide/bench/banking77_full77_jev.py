"""BANKING77 full-77-way — the live Jev leg.

The full77 bench ran readout-vs-greedy only; Jev appeared as a published
number (79.7 macro-F1, Cloudflare's table — different harness/split). This
driver measures Jev on the SAME 500 items with the SAME 77 options via the
OpenRouter decisions API (same client/dialect as bench_banking77.py's 10-way
leg), joins the answers with both stored artifacts (4B + 9B rows), and
scores: Jev accuracy/macro-F1 vs gold, Jev-vs-readout and Jev-vs-greedy
agreement, paired exact McNemar on gold accuracy per model.

Items are re-derived from load_items_full(500) and ASSERTED identical to the
stored rows (deterministic selection) — no model is loaded, nothing fit.

Cost: ~500 decisions x ~$5e-6 = ~$0.003. Run inside nix develop:
  python decide/bench/banking77_full77_jev.py
-> runs/banking77_full77.jev.json (results/ is frozen and its stored artifacts
are only read; see decide/runio.py). A resumable checkpoint
(*.partial.jsonl) sits next to the output; failed calls in it are retried.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.bench.bench_banking77 import load_items_full, macro_f1
from decide.bench.eval_vs_jev import normalize_jev, post, to_jev_dialect, wilson
from decide.runio import abort_if_all_failed, add_out, check_out, require_key, warn_failures

QUESTION = "Which customer intent does this message express?"
STORED = {"4b": ROOT / "results/banking77_full77.json",
          "9b": ROOT / "results/banking77_full77.9b.json"}


def mcnemar(a_correct, b_correct):
    """Paired exact McNemar between two correctness vectors."""
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    nd = only_a + only_b
    p = (sum(comb(nd, k) for k in range(min(only_a, only_b) + 1)) / 2 ** nd * 2
         if nd else 1.0)
    return only_a, only_b, min(p, 1.0)


def call_jev(jev_url, key, model, text, options, retries=3):
    body = {"model": model, "state": text, "questions": {"intent": {
        "type": "Choice", "instructions": QUESTION, "options": options}}}
    for attempt in range(retries):
        rem, ms, err = post(jev_url, to_jev_dialect(body), key)
        if err is None and rem is not None:
            a, _, _ = normalize_jev((rem or {}).get("answers", {}).get(
                "intent", {}) or {})
            if a is not None:
                return a, ms, None
            err = f"unparsed answer: {json.dumps(rem)[:200]}"
        time.sleep(2.0 * (attempt + 1))
    return None, ms, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-url",
                    default="https://openrouter.ai/api/alpha/decisions")
    ap.add_argument("--jev-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--model", default="~typesafe/jev-latest")
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.4)
    add_out(ap, "banking77_full77.jev.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    partial = out.with_suffix(".partial.jsonl")
    require_key(ap, args.jev_key)

    classes, options, items = load_items_full(args.n_target)
    stored = {}
    for tag, path in STORED.items():
        d = json.load(open(path))
        rows = d["rows"]
        assert len(rows) == len(items), f"{tag}: row count mismatch"
        for r, (t, g) in zip(rows, items):
            assert r["text"] == t and r["gold"] == g, f"{tag}: item mismatch"
        stored[tag] = d
    print(f"{len(items)} items x {len(classes)} options aligned with stored "
          f"artifacts {list(STORED)}", flush=True)

    done = {}
    if partial.exists():
        for line in open(partial):
            if line.strip():
                r = json.loads(line)
                if r["jev"] is not None:          # failed calls are retried
                    done[r["i"]] = (r["jev"], r["ms"])
        print(f"resuming: {len(done)} calls already made", flush=True)

    preds, lat, fails = [None] * len(items), [], 0
    with open(partial, "a") as ckpt:
        for i, (text, gold) in enumerate(items):
            if i in done:
                preds[i], _ = done[i]
                continue
            a, ms, err = call_jev(args.jev_url, args.jev_key, args.model,
                                  text, options)
            if a is None:
                fails += 1
                print(f"  {i}: ERR {err}", flush=True)
                if not done and fails == 1 and i == 0:
                    sys.exit(f"error: the first Jev call failed ({err}); "
                             "check $OPENROUTER_API_KEY and --jev-url. "
                             "Nothing written.")
            preds[i] = a
            lat.append(ms)
            ckpt.write(json.dumps(dict(i=i, jev=a, ms=ms)) + "\n")
            ckpt.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                ok = sum(p == g for p, g in zip(preds[:i + 1],
                                                [g for _, g in items][:i + 1]))
                print(f"  jev {i + 1}/{len(items)}  acc={ok / (i + 1):.3f}"
                      f"  fails={fails}", flush=True)
            time.sleep(args.sleep)

    n = len(items)
    abort_if_all_failed(sum(p is not None for p in preds), n, "Jev API request")
    warn_failures(sum(p is None for p in preds), n, "Jev API requests")
    golds = [g for _, g in items]
    j_acc = sum(p == g for p, g in zip(preds, golds))
    j_f1, _ = macro_f1(preds, golds)

    def pcts(v):
        v = sorted(x for x in v if x)
        return dict(p50=round(v[len(v) // 2]), p95=round(v[int(len(v) * .95)]))

    legs = {}
    for tag, d in stored.items():
        rows = d["rows"]
        r = [x["readout"] for x in rows]
        g = [x["greedy"] for x in rows]
        j_ok = [p == gold for p, gold in zip(preds, golds)]
        r_ok = [p == gold for p, gold in zip(r, golds)]
        g_ok = [p == gold for p, gold in zip(g, golds)]
        oj_rr, or_jr, p1 = mcnemar(j_ok, r_ok)
        oj_gr, or_jg, p2 = mcnemar(j_ok, g_ok)
        legs[tag] = dict(
            agreement_jev_readout=round(sum(a == b for a, b in
                                            zip(preds, r)) / n, 4),
            agreement_jev_greedy=round(sum(a == b for a, b in
                                           zip(preds, g)) / n, 4),
            mcnemar_jev_vs_readout=dict(only_jev_right=oj_rr,
                                        only_readout_right=or_jr,
                                        p=round(p1, 5)),
            mcnemar_jev_vs_greedy=dict(only_jev_right=oj_gr,
                                       only_greedy_right=or_jg,
                                       p=round(p2, 5)),
            readout=dict(accuracy=d["readout"]["accuracy"],
                         macro_f1=d["readout"]["macro_f1"]),
            greedy=dict(accuracy=d["greedy"]["accuracy"],
                        macro_f1=d["greedy"]["macro_f1"]))

    conf = {}
    for p, gold in zip(preds, golds):
        if p is not None and p != gold:
            conf[(gold, p)] = conf.get((gold, p), 0) + 1

    report = dict(
        benchmark="BANKING77 full-77-way live Jev leg — same 500 items, same "
                  "77 prettified options as banking77_full77{,.9b}.json; raw "
                  "message text as state (mirrors the 10-way bench's Jev "
                  "path and the published task form)",
        model=args.model,
        jev=dict(accuracy=round(j_acc / n, 4),
                 accuracy_wilson95=[round(x, 3) for x in wilson(j_acc, n)],
                 macro_f1=round(j_f1, 4),
                 unparsed_or_failed=fails,
                 latency_ms=pcts(lat)),
        context="published full-77-way macro-F1 (Cloudflare Clef table): Clef "
                "94.2 / Clef-flash 90.9 / Jev 79.7 — published harness/split, "
                "this run's number is the same-API same-items comparator",
        per_model=legs,
        top_confusions_jev=[[g, p, c] for (g, p), c in
                            sorted(conf.items(), key=lambda kv: -kv[1])[:10]],
    )
    out.write_text(json.dumps(
        dict(report, rows=[dict(text=t, gold=g, jev=jp,
                                readout_4b=s4["readout"], greedy_4b=s4["greedy"],
                                readout_9b=s9["readout"], greedy_9b=s9["greedy"])
                           for (t, g), jp, s4, s9
                           in zip(items, preds, stored["4b"]["rows"],
                                  stored["9b"]["rows"])]),
        indent=1))
    partial.unlink(missing_ok=True)
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
