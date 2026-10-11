"""BANKING77 ground-truth benchmark — the first ACCURACY eval for this
project (gold labels from the dataset), and a neutral-comparison row vs
real Jev on identical items.

Source: huggingface.co/datasets/mteb/banking77 (test split, 3,080 rows,
77 intents, 40/class) — one of the public benchmarks in Cloudflare's Clef
table (BANKING77 macro-F1: Clef 94.2 / Clef-flash 90.9 / Jev 79.7 on the
full 77-way task).

Design honesty: our letter readout serves Choice with <=26 options and one
prefill per rotation, so we run the RESTRICTED task — top-10 intents
(alphabetical among the 40-count ties), 40 examples each = 400 items — and
send the SAME 10 options to BOTH our endpoint and Jev. The published
77-way numbers are context, not the comparator; the comparator is
Jev-on-this-subset, measured in the same run. Both sides classify against
gold; we report accuracy + macro-F1 (the leaderboard metric, stdlib-
computed) + pairwise agreement.

Run:  endpoint on :8095, then
  python decide/bench/bench_banking77.py            # key from $OPENROUTER_API_KEY
-> runs/banking77.json (results/ is frozen; see decide/runio.py). The
endpoint and the key are checked before the first request.
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

from decide.bench.eval_vs_jev import post, to_jev_dialect, normalize_jev, wilson
from decide.runio import (abort_early, abort_if_all_failed, add_out,
                                check_out, require_endpoint, require_key,
                                warn_failures)

DATA = ROOT / "data/eval/banking77/test.jsonl"
N_CLASSES = 10
PER_CLASS = 40


def prettify(label_text: str) -> str:
    return label_text.replace("_", " ")


def macro_f1(preds: list[str], golds: list[str]) -> tuple[float, dict]:
    classes = sorted(set(golds))
    f1s, table = {}, {}
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s[c], table[c] = f1, dict(tp=tp, fp=fp, fn=fn,
                                    f1=round(f1, 3))
    return sum(f1s.values()) / len(f1s), table


def load_items(n_classes: int = N_CLASSES, per_class: int = PER_CLASS,
               data: Path = DATA):
    """Deterministic selection: top-N classes (alphabetical among count
    ties), first `per_class` test rows each -> (options, [(text, gold)])."""
    rows = [json.loads(l) for l in open(data) if l.strip()]
    from collections import Counter
    counts = Counter(r["label_text"] for r in rows)
    top = sorted(sorted(counts, key=lambda c: (-counts[c], c))[:n_classes])
    options = [prettify(c) for c in top]
    by_class: dict[str, list] = {c: [] for c in top}
    for r in rows:
        if r["label_text"] in top and len(by_class[r["label_text"]]) < per_class:
            by_class[r["label_text"]].append(r)
    items = [(r["text"], prettify(r["label_text"]))
             for c in top for r in by_class[c]]
    return top, options, items


def load_items_full(n_target: int = 500, data: Path = DATA):
    """Full-77-way selection (the banking77_full77 bench): every class gets
    n_target // 77 test rows, the first (n_target % 77) classes alphabetically
    get one more — deterministic, near-balanced, ~500 rows to match Unsloth's
    published 500-row cut. -> (classes, options, [(text, gold)])."""
    rows = [json.loads(l) for l in open(data) if l.strip()]
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["label_text"], []).append(r)
    classes = sorted(by_class)
    base, rem = divmod(n_target, len(classes))
    options = [prettify(c) for c in classes]
    items = []
    for ci, c in enumerate(classes):
        for r in by_class[c][: base + (1 if ci < rem else 0)]:
            items.append((r["text"], prettify(c)))
    return classes, options, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-url", default="http://127.0.0.1:8095")
    ap.add_argument("--jev-url", default="https://openrouter.ai/api/alpha/decisions")
    ap.add_argument("--jev-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--model", default="~typesafe/jev-latest")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--no-jev", action="store_true",
                    help="profile our endpoint only (gold labels still "
                         "scored; used for cross-model runs where the "
                         "stored Jev rows are already paid for)")
    ap.add_argument("--data", default=str(DATA))
    add_out(ap, "banking77.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    require_endpoint(ap, args.local_url)
    if not args.no_jev:
        require_key(ap, args.jev_key, skip="--no-jev skips the reference side")

    top, options, items = load_items(data=Path(args.data))
    print(f"{len(items)} items, {len(top)} classes: {top}", flush=True)

    preds_l, preds_j, golds, lat_l, lat_j = [], [], [], [], []
    agree = n_err_l = n_err_j = 0
    for i, (text, gold) in enumerate(items, 1):
        body = {"state": text, "questions": {"intent": {
            "type": "Choice",
            "instructions": "Which customer intent does this message express?",
            "options": options}}}
        loc, ms_l, err_l = post(f"{args.local_url}/v1/systemone", body, None)
        if args.no_jev:
            rem, ms_j, err_j, a_j = None, 0.0, None, None
        else:
            rem, ms_j, err_j = post(args.jev_url, to_jev_dialect(
                {"model": args.model, **body}), args.jev_key)
            if i == 1 and err_j:
                sys.exit(f"error: the first Jev call failed ({err_j}); check "
                         "$OPENROUTER_API_KEY and --jev-url. Nothing written.")
            a_j, _, _ = normalize_jev((rem or {}).get("answers", {}).get(
                "intent", {}) or {}) if rem else (None, None, None)
        a_l = (loc or {}).get("answers", {}).get("intent", {}).get("answer")
        preds_l.append(a_l); preds_j.append(a_j); golds.append(gold)
        lat_l.append(ms_l); lat_j.append(ms_j)
        ok_l = a_l == gold; ok_j = a_j == gold
        agree += int(a_l == a_j)
        n_err_l += int(bool(err_l))
        n_err_j += int(bool(err_j))
        abort_early(n_err_l, i, "local endpoint")
        if err_l or err_j:
            print(f"  {i}: ERR local={err_l} jev={err_j}", flush=True)
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  acc_l={sum(x==g for x,g in zip(preds_l,golds))/i:.3f}"
                  f" acc_j={sum(x==g for x,g in zip(preds_j,golds))/i:.3f}", flush=True)
        time.sleep(args.sleep)

    n = len(golds)
    abort_if_all_failed(n - n_err_l, n, "local endpoint request")
    if not args.no_jev:
        abort_if_all_failed(n - n_err_j, n, "Jev API request")
    warn_failures(max(n_err_l, n_err_j), n, "requests")
    acc_l = sum(p == g for p, g in zip(preds_l, golds))
    acc_j = sum(p == g for p, g in zip(preds_j, golds))
    f1_l, tab_l = macro_f1(preds_l, golds)
    f1_j, tab_j = macro_f1(preds_j, golds)
    # confusions (ours): top wrong-direction pairs
    conf = {}
    for p, g in zip(preds_l, golds):
        if p != g:
            conf[(g, p)] = conf.get((g, p), 0) + 1
    top_conf = sorted(conf.items(), key=lambda kv: -kv[1])[:8]

    def pcts(v):
        v = sorted(v)
        return dict(p50=round(v[len(v) // 2]), p95=round(v[int(len(v) * .95)]))

    report = dict(
        benchmark="BANKING77 (mteb/banking77 test split) — restricted: "
                  f"top-{N_CLASSES} intents x {PER_CLASS} = {n} items, same "
                  "options to both sides",
        context="published full-77-way macro-F1 (Cloudflare Clef table): "
                "Clef 94.2 / Clef-flash 90.9 / Jev 79.7 — NOT directly "
                "comparable (77-way vs 10-way); the comparator here is "
                "Jev-on-the-same-subset",
        classes=top,
        ours=dict(accuracy=round(acc_l / n, 4),
                  accuracy_wilson95=[round(x, 3) for x in wilson(acc_l, n)],
                  macro_f1=round(f1_l, 4), per_class=tab_l),
        **({} if args.no_jev else dict(
            jev=dict(accuracy=round(acc_j / n, 4),
                     accuracy_wilson95=[round(x, 3) for x in wilson(acc_j, n)],
                     macro_f1=round(f1_j, 4), per_class=tab_j),
            pairwise=dict(agreement=round(agree / n, 4),
                          wilson95=[round(x, 3) for x in wilson(agree, n)]))),
        latency_ms=dict(ours=pcts(lat_l), **(
            {} if args.no_jev else dict(jev=pcts(lat_j)))),
        top_confusions_ours=[[g, p, c] for (g, p), c in top_conf],
    )
    out.write_text(json.dumps(
        dict(report, rows=[dict(text=t, gold=g, ours=pl, jev=pj)
                           for (t, g), pl, pj in zip(items, preds_l, preds_j)]),
        indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
