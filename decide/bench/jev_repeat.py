"""Jev self-agreement ceiling: repeat the reference side of the 105-question
eval N times against the live Jev API and measure agreement-of-the-reference-
with-itself. No local MLX needed (our side is deterministic: readout
distributions identical 10/10 in bench_vs_generate, 105/105 answer match vs
jev_compare.json in l0_ablation — replayed from the stored rows instead).

Why: our 79% is fidelity to ONE sample of a noisy reference. If Jev itself
only agrees with itself X% run-to-run, no deterministic system can beat ~X%
against single samples — that is the ceiling our number sits under, and
nobody in this ecosystem publishes it.

Stores runs + analysis in runs/jev_repeat.json (results/ is frozen and
jev_compare.json is only read; see decide/runio.py). Cost: ~$0.0005 per
full 105-question run.

Run:  python decide/bench/jev_repeat.py --runs 3     # key from $OPENROUTER_API_KEY
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.bench.eval_vs_jev import (normalize_jev, post, to_jev_dialect, wilson)
from decide.runio import abort_if_all_failed, add_out, check_out, require_key

QUESTIONS = ROOT / "data/eval/questions.jsonl"
STORED = ROOT / "results/jev_compare.json"


def norm(ans) -> str:
    if isinstance(ans, bool):
        return "Yes" if ans else "No"
    return ans


def agreement(a: dict, b: dict) -> tuple[int, int]:
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    return sum(a[k] == b[k] for k in keys), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--questions", default=str(QUESTIONS),
                    help="question file (default: the short 105; use "
                         "data/eval/questions_long.jsonl for the long "
                         "regime with --stored jev_compare.long.json)")
    ap.add_argument("--stored", default=str(STORED),
                    help="stored jev_compare results to replay against")
    ap.add_argument("--jev-url", default="https://openrouter.ai/api/alpha/decisions")
    ap.add_argument("--jev-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--model", default="~typesafe/jev-latest")
    ap.add_argument("--sleep", type=float, default=0.6)
    add_out(ap, "jev_repeat.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    require_key(ap, args.jev_key)

    reqs = [json.loads(l) for l in open(args.questions) if l.strip()]
    stored = {}                                   # (i, qid) -> stored jev answer
    for r in json.load(open(args.stored))["rows"]:
        for x in r.get("questions", []):
            stored[(r["i"], x["qid"])] = norm(x["answer_j"])
    qtype = {(i, qid): spec["type"] for i, b in enumerate(reqs, 1)
             for qid, spec in b["questions"].items()}

    runs: list[dict] = []
    for run in range(args.runs):
        answers: dict = {}
        errs = 0
        for i, body in enumerate(reqs, 1):
            payload = {"model": args.model, **body}
            rem, _, err = post(args.jev_url, to_jev_dialect(payload), args.jev_key)
            if err or rem is None:
                errs += 1
                rem, _, err2 = post(args.jev_url, to_jev_dialect(payload), args.jev_key)
                if err2:
                    print(f"  run {run + 1} req {i}: FAILED {err2}", flush=True)
                    if run == 0 and i == 1:
                        sys.exit(f"error: the first Jev call failed ({err2}); "
                                 "check $OPENROUTER_API_KEY and --jev-url. "
                                 "Nothing written.")
                    continue
            for qid, a in rem["answers"].items():
                ans, conf, _ = normalize_jev(a)
                answers[(i, qid)] = norm(ans) if ans is not None else None
            time.sleep(args.sleep)
        runs.append(answers)
        print(f"  run {run + 1}/{args.runs}: {len(answers)} answers, {errs} retried",
              flush=True)
    abort_if_all_failed(sum(len(a) for a in runs), len(runs), "Jev run")

    # pairwise self-agreement: run x run, and run x stored
    pairs = []
    labels = [f"run{r + 1}" for r in range(len(runs))]
    for (x, lx), (y, ly) in itertools.combinations(list(enumerate(labels)) + [(-1, "stored")], 2):
        A = runs[x] if x >= 0 else stored
        B = runs[y] if y >= 0 else stored
        h, n = agreement(A, B)
        lo, hi = wilson(h, n)
        by_type = {}
        for t in ("Choice", "Noul", "Score"):
            keys = [k for k in A if k in B and A[k] is not None and B[k] is not None
                    and qtype[k] == t]
            if keys:
                by_type[t] = round(sum(A[k] == B[k] for k in keys) / len(keys), 3)
        pairs.append(dict(pair=f"{lx}-vs-{ly}", agree=h, n=n,
                          agreement=round(h / n, 4),
                          wilson95=[round(lo, 3), round(hi, 3)],
                          by_type=by_type))

    # per-item stability across [stored + runs]
    all_samples = [stored] + runs
    stable = 0
    unstable_items = []
    keys = set(stored)
    for r in runs:
        keys |= set(r)
    for k in keys:
        vals = {m[k] for m in all_samples if m.get(k) is not None}
        if len(vals) == 1:
            stable += 1
        else:
            unstable_items.append(dict(
                i=k[0], qid=k[1], type=qtype.get(k),
                answers=[m.get(k) for m in all_samples]))
    n_items = len(keys)

    report = dict(
        n_runs=args.runs, n_items=n_items,
        self_agreement_pairs=pairs,
        self_agreement_mean=round(sum(p["agreement"] for p in pairs) / len(pairs), 4),
        item_stability=dict(
            stable=stable, unstable=n_items - stable,
            unstable_rate=round((n_items - stable) / n_items, 4),
            items=unstable_items),
        note="agreement-of-reference-with-itself on identical inputs; the "
             "ceiling for any deterministic system evaluated against single "
             "Jev samples. Our side is deterministic and replayed from "
             "jev_compare.json (see l0_ablation sanity 105/105).",
    )
    out.write_text(json.dumps(
        dict(report, runs={f"run{r + 1}": {f"{i}:{qid}": a
                                            for (i, qid), a in ans.items()}
                          for r, ans in enumerate(runs)}), indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
