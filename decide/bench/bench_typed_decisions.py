"""typed-decisions public benchmark (LocalLLaMA/typed-decisions, Apache-2.0)
— the zero-training head-to-head: our frozen readout on THEIR benchmark,
against fine-tuned decision models' published numbers.

Dataset: 4 workflows (agent_trace_observability / customer_service /
invoice_processing / security_incidents) x 100 test states x 5 questions
(choice action / noul needs_review / choice outcome / score risk / score
urgency) = 2,000 test decisions. Gold = teacher-derived soft labels (the
dataset ships probabilities, a continuous score expectation, and
label_agreement diagnostics — gold itself is a measured consensus, not
oracle truth).

Reference lines (published, NOT this run; both trained IN-distribution on
typed-decisions + a 12-source mix): Unsloth fine-tuned (LoRA + Clef head)
Qwen3.5-0.8B 73% / 2B 78% on this benchmark, before training 36% / 33%;
HF dataset leaderboard top (2026-10-08) 0.74 (FINAL-Bench/Darwin-27B).
Ours trains on NOTHING: frozen 4-bit base, label-free L0 readout — the
train-in vs read-out comparison (docs/PRIOR_ART.md).

Dialect: the dataset speaks the lowercase/criteria dialect (choice
criteria = {option_key: description}; noul criteria = {true/false}
glosses; score criteria = level-description list; gold labels are option
keys / "true"|"false" / level-index strings). Translated to our canonical
questions here; gold mapped back by key / index. Option DESCRIPTIONS are
folded into the instructions so the model sees the same criteria text the
fine-tunes train on.

Run:  endpoint on :8095 (swap --model for the cross-model axis), then
  python decide/bench/bench_typed_decisions.py
-> runs/typed_decisions.json (results/ is frozen; see decide/runio.py).
The endpoint is checked before the first request.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.bench.eval_vs_jev import jsd, post, wilson
from decide.runio import (abort_early, abort_if_all_failed, add_out,
                                check_out, require_endpoint, warn_failures)

DATA = ROOT / "data/eval/typed_decisions/test.jsonl"
TYPES = ("choice", "noul", "score")


def to_canonical(questions: dict) -> dict:
    """lowercase/criteria dialect -> our endpoint's canonical questions."""
    out = {}
    for qid, q in questions.items():
        t = q["type"].lower()
        crit = q.get("criteria") or {}
        instr = (q.get("instructions") or "").strip()
        if t == "choice":
            gloss = "; ".join(f"{k} = {v}" for k, v in crit.items())
            out[qid] = {"type": "Choice",
                        "instructions": f"{instr}\n({gloss})",
                        "options": list(crit.keys())}
        elif t == "noul":
            gloss = "; ".join(f"{k} = {v}" for k, v in crit.items())
            out[qid] = {"type": "Noul",
                        "instructions": f"{instr}\n({gloss})"
                        if gloss else instr}
        elif t == "score":
            out[qid] = {"type": "Score", "instructions": instr,
                        "levels": [str(x) for x in crit]}
        else:
            raise ValueError(f"unknown question type {t!r}")
    return out


def gold_dist(qtype: str, gold_q: dict, levels: list[str]) -> dict:
    """gold probabilities -> our key space (choice keys; Yes/No; level text)."""
    p = gold_q.get("probabilities") or {}
    if qtype == "noul":
        t = float(p.get("true", 0.0))
        return {"Yes": t, "No": 1.0 - t}
    if qtype == "score":
        return {levels[int(k)]: float(v) for k, v in p.items()}
    return {k: float(v) for k, v in p.items()}


def gold_answer(qtype: str, gold_q: dict, levels: list[str]):
    if qtype == "choice":
        return gold_q["label"]
    if qtype == "noul":
        return str(gold_q["label"]).lower() == "true"
    return levels[int(gold_q["label"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-url", default="http://127.0.0.1:8095")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--data", default=str(DATA))
    add_out(ap, "typed_decisions.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    served = require_endpoint(ap, args.local_url).get("model", "?")

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"{len(rows)} states x 5 questions = {len(rows) * 5} decisions",
          flush=True)
    print(f"endpoint model: {served}", flush=True)

    n_ok = n_all = n_err = 0
    acc = {t: [0, 0] for t in TYPES}
    acc_wf: dict[str, list[int]] = {}
    jsds, lat = [], []
    misses = []
    for i, row in enumerate(rows, 1):
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        canon = to_canonical(questions)
        body = {"state": row["state"], "questions": canon}
        loc, ms, err = post(f"{args.local_url}/v1/systemone", body, None)
        lat.append(ms)
        answers = (loc or {}).get("answers", {})
        wf = row["workflow"]
        acc_wf.setdefault(wf, [0, 0])
        if err:
            n_err += 1
            print(f"  {i} ({row['id']}): ERR {err}", flush=True)
        abort_early(n_err, i, "endpoint")
        for qid, q in questions.items():
            t = q["type"].lower()
            levels = [str(x) for x in (q.get("criteria") or [])] \
                if t == "score" else []
            g = gold_answer(t, gold[qid], levels)
            a = answers.get(qid, {}).get("answer")
            ok = a is not None and a == g
            n_ok += int(ok)
            n_all += 1
            acc[t][0] += int(ok)
            acc[t][1] += 1
            acc_wf[wf][0] += int(ok)
            acc_wf[wf][1] += 1
            probs = (answers.get(qid) or {}).get("probabilities") or {}
            if probs and gold[qid].get("probabilities"):
                jsds.append(jsd(probs, gold_dist(t, gold[qid], levels)))
            if not ok:
                misses.append(dict(id=row["id"], workflow=wf, qid=qid,
                                   type=t, gold=g, ours=a))
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  acc={n_ok / n_all:.4f}", flush=True)
        time.sleep(args.sleep)

    abort_if_all_failed(len(rows) - n_err, len(rows), "endpoint request")
    warn_failures(n_err, len(rows), "requests")

    def pcts(v):
        v = sorted(v)
        return dict(p50=round(v[len(v) // 2]), p95=round(v[int(len(v) * .95)]))

    report = dict(
        benchmark="LocalLLaMA/typed-decisions (Apache-2.0) test split: "
                  f"4 workflows x 100 states x 5 questions = {n_all} decisions",
        protocol="zero-training frozen-base readout (label-free L0), "
                 "POST /v1/systemone, one request per state (state-share on)",
        gold="teacher-derived soft labels (dataset consensus, not oracle)",
        reference_lines_published=(
            "NOT this run — trained in-distribution on typed-decisions + "
            "12-source mix: Unsloth Qwen3.5-0.8B 73% / 2B 78% (before "
            "training 36%/33%); HF leaderboard top 0.74 (2026-10-08)"),
        model=served,
        ours=dict(accuracy=round(n_ok / n_all, 4),
                  accuracy_wilson95=[round(x, 3) for x in wilson(n_ok, n_all)],
                  per_type={t: dict(acc=round(a / n, 4), n=n)
                            for t, (a, n) in acc.items()},
                  per_workflow={w: dict(acc=round(a / n, 4), n=n)
                                for w, (a, n) in acc_wf.items()},
                  mean_jsd_vs_gold=round(sum(jsds) / len(jsds), 4)
                  if jsds else None),
        latency_ms=dict(request=pcts(lat),
                        per_decision_mean=round(
                            sum(lat) / max(n_all / 5, 1))),
        n_states=len(rows), n_decisions=n_all,
    )
    out.write_text(json.dumps(dict(report, misses=misses), indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
