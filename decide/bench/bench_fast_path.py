"""Serving fast path: sequential vs batched suffix passes vs batched + early stop.

Same model, same requests, one process (stop the server first: one MLX instance
at a time). The three modes are the three ways the endpoint can read a request:

  sequential   one forward per (question, rotation) off a copy of the state cache
               (the endpoint's behaviour before 2026-10-11; --no-batch)
  batched      the suffix passes of a request in ONE forward over the broadcast
               prefix cache, byte-budgeted (the endpoint's default now)
  early_stop   batched + read one rotation per Choice/Noul question first, the
               rest only where the confidence is below tau (--early-stop, opt-in)

Legs
  eval   the 70 requests / 350 questions of data/eval/questions{,_v2,_long}.jsonl,
         scored against the STORED Jev answers (results/jev_compare*.json): agreement
         with the reference (fidelity, not accuracy), answer changes against the
         sequential mode, confidence drift and reliability, passes and ms per request,
         peak GPU memory on the longest state.
  bank   BANKING77 10-way (top-10 intents, the bench_banking77 selection, every
         k-th item), scored against the GOLD labels: accuracy, passes, ms per item.
         All 10 rotations of a Choice are what early stopping can skip here.

Requests alternate the mode order so drift cancels; each (dataset, mode) has its
own long-lived Decider, like the served endpoint (its batch prior accumulates).

Run (nothing else on the GPU; ~9 min for the 4B with the defaults):
  python decide/bench/bench_fast_path.py
-> runs/fast_path.json (results/ is frozen; see decide/runio.py)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np

from anyjev import Decider, Question
from anyjev.state import render_state

from decide.bench.bench_banking77 import load_items
from decide.bench.bench_state_share import build_q
from decide.bench.eval_vs_jev import wilson
from decide.core.mlx_backend import MLXBackend
from decide.core.state_shared import EarlyStop, decide_state_shared
from decide.runio import add_out, check_out

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
MODES = ("sequential", "batched", "early_stop")
SETS = (("short", "questions", "jev_compare.json"),
        ("v2", "questions_v2", "jev_compare.n300.json"),
        ("long", "questions_long", "jev_compare.long.json"))


def ece(conf, ok, bins: int = 10) -> float:
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def setup_mode(be, mode, es):
    be.batch_suffixes = mode != "sequential"
    return es if mode == "early_stop" else None


def answer_conf(q, d):
    if q.kind == "noul":
        return bool(d.answer), max(d.p_true, 1.0 - d.p_true)
    return d.argmax, float(d.confidence)


def eval_leg(be, modes, es, max_requests=None):
    refs = {}
    reqs = []
    for kind, qfile, jfile in SETS:
        for r in json.load(open(ROOT / "results" / jfile))["rows"]:
            for q in r["questions"]:
                refs[(kind, r["i"], q["qid"])] = q["answer_j"]
        reqs += [(kind, json.loads(l)) for l in open(ROOT / f"data/eval/{qfile}.jsonl") if l.strip()]
    reqs = reqs[:max_requests] if max_requests else reqs
    b0 = reqs[0][1]
    for m in modes:                                              # warm every path on a throwaway decider
        es_m = setup_mode(be, m, es)
        decide_state_shared(Decider(be, level="auto", shared_prefix=True), be, render_state(b0["state"]),
                            [build_q(s, i) for i, s in b0["questions"].items()], early_stop=es_m)
    decs, idx, rows = {}, {}, {m: [] for m in modes}
    for n, (kind, body) in enumerate(reqs):
        idx[kind] = idx.get(kind, 0) + 1
        state = render_state(body["state"])
        rot = n % len(modes)
        for m in modes[rot:] + modes[:rot]:
            es_m = setup_mode(be, m, es)
            dec = decs.setdefault((kind, m), Decider(be, level="auto", shared_prefix=True))
            qs = [build_q(s, i) for i, s in body["questions"].items()]
            t0 = time.perf_counter()
            out = decide_state_shared(dec, be, state, qs, early_stop=es_m)
            ms = (time.perf_counter() - t0) * 1000
            qd = {}
            for q in qs:
                a, c = answer_conf(q, out[q.id])
                qd[q.id] = dict(ans=a, conf=c, reads=int(out[q.id].diagnostics["permutations"]),
                                jev=refs[(kind, idx[kind], q.id)],
                                type=body["questions"][q.id]["type"])
            rows[m].append(dict(set=kind, i=idx[kind], ms=ms, q=qd))
        if (n + 1) % 10 == 0:
            print(f"  eval [{n + 1}/{len(reqs)}] " + "  ".join(
                f"{m} {np.mean([r['ms'] for r in rows[m]]):.0f} ms" for m in modes), flush=True)

    base = {(r["set"], r["i"]): r for r in rows[modes[0]]}
    summary = {}
    for m in modes:
        rs = rows[m]
        qs_ = [(r, qid, v) for r in rs for qid, v in r["q"].items()]
        ok = [v["ans"] == v["jev"] for _, _, v in qs_]
        agree = float(np.mean(ok))
        chg = sum(int(v["ans"] != base[(r["set"], r["i"])]["q"][qid]["ans"]) for r, qid, v in qs_)
        dconf = float(np.mean([abs(v["conf"] - base[(r["set"], r["i"])]["q"][qid]["conf"]) for r, qid, v in qs_]))
        hi = np.array([v["conf"] >= 0.9 for _, _, v in qs_])
        by_type = {t: float(np.mean([v["ans"] == v["jev"] for _, _, v in qs_ if v["type"] == t]))
                   for t in ("Choice", "Noul", "Score")}
        b, c = (sum(int(v["ans"] == v["jev"] and base[(r["set"], r["i"])]["q"][qid]["ans"] != v["jev"]) for r, qid, v in qs_),
                sum(int(v["ans"] != v["jev"] and base[(r["set"], r["i"])]["q"][qid]["ans"] == v["jev"]) for r, qid, v in qs_))
        summary[m] = dict(
            n_requests=len(rs), n_questions=len(qs_),
            ms_per_request={k: round(float(np.mean([r["ms"] for r in rs if r["set"] == k])), 1) for k in ("short", "v2", "long")}
                          | {"all": round(float(np.mean([r["ms"] for r in rs])), 1)},
            passes_per_request=round(float(np.mean([sum(v["reads"] for v in r["q"].values()) for r in rs])), 2),
            agreement_with_jev=round(agree, 4), agreement_wilson95=[round(x, 3) for x in wilson(sum(ok), len(ok))],
            agreement_by_type={k: round(v, 3) for k, v in by_type.items()},
            answer_changes_vs_sequential=f"{chg}/{len(qs_)}",
            paired_vs_sequential=dict(only_this_right=b, only_sequential_right=c, mcnemar_p=round(mcnemar_p(b, c), 3)),
            mean_abs_conf_delta_vs_sequential=round(dconf, 4),
            ece_vs_agreement=round(ece([v["conf"] for _, _, v in qs_], ok), 4),
            share_conf_ge_0_9=round(float(hi.mean()), 3),
            agreement_when_conf_ge_0_9=round(float(np.mean(np.array(ok)[hi])), 3) if hi.any() else None)
    return summary


def memory_leg(be, modes, es):
    """Peak GPU memory of one request on the longest eval state and on a mid-length one
    (about 3k tokens, where the batched forward runs many rows), per mode."""
    bodies = sorted((json.loads(l) for l in open(ROOT / "data/eval/questions_long.jsonl")),
                    key=lambda b: -len(b["state"]))
    out = {}
    for label, body in (("longest", bodies[0]), ("mid", bodies[4])):
        state = render_state(body["state"])
        row = {"state_tokens": len(be.tokenizer.encode(state, add_special_tokens=False))}
        for m in modes:
            es_m = setup_mode(be, m, es)
            run = lambda: decide_state_shared(Decider(be, level="auto", shared_prefix=True), be, state,  # noqa: E731
                                              [build_q(s, i) for i, s in body["questions"].items()], early_stop=es_m)
            run()
            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            run()
            row[m] = dict(peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2), ms=round((time.perf_counter() - t0) * 1000))
        out[label] = row
    return out


def bank_leg(be, modes, es, n_items):
    top, options, items = load_items()
    step = max(1, len(items) // n_items)
    items = items[::step][:n_items]
    q = Question.choice("Which customer intent does this message express?", options, name="intent")
    decs = {m: Decider(be, level="auto", shared_prefix=True) for m in modes}
    for m in modes:
        es_m = setup_mode(be, m, es)
        decide_state_shared(Decider(be, level="auto", shared_prefix=True), be, items[0][0], [q], early_stop=es_m)
    res = {m: dict(ok=[], ms=[], reads=[], ans=[]) for m in modes}
    for n, (text, gold) in enumerate(items):
        rot = n % len(modes)
        for m in modes[rot:] + modes[:rot]:
            es_m = setup_mode(be, m, es)
            t0 = time.perf_counter()
            d = decide_state_shared(decs[m], be, text, [q], early_stop=es_m)["intent"]
            res[m]["ms"].append((time.perf_counter() - t0) * 1000)
            res[m]["ok"].append(d.argmax == gold)
            res[m]["ans"].append(d.argmax)
            res[m]["reads"].append(int(d.diagnostics["permutations"]))
        if (n + 1) % 50 == 0:
            print(f"  bank [{n + 1}/{len(items)}] " + "  ".join(
                f"{m} acc {100 * np.mean(res[m]['ok']):.1f}%" for m in modes), flush=True)
    base = res[modes[0]]["ok"]
    out = {}
    for m in modes:
        ok = res[m]["ok"]
        b = sum(int(x and not y) for x, y in zip(ok, base))
        c = sum(int((not x) and y) for x, y in zip(ok, base))
        out[m] = dict(n=len(ok), accuracy=round(float(np.mean(ok)), 4),
                      accuracy_wilson95=[round(x, 3) for x in wilson(int(sum(ok)), len(ok))],
                      passes_per_item=round(float(np.mean(res[m]["reads"])), 2),
                      ms_per_item=dict(mean=round(float(np.mean(res[m]["ms"])), 1), p50=round(float(np.median(res[m]["ms"])), 1)),
                      vs_first_mode=dict(only_this_right=b, only_first_right=c, mcnemar_p=round(mcnemar_p(b, c), 3)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Sequential vs batched suffix passes vs batched + early stop, in process: "
                    "agreement with the stored Jev answers (350 questions), BANKING77 gold "
                    "accuracy, passes, latency, peak memory. Loads its own model: stop the "
                    "endpoint first.")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--modes", default=",".join(MODES),
                    help="eval-leg modes, comma separated (the first is the baseline); default %(default)s")
    ap.add_argument("--bank-modes", default="batched,early_stop",
                    help="BANKING77-leg modes (default %(default)s)")
    ap.add_argument("--bank-n", type=int, default=200,
                    help="BANKING77 items (every k-th of the 400-item selection; 0 skips the leg)")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="only the first N of the 70 eval requests (smoke test; the numbers are then not comparable)")
    ap.add_argument("--choice-tau", type=float, default=0.9, help="early_stop Choice threshold")
    ap.add_argument("--noul-tau", type=float, default=0.8, help="early_stop yes/no threshold")
    add_out(ap, "fast_path.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    modes = args.modes.split(",")
    bank_modes = args.bank_modes.split(",")
    for m in modes + bank_modes:
        if m not in MODES:
            ap.error(f"unknown mode {m!r}; choose from {MODES}")
    es = EarlyStop(args.choice_tau, args.noul_tau)

    be = MLXBackend(args.model)
    report = dict(model=args.model, early_stop=dict(choice_tau=es.choice_tau, noul_tau=es.noul_tau),
                  kv_budget_gib=round(be.kv_budget_bytes / 2 ** 30, 2), modes=modes)
    print("eval leg (70 requests, 350 questions) ...", flush=True)
    report["eval"] = eval_leg(be, modes, es, args.max_requests)
    report["memory"] = memory_leg(be, modes, es)
    if args.bank_n:
        print(f"BANKING77 leg ({args.bank_n} items) ...", flush=True)
        report["bank"] = bank_leg(be, bank_modes, es, args.bank_n)
    report["backend"] = dict(batched_forwards=be.batched_forwards, batched_fallbacks=be.batched_fallbacks)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
