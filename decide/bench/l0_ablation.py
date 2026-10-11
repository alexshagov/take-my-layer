"""L0-vs-raw ablation: what does the debiasing stack buy over the naive
single-prompt logit read (the Mini-Jev letter-logit read) on THIS eval?

For each of the 105 eval questions (data/eval/questions.jsonl), read at:
  - raw  : one prompt, identity listing, restricted softmax, NO correction
           (decider level="raw" — what every naive clone does);
  - L0   : the served endpoint config (permutation marginalization, K cyclic
           shifts; batch prior INACTIVE here — each question layout appears
           once, min_prior_n=8 — verified and reported, not assumed);
  - L0+cf: the full stack with the content-free prior (Zhao et al. probes)
           — NOT what the endpoint serves; isolates the prior-correction leg.
Jev's stored answers are replayed from results/jev_compare.json (same
pattern as bench_vs_generate.py — no API calls). Score questions are ordered
(1 perm at every level): raw == L0 there by construction; the delta comes
from Choice (K shifts) and Noul (2 phrasing orders) — reported per type.

K-curve: agreement and measured latency for the first m rotations (m=1..K),
offline from the full-K read (exact when the prior is inactive) plus served
wall-ms for m in {1,2,3}.

Run (memory discipline: no other MLX instance running):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/bench/l0_ablation.py'
-> runs/l0_ablation.json (results/ is frozen; see decide/runio.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev import Decider, Question
from anyjev.calibrate.permute import marginalize
from anyjev.state import render_state

from decide.bench.eval_vs_jev import wilson
from decide.runio import add_out, check_out

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
QUESTIONS = ROOT / "data/eval/questions.jsonl"
JEV = ROOT / "results/jev_compare.json"
BENCH = ROOT / "results/bench_vs_generate.json"


def build_q(spec: dict, qid: str) -> Question:
    t = spec["type"]
    text = str(spec.get("instructions") or "").strip()
    if t == "Choice":
        return Question.choice(text or "Which option applies?", spec["options"], name=qid)
    if t == "Noul":
        return Question.noul(text or "Is this true?", name=qid)
    crit = str(spec.get("criteria") or "").strip()
    return Question.score((text + ("\nCriteria: " + crit if crit else "")).strip()
                          or "Rate this.", levels=spec["levels"], name=qid)


def norm(ans) -> str:
    """bools (noul) -> Yes/No; everything else stays the option/level text."""
    if isinstance(ans, bool):
        return "Yes" if ans else "No"
    return ans


def stats(rows: list[dict], kmap: dict) -> dict:
    """agreement / wilson / kappa (uniform-1/k chance, as eval_vs_jev) + conf."""
    n = len(rows)
    hits = sum(r["agree"] for r in rows)
    p_e = sum(1.0 / kmap[(r["i"], r["qid"])] for r in rows) / n if n else None
    by_type = {}
    kappa_by_type = {}
    for t in ("Choice", "Noul", "Score"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        h = sum(r["agree"] for r in sub)
        pe = sum(1.0 / kmap[(r["i"], r["qid"])] for r in sub) / len(sub)
        by_type[t] = round(h / len(sub), 3)
        kappa_by_type[t] = round(((h / len(sub)) - pe) / (1 - pe), 3)
    agree_rows = [r for r in rows if r["agree"]]
    disagree_rows = [r for r in rows if not r["agree"]]
    return dict(
        n=n, agreement=round(hits / n, 4) if n else None,
        agreement_wilson95=[round(x, 3) for x in wilson(hits, n)] if n else None,
        kappa=round(((hits / n) - p_e) / (1 - p_e), 3) if n and p_e is not None else None,
        kappa_by_type=kappa_by_type,
        agreement_by_type=by_type,
        conf_mean=round(np.mean([r["conf"] for r in rows]), 3) if n else None,
        conf_on_agree=round(np.mean([r["conf"] for r in agree_rows]), 3) if agree_rows else None,
        conf_on_disagree=round(np.mean([r["conf"] for r in disagree_rows]), 3) if disagree_rows else None,
    )


def main():
    global MODEL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    add_out(ap, "l0_ablation.json")
    args = ap.parse_args()
    MODEL = args.model
    out = check_out(ap, args.out, args.force)
    reqs = [json.loads(l) for l in QUESTIONS.open() if l.strip()]
    jev_rows, our_stored = {}, {}
    for r in json.load(JEV.open())["rows"]:
        m = {x["qid"]: norm(x["answer_j"]) for x in r.get("questions", [])}
        jev_rows[r["i"]] = m
        our_stored[r["i"]] = {x["qid"]: norm(x["answer_l"]) for x in r.get("questions", [])}
    # chance term needs k per question definition (options/levels count; noul=2)
    kmap = {}
    for i, body in enumerate(reqs, 1):
        for qid, spec in body["questions"].items():
            kmap[(i, qid)] = (len(spec["levels"]) if spec["type"] == "Score"
                              else 2 if spec["type"] == "Noul"
                              else len(spec["options"]))

    from decide.core.mlx_backend import MLXBackend
    be = MLXBackend(MODEL)
    dec_raw = Decider(be, level="raw")                          # naive clone read
    dec_l0 = Decider(be, level="L0", shared_prefix=True)        # endpoint config
    dec_cf = Decider(be, level="L0", prior="content_free", shared_prefix=True)
    dec_km = {m: Decider(be, level="L0", max_permutations=m, shared_prefix=True)
              for m in (1, 2, 3)}                               # served K-curve

    # warmup (compile + first forward) on one question, untimed
    q0 = build_q(list(reqs[0]["questions"].values())[0], "warmup")
    dec_raw.decide(render_state(reqs[0]["state"]), [q0])
    dec_l0.decide(render_state(reqs[0]["state"]), [q0])

    raw_rows, l0_rows, cf_rows, curve_rows = [], [], [], []
    km_timing = {m: [] for m in (1, 2, 3)}
    prior_methods, sanity_match, sanity_n = {}, 0, 0
    score_raw_eq_l0 = 0
    for i, body in enumerate(reqs, 1):
        state = render_state(body["state"])
        jmap = jev_rows.get(i, {})
        for qid, spec in body["questions"].items():
            q = build_q(spec, qid)
            ref = jmap.get(qid)

            t0 = time.perf_counter()
            d_raw = dec_raw.decide(state, [q], level="raw")[q.name]
            ms_raw = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            d_l0 = dec_l0.decide(state, [q])[q.name]
            ms_l0 = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            d_cf = dec_cf.decide(state, [q])[q.name]
            ms_cf = (time.perf_counter() - t0) * 1000

            # served-K latency (choice only; m=1..3)
            ms_km = {}
            if q.kind == "choice":
                for m, dec in dec_km.items():
                    t0 = time.perf_counter()
                    dec.decide(state, [q])
                    ms_km[m] = round((time.perf_counter() - t0) * 1000, 1)

            if q.kind == "choice":
                for m, ms in ms_km.items():
                    km_timing[m].append(ms)

            base = dict(i=i, qid=qid, type=spec["type"], k=q.k, ref=ref)
            raw_rows.append(dict(base, answer=d_raw.argmax, conf=round(float(d_raw.confidence), 4),
                                 agree=d_raw.argmax == ref, ms=round(ms_raw, 1)))
            l0_rows.append(dict(base, answer=d_l0.argmax, conf=round(float(d_l0.confidence), 4),
                                agree=d_l0.argmax == ref, ms=round(ms_l0, 1)))
            cf_rows.append(dict(base, answer=d_cf.argmax, conf=round(float(d_cf.confidence), 4),
                                agree=d_cf.argmax == ref, ms=round(ms_cf, 1)))

            pm = str(d_l0.diagnostics.get("prior_method"))
            prior_methods[pm] = prior_methods.get(pm, 0) + 1
            if qid in our_stored.get(i, {}):
                sanity_n += 1
                sanity_match += int(d_l0.argmax == our_stored[i][qid])
            if q.kind == "score":
                score_raw_eq_l0 += int(d_raw.argmax == d_l0.argmax)

            # offline prefix curve from the full-K read (exact: prior inactive)
            if q.kind in ("choice", "noul"):
                p_pos = np.asarray(d_l0.diagnostics["p_pos_raw"])       # [P, K]
                perms = d_l0.diagnostics["perms"]
                for m in range(1, len(perms) + 1):
                    marg = marginalize(p_pos[:m], perms[:m], "logmean")
                    ans = q.options[int(np.argmax(marg))]
                    curve_rows.append(dict(i=i, qid=qid, type=spec["type"], m=m,
                                           agree=ans == ref))
            if q.kind == "choice":
                ms_prefill = ms_l0 / max(len(d_l0.diagnostics["perms"]), 1)
                curve_rows[-1]["ms_per_prefill_full_k"] = round(ms_prefill, 1)
            print(f"  {i}/21 {qid:<22} raw={d_raw.argmax!r:<28} l0={d_l0.argmax!r:<28}"
                  f" ref={ref!r:<28} {'=' if d_raw.argmax == d_l0.argmax else 'DELTA'}",
                  flush=True)

    def ms_stats(rows, key="ms"):
        v = sorted(r[key] for r in rows)
        return dict(mean=round(np.mean(v), 1), p50=round(v[len(v) // 2], 1))

    curve = {}
    for kind in ("choice", "noul"):
        per_m = {}
        for m in range(1, 6):
            sub = [r for r in curve_rows if r["type"].lower() == kind and r["m"] == m]
            if sub:
                per_m[m] = dict(n=len(sub),
                                agreement=round(sum(r["agree"] for r in sub) / len(sub), 4))
        curve[kind] = per_m
    # measured served latency for the first-m-rotations configs (choice only)
    measured_ms = {}
    for m in (1, 2, 3):
        v = sorted(km_timing[m])
        if v:
            measured_ms[m] = dict(n=len(v), mean_ms=round(float(np.mean(v)), 1),
                                  p50_ms=round(v[len(v) // 2], 1),
                                  prefills=m)

    bench = json.loads(BENCH.read_text()) if BENCH.exists() else {}
    report = dict(
        n_questions=len(raw_rows),
        configs=dict(
            raw=dict(**stats(raw_rows, kmap), ms_per_decision=ms_stats(raw_rows),
                     prefills_per_decision=1,
                     what="one prompt, identity listing, no correction"),
            l0_served=dict(**stats(l0_rows, kmap), ms_per_decision=ms_stats(l0_rows),
                           prefills_per_decision=round(
                               np.mean([r["k"] if r["type"] == "Choice" else
                                        2 if r["type"] == "Noul" else 1
                                        for r in l0_rows]), 2),
                           prior_method_counts=prior_methods,
                           sanity_match_vs_jev_compare=f"{sanity_match}/{sanity_n}",
                           what="endpoint config: K cyclic shifts, batch prior "
                                "(inactive on this eval — see prior_method_counts)"),
            l0_content_free=dict(**stats(cf_rows, kmap), ms_per_decision=ms_stats(cf_rows),
                                 what="full stack: marginalization + content-free "
                                      "prior (N/A,'' and [MASK] probes) — not served"),
        ),
        score_raw_equals_l0=f"{score_raw_eq_l0}/{sum(1 for r in raw_rows if r['type'] == 'Score')}",
        k_curve=dict(agreement=curve, served_ms_choice=measured_ms,
                     note="agreement: prefix-m rotations marginalized offline from "
                          "the full-K read (exact — prior inactive); served_ms_choice: "
                          "wall ms of a real Decider(max_permutations=m) per choice "
                          "question. m=1 choice == the raw read."),
        greedy_reference=dict(
            decide_l0=bench.get("agreement_with_jev", {}).get("decide"),
            greedy=bench.get("agreement_with_jev", {}).get("generate"),
            note="stored bench_vs_generate.json (same questions, same model)"),
        note="raw = the Mini-Jev-style naive letter-logit read; the L0-raw delta "
             "per type decomposes the debiasing gain (Score is 1-perm at every "
             "level by construction).",
    )
    out.write_text(json.dumps(dict(report, rows=dict(raw=raw_rows, l0=l0_rows, cf=cf_rows),
                                   curve_rows=curve_rows), indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
