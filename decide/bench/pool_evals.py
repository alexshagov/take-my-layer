"""Pool the stored eval runs (v1 105 + v2 195 = 300) into one report:
agreement / Wilson / kappa for the readout side and (from the stored
bench_vs_generate runs) the greedy side, per type and per source file, plus a
paired McNemar test on readout-vs-greedy (decide right & generate wrong vs
the reverse — the +4.x-point claim's significance test, which the n=105 CIs
could not settle).

Reads only stored artifacts (no MLX, no API):
  results/jev_compare.json            (v1, 105)
  results/jev_compare.n300.json       (v2, 195)
  results/bench_vs_generate{,.n300}.json
-> runs/jev_compare.pooled.json (results/ is frozen; see decide/runio.py)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.bench.eval_vs_jev import wilson
from decide.runio import add_out, check_out, rel

V1_Q, V2_Q = ROOT / "data/eval/questions.jsonl", ROOT / "data/eval/questions_v2.jsonl"
V1 = ROOT / "results/jev_compare.json"
V2 = ROOT / "results/jev_compare.n300.json"
B1 = ROOT / "results/bench_vs_generate.json"
B2 = ROOT / "results/bench_vs_generate.n300.json"


def k_of(qspec: dict) -> int:
    t = qspec["type"]
    if t == "Score":
        return len(qspec["levels"])
    if t == "Noul":
        return 2
    return len(qspec["options"])


def eval_rows(path: Path):
    """((type, k, agree), ...) per question from a stored eval run."""
    out = []
    for r in json.load(path.open())["rows"]:
        for x in r.get("questions", []):
            if x["agree"] is None:
                continue
            out.append((x["type"], x["k"] if "k" in x else None, bool(x["agree"])))
    return out


def kappa(rows):
    if not rows:
        return None
    p_e = sum(1.0 / k for _, k, _ in rows) / len(rows)
    p_o = sum(a for _, _, a in rows) / len(rows)
    return round((p_o - p_e) / (1 - p_e), 3)


def agr(rows):
    h = sum(a for _, _, a in rows)
    return h, len(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Pool the stored eval runs (105 + 195 = 300) and run the "
                    "paired readout-vs-greedy McNemar test. CPU only; reads "
                    "frozen results/ artifacts, no MLX, no API.")
    add_out(ap, "jev_compare.pooled.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    qs = {}
    for tag, qp in (("v1", V1_Q), ("v2", V2_Q)):
        for i, l in enumerate(qp.open(), 1):
            if l.strip():
                b = json.loads(l)
                for qid, spec in b["questions"].items():
                    qs[(tag, i, qid)] = (spec["type"], k_of(spec))

    # eval-side rows: attach k from the dataset definitions
    per_file, pooled = {}, []
    for tag, ep in (("v1", V1), ("v2", V2)):
        rows = []
        ri = 0
        for r in json.load(ep.open())["rows"]:
            ri += 1
            for x in r.get("questions", []):
                if x["agree"] is None:
                    continue
                t, k = qs[(tag, r["i"], x["qid"])]
                rows.append((t, k, bool(x["agree"])))
        per_file[tag] = rows
        pooled += rows

    def block(rows):
        h, n = agr(rows)
        by_type = {t: dict(n=len(sub), agreement=round(sum(a for _, _, a in sub) / len(sub), 3))
                   for t, sub in ((t, [r for r in rows if r[0] == t])
                                  for t in ("Choice", "Noul", "Score")) if sub}
        return dict(n=n, hits=h, agreement=round(h / n, 4),
                    agreement_wilson95=[round(x, 3) for x in wilson(h, n)],
                    kappa=kappa(rows), kappa_by_type={
                        t: kappa([r for r in rows if r[0] == t]) for t in by_type},
                    agreement_by_type=by_type)

    # greedy side from the stored benches: per_q has dec_ok / gen_ok (-1 = no ref)
    b_rows = []
    for tag, bp in (("v1", B1), ("v2", B2)):
        d = json.load(bp.open())
        for q in d.get("per_q", []):
            if q.get("dec_ok", -1) >= 0 and q.get("gen_ok", -1) >= 0:
                b_rows.append((tag, q["type"], q["dec_ok"], q["gen_ok"]))
    n_dec = sum(d for *_, d, _ in b_rows)
    n_gen = sum(g for *_, _, g in b_rows)
    n_b = len(b_rows)
    both = sum(1 for *_, d, g in b_rows if d and g)
    neither = sum(1 for *_, d, g in b_rows if not d and not g)
    only_dec = sum(1 for *_, d, g in b_rows if d and not g)
    only_gen = sum(1 for *_, d, g in b_rows if not d and g)
    # exact McNemar: binomial two-sided p on the discordant pairs
    nd = only_dec + only_gen
    p_two = (sum(math.comb(nd, i) for i in range(0, min(only_dec, only_gen) + 1)) * 2
             / 2 ** nd) if nd else 1.0
    p_two = min(1.0, p_two)

    report = dict(
        n_total=len(pooled),
        readout=dict(
            pooled=block(pooled), v1=block(per_file["v1"]), v2=block(per_file["v2"]),
            note="agreement with the real Jev API (fidelity), pooled over the "
                 "stored runs; v2 items measured harder (new-item effect: "
                 "different authoring batch, more multi-factor items)"),
        readout_vs_greedy=dict(
            n=n_b, readout_hits=n_dec, greedy_hits=n_gen,
            readout=round(n_dec / n_b, 4), greedy=round(n_gen / n_b, 4),
            delta_points=round(100 * (n_dec - n_gen) / n_b, 1),
            both_right=both, neither_right=neither,
            only_readout=only_dec, only_greedy=only_gen,
            mcnemar_exact_p=round(p_two, 5),
            note="paired on identical questions; exact binomial McNemar over "
                 "the discordant pairs"),
        sources=dict(v1=rel(V1), v2=rel(V2),
                     bench=[rel(B1), rel(B2)]),
    )
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
