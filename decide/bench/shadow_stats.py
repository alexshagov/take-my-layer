"""Shadow-log re-read: agreement + router curve + ECE on a shadow batch.

Usage:
  python decide/bench/shadow_stats.py [log.jsonl ...]
  (default: the frozen results/shadow_log.v3.jsonl; a live proxy's log is
  logs/shadow_log.jsonl)

Per log: level mix, agreement of pred vs the coarse label of actual_tool,
confidence percentiles, would_route (coverage) at the logged tau with the
routed error rate, 10-bin ECE, answer-mass percentiles, decide_ms. Rows
without a mapped actual_tool (no tool call / unknown tool) are counted but
excluded from agreement/ECE. The renderer-shift check after any head
promotion, and the router-curve/ECE reporting, folded into one place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.core.labels import CLASSES, MERGE


def read(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or r.get("worker_down"):
                continue
            if r.get("pred") and r.get("actual_tool"):
                continue                    # old-format rows carried pred+tool directly
            rows.append(r)
    return rows


def ece(confs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (confs > lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        e += m.sum() / len(confs) * abs(confs[m].mean() - correct[m].mean())
    return float(e)


def stats(path: str) -> dict:
    raw = [json.loads(l) for l in open(path)]
    rows = [r for r in raw if not r.get("error") and not r.get("worker_down")]
    have = [r for r in rows if r.get("pred") and MERGE.get(r.get("actual_tool") or "")]
    gold = np.array([int(MERGE[r["actual_tool"]] == r["pred"]) for r in have])
    conf = np.array([float(r["conf"]) for r in have])
    tau = rows[0].get("tau", 0.8) if rows else 0.8
    route_mask = conf >= tau if len(have) else np.array([])
    out = dict(
        path=path, n_rows=len(rows),
        n_labeled=len(have),
        levels={lv: sum(1 for r in rows if r.get("level") == lv)
                for lv in ("L2", "L0", "guard")},
        agreement=float(gold.mean()) if len(have) else float("nan"),
        conf_p50=float(np.percentile(conf, 50)) if len(have) else float("nan"),
        conf_p90=float(np.percentile(conf, 90)) if len(have) else float("nan"),
        cov_at_tau=float(route_mask.mean()) if len(have) else float("nan"),
        routed_err=float(1 - gold[route_mask].mean()) if route_mask.any() else float("nan"),
        n_routed=int(route_mask.sum()),
        ece=ece(conf, gold) if len(have) else float("nan"),
        mass_p50=float(np.percentile([r["mass"] for r in rows if r.get("mass")], 50))
        if any(r.get("mass") for r in rows) else float("nan"),
        decide_ms_p50=float(np.percentile([r["decide_ms"] for r in rows
                                           if r.get("decide_ms")], 50))
        if any(r.get("decide_ms") for r in rows) else float("nan"),
        tau=tau,
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Shadow-log re-read: agreement, router curve, ECE.")
    ap.add_argument("logs", nargs="*",
                    default=[str(ROOT / "results/shadow_log.v3.jsonl")],
                    help="shadow log(s) (default: the frozen "
                         "results/shadow_log.v3.jsonl; live: "
                         "logs/shadow_log.jsonl)")
    for p in ap.parse_args().logs:
        s = stats(p)
        print(f"== {s['path']}")
        print(f"   rows {s['n_rows']} (labeled {s['n_labeled']})  levels {s['levels']}")
        print(f"   agreement {s['agreement']:.3f}  ECE {s['ece']:.3f}  "
              f"conf p50/p90 {s['conf_p50']:.2f}/{s['conf_p90']:.2f}")
        print(f"   router @tau={s['tau']}: coverage {s['cov_at_tau']:.2f} "
              f"({s['n_routed']}/{s['n_labeled']}), routed error {s['routed_err']:.3f}")
        print(f"   mass p50 {s['mass_p50']:.3f}  decide p50 {s['decide_ms_p50']:.0f}ms")
