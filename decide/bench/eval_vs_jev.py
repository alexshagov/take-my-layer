"""Eval loop: our SystemOne endpoint vs the real Jev API (docs/RESULTS.md).

Runs each request from a dataset (default data/eval/questions.jsonl — 21
states x 5 questions = 105, deliberately NON-coding domains) against BOTH the
local endpoint and the real Jev SystemOne API, then reports:

  - answer agreement overall and per question type (exact answer match),
    with Wilson 95% CI and chance-corrected kappa (chance term = uniform
    1/k over each question's own option/level count — agreement with a
    reference is FIDELITY, not accuracy),
  - mean |confidence delta| and mean JSD between probability distributions,
  - latency p50/p95 for both sides, input_tokens for our side,
  - resource profile of our server process — PURE STDLIB, nothing
    installed: RSS + CPU via `ps` cputime deltas, GPU% via stock `ioreg`
    (AGXDeviceUserClient AppUsage accumulatedGPUTime per pid, see
    ResourceSampler for the mechanism and the measured-broken alternatives
    it avoids).

The reference is Jev's own output (agreement, not accuracy — for a local
drop-in that IS the metric). With --no-jev it profiles our endpoint alone.

Run (full comparison; key from $OPENROUTER_API_KEY by default):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/serve/systemone.py --port 8095 &'        # our side
  python decide/bench/eval_vs_jev.py --local-url http://127.0.0.1:8095
-> runs/jev_compare.json (results/ is frozen; see decide/runio.py). Both
sides are checked before the first request, and a run where every call failed
exits non-zero without writing.
Remote dialect (openrouter.ai/api/alpha/decisions, model ~typesafe/jev-latest):
translated automatically — lowercase types, Choice options -> criteria object
(that object DEFINES the options), Score levels -> criteria list; responses
normalized back (noul = raw p(yes), score = continuous value + legend).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decide.runio import (abort_early, abort_if_all_failed, add_out,  # noqa: E402
                                check_out, require_endpoint, require_key,
                                warn_failures)

QUESTIONS = ROOT / "data/eval/questions.jsonl"


def to_jev_dialect(body: dict) -> dict:
    """canonical (TypeSafe-style, our dataset) -> OpenRouter decisions dialect:
    lowercase types; Choice options become `criteria` {option: description}
    (that object DEFINES the options there); Score levels become the `criteria`
    list; Noul takes instructions only (criteria verified optional)."""
    qs = {}
    for qid, q in body["questions"].items():
        t = q["type"].lower()
        out = {"type": t, "instructions": q.get("instructions") or ""}
        if t == "choice":
            out["criteria"] = {o: o for o in q["options"]}
        elif t == "score":
            out["criteria"] = q["levels"]
        qs[qid] = out
    return {"model": body.get("model", "~typesafe/jev-latest"),
            "state": body["state"], "questions": qs}


def normalize_jev(a: dict):
    """OpenRouter answer -> (answer, confidence, probabilities) comparable
    with ours. noul carries a raw p(yes); score carries a continuous value +
    index-keyed probabilities + legend (criteria order defines indices)."""
    t = a.get("type")
    if t == "noul":
        p = float(a["noul"])
        return p > 0.5, max(p, 1 - p), {"Yes": p, "No": 1 - p}
    if t == "choice":
        probs = a.get("probabilities") or {}
        return a["choice"], a.get("confidence") or (max(probs.values()) if probs else None), probs
    if t == "score":
        legend = a.get("legend") or {}
        probs = {legend.get(str(k), str(k)): v for k, v in (a.get("probabilities") or {}).items()}
        n = len(legend) or len(probs)
        idx = max(0, min(n - 1, round(float(a["score"]))))
        answer = legend.get(str(idx), str(idx))
        return answer, a.get("confidence") or (max(probs.values()) if probs else None), probs
    return None, None, {}


def post(url: str, body: dict, key: str | None, timeout: float = 300):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "X-OpenRouter-Title": "take-my-layer-eval",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, (time.perf_counter() - t0) * 1000, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:                       # noqa: BLE001
        return None, (time.perf_counter() - t0) * 1000, repr(e)


def jsd(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    pu = [max(float(p.get(k, 0.0)), 1e-12) for k in keys]
    qu = [max(float(q.get(k, 0.0)), 1e-12) for k in keys]
    sp, sq = sum(pu), sum(qu)
    pu = [x / sp for x in pu]
    qu = [x / sq for x in qu]
    m = [(a + b) / 2 for a, b in zip(pu, qu)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)  # noqa: E731
    return math.sqrt((kl(pu, m) + kl(qu, m)) / 2)


class ResourceSampler:
    """Samples RSS + instantaneous CPU + per-process GPU of a pid — PURE
    STDLIB, nothing installed.

    CPU: `ps -o time=` cputime deltas over a monotonic wall interval (macOS
    `ps -o pcpu` is a decaying lifetime average; third-party per-pid
    cputime helpers measured 0 for non-self pids).

    GPU: stock `ioreg -a -r -c AGXDeviceUserClient -l` parsed with plistlib.
    Each registry entry carries IOUserClientCreator ("pid N, name") and an
    AppUsage array whose accumulatedGPUTime values (ns) we sum per pid;
    GPU% = delta over the monotonic interval. Same IORegistry data source
    as Activity Monitor; world-readable, no sudo, no third-party code.
    Restart/reset starts a new baseline instead of a bogus sample.
    CPU% is percent of one core; GPU% is percent of the GPU device."""

    def __init__(self, pid: int):
        self.pid = pid
        self.samples: list[tuple] = []   # (rss_mb, cpu_pct_1core, gpu_pct|None)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._last_cpu = None
        self._last_at = None
        self._last_gpu_ns = None

    @staticmethod
    def _gpu_ns_by_pid() -> dict[int, int]:
        import plistlib
        import re
        out = subprocess.run(["ioreg", "-a", "-r", "-c", "AGXDeviceUserClient",
                              "-l"], capture_output=True, timeout=10).stdout
        per_pid: dict[int, int] = {}

        def walk(o):
            if isinstance(o, list):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                creator = o.get("IOUserClientCreator")
                usage = o.get("AppUsage")
                if isinstance(creator, str) and isinstance(usage, list):
                    m = re.match(r"pid (\d+)", creator)
                    if m:
                        pid = int(m.group(1))
                        per_pid[pid] = per_pid.get(pid, 0) + sum(
                            int(u.get("accumulatedGPUTime", 0)) for u in usage
                            if isinstance(u, dict))
                for v in o.values():
                    walk(v)

        try:
            walk(plistlib.loads(out))
        except plistlib.InvalidFileException:
            pass
        return per_pid

    @staticmethod
    def _cputime_s(field: str) -> float:
        # formats: MM:SS.ss, HH:MM:SS, DD-HH:MM:SS
        parts = field.split("-")
        days = float(parts[0]) if len(parts) == 2 else 0.0
        hms = parts[-1].split(":")
        sec = 0.0
        for x in hms:
            sec = sec * 60 + float(x)
        return days * 86400 + sec

    def _gpu_ns(self) -> int | None:
        try:
            return self._gpu_ns_by_pid().get(self.pid)
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ps", "-o", "rss=,time=", "-p", str(self.pid)],
                    capture_output=True, text=True, timeout=5).stdout.split()
                rss = float(out[0]) / 1024
                cpu = self._cputime_s(out[1])
                gpu = self._gpu_ns()
                now = time.monotonic()
                if self._last_cpu is not None:
                    d_cpu, d_t = cpu - self._last_cpu, now - self._last_at
                    if d_cpu < 0 or d_t <= 0:
                        # process restarted / clock anomaly: new baseline
                        self._last_cpu, self._last_at, self._last_gpu_ns = None, None, None
                    else:
                        gpu_pct = None
                        if gpu is not None and self._last_gpu_ns is not None and gpu >= self._last_gpu_ns:
                            gpu_pct = (gpu - self._last_gpu_ns) / 1e9 / d_t * 100
                        self.samples.append((rss, d_cpu / d_t * 100, gpu_pct))
                        self._last_cpu, self._last_at, self._last_gpu_ns = cpu, now, gpu
                else:
                    self._last_cpu, self._last_at, self._last_gpu_ns = cpu, now, gpu
            except (IndexError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=3)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        import os
        rss = [s[0] for s in self.samples]
        cpu = [s[1] for s in self.samples]
        gpu = [s[2] for s in self.samples if s[2] is not None]
        ncpu = os.cpu_count() or 1
        out = {"rss_mb_mean": round(sum(rss) / len(rss), 1),
               "rss_mb_peak": round(max(rss), 1),
               "cpu_pct_1core_mean": round(sum(cpu) / len(cpu), 1),
               "cpu_pct_1core_peak": round(max(cpu), 1),
               "cpu_pct_machine_mean": round(sum(cpu) / len(cpu) / ncpu, 1),
               "n_samples": len(self.samples)}
        if gpu:
            out.update({"gpu_pct_of_gpu_mean": round(sum(gpu) / len(gpu), 1),
                        "gpu_pct_of_gpu_peak": round(max(gpu), 1),
                        "gpu_samples": len(gpu)})
        return out


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--local-url", default="http://127.0.0.1:8095")
    ap.add_argument("--local-pid", type=int, default=None,
                    help="server pid to profile (default: pgrep systemone.py)")
    ap.add_argument("--jev-url",
                    default="https://openrouter.ai/api/alpha/decisions")
    ap.add_argument("--jev-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--model", default="~typesafe/jev-latest")
    ap.add_argument("--sleep", type=float, default=0.6,
                    help="pause between remote calls (rate-limit friendly)")
    add_out(ap, "jev_compare.json")
    ap.add_argument("--no-jev", action="store_true",
                    help="profile our endpoint only")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    require_endpoint(ap, args.local_url)
    if not args.no_jev:
        require_key(ap, args.jev_key, skip="--no-jev skips the reference side")

    reqs = [json.loads(l) for l in open(args.questions) if l.strip()]
    pid = args.local_pid
    if pid is None:
        got = subprocess.run(["pgrep", "-f", "serve/systemone.py"],
                             capture_output=True, text=True).stdout.split()
        pid = int(got[0]) if got else None
        if pid:
            print(f"profiling server pid {pid}")

    rows = []
    sampler = ResourceSampler(pid) if pid else None
    ctx = sampler if sampler is not None else _null()
    with ctx:
        t_start = time.time()
        for i, body in enumerate(reqs, 1):
            body = {"model": args.model, **body}
            loc, ms_loc, err_loc = post(f"{args.local_url}/v1/systemone", body, None)
            row = dict(i=i, n_q=len(body["questions"]),
                       local_ms=round(ms_loc), local_err=err_loc,
                       input_tokens=(loc or {}).get("usage", {}).get("input_tokens"))
            if args.jev_url and not args.no_jev:
                rem, ms_jev, err_jev = post(args.jev_url, to_jev_dialect(body),
                                            args.jev_key)
                if i == 1 and err_jev:
                    sys.exit(f"error: the first Jev call failed ({err_jev}); "
                             "check $OPENROUTER_API_KEY and --jev-url. "
                             "Nothing written.")
                row["jev_ms"] = round(ms_jev)
                row["jev_err"] = err_jev
                u = (rem or {}).get("usage") or {}
                row["jev_input_tokens"] = u.get("input_tokens")
                row["jev_cost"] = u.get("cost")
                time.sleep(args.sleep)
                if loc and rem:
                    qa = []
                    for qid, a in loc["answers"].items():
                        b = rem["answers"].get(qid)
                        if b is None:
                            continue
                        ans_j, conf_j, probs_j = normalize_jev(b)
                        qa.append(dict(
                            qid=qid, type=a["type"],
                            agree=a["answer"] == ans_j,
                            answer_l=a["answer"], answer_j=ans_j,
                            conf_l=a["confidence"], conf_j=conf_j,
                            jsd=round(jsd(a.get("probabilities") or {}, probs_j), 4)))
                    row["questions"] = qa
            rows.append(row)
            abort_early(sum(1 for r in rows if r["local_err"]), i,
                        "local endpoint")
            errs = [e for e in (err_loc, row.get("jev_err")) if e]
            print(f"  {i}/{len(reqs)} local {ms_loc:.0f}ms"
                  + (f" jev {row.get('jev_ms', 0):.0f}ms" if not args.no_jev else "")
                  + (" ERR " + "; ".join(errs) if errs else ""), flush=True)
        wall = time.time() - t_start

    abort_if_all_failed(sum(1 for r in rows if not r["local_err"]), len(rows),
                        "local endpoint request")
    if not args.no_jev:
        abort_if_all_failed(sum(1 for r in rows if not r.get("jev_err")),
                            len(rows), "Jev API request")
    warn_failures(sum(1 for r in rows if r["local_err"] or r.get("jev_err")),
                  len(rows), "requests")

    report = dict(n_requests=len(rows),
                  n_questions=sum(r["n_q"] for r in rows),
                  wall_s=round(wall, 1),
                  local_latency_ms={k: round(v) for k, v in {
                      "p50": pct([r["local_ms"] for r in rows], .5),
                      "p95": pct([r["local_ms"] for r in rows], .95)}.items()},
                  input_tokens_total=sum(r.get("input_tokens") or 0 for r in rows))

    if not args.no_jev:
        allq = [q for r in rows for q in r.get("questions", [])]
        ok = [q for q in allq if q["agree"] is not None]
        by_type: dict[str, list] = {}
        for q in ok:
            by_type.setdefault(q["type"], []).append(q)
        costs = [r["jev_cost"] for r in rows if r.get("jev_cost") is not None]
        # chance-corrected agreement: P_e = mean uniform 1/k over each
        # question's own option/level count (k from the dataset definitions)
        qk = []  # (type, k, agree)
        for r in rows:
            qdefs = reqs[r["i"] - 1]["questions"]
            for q in r.get("questions", []):
                if q["agree"] is None:
                    continue
                qd = qdefs[q["qid"]]
                k = (len(qd["levels"]) if q["type"] == "Score"
                     else 2 if q["type"] == "Noul" else len(qd["options"]))
                qk.append((q["type"], k, q["agree"]))
        n_qk = len(qk)
        p_e = sum(1.0 / k for _, k, _ in qk) / n_qk if qk else None
        kappa_by_type = {}
        for t, qs_ in by_type.items():
            sub = [(k, a) for tt, k, a in qk if tt == t]
            if sub:
                pe_t = sum(1.0 / k for k, _ in sub) / len(sub)
                po_t = sum(a for _, a in sub) / len(sub)
                kappa_by_type[t] = round((po_t - pe_t) / (1 - pe_t), 3)
        agree_n = sum(q["agree"] for q in allq)
        report["comparison"] = dict(
            n=len(allq),
            agreement=agree_n / len(allq) if allq else None,
            agreement_wilson95=[round(x, 3) for x in wilson(agree_n, len(allq))]
            if allq else None,
            chance_agreement=round(p_e, 3) if p_e is not None else None,
            kappa=round(((agree_n / len(allq)) - p_e) / (1 - p_e), 3)
            if allq and p_e is not None else None,
            kappa_by_type=kappa_by_type,
            agreement_by_type={t: round(sum(q["agree"] for q in qs) / len(qs), 3)
                               for t, qs in by_type.items()},
            abs_conf_delta_mean=round(sum(abs(q["conf_l"] - (q["conf_j"] or 0))
                                          for q in allq) / max(len(allq), 1), 3),
            jsd_mean=round(sum(q["jsd"] for q in allq) / max(len(allq), 1), 4),
            jev_latency_ms={k: round(v) for k, v in {
                "p50": pct([r["jev_ms"] for r in rows if "jev_ms" in r], .5),
                "p95": pct([r["jev_ms"] for r in rows if "jev_ms" in r], .95)}.items()},
            jev_input_tokens=sum(r.get("jev_input_tokens") or 0 for r in rows),
            jev_cost_usd_total=round(sum(costs), 6) if costs else None)

    if sampler is not None:
        report["resources"] = sampler.summary()

    out.write_text(json.dumps(dict(report, rows=rows), indent=1))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
