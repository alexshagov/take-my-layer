"""Jev SystemOne-style single-endpoint decision API — THE entry point.

POST /v1/systemone  {state, questions} -> answers with probabilities — no
chat, no generation. The AnyJev readout machinery behind one HTTP surface:

  - Choice (2-26 options, single-letter readout) / Noul / Score (2-10
    levels): L0 readout — permutation marginalization + prior correction,
    label-free, any question.
  - The registry next_move head layout serves at L2 (level="auto": L2 where
    a head routes, else L0 — the same degradation chain as the proxy).
  - usage.input_tokens = state + one prompt per question (prefill billing).

Auth: set SYSTEMONE_API_KEY to require `Authorization: Bearer <key>`.
Binds 127.0.0.1 only. Limits (all enforced; GET /healthz reports them): <= 8
questions, state <= --max-state-chars (default 32000; the long-context evals
run up to 17.6k), request body <= 1 MiB. Full reference: docs/API.md.
Telemetry: one row per request to logs/systemone_log.jsonl (gitignored)
(level, conf, answer_mass, ms per question) — the mass-watch for the
measured non-separating guard (proxy.py MASS_TAU note).

Serving defaults (docs/RESULTS.md, "Speed/precision sweep"): state-level prefix sharing ON; the
suffix passes of a request run as ONE batched forward (byte-budgeted, falls back to one-by-one;
--no-batch / --kv-budget-gib); every rotation is read. --early-stop (OFF by default) reads one
rotation per Choice/Noul question first and the rest only where confidence is below tau.

Memory discipline: ONE MLX instance, eval serialized behind a lock.

Run:  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
        python decide/serve/systemone.py --port 8095'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
REG = ROOT / "registry" / MODEL.replace("/", "__") / "base" / "next_move.json"
LOG = ROOT / "logs/systemone_log.jsonl"

NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
MAX_QUESTIONS = 8
MAX_STATE_CHARS = 32000      # --max-state-chars; validated up to 17.6k chars
#                             (data/eval/questions_long.jsonl), 2.8 s at 20k
MAX_INSTRUCTIONS = 1800
MAX_CRITERIA = 2000
MAX_BODY_BYTES = 1 << 20     # 1 MiB; checked from Content-Length before reading

D: dict = {}                 # decider, tokenizer, head-question, lock
LOCK = threading.Lock()
API_KEY = os.environ.get("SYSTEMONE_API_KEY") or None
STATE_SHARED = True          # state-level prefix sharing (bench_state_share.py:
#                             long 5.3x, short 1.55x, argmax flips only on
#                             exact 0.50 ties; --no-state-shared disables)
BATCH_SUFFIXES = True        # one batched forward per scoring round (bench_fast_path.py:
#                             ~1.9x on short requests; --no-batch disables)
KV_BUDGET_GIB = 1.5          # cap on the broadcast prefix cache of one batched forward
EARLY_STOP = None            # EarlyStop(...) when --early-stop is given (default: every rotation)


def build(max_permutations=None, prior_strength=None):
    from anyjev import Decider, Question

    from decide.core.mlx_backend import MLXBackend

    be = MLXBackend(MODEL)
    be.batch_suffixes = BATCH_SUFFIXES
    be.kv_budget_bytes = int(KV_BUDGET_GIB * 2 ** 30)
    # shared_prefix=True: >=2 permutations of one question share the prefix KV
    # (score_shared) with no minimum-prefix floor — local backend, no round
    # trips, so the HF 256-token floor does not apply.
    # max_permutations: cap the L0 rotation budget (rotation_budget.py on
    # BANKING77: m=2 keeps 86.3 vs 87.6 macro-F1 at 3.8x less latency and
    # beats Jev's p50; default None = full K).
    # prior_strength: batch-prior correction exponent (Decider default 0.75
    # partial; 1.0 = full marginal correction — the typed-decisions
    # score-miss experiment knob, decider.py DEFAULT_PRIOR_STRENGTH).
    dec = Decider(be, level="auto", shared_prefix=True,
                  **({"max_permutations": max_permutations,
                      "prior_strength": prior_strength}
                     if max_permutations is not None or prior_strength is not None
                     else {}))
    head_q = None
    if REG.exists():
        reg = json.loads(REG.read_text())
        head_q = Question.choice(reg["question"]["text"], reg["question"]["options"],
                                 name=reg["question"]["name"])
        dec.load_artifact(head_q, reg["artifact"])
    return dict(be=be, dec=dec, tok=be.tokenizer, head_q=head_q)


def serving_config() -> dict:
    """What GET /healthz reports about how reads are served (the flags in make_parser())."""
    return {"state_shared": STATE_SHARED,
            "batched_suffixes": STATE_SHARED and BATCH_SUFFIXES,
            "kv_budget_gib": KV_BUDGET_GIB,
            "early_stop": (None if EARLY_STOP is None else
                           {"choice_tau": EARLY_STOP.choice_tau, "noul_tau": EARLY_STOP.noul_tau})}


def _err(msg: str, code: str = "invalid_request", status: int = 400):
    return status, {"error": {"type": code, "message": msg}}


def parse_request(body: bytes):
    try:
        req = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        return _err(f"request body is not valid JSON: {e}")
    if not isinstance(req, dict):
        return _err("request body must be a JSON object")
    state = req.get("state")
    if state is None:
        return _err("'state' is required")
    if not isinstance(state, (str, dict, list)):
        return _err("'state' must be a string, object or array")
    n_chars = (len(state) if isinstance(state, str)
               else len(json.dumps(state, ensure_ascii=False)))
    if n_chars > MAX_STATE_CHARS:
        return _err(f"'state' is {n_chars} characters; the limit is "
                    f"{MAX_STATE_CHARS} (server flag --max-state-chars)")
    qs = req.get("questions")
    if not isinstance(qs, dict) or not qs:
        return _err("'questions' must be a non-empty map")
    if len(qs) > MAX_QUESTIONS:
        return _err(f"at most {MAX_QUESTIONS} questions per request")
    from anyjev import Question
    from anyjev.question import QuestionError

    parsed = {}
    for qid, spec in qs.items():
        if not NAME_RE.match(str(qid)):
            return _err(f"question name {qid!r}: letters/digits/_ only, <=64 chars")
        if not isinstance(spec, dict):
            return _err(f"question {qid!r} must be an object")
        t = spec.get("type")
        text = str(spec.get("instructions") or "").strip()
        if len(text) > MAX_INSTRUCTIONS:
            return _err(f"question {qid!r}: instructions > {MAX_INSTRUCTIONS} chars")
        try:
            if t == "Choice":
                opts = spec.get("options")
                if not isinstance(opts, list):
                    return _err(f"question {qid!r}: 'options' list required")
                q = Question.choice(text or "Which option applies?", opts, name=qid)
            elif t == "Noul":
                q = Question.noul(text or "Is this statement true?", name=qid)
            elif t == "Score":
                levels = spec.get("levels")
                if not isinstance(levels, list):
                    return _err(f"question {qid!r}: 'levels' list required")
                crit = str(spec.get("criteria") or "").strip()
                if len(crit) > MAX_CRITERIA:
                    return _err(f"question {qid!r}: criteria > {MAX_CRITERIA} chars")
                full = (text + ("\nCriteria: " + crit if crit else "")).strip()
                q = Question.score(full or "Rate this.", levels=levels, name=qid)
            else:
                return _err(f"question {qid!r}: type must be Choice|Score|Noul")
        except QuestionError as e:
            return _err(f"question {qid!r}: {e}")
        parsed[qid] = (t, q)
    return 200, (state, parsed)


def answer_for(t: str, q, d):
    probs = d.distribution
    if t == "Noul":
        return {"type": t, "answer": bool(d.answer),
                "confidence": max(d.p_true, 1.0 - d.p_true),
                "probabilities": probs}
    return {"type": t, "answer": d.argmax, "confidence": d.confidence,
            "probabilities": probs}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _relay(self, status, obj, headers=None, close=False):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if close:                       # request body left unread: drop the connection
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._relay(200, {"ok": True, "model": MODEL,
                              "head": D.get("head_q") is not None,
                              "serving": serving_config(),
                              "limits": {"max_questions": MAX_QUESTIONS,
                                         "max_state_chars": MAX_STATE_CHARS,
                                         "max_body_bytes": MAX_BODY_BYTES}})
        elif self.path == "/v1/systemone":
            self._relay(405, {"error": {"type": "method_not_allowed",
                                        "message": "POST /v1/systemone with a JSON body "
                                                   "{state, questions}; see docs/API.md"}},
                        headers={"Allow": "POST"})
        else:
            self._relay(404, {"error": {"type": "not_found",
                                        "message": "POST /v1/systemone, GET /healthz"}})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._relay(404, {"error": {"type": "not_found",
                                        "message": "POST /v1/systemone, GET /healthz"}},
                        close=True)
            return
        if API_KEY and self.headers.get("Authorization") != f"Bearer {API_KEY}":
            self._relay(401, {"error": {"type": "auth", "message": "bad or missing bearer token"}},
                        close=True)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:
            self._relay(*_err("invalid Content-Length"), close=True)
            return
        if n > MAX_BODY_BYTES:
            self._relay(*_err(f"request body is {n} bytes; the limit is {MAX_BODY_BYTES}",
                              "payload_too_large", 413), close=True)
            return
        status, payload = parse_request(self.rfile.read(n))
        if status != 200:
            self._relay(status, payload)
            return
        state, parsed = payload

        t0 = time.perf_counter()
        with LOCK:                                  # MLX eval is serialized
            try:
                merged = None
                if STATE_SHARED:
                    # state-level prefix sharing (P2): ONE state forward, all
                    # L0 questions+rotations as suffix fan-outs. L2-routed
                    # questions (registry head layout) keep the ordinary path.
                    l2 = [q for _, q in parsed.values() if D["dec"].route(q)]
                    rest = [q for _, q in parsed.values() if not D["dec"].route(q)]
                    from decide.core.state_shared import decide_state_shared
                    merged = {}
                    if l2:
                        for d in D["dec"].decide(state, l2, level="L2"):
                            merged[d.question.id] = d
                    if rest:
                        merged.update(decide_state_shared(
                            D["dec"], D["be"], state, rest, early_stop=EARLY_STOP))
            except Exception:                       # noqa: BLE001 — any break in the
                merged = None                       # shared path -> ordinary pipeline
            if merged is None:
                try:
                    dset = D["dec"].decide(state, [q for _, q in parsed.values()])
                    merged = {d.question.id: d for d in dset}
                except Exception as e:              # noqa: BLE001
                    self._relay(500, {"error": {"type": "decide_failed", "message": repr(e)}})
                    return
            # token accounting: state + one readout prompt per question
            from anyjev.readout import build_prompt, render_chat_parts, resolve_labels
            from anyjev.state import render_state
            st_text = render_state(state)
            n_tok = len(D["tok"].encode(st_text, add_special_tokens=False))
            for t, q in parsed.values():
                labels, _ = resolve_labels(D["tok"], q)
                spec = build_prompt(st_text, q, list(range(q.k)), labels=labels)
                pre, suf = render_chat_parts(D["tok"], spec)
                n_tok += len(D["tok"].encode(pre + suf, add_special_tokens=False))

        answers, rows = {}, []
        for qid, (t, q) in parsed.items():
            d = merged[q.name]
            answers[qid] = answer_for(t, q, d)
            rows.append(dict(qid=qid, type=t, level=d.level,
                             answer=answers[qid]["answer"],
                             confidence=round(answers[qid]["confidence"], 4),
                             answer_mass=round(float(d.diagnostics.get("answer_mass") or 0.0), 6),
                             reads=d.diagnostics.get("permutations"),
                             ms=None))
        out = {"model": MODEL, "answers": answers,
               "usage": {"input_tokens": n_tok}}
        try:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a") as f:
                f.write(json.dumps(dict(ts=time.time(), ms=int((time.perf_counter() - t0) * 1000),
                                        n_questions=len(parsed), rows=rows)) + "\n")
        except OSError:
            pass
        self._relay(200, out)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Local decision endpoint: POST /v1/systemone {state, "
                    "questions} -> typed answers with probabilities, read "
                    "from the model's own logits (no text generated). Binds "
                    "127.0.0.1 only; SYSTEMONE_API_KEY enables bearer auth. "
                    "Reference: docs/API.md.")
    ap.add_argument("--port", type=int, default=8095,
                    help="TCP port on 127.0.0.1 (default 8095)")
    ap.add_argument("--max-state-chars", type=int, default=MAX_STATE_CHARS,
                    help="reject states longer than this many characters "
                         "(default %(default)s; longer states cost time and "
                         "memory, validated up to ~17.6k)")
    ap.add_argument("--model", default=MODEL,
                    help="any MLX model id/path (must be in the HF cache or "
                         "downloadable; a model with no registry head artifact "
                         "serves L0-only — 'fit on what you serve' applies to "
                         "L2 heads)")
    ap.add_argument("--max-permutations", type=int, default=None,
                    help="cap L0 rotations (default: full K). BANKING77 "
                         "measured: m=2 keeps 86.3 vs 87.6 macro-F1 at "
                         "3.8x less latency (rotation_budget.py).")
    ap.add_argument("--no-state-shared", action="store_true",
                    help="disable state-level prefix sharing (one state "
                         "forward shared across the questions of a request; "
                         "bench_state_share.py: long 5.3x)")
    ap.add_argument("--no-batch", action="store_true",
                    help="score the suffix passes of a request one after another "
                         "instead of as one batched forward (the pre-2026-10-11 "
                         "behaviour; same answers up to bf16 kernel noise, ~1.9x "
                         "slower on short requests)")
    ap.add_argument("--kv-budget-gib", type=float, default=KV_BUDGET_GIB,
                    help="memory cap for one batched forward (the prefix cache is "
                         "broadcast to one row per suffix); a prefix too long to "
                         "fit two rows is scored sequentially. Default %(default)s")
    ap.add_argument("--early-stop", action="store_true",
                    help="read ONE rotation per Choice/Noul question first and the "
                         "remaining rotations only where the confidence is below "
                         "the thresholds (fewer passes, answers essentially "
                         "unchanged, reported confidences slightly more optimistic; "
                         "off by default — docs/RESULTS.md)")
    ap.add_argument("--early-stop-choice-tau", type=float, default=0.9,
                    help="--early-stop: Choice confidence needed to stop (default %(default)s)")
    ap.add_argument("--early-stop-noul-tau", type=float, default=0.8,
                    help="--early-stop: yes/no confidence needed to stop (default %(default)s)")
    ap.add_argument("--prior-strength", type=float, default=None,
                    help="batch-prior correction exponent (default: Decider's "
                         "0.75 partial; 1.0 = full marginal correction)")
    ap.add_argument("--head", default="",
                    help="registry head to load (default: derived from "
                         "--model; empty -> L0-only serving when that model "
                         "has no artifact)")
    return ap


def main():
    global REG, STATE_SHARED, MODEL, D, MAX_STATE_CHARS, BATCH_SUFFIXES, KV_BUDGET_GIB, EARLY_STOP
    ap = make_parser()
    args = ap.parse_args()
    MODEL = args.model
    MAX_STATE_CHARS = args.max_state_chars
    if args.head:
        REG = Path(args.head)
    else:
        REG = ROOT / "registry" / MODEL.replace("/", "__") / "base" / "next_move.json"
    if args.no_state_shared:
        STATE_SHARED = False
    if args.no_batch:
        BATCH_SUFFIXES = False
    if args.kv_budget_gib <= 0:
        ap.error("--kv-budget-gib must be positive")
    KV_BUDGET_GIB = args.kv_budget_gib
    if args.early_stop:
        from decide.core.state_shared import EarlyStop
        try:
            EARLY_STOP = EarlyStop(args.early_stop_choice_tau, args.early_stop_noul_tau)
        except ValueError as e:
            ap.error(str(e))

    print("loading model + head ...", flush=True)
    D.update(build(max_permutations=args.max_permutations,
                   prior_strength=args.prior_strength))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"systemone on :{args.port} (head loaded: {D['head_q'] is not None}; "
          f"auth {'on' if API_KEY else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
