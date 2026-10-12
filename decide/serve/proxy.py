"""Shadow proxy + the safety/adaptation legs.

See docs/DESIGN.md, 'Shadow proxy and the supervised worker'.

Transparent pass-through in front of mlx_lm.server. Routes NOTHING: while the
upstream generates, the proxy renders the conversation through the v2
pipeline, asks the registry head what it would do, and logs that against what
the model actually called. Live agreement is the renderer-shift check that
gates enabling routing. (The fabrication leg — extractors +
fabricated tool_calls — was removed at fork; see docs/POLICY.md.)

Legs:
  - answer-mass guard: the worker's single L2 forward also reads the
    label-token log-probs; mass < --mass-tau means "not a readable decision"
    → would_route=false (generation stays authoritative). Policy §5.
  - level="auto" degradation: L2 → L0 (same worker, no head / broken path) →
    pass-through (worker down/slow, Head.decide -> None).
  - observe(): every shadow row with a mapped actual_tool is appended (with
    the FULL rendered state) to logs/observe_log.jsonl;
    decide/adapt/observe.py --log logs/observe_log.jsonl replays it into
    Decider.observe (auto-solve at 30 labels, refit at 60/120/...) — the
    permanent adaptation engine.

Live logs go to the gitignored logs/ (never results/, which holds frozen,
committed evidence; copy a log there deliberately).

  upstream (:8080 mlx_lm.server)  <-  proxy (:8090)  <-  opencode
                                           | own 4-bit instance for the head

Usage:
  nix develop -c bash -c 'python decide/serve/proxy.py --port 8090 \
      --upstream http://127.0.0.1:8080'
  then point opencode at http://127.0.0.1:8090/v1
  (decide/serve/opencode-shadow.json, next to this file).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
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
LOG = ROOT / "logs/shadow_log.jsonl"
OBSERVE_LOG = ROOT / "logs/observe_log.jsonl"
MASS_TAU = 0.0           # answer-mass guard floor. MEASURED NON-SEPARATING on
# this model (docs/RESULTS.md "Answer-mass guard measurement"): seed/foreign/degenerate states all
# put 0.92-0.99 mass on the label alphabet at full depth (the model complies
# with the readout format unconditionally), and at the head's exit layer the
# mass is ~1e-5 noise. Default 0.0 = log-only, never gate. Kept gateable so a
# future model/artifact that DOES separate can enable it without code change.


def _worker_main(reg_path, model_path, q_req, q_rep, mass_tau, no_mass=False):
    """Runs in its own process: owns the ONLY extra MLX instance. A wedged
    forward here can only cost the parent one timeout, never the request path
    (docs/DESIGN.md, 'Shadow proxy and the supervised worker': an in-process decide held the Head
    lock and silenced every later request for the rest of the batch).

    L2 read = ONE forward: the hidden state for the registry head at its exit
    layer AND the label-token log-probs at full depth (the answer position,
    where mass is meaningful) come out of the same hidden_states_to call.
    Degrades L2 -> L0 -> "down" (parent pass-through).

    DECIDE_NO_MASS=1 (or --no-mass): skip the full-depth mass leg — the
    forward early-exits at the head's block (measured 2.05x,
    bench_mass_removal.py), `mass` is logged as None. Behavior-safe:
    MASS_TAU=0.0 is log-only (guard measured NON-separating,
    docs/RESULTS.md), so nothing gates on it."""
    import numpy as np

    from anyjev import Decider, Question
    from anyjev.heads import LinearHead
    from anyjev.readout import build_prompt, render_chat_parts, resolve_labels

    from decide.core.mlx_backend import MLXBackend

    no_mass = bool(no_mass) or bool(int(os.environ.get("DECIDE_NO_MASS", "0") or 0))

    reg = json.loads(Path(reg_path).read_text())
    q = Question.choice(reg["question"]["text"], reg["question"]["options"],
                        name=reg["question"]["name"])
    be = MLXBackend(model_path)
    dec = Decider(be, level="auto")            # L2 where the head routes, else L0
    dec.load_artifact(q, reg["artifact"])      # validates model + layout at startup
    head = LinearHead.from_dict(reg["artifact"])
    layer = int(reg["artifact"]["layer_abs"])
    labels, label_ids = resolve_labels(be.tokenizer, q)
    layers = [layer] if no_mass else sorted({layer, be.n_layers})

    def guarded(state_text):
        # canonical listing (perm = identity), the layout every registry head reads
        spec = build_prompt(state_text, q, list(range(q.k)), dec.system, labels)
        pre, suf = render_chat_parts(be.tokenizer, spec)
        feats, lps, _, _ = be.hidden_states_to([pre + suf], layers,
                                               token_ids=None if no_mass else [label_ids],
                                               max_layer=layers[-1])
        if no_mass:
            mass = None
        else:
            mass = float(np.exp(lps[0]).sum())  # at full depth (ls[-1])
            if mass_tau > 0.0 and mass < mass_tau:
                # not a readable decision by the mass criterion — fall back to
                # generation (policy §5). Non-firing by default: see MASS_TAU note.
                return dict(pred=None, conf=0.0, exit_layer=layer, mass=mass,
                            level="guard")
        probs = head.probs(feats[:, 0])[0]
        return dict(pred=q.options[int(np.argmax(probs))],
                    conf=float(np.max(probs)), exit_layer=layer, mass=mass,
                    level="L2")

    q_rep.put(("ready", reg["tau"]))
    while True:
        msg = q_req.get()
        if msg is None:
            return
        kind, payload = msg
        if kind == "decide":
            try:
                out = guarded(payload)
            except Exception:                   # noqa: BLE001 — head path broken: L0
                try:
                    d = dec.decide(payload, [q], level="L0")[q.name]
                    out = dict(pred=d.argmax, conf=d.confidence,
                               exit_layer=be.n_layers,
                               mass=float(d.diagnostics.get("answer_mass") or 0.0),
                               level="L0")
                except Exception:               # noqa: BLE001 — decide stack broken
                    out = dict(pred=None, conf=0.0, exit_layer=0, mass=0.0,
                               level="down")
            q_rep.put(("decide", out))


class Head:
    """Parent-side handle: supervised worker, hard decide timeout, degrades to
    no-routing (pass-through) whenever the worker is down or slow."""

    def __init__(self, reg_path=REG, mass_tau=MASS_TAU, no_mass=False):
        self.reg_path = Path(reg_path)
        reg = json.loads(self.reg_path.read_text())
        self.tau = reg["tau"]
        self.mass_tau = mass_tau
        self.no_mass = no_mass
        self.ctx = mp.get_context("spawn")
        self.q_req = self.q_rep = None
        self.proc = None
        self.ready = False
        self._spawn()

    def _spawn(self):
        self.ready = False
        self.q_req = self.ctx.Queue()
        self.q_rep = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(str(self.reg_path), MODEL, self.q_req, self.q_rep,
                  self.mass_tau, self.no_mass),
            daemon=True)
        self.proc.start()
        threading.Thread(target=self._await_ready, daemon=True).start()

    def _await_ready(self):
        try:
            kind, _ = self.q_rep.get(timeout=180)
            if kind == "ready":
                self.ready = True
        except Exception:                       # noqa: BLE001
            pass

    def _restart(self):
        try:
            self.proc.kill()
        except Exception:                       # noqa: BLE001
            pass
        self._spawn()

    def decide(self, state_text, timeout=15.0):
        """(pred, conf, layer) or None (worker down/slow -> do not route)."""
        if not self.ready:
            return None
        self.q_req.put(("decide", state_text))
        try:
            kind, val = self.q_rep.get(timeout=timeout)
            return val if kind == "decide" else None
        except queue_mod.Empty:
            self._restart()
            return None


HEAD: Head | None = None
MODE = "shadow"          # shadow | off  (toggle live via POST /admin/mode)


def parse_actual_tool(resp_body: bytes, was_stream: bool):
    """The tool the model actually called (first tool_call), or None."""
    try:
        if not was_stream:
            r = json.loads(resp_body)
            tcs = (r["choices"][0].get("message") or {}).get("tool_calls") or []
            if tcs:
                return tcs[0]["function"]["name"]
            return None
        name = None
        for line in resp_body.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
                chs = chunk.get("choices") or []
                if not chs:                        # usage/role chunks carry no choices
                    continue
                d = chs[0].get("delta") or {}
                for tc in d.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        name = fn["name"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return name
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                      # quiet
        pass

    def _relay(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._relay(200, "application/json",
                        json.dumps({"ok": True, "mode": MODE}).encode())
        elif self.path == "/admin/health":
            up_ok = False
            try:
                with urllib.request.urlopen(f"{UPSTREAM}/v1/models", timeout=5) as r:
                    up_ok = r.status == 200
            except Exception:                     # noqa: BLE001
                pass
            self._relay(200, "application/json", json.dumps({
                "mode": MODE, "upstream_ok": up_ok,
                "worker_ready": bool(HEAD and HEAD.ready)}).encode())
        else:
            self._relay(404, "application/json", b"{}")

    def do_POST(self):
        if self.path == "/admin/mode":
            n = int(self.headers.get("Content-Length") or 0)
            global MODE
            try:
                MODE = json.loads(self.rfile.read(n) or b"{}").get("mode", MODE)
                if MODE not in ("shadow", "off"):
                    MODE = "shadow"
            except json.JSONDecodeError:
                pass
            self._relay(200, "application/json",
                        json.dumps({"mode": MODE}).encode())
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        req = urllib.request.Request(f"{UPSTREAM}{self.path}", data=body, method="POST",
                                     headers={"Content-Type":
                                              self.headers.get("Content-Type",
                                                               "application/json")})
        shadow_row = None
        if self.path.endswith("/chat/completions") and HEAD is not None and MODE != "off":
            shadow_row = self._shadow(body)

        box: dict = {}
        th = threading.Thread(target=self._fetch, args=(req, box))
        th.start()
        th.join()
        if "error" in box:
            self._relay(502, "application/json",
                        json.dumps({"error": box["error"]}).encode())
        else:
            if shadow_row is not None:
                actual = parse_actual_tool(box["body"], box.get("stream", False))
                shadow_row["actual_tool"] = actual
                shadow_row["mode"] = MODE
                self._observe(shadow_row, actual)
                shadow_row.pop("_state", None)
                LOG.parent.mkdir(parents=True, exist_ok=True)
                with LOG.open("a") as f:
                    f.write(json.dumps(shadow_row) + "\n")
            self._relay(box["status"], box["ctype"], box["body"])

    @staticmethod
    def _observe(shadow_row, actual_tool):
        """observe() leg: pair the rendered state with the coarse label of
        the tool the model ACTUALLY called, appended to observe_log.jsonl for
        the offline replay (decide/adapt/observe.py). Written even when the
        worker was down — the label comes from upstream generation, not the
        head."""
        from decide.core.labels import MERGE
        state = shadow_row.get("_state")
        label = MERGE.get(actual_tool or "")
        if state is None or label is None:
            return
        row = dict(ts=shadow_row.get("ts"), repo=shadow_row.get("repo"),
                   tool=actual_tool, label=label, state=state,
                   pred=shadow_row.get("pred"), conf=shadow_row.get("conf"),
                   level=shadow_row.get("level"), mass=shadow_row.get("mass"))
        OBSERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with OBSERVE_LOG.open("a") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def _fetch(req, box):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                box["body"] = r.read()
                box["status"] = r.status
                box["ctype"] = r.headers.get("Content-Type", "application/json")
                box["stream"] = "event-stream" in box["ctype"]
        except Exception as e:                     # noqa: BLE001
            box["error"] = repr(e)

    def _shadow(self, body):
        from decide.core.state_feed import calls_from_messages, infer_repo, render
        t0 = time.perf_counter()
        try:
            payload = json.loads(body)
            task, calls = calls_from_messages(payload.get("messages") or [])
            repo = infer_repo(payload.get("messages") or [], calls)
            if not task or not repo:
                return None
            state_text = render(task, repo, calls)
            got = HEAD.decide(state_text)
            decide_ms = int((time.perf_counter() - t0) * 1000)
            base = dict(ts=time.time(), repo=repo, n_calls=len(calls),
                        task=task[:100], tau=HEAD.tau,
                        decide_ms=decide_ms, actual_tool=None,
                        _state=state_text)
            if got is None:
                return dict(base, worker_down=True, would_route=False)
            pred, conf = got["pred"], got["conf"]
            level, mass = got["level"], got["mass"]
            return dict(base, pred=pred, conf=round(conf, 3),
                        exit_layer=got["exit_layer"], level=level,
                        mass=(round(mass, 4) if mass is not None else None),
                        would_route=bool(conf >= HEAD.tau and level == "L2"),
                        guard=bool(level == "guard"))
        except Exception as e:                     # noqa: BLE001
            return dict(ts=time.time(), error=repr(e))


UPSTREAM = "http://127.0.0.1:8080"


def main():
    global HEAD, UPSTREAM, LOG, OBSERVE_LOG
    ap = argparse.ArgumentParser(
        description="Shadow proxy: pass-through in front of mlx_lm.server "
                    "that logs what the registry head would have done next "
                    "against what the model actually called. Routes nothing. "
                    "Binds 127.0.0.1 only.")
    ap.add_argument("--port", type=int, default=8090,
                    help="TCP port on 127.0.0.1 (default 8090)")
    ap.add_argument("--upstream", default="http://127.0.0.1:8080",
                    help="base URL of the generation server (a full URL, "
                         "not a bare port)")
    ap.add_argument("--head", default=str(REG),
                    help="registry head artifact (e.g. next_move_observed.json "
                         "after an observe refit)")
    ap.add_argument("--mass-tau", type=float, default=MASS_TAU,
                    help="answer-mass guard floor (see docs/RESULTS.md, "
                         "answer-mass measurement)")
    ap.add_argument("--no-mass", action="store_true",
                    help="skip the full-depth answer-mass read: the L2 forward "
                         "early-exits at the head's block (measured 2.05x, "
                         "bench_mass_removal.py); mass logs as None. Also via "
                         "DECIDE_NO_MASS=1. Behavior-safe: MASS_TAU=0.0 is "
                         "log-only (guard measured non-separating).")
    ap.add_argument("--log", default=str(LOG),
                    help="shadow log path (default is gitignored)")
    ap.add_argument("--observe-log", default=str(OBSERVE_LOG),
                    help="observe log path (default is gitignored)")
    ap.add_argument("--no-head", action="store_true",
                    help="pure pass-through (no shadow logging)")
    args = ap.parse_args()
    UPSTREAM = args.upstream
    LOG, OBSERVE_LOG = Path(args.log), Path(args.observe_log)
    if not args.no_head:
        print("loading head + model (worker process) ...", flush=True)
        HEAD = Head(reg_path=args.head, mass_tau=args.mass_tau, no_mass=args.no_mass)
        print(f"head handle created (tau={HEAD.tau}); worker warms up in the "
              f"background — routing starts when /admin/health shows worker_ready",
              flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"shadow proxy on :{args.port} -> {UPSTREAM}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
