"""Synthetic state generation + teacher labelling (docs/POLICY.md §3).

The measured AnyJev generalization recipe: keep verifier-backed gold labels as
the corpus, SUPPLEMENT with teacher-labelled synthetic states across domains
(~3 exemplars -> N states at temp 1.0; gain appears at >=300 synthetic; never
synthetic-only). This script:

  1. generates synthetic agent-session states in the v3 renderer's exact
     compact format, few-shot off 2 hand-written exemplars per domain
     (python-pytest, typescript-jest, go-test, sysadmin, terraform-k8s);
  2. teacher-labels each with the base model's own L0 readout over the coarse
     question (hard label + conf; hard-or-soft is the recipe, gold stays the
     strong signal);
  3. appends to data/synth/states.jsonl: {ts, domain, repo, state, label,
     conf, probs}.

Memory discipline: one MLX instance, one short-lived supervised process.

Run:  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
        python decide/adapt/synth_states.py --per-domain 12'      # pilot
      ... --per-domain 60                                   # full (300)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
OUT = ROOT / "data/synth/states.jsonl"
TEMP = 1.0
MAX_TOKENS = 420

REPOS = {
    "python-pytest": ["djangorest-api", "ml-pipeline", "flask-billing", "pandas-etl",
                      "fastapi-users", "sqlalchemy-utils", "celery-beat-ext", "pydantic-forms"],
    "typescript-jest": ["react-checkout", "node-graphQL-gw", "nextjs-blog", "vue-dashboard",
                        "express-orders", "ts-orm-lite", "react-native-chat", "jest-fixtures"],
    "go-test": ["k8s-reconciler", "grpc-uploader", "go-http-lb", "etcd-watcher",
                "prom-exporter", "slog-rotate", "gochan-pool", "terraform-prov"],
    "sysadmin": ["nginx-php-fpm", "postgres-primary", "mail-relay", "nfs-homes",
                 "vpn-gateway", "ci-runner", "dns-cache", "log-shipper"],
    "terraform-k8s": ["infra-staging", "payments-platform", "data-warehouse", "edge-cdn",
                      "auth-tenant", "observability-stack", "batch-jobs", "vault-pki"],
}

# 2 exemplars per domain, hand-written in the v3 renderer's exact output shape
# (compact events: read/bash/edit/grep/glob/write, numbered, tails+exits).
EXEMPLARS = {
    "python-pytest": [
        "Repo: ml-pipeline. Task: Two tests in test_transform.py fail after the pandas upgrade. Find the cause and fix it.\nActions so far this session:\n1. read tests/test_transform.py (84 lines)\n2. bash `pytest tests/test_transform.py -x` -> exit 1; FAILED test_fill_missing::test_forward - AssertionError: DataFrames are not equal | assert 3.0 == 3.0000001\n3. read src/ml_pipeline/transform.py (212 lines)\n4. grep `fillna` -> src/transform.py:44, src/transform.py:91, tests/test_transform.py:12",
        "Repo: flask-billing. Task: The CI workflow fails on test_proration; reproduce and repair.\nActions so far this session:\n1. bash `python -m pytest tests/test_proration.py -q` -> exit 1; 1 failed, 33 passed in 4.02s | assert plan.prorate(30) == 15.0\n2. read src/flask_billing/plans.py (150 lines)\n3. edit src/flask_billing/plans.py: `def prorate(self, days): return days * self.rate` -> `def prorate(self, days: int) -> Decimal: (applied)`\n4. bash `python -m pytest tests/test_proration.py -q` -> exit 1; 1 failed, 33 passed | assert Decimal('14.999') == 15.0",
    ],
    "typescript-jest": [
        "Repo: react-checkout. Task: jest suite for the cart reducer fails on the new coupon logic.\nActions so far this session:\n1. bash `yarn jest src/cart` -> exit 1; expect(reducer(...)).toEqual({...coupon}) | Object {\"applied\": true} vs {\"applied\": false}\n2. read src/cart/reducer.ts (96 lines)\n3. grep `COUPON_` -> src/cart/reducer.ts:31, src/cart/reducer.ts:55, src/hooks/useCart.ts:18",
        "Repo: express-orders. Task: Snapshot tests broke after the packaging bump; fix without weakening the assertions.\nActions so far this session:\n1. bash `npm test -- orderService` -> exit 1; 4 snapshots failed | - \"status\": \"processing\" | + \"status\": \"queued\"\n2. read src/services/orderService.ts (240 lines)\n3. read src/services/orderService.spec.ts (180 lines)\n4. edit src/services/orderService.ts: `return { status: 'processing',` -> `return { status: 'queued', (applied)`",
    ],
    "go-test": [
        "Repo: k8s-reconciler. Task: go test ./... fails in the controller package after the client-go bump.\nActions so far this session:\n1. bash `go test ./...` -> exit 1; --- FAIL: TestReconcile | pods \"demo-0\" not found\n2. read internal/controller/reconciler_test.go (310 lines)\n3. grep `demo-0` -> internal/controller/reconciler_test.go:88, internal/controller/fixtures.go:21\n4. bash `go vet ./internal/controller` -> exit 0;",
        "Repo: grpc-uploader. Task: Race detector fires on TestStreamChunk; make the suite green with -race.\nActions so far this session:\n1. bash `go test -race ./upload` -> exit 1; WARNING: DATA RACE | Write at 0x00c0000b4018 by goroutine 7: upload.(*Writer).Flush()\n2. read upload/writer.go (120 lines)\n3. grep `sync.` -> upload/writer.go:22, upload/stream.go:64, upload/stream.go:65",
    ],
    "sysadmin": [
        "Repo: None (sysadmin). Task: Users report 502s from the app tier since the last change; find the cause.\nActions so far this session:\n1. bash `tail -50 /var/log/nginx/error.log` -> exit 0; connect() failed (111: Connection refused) while connecting to upstream 127.0.0.1:9000\n2. bash `systemctl status php8.2-fpm` -> exit 3; Active: failed (Result: exit-code) | Main PID 4472 (code=exited, status=78)\n3. bash `journalctl -u php8.2-fpm -n 30` -> exit 0; ERROR: failed to open configuration file /etc/php/8.2/fpm/pool.d/www.conf",
        "Repo: None (sysadmin). Task: Nightly backups on db01 are slower than last week; diagnose.\nActions so far this session:\n1. bash `systemctl list-timers --all | grep backup` -> exit 0; backup.service backup.timer\n2. bash `grep vacuumdb /var/log/backup.log | tail -5` -> exit 0; vacuumdb: vacuuming database \"analytics\" | WARNING: skipping analysis of table \"events_2026_09\" | copy 14820012 rows\n3. bash `psql -c \"SELECT pg_size_pretty(pg_total_relation_size('events'));\"` -> exit 0; 412 GB",
    ],
    "terraform-k8s": [
        "Repo: infra-staging. Task: terraform plan shows a spurious diff on the node pool after the provider upgrade; make it clean.\nActions so far this session:\n1. bash `terraform plan -out=tfplan` -> exit 0; ~ module.gke.google_container_node_pool.primary | machine_type: \"n2-standard-4\" -> \"n2-standard-4\" (no-op detect) | ~ node_count\n2. read modules/gke/node_pool.tf (95 lines)\n3. grep `machine_type` -> modules/gke/node_pool.tf:12, modules/gke/variables.tf:30",
        "Repo: observability-stack. Task: The prometheus pod is CrashLoopBackOff after last night's apply; find why.\nActions so far this session:\n1. bash `kubectl get pods -n monitoring` -> exit 0; prometheus-0 1/2 CrashLoopBackOff 14 46m\n2. bash `kubectl logs prometheus-0 -n monitoring -c prometheus | tail -5` -> exit 1; err=\"opening storage failed: lock DB directory: resource temporarily unavailable\"\n3. bash `kubectl describe pvc -n monitoring` -> exit 0; prometheus-db-prometheus-0 Bound vol-04ac",
    ],
}


def build_prompt(domain: str, repo: str, rng: random.Random) -> str:
    ex = rng.sample(EXEMPLARS[domain], 2)
    other = ", ".join(sorted(set(REPOS[domain]) - {repo})[:3])
    return (
        "Below are example compact states of an AI coding/sysadmin agent session. "
        "Each state is:\n"
        '"Repo: <name>. Task: <one-line instruction>" followed by numbered compact '
        "tool-event lines (read <path> (N lines) / bash `<cmd>` -> exit <code>; <tail> / "
        "edit <path>: `<old>` -> `<new>` (applied) / grep `<pat>` -> <hits> / glob `<pat>` -> <hits> / "
        "write <path>).\n\n"
        f"Example 1:\n{ex[0]}\n\nExample 2:\n{ex[1]}\n\n"
        f"Now write ONE new state for the project \"{repo}\" (a different project than "
        f"{other}), same domain and format, a NEW realistic task and a plausible "
        "mid-session action history (2-6 actions, some failed commands where it makes "
        "sense). Output ONLY the state, nothing else.\n\n"
        f"Repo: {repo}. Task:"
    )


_STATE_RE = re.compile(r"^Repo: [\w.\-/]+\. Task: .+\nActions so far this session:\n"
                       r"(?:\(earlier actions omitted\)\n)?\d+\. ", re.S)

_HEADER_RE = re.compile(r"Actions[^\n:]{0,40}session:", re.I)


def validate(text: str, repo: str) -> str | None:
    """`text` is the CONTINUATION after 'Repo: <repo>. Task:' — reconstruct the
    full state, normalize the (often slightly malformed) history header, trim
    trailing chatter after the last numbered action, and shape-check. Returns
    the canonical state text or None."""
    full = f"Repo: {repo}. Task: {text.strip()}"
    lines = full.splitlines()
    # the history header lands on its own line (or rarely inline); normalize
    # temp-1.0 malformations like "Actions so so far this session:"
    for i, l in enumerate(lines):
        if "session" in l.lower() and "Actions" in l:
            lines[i] = _HEADER_RE.sub("Actions so far this session:", l, count=1)
            break
    last_action = None
    for i, l in enumerate(lines):
        if re.match(r"^\d+\. ", l):
            last_action = i
    if last_action is None:
        return None
    state = "\n".join(lines[: last_action + 1]).strip()
    if not _STATE_RE.match(state):
        return None
    if len(state) > 4000:
        return None
    non_ascii = sum(1 for c in state if ord(c) > 127)
    if non_ascii > len(state) * 0.02:        # temp-1.0 glyph noise
        return None
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler

    from anyjev import Decider, Question

    from decide.core.labels import CLASSES
    from decide.core.mlx_backend import MLXBackend

    model, tok = mlx_load(MODEL)
    sampler = make_sampler(temp=TEMP)
    rng = random.Random(args.seed)

    reg = json.loads((ROOT / "registry" / MODEL.replace("/", "__") /
                      "base" / "next_move.json").read_text())
    q = Question.choice(reg["question"]["text"], reg["question"]["options"],
                        name=reg["question"]["name"])
    dec = Decider(MLXBackend(MODEL), level="L0")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    t0 = time.time()
    with out_path.open("a") as fout:
        for domain in EXEMPLARS:
            repos = rng.sample(REPOS[domain], len(REPOS[domain]))
            for k in range(args.per_domain):
                repo = repos[k % len(repos)]
                prompt = build_prompt(domain, repo, rng)
                for attempt in range(2):
                    gen = mlx_lm_generate(model, tok, prompt, sampler,
                                          max_tokens=MAX_TOKENS)
                    state = validate(gen, repo)
                    if state:
                        break
                    if attempt == 0:
                        prompt += "\n(remember: output ONLY the state, starting \"Repo: "
                        prompt += f'{repo}. Task: ...\")\n\nRepo: {repo}. Task:'
                else:
                    n_bad += 1
                    continue
                d = dec.decide(state, [q], level="L0")[q.name]
                probs = {o: round(float(p), 4) for o, p in d.distribution.items()}
                fout.write(json.dumps(dict(
                    ts=time.time(), domain=domain, repo=repo, state=state,
                    label=d.argmax, conf=round(d.confidence, 4),
                    probs=probs)) + "\n")
                fout.flush()
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"  {n_ok} states ({n_bad} rejected), "
                          f"{time.time() - t0:.0f}s", flush=True)
    print(f"done: {n_ok} synthetic states -> {out_path} ({n_bad} rejected, "
          f"{time.time() - t0:.0f}s)")


def mlx_lm_generate(model, tok, prompt, sampler, max_tokens):
    from mlx_lm import generate as _g
    return _g(model, tok, prompt=prompt, max_tokens=max_tokens,
              sampler=sampler, verbose=False)


if __name__ == "__main__":
    main()
