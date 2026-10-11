# HARNESS.md — the decision-layer harness (DECIDE.md §7, written after steps 1–3)

> Historical design record, kept verbatim. File paths predate the
> 2026-10-10 reorganization (see docs/DECIDE.md's header note for the
> mapping). The substrate-specific legs (extractors, FIX-AB) were removed
> at fork — docs/POLICY.md carries what replaced them.

**Scope:** how the decide-layer becomes a serving harness around the real agent loop —
shape, components, flow, args policy, refit mechanics, staged rollout, evaluation,
guardrails. Everything here builds on DECIDE.md §8 (steps 1–3 measured facts).

## 1. Assets in hand (all measured, DECIDE.md §8)

| asset | state |
|---|---|
| `decide/mlx_backend.py` | raw/L0/L1 + L2 (`hidden_states_to` early-stop), parity-checked |
| `decide/trace_states.py` | 2,306 decision points from 595 traces; compact renderer v1 |
| L2 head, 3-class `next_move` | 0.89 merged acc, conf .88/.64, exits block 19/32, 0.25s/decision |
| Router curve | conf ≥0.8 → 80% auto at 3.1% err (n=120) |
| Args extractability (measured this session) | read 710/855 (83%) path-from-context; bash 188/710 (26%) test-command-shaped; combined 57% of read+bash volume |
| Serving today | `mlx_lm.server` :8080 (4-bit+adapter), supervisor scripts, hardened |

## 2. Shape decision: fallback proxy in front of the existing server

The agent loop is opencode's, not ours → the router must be **transparent to the client**.
Therefore: a proxy that speaks the OpenAI-compatible API opencode already uses.

**Framing (added after the steps 1–3 review): opencode is the *instrumentation* surface,
not the product surface.** The proxy caps the payoff at "accelerant" (~30% of calls) —
it can short-circuit the LLM call, never the loop policy (termination, cadence, the final
generation). The pivot's full shape — a head-driven fast loop for inspect/run/verify with
generation reserved for modify, `done?`/`needs_review` as *policy* — needs an owned loop,
which we already have (`eval_harness.py`, in-process). So: H2 under opencode buys the
real-world evidence (live agreement, fabrication contract, wall-clock A/B); **H4 moves the
router into the owned loop as policy** (§5). Every asset (backend, heads, renderer,
registry, router rule) is engine-agnostic — the proxy is one deployment, not the idea.

```
opencode ──> decide-proxy :8090 ──(confident + args resolvable)──> fabricated tool_call (~0.3s, 0 tokens decoded)
                    │
                    └──(else: pass-through)──> mlx_lm.server :8080 ──> normal generation (~8s/call)
```

- **Why a separate process, not inside mlx_lm.server:** the :8080 path is hardened
  (supervisors, descriptor-cliff restarts, health checks); the proxy is stateless and
  restartable without touching it. Cost: a second 4-bit model instance (~2.2GB wired) —
  fine on 32GB. Single-process merge is a later optimization (§7 H3), not v1.
- **What opencode sees:** identical API. Fabricated responses are native
  `tool_calls` JSON (parsed form) — NOT XML content. Contract risk #1 (§9): verify exactly
  what `@ai-sdk/openai-compatible` + mlx_lm.server's dialect parsing produce today, and
  replicate it byte-shape for byte-shape before any live traffic.
- Proxy additionally exposes `/decide` (debug/telemetry) and `/healthz`.

## 3. Components

1. **Renderer v2** — messages[] → state text at FULL fidelity (the proxy holds the real
   conversation; the trace renderer v1 was a lower bound). Rules mirror v1 (paths, commands,
   exit codes, tails, diffs; assistant prose dropped) but read from live message structure.
   **The bridge that kills offline→live shift:** traces are rebuilt into messages-shaped
   arrays (task → user msg; tool events → assistant tool_calls + tool results), then rendered
   by the SAME v2 code — calibration data and live traffic go through one pipeline. Heads
   are refit on v2 states before H2 (§6).
2. **Head registry** — `decide/registry/<model_id>/<adapter-hash|base>/<question>.json`:
   the AnyJev head artifact + `renderer_version`, per-question threshold `tau` (from
   calibration curve at target error), `refit_spec.json` (exact decision-point ids used),
   validation row. Refuse to route on any key mismatch (the reference server's hash-keying discipline).
3. **Router** — decide → apply rule: **auto iff conf ≥ tau AND move-class has a registered
   args extractor AND the extractor returns a candidate.** Log the failing leg otherwise
   (low-conf | no-extractor | extractor-miss).
4. **Args extractors** (per-tool, deterministic v1):
   - `read`: path from task text; else top .rb path in prior results not yet read.
   - `bash`(run): per-repo test command template + spec path from task/state (the
     edit→verify moment; late-third where the head is 1.00).
   - `edit`/`write`/modify: **never automated** — content generation required, always
     pass-through.
5. **Fabricator** — builds the tool_calls JSON response + logs provenance
   (question, conf, exit-layer, args source, latency).
6. **Telemetry + circuit breaker** — ring buffer of decisions; trailing-window router-error
   estimate (fabricated call later contradicted/fails) exceeding threshold → routing OFF
   (pure pass-through) + alert. Supervisors unchanged (they own :8080; proxy is stateless).

## 4. Per-turn flow (the answer to "what would the flow look like")

```
POST /v1/chat/completions (messages, tools)
  1. renderer v2: messages -> state text            (~ms)
  2. head: hidden_states_to @ registry depth        (~0.2-0.3s, exits block 16-19/32)
  3. conf >= tau ?  no  -> 6
  4. extractor for the move-class?  no/miss  -> 6
  5. FABRICATE tool_call, log, return               (total ~0.3s; 0 tokens decoded)
  6. pass-through to :8080, generate (~8s), log fallback reason, return
```

Expected v1 envelope (from measured pieces): head auto-eligible ~80% × args-extractable
57% × read+bash share 68% ≈ **~30% of all tool calls fabricated at ~0.3s instead of ~8s**,
concentrated in late-phase verify loops where the head is most accurate (late-third 1.00).

## 5. Staged rollout

**H1 — offline router replay (`decide/router_replay.py`).** No proxy yet. All 2,306 points
through the full router rule (head + extractors) under k-fold refits (heads re-fit per fold
on v2-rendered states; ~75–90 min total). Deliverables: tau table at target errors, auto
rate / error / estimated speedup (join `data/telemetry` decode latencies), by-third. **Gate:
estimated ≥2× mean episode speedup at ≤5% added wrong-action rate → H2.**

**H2 — live proxy, shadow first.** Phase 0 (shadow): proxy routes NOTHING; logs what it
would decide + conf alongside the model's actual call; e2e unchanged. Measures live
agreement vs offline (renderer-shift check) on real sessions. **Gate: live agreement within
a few points of H1 → enable routing.** Phase 1: **FIX-AB** (below) through the proxy —
`e2e_compare.py` arms base | adapter | adapter+router, tamper guard active.

**H3 — expansions (each independently gated):** `done?`/`needs_review` noul heads (episode
termination policy; interacts with the DPO persistence axis — measure, don't assume);
file-shortlist readout for read-args (K = repo paths — variable-K question, design needed);
`observe()` flywheel from proxy telemetry (auto-refit at label-count doublings); KV-prefix
reuse between decide-prefill and fallback generation; single-process convergence (merge
proxy into the model server; decide prefill becomes the generation prefix).

**H4 — the owned loop: router as policy (`eval_harness`-native).** Integrate the router as
a first-class decision point inside the in-process loop we own: head-driven fast path for
inspect/run/verify (no API boundary, no fabrication contract — the loop calls `decide`
directly and acts); generation reserved for modify and low-conf moves; `done?`/`needs_review`
heads become loop policy (termination, escalation); speculative tool execution (fire the
predicted read/bash while the modifying generation is still streaming) becomes reachable.
This is where the runner/thinker architecture actually lives. Reuses FIX-AB (below) as its
A/B; the H2 proxy remains as the opencode-compatible serving surface.

### The A/B protocol — FIX-AB (the mutation-fix revert methodology, with/without the decide layer)

Same paradigm as the original pipeline (mutation-fix tasks + git state restore), applied to
the router. This is the mandatory measurement for BOTH H2-phase-1 and H4 — no claim about
the pivot ships without a FIX-AB run.

- **Task subset:** 30–40 tasks sampled from the 595, stratified by repo × calls-per-trace
  bucket (1–2 / 3–5 / 6–9 / 10+) so early/mid/late regimes are all represented; fixed seed,
  frozen list committed to `decide/registry/fixab_tasks.json` before any run.
- **Arms:** A = the loop without the decide layer (pure generation — the current behavior,
  same model + adapter); B = the identical stack with the router enabled (proxy in H2,
  native in H4). Nothing else differs: same model, same adapter, same temperature/sampling,
  same tool set.
- **State restore between every run:** repo checked out to the task's mutated snapshot
  (the existing `setup_repos`/snapshot mechanism — same one the traces' `step_start`
  snapshots record), plus the spec-restore tamper guard active during verification. A run
  never sees a repo state its twin didn't.
- **Order:** interleaved and randomized (A,B,B,A,…) so drift (server restarts, thermal,
  descriptor-cliff cadence) hits both arms equally; restarts logged per run and carried as
  a covariate.
- **Metrics per task (both arms):** spec-green rate; wall-clock to green; tool-call count;
  fabricated-call share + wrong-fabrication incidents (arm B); calls-to-first-edit; tokens
  decoded. Aggregate: mean/median speedup, added-failure delta (arm B green rate must not
  drop >2pts vs arm A — hard gate), and the router report (fallback-leg distribution,
  circuit-breaker trips).
- **Scale:** 30 tasks × 2 arms ≈ 60 episodes ≈ ~2h at historical 25–30 eps/h (arm B
  faster); extend to 2 seeds / 40 tasks only if the speedup confidence interval straddles
  the gate.
- **Honesty rules:** tasks whose green outcome differs between arms are replayed once from
  the transcript to classify the cause (router error vs model variance); zero-fabrication
  episodes in arm B are counted as "router chose not to route" (a valid outcome, reported
  separately from failures).

## 6. Per-adapter refit mechanics

`decide/refit.py --model M --adapter A`: rebuild states from `refit_spec.json` ids through
renderer v2 → extract hidden states → closed-form solve → validate held-out → write
registry artifact. Minutes per adapter. Runbook hook: run after every `mlx_lm.fuse`.
Base-model heads never serve an adapter-backed session (registry keying refuses).

## 7. Evaluation & guardrails (summary)

- Offline: k-fold, multi-seed; router metrics as in H1.
- Online: **FIX-AB** (§5) is the standard A/B — used unchanged at H2 (proxy) and H4 (native
  loop); `openended_ab` for synthesis tasks; per-session router report appended to
  `data/telemetry/`.
- Guardrails: tau from calibration (never hand-picked); circuit breaker; shadow-before-live;
  modify-class never fabricated; every fabricated call logged with full provenance.

## 8. Risks / open questions

1. **Fabrication contract** — RETIRED with evidence (DECIDE.md §8 step 6): the stub test
   proved @ai-sdk parses the fabricated tool_calls shape and the loop continues executing
   them.
2. **Actor-agreement ≠ task-correctness** — the head predicts what the actor did; a
   fabricated call must also be RIGHT for the episode. Shadow mode + e2e measure this;
   circuit breaker bounds the damage.
3. **Renderer shift** — killed in principle by the one-pipeline bridge (§3.1); shadow mode
   is the empirical check.
4. **Mid-third weakness (0.72)** — router under-routes there by design; acceptable.
5. **Tool-schema drift** (opencode tool changes) — registry keyed per question+options;
   extractor registry per tool name; mismatches fail to pass-through, never to fabricate.
6. **Memory / wedge** — RESOLVED operationally (step 6 postmortem): decide runs in a
   supervised worker subprocess with a hard timeout (wedge can only cost one timeout, then
   pass-through); the runner restarts both servers every 5 episodes and on wedge symptoms.
   Long-term answer remains the single-instance convergence (H3).

## 9. Out of scope v1

Attention-head readouts (reference locate-style); fine-tool-identity heads (proven label
noise); constrained-argument generation; adaptive rotation budget; upstream AnyJev PR.
All revisit after H2 numbers.
