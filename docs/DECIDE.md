# DECIDE.md — pivot idea capture: read the next action from the model, don't generate it

> Historical record (2026-10-06), kept verbatim. File paths predate the
> 2026-10-10 reorganization: `decide/<name>.py` = today's
> `decide/{core,serve,adapt,bench,research,authoring}/<name>.py`,
> `decide/results/` = `results/`, `decide/registry/` = `registry/`.
> §8 references files deleted at fork (`fixab.py`, `args_extractors.py`,
> `router_replay.py`, `pipeline/`) and, archived 2026-10-10, the
> seed-dependent `l0_agreement.py` / `l2_head.py` / `guard_stats.py` —
> do not chase those; current work is the README roadmap (docs/ROADMAP.md).

**Status: idea capture only (2026-10-06). Harness/design doc is the NEXT step — deliberately
not attempted here.** Sources: the AnyJev repository (perused in full), a private
production decision-server (read in full for methodology; no code from it
is used — references anonymized 2026-10-08), our PLAN/RESULTS/RESEARCH. The
question this doc answers: *what would we get if the agent asked the local Qwen a typed
question and read the answer out of logits/hidden states in one prefill — instead of decoding
a ~200-token tool call at 25–30 tok/s?*

---

## 1. The idea

Today every agent step = full generation: state + question → 100–300 decoded tokens (think +
XML tool call) → parse → validate. Decode is bandwidth-bound on this machine (PLAN.md §perf),
so ~8s per call no matter how simple the decision.

The pivot: for decisions that are really *choices* (which tool? done yet? which file next?),
replace generation with a **readout** — one prefill, look at next-token logits or a hidden
state at an intermediate block, apply a small fitted head, return a calibrated distribution.
"Nothing is generated and nothing is parsed" (AnyJev README). AnyJev proves the method; the reference server read during this
investigation proves it can be a production serving primitive
(~36.5ms/decision, 0 tokens decoded).

This is NOT a replacement for the fine-tuning investment — the SFT'd model is the substrate
the heads read from, and the verified trace dataset is exactly the calibration label set.
Section 4 maps this in detail.

## 2. What AnyJev is, mechanically (from the repo)

PyPI `anyjev` v0.2.0 (Nokia Applied Research, Apache-2.0, pre-alpha). Core dep: numpy only.
A `Decider` over a pluggable backend; typed questions (`choice` / `noul` yes-no / `score`)
return calibrated probabilities from **one prefill, zero decoded tokens**.

### The levels (docs/levels.md, decider.py)

| Level | Cost per decision | Needs | Quality (20-way, Qwen3-8B, BANKING-style) |
|---|---|---|---|
| raw | 1 prefill | nothing | 0.747 acc, flip 0.230, ECE 0.240 |
| L0 | K prefills (~7 of 18 w/ adaptive rotation budget) | **zero labels** | 0.803 acc, flip 0.073, ECE 0.184 |
| L1 | same as L0 | 100–500 labels (temperature artifact) | 0.807, ECE 0.095 |
| **L2** | **1 prefill through ~⅔ of blocks (0.68× of one forward)** | 100–300 labels + hidden states | 0.771–0.786 (Qwen3-4B 0.786) — above Jev 0.727, fine-tuned Laya 0.768 |

Auto-decidable traffic at ≤5% error: **7.7% → 52.0%** (their headline). Note their "accuracy"
= agreement with a teacher LLM, not ground truth — we can do better (verifier labels, §4).

- **L0 debias:** cyclic-shift permutation marginalization (rotate the K options through the
  letter positions, combine in log space) + label-prior division. Zero training.
- **L1:** temperature scaling on L0, fit on labels. Frozen into an artifact.
- **L2 head:** linear probe on the standardized last-position hidden state at a fixed block
  ≈ 2/3 depth; closed-form solvers (shrunk LDA / ridge / diff-means / RRR — CPU seconds, no
  gradients); layer + solver + temperature selected by 5-fold out-of-fold NLL on the
  calibration set alone. Forward **stops at that block** — no later blocks, no lm_head.
- **Adaptive rotation budget** (L0, opt-in `adaptive_shifts=True`): stop reading rotations
  when log-odds margin ≥ threshold; threshold *certified label-free* (Clopper–Pearson bound
  on disagreement vs full-K). Measured 7.2/18 shifts, 2.2–2.7× decisions/s, accuracy unchanged.

### Backend contract (anyjev/backends/base.py) — genuinely thin

Required: `next_token_logprobs(prompts, token_ids)` — full-vocab log-softmax at the LAST
position, gather requested ids. Unlocks raw/L0/L1. Optional, probed by hasattr:
`score_shared` (prefix-KV fast path for shared state prefixes), `hidden_states` (L2 fitting),
`hidden_states_to(..., max_layer=b)` (early-stop block loop → cheap L2), attrs `n_layers`,
`hidden_size`. MLX is an explicit help-wanted slot (ROADMAP.md lines 60, 77): one file, one
method minimum, parity script vs HFBackend, `@pytest.mark.engine` test.

### Shipped heads (anyjev-heads/) — NOT for our model

Five dense Qwen3 checkpoints only (1.7B/4B/8B/30B-A3B/32B), 23 heads each (20 workflow
questions + 3 bench), 1.8–4.4MB JSON. **No Qwen3.5, no hybrids** → we fit our own, which is
the intended primary workflow anyway (`Decider.fit_head`, min `max(8, 2K)` labels, in
practice 100–300; their curve on Qwen3-8B: 20 labels → 0.654, 100 → 0.740, 300 → 0.772).

Head format: base64 float32 `W [hidden,K]`, `b [K]`, feature `mean`/`scale`, `temperature`,
keyed to exact model ID. `load_artifacts` refuses a model-name mismatch.

### Other machinery worth knowing exists

- `truncate.py`: writes an on-disk truncated checkpoint (blocks 0..b) any server can serve.
  Caveat we must respect: **fit the head on the artifact you deploy** (truncated ckpt pools
  `norm(h_b)`, HF block-loop captures raw `h_b`; likewise 4-bit vs bf16 hidden states differ).
- Label-free test-time adaptation (`adapt="routed"`: re-estimate mean/scale from ≥30
  unlabelled states); online label collection (`observe()` → auto-solves a head at 30 labels,
  refits at 60/120/…).
- Routing: one stored head serves reworded questions / reordered options / truncated builds.
- Demo runs end-to-end on the FakeBackend in <1s on CPU (`python -m demo.jev_mode --backend
  fake`) — zero-GPU way to learn the API.

## 3. What the reference server is, and what it contributes (methodology, not code)

The reference (Rust + CUDA) is a single-GPU inference server for one 27B GDN-hybrid Qwen on
Blackwell SM120a — **cannot run on macOS at all** (no Metal/MLX/CPU path; the CPU backend is
a deterministic mock). None of its compute ports. What transfers is the most engineered
reference implementation of "decide, don't generate" (`POST /v1/decide`, ADR 0034 + 0038–0042
+ docs/specs/decide/01–23):

- **Answer mass** = `exp(lse(answer logits) − lse(all))` — measures how much probability the
  model actually puts on your label alphabet. Doubles as a built-in *fallback signal*: low
  mass → this isn't a readable decision → route to full generation. Median 99.8% mass over
  144 decisions; scales to 256 options.
- **Prompt layout is a measured lever:** evidence in the system block (not user) lifted
  balanced accuracy 0.934 → 0.963. AnyJev puts state in user, the reference in system — A/B on ours.
- **Single-token label alphabet discipline:** admit a label only if it encodes to exactly one
  token that decodes back to itself. (Our tool names — read/edit/bash/grep/… — qualify.)
- **Attention-head readouts for the "where" question** (their `locate`/`point`/`box`):
  calibrated sets of (layer, head) read pre-softmax attention over a key span; content-free
  N/A baseline subtracted; template folding to scale to huge logs. This is the complement to
  AnyJev's choice-heads: choice = *what to do*, attention = *where to look*. Keyed to
  checkpoint content-hash; refused on uncalibrated artifacts.
- **Their honest negative result** (docs/findings/2026-09-21): linear probes on hidden states
  FAILED for spatial pointing (97/240 scenes vs 218/240 for the constrained chain, flat
  learning curve — "the one-pass point via a latent probe is dead on this model"). A fitted
  probe upper-bounds every fixed readout of that residual. Lesson: choice-type reads are the
  proven ones; do not assume every question is readable from a hidden state.
- **Serving pattern:** decisions never occupy a decode lane and hold no residency; fan-out
  questions share one prefilled evidence prefix (measured: 4 questions over one screenshot
  27.9s → 13.2s).
- The `Compute` seam + step ABI (its scheduler and kernel step headers) is the
  architectural template for any "engine leaf answers without generating" design.

## 4. What WE'd get on our substrate (Qwen3.5-4B MLX 4-bit, 32GB, v4 adapter)

**Latency.** One L2 read = prefill of a short state+question prompt through ~⅔ of blocks.
Even at a conservative 500–1,000 tok/s prefill on this machine: ~0.2–0.4s vs ~8s for a
200-token call at 25–30 tok/s → **~25–40× per decision**. Multiple questions over the same
agent state share the prefilled prefix (AnyJev `score_shared` pattern) → each extra question
nearly free. If the agent loop keeps its KV cache across turns, the marginal state prefill
shrinks further (mlx-lm server prompt caching).

**The fast-path router (the realistic product shape).** Not "replace generation" —
short-circuit it: head answers the obvious actions (read the file, run the tests, call done,
needs_review?) when confidence ≥ a calibrated threshold; full generation handles everything
else. AnyJev's 7.7%→52% auto-decidable lift is the reference point; our threshold can be
certified on our own traces label-free (their `calibrate_adaptive` does exactly this shape of
certification).

**Our dataset IS the calibration set.** 532 verified teacher traces = (state → correct next
action) pairs, and *verifier-labeled* (rspec/minitest green) — stronger supervision than
AnyJev's teacher-agreement. Each trace has multiple decision points → thousands of labels vs
the 100–300 per question `fit_head` wants. DPO pair axes (canonical/counter) even suggest
negative/supplementary states for free.

**Our SFT is the substrate, not a sunk cost.** Our measured story: SFT bought weights-level
tool robustness/efficiency/grounding at capability-par — precisely the property that should
make "which tool next" linearly separable in hidden space. Heads read the fused v4 model.
Consequence: **heads are per-adapter** (hidden states shift with weights; the reference keys head
tables to artifact hash for this reason). Refit = re-extract hidden states over N calibration
prompts (minutes) + closed-form solve (CPU seconds). Our candidate-per-adapter workflow
(v4-1400 etc.) needs that discipline baked in.

**Cheap first signal.** L0 needs ZERO labels — run it day-one against the fused model on
held-out trace states and measure agreement with the model's own actual choices. That number
decides whether to invest in L2 heads.

## 5. Idea inventory (everything on the table; harness picks later)

1. **MLX backend for AnyJev** — `anyjev/backends/mlx.py`: `next_token_logprobs` (min),
   `hidden_states_to` early-stop block loop, `score_shared` KV reuse. We already write MLX
   internals (TBPTT patch); GDN hybrid is fine because we own the layer loop (AnyJev's HF
   block-loop balks at exotic archs; MLX has no such constraint). Upstreamable (help-wanted).
2. **L0 zero-label bootstrap** — agreement-with-model measurement before any labeling work.
3. **L2 heads fit on verifier-labeled traces** — labels = tool chosen by teacher at each
   decision point; one head per question kind. Depth self-selects via OOF NLL over
   {50,60,70,85,100}% candidates — empirically answers "is ⅔ depth right for the GDN hybrid?"
4. **Hybrid decision/generation router** — confidence threshold (+ answer-mass guard) splits
   traffic: head fast-path vs full decode. Threshold certified on traces.
5. **Arguments problem (the main functional gap)** — classification gives the action, not the
   args. Candidate ideas, all deferred to harness doc:
   - a) head picks tool + short *constrained* generation fills args only;
   - b) attention-head readout (the reference's `locate` style) for "which file/line" over the context;
   - c) second head over a shortlist when args come from a small set (files in repo, tests);
   - d) digit-constrained readout for numbers/scalars (a reference-server idea).
6. **Answer-mass guard** as the always-on sanity check + fallback trigger.
7. **Prompt-layout A/B** — evidence-in-system vs evidence-in-user on our model (the reference measured +0.029
   bal-acc says this is worth a measured check).
8. **Per-adapter head registry** keyed to adapter/config hash; refuse-on-mismatch.
9. **Online label collection** (`observe`-style) — heads improve from agent telemetry as the
   loop runs; auto-refit at label-count doublings.
10. **Label-free adaptation** — re-standardize mean/scale from live unlabelled states when
    the question is reworded or the listing reordered, no new labels.
11. **Adaptive rotation budget** for L0/L1 multi-option reads — ~2.2–2.7× more decisions/s at
    certified ≤1% disagreement; matters only if L0/L1 become the workhorse.
12. **Head on the exact deployed artifact** — fit on `mlx-community/Qwen3.5-4B-MLX-4bit` (or
    the fused bf16 we actually serve), never on a different quant.
13. **Truncated-checkpoint serving** (AnyJev `truncate.py` idea) — ship a blocks-0..b model
    as a plain smaller MLX model for the decision service; same caveat as #12.
14. **Upstream contribution** — MLX backend + parity script back to AnyJev (their slot).
15. **Evaluation as a third arm** — decision-layer vs generation A/B inside the existing
    `eval_harness` / `e2e_compare` on identical states (spec-restore guard etc. reused).

## 6. Risks / open questions

- **Accuracy = teacher agreement** in AnyJev's bench, not ground truth. Our verifier labels
  are better, but "best possible action" has no oracle beyond outcome; expect noisy labels
  for genuinely contested decision points.
- **Probe readability is task-dependent** — the reference's spatial-probe negative result is the
  caution; choice-type questions are the proven surface. Don't promise reads for everything.
- **Pre-alpha dependency** (v0.2.0); backend contract could shift. Mitigation: the contract
  is tiny; vendor/fork if needed.
- **No shipped Qwen3.5 head; hybrid depth unknown** — resolved empirically by fit_head's
  layer selection; budget for the possibility that the good depth is unusual.
- **Quantization drift** — hidden states at 4-bit differ from bf16; strictly fit-on-what-you-
  serve (#12). Also 4-bit tool-call reliability tax (RESEARCH.md §2) may or may not extend to
  readout quality — measure.
- **Prefill speed on this machine** — ~~assumed ~500–1,000 tok/s from external M4-class
  benchmarks~~ **measured 2026-10-06: 1,015–1,059 tok/s, 165–172 ms/prompt** (short
  175–525-token prompts, per-prompt loop, no KV reuse; longer prompts amortize better). See §8.
- **State length eats the win** — agent contexts are long; the rotation-budget saving itself
  shrinks with context (1.9× at 1,500 tokens). Prefix/KV reuse is not optional.
- **Does the router actually help end-to-end?** — only if a large fraction of agent steps are
  "obvious" actions. Our own telemetry (call sequences per task) can answer this from logs
  before any head is fit.

## 7. Deferred to the NEXT doc: the harness — now written (see HARNESS.md)

Designed 2026-10-06 after steps 1–3, as `HARNESS.md`: fallback-proxy shape in front of the
existing mlx_lm.server, renderer v2 + trace→messages bridge, head registry with per-adapter
refit, deterministic args extractors, staged rollout H1 (offline replay) → H2 (shadow →
live A/B) → H3 (expansions), circuit breaker + tamper-guard guardrails.

## 8. Implementation log

### Step 1 (2026-10-06) — MLX backend + L0 smoke: DONE, pipeline live on the real model

**Built:** `anyjev` 0.2.0 added to `requirements.txt` (nix+uv convention). Fake-backend demo
runs end-to-end in 0.4s CPU (`AnyJev/demo/jev_mode.py --backend fake`). `decide/mlx_backend.py`
(~55 lines): `next_token_logprobs` via one MLX forward per prompt, full-vocab log-softmax at
last position (`last − logsumexp(last)`, float32), gather ids, float64 — plus `n_layers` /
`hidden_size` attrs (needed by `fit_head` later). No padding gymnastics: per-prompt loop, the
Decider dedups before calling. L2 (`hidden_states_to`) and `score_shared` NOT yet implemented.
**Gotcha found:** mlx_lm `qwen3_5` nests config under `args.text_config` (hybrid wrapper);
`mx` has no `log_softmax` primitive.

**Smoke** (`decide/l0_smoke.py`): Qwen3.5-4B-MLX-4bit, K=6 tools (read/bash/grep/edit/glob/
write — the real trace vocabulary), 3 hand-written aasm-agent states, `next_tool` choice
question. Full-vocab = 248,320 (tied embeddings).

| state | raw | L0 | verdict |
|---|---|---|---|
| fresh-task-inspect | bash 0.48 | **read 0.35** | L0 fixes raw's wrong pick (trace stats: read-first) |
| fault-located | **edit 0.88** | **edit 0.88** | both right, high conf |
| edit-applied | bash 0.47 | **write 0.56** | expected bash (run rspec) — L0's debias flips raw's plausible answer ✗ |

**Measured:** prefill 1,015–1,059 tok/s, 165–172 ms/prompt. Per decision wall time: raw =
172ms, L0 (K=6, no adaptive budget, no prefix reuse) = 6 prompts ≈ 1.0s — vs ~8s generated
call → raw ~46×, L0 ~8×, projected L2 (1 prompt × ~0.68 forward) ~130ms ≈ **~60×**.

**Model facts learned (matter for L2):** qwen3_5 = 32 blocks, hidden 2560,
`full_attention_interval: 4` — full attention ONLY at blocks 3, 7, …, 31 (rest GDN linear
attention). Truncating at the usual ~⅔ depth (block 21) drops the last three global-attention
blocks; on this hybrid early-exit loses disproportionately more context mixing than on dense
Qwen3 — `fit_head`'s layer self-selection {50,60,70,85,100}% is the empirical answer, and
depth candidates near full-attention boundaries are the interesting ones.

**Next (step 2 candidates):** real trace states → L0 agreement-with-teacher measurement
(the actual §5.2 bootstrap); `hidden_states_to` block loop for L2; `score_shared` (state
prefix shared by all K rotations — L0's 6 prefills → ~1 + 6 cheap suffixes); re-examine the
edit-applied flip on real states before trusting hand-written ones.

### Step 2 (2026-10-06) — L0 agreement on real trace states: DONE, green light for L2

**Built:** `decide/trace_states.py` — 2,306 decision points from 595 traces (15 repos;
labels: read 855, bash 710, grep 519, edit 194, glob 27, write 1). State = task text +
compact action history (paths, commands, exit codes, output tails, edit diffs — outcome
signals, not file bodies). Fidelity is a documented **lower bound**: the actor saw full
outputs; assistant `text` events deliberately excluded (at router time the decision replaces
the generation that would produce that preamble). `decide/l0_agreement.py` — 120 points,
40 per third (early/mid/late), round-robin over repos, seed 0; results in
`decide/results/l0_agreement.json`.

**Measured (agreement with the actor's actual next tool):**

| metric | raw | L0 | majority baseline |
|---|---|---|---|
| exact 6-class | 0.57 | 0.47 | 0.33 |
| merged 4-class (read+grep→inspect) | 0.61 | **0.72** | 0.42 |
| merged, early third | 0.78 | 0.72 | — |
| merged, mid third | 0.53 | 0.50 | — |
| merged, late third | 0.53 | **0.95** | — |
| s/decision | 0.30 | 1.67 | — |

**Findings:**
1. **raw is a degenerate bash-magnet** — predicts `bash` 74/120 (position/label prior
   dominance; its 0.57 ≈ bash's base rate where it's right). L0's debiasing kills the
   degeneracy (predictions spread to match label distribution) — visible in confusion
   matrices; this is precisely the raw→L0 design story.
2. **The read↔grep split is label noise, not model failure**: 31 of L0's 65 errors are
   read↔grep — the actor's choice between them is interchangeable given the same state.
   Merged, L0 = 0.72 vs 0.42 baseline. **Question-design lesson: ask coarse questions**
   (inspect / run / modify / other), or accept a noise ceiling on fine tool identity.
3. **Late-third 0.95** — with a rich action history the zero-label readout nails the coarse
   next action 19/20 times, exactly where a router pays off most (deep in a task, many
   steps, decode cost compounds). Mid-third is the hard regime (partial history, ambiguous
   phase). Step 1's hand-written `edit-applied` flip was a synthetic-state artifact — real
   late states don't show it.
4. **Confidence is unseparated** (L0: 0.40 right vs 0.36 wrong) — cannot threshold on it
   yet; calibration (L1 temperature, or L2 head with OOF-fitted temperature) is required
   before any routing. Matches AnyJev's own raw/L0 ECE numbers.
5. Timing reconfirmed at scale: raw 0.30s, L0 1.67s per decision (K=6 rotations, no prefix
   reuse — `score_shared` would cut L0 toward ~1 prefix + 6 suffixes).

**Verdict:** signal is present in last-position logits, zero labels used. Proceed to
**step 3: L2** — `hidden_states` + `hidden_states_to` early-stop block loop in the MLX
backend, `fit_head` on these same decision points (labels free), compare exact/merged and
per-third against L0, and let layer self-selection find the right depth for the GDN hybrid
(§8 step-1 finding: full attention only at blocks 3,7,…,31). Calibration comes with the
head (OOF temperature) — re-check the confidence separation then.

### Step 3 (2026-10-06) — L2 heads on the GDN hybrid: DONE, pivot feasibility demonstrated

**Built:** L2 in `decide/mlx_backend.py` — `hidden_states` / `hidden_states_to` block loop
(early stop), HF layer convention honored (0=embeddings, i=block i, n=after final norm,
negatives converted). qwen3_5 backbone path: `Model.language_model.model.{embed_tokens,
layers, norm}`; per-layer mask = `ssm_mask` for GDN layers / `fa_mask` for attention
(mirrors the model's own `__call__`); GDN layers called with `cache=None` — identical math
to a cache-less prefill. **Parity:** manual full loop vs stock forward: max |Δlogit| ≈ 0.037
(quant-noise scale), argmax identical; heads are fit on OUR features, so this only matters
for future artifact portability. `decide/l2_head.py`: fits on 450 points disjoint from the
eval sample, evaluates on the **identical 120 states** as step 2. Two questions: the 6-tool
exact question and the merged 3-class (inspect/run/modify) question. Labels = actor's actual
choice, free from existing traces.

**Measured (eval = step-2's 120 states; all scores on 3-class comparison space where noted):**

| | raw | L0 | L2 6-tool | L2 3-class |
|---|---|---|---|---|
| exact (own classes) | 0.57 | 0.47 | 0.81 | **0.89** |
| 3-class merged | 0.61 | 0.77 | 0.89 | 0.89 |
| early / mid / late | .78/.53/.53 | .72/.50/.95 | .95/.72/1.00 | .95/.72/1.00 |
| confidence right vs wrong | .58/.54 | .40/.36 | **.89/.68** | **.88/.64** |
| s/decision | 0.30 | 1.67 | 0.21 | 0.25 |
| exit depth | full | full | **block 16/32** | **block 19/32** |

Head details: LDA, T ≈ 7, OOF acc 0.74 / 0.80 (≈ eval — no overfit at 450 labels); fit time
~2.5 min per question on this machine (one full forward per calibration state capturing all
5 candidate layers simultaneously).

**Router curve (the §5.4 product question, on the 3-class head):** conf ≥ 0.8 → **80% of
decisions auto-handled at 3.1% error**; ≥ 0.9 → 61% at 0% error; ≥ 0.6 → 92% at 6.4%. Same
regime as AnyJev's 7.7%→52% headline, measured locally.

**Findings:**
1. **L2 clears the bar**: exact 0.81 / merged 0.89 vs L0's 0.47 / 0.77 — squarely in
   AnyJev's published L2 range (0.77–0.79), on a 4-bit quantized GDN hybrid they never
   tested. The pivot's core claim — read the next action from an intermediate layer of our
   local Qwen — works.
2. **The signal lives in the GDN trunk**: layer self-selection picked blocks 16 and 19 of
   32 — BEFORE the last full-attention blocks (23, 27, 31). The step-1 worry (early exit
   disproportionately cuts global attention) is moot for this question: the cheapest
   candidates won, halving compute — 0.21–0.25s/decision ≈ 30–40× vs a ~8s generated call.
3. **Confidence is calibrated enough to route** (0.88–0.89 right vs 0.64–0.68 wrong) —
   step 2's blocker resolved by the OOF-fitted temperature.
4. **Fine tool identity is confirmed label noise at head level too**: 6-tool merged (0.89)
   ≡ 3-class exact (0.89). Ask coarse questions; if the fine split matters, it needs its
   own second-stage question, not a bigger head.

**Caveats:** eval n=120 (router-curve denominators small at high thresholds); truth =
actor's choice (noise ceiling — read↔grep flips are unpenalized in the merged space but
early-vs-mid genuinely ambiguous states exist); single seed; heads keyed to the exact
`mlx-community/Qwen3.5-4B-MLX-4bit` artifact (DECIDE §5.12 respected); base model, not the
v4 adapter (adapter changes hidden states — refit required per §5.8, unchanged procedure).

**Status: steps 1–3 complete — feasibility demonstrated end-to-end** (backend, dataset,
zero-label probe, fitted heads, calibration, router curve). Remaining work is §7 harness
territory: serving shape, router integration into the agent loop, args strategy (§5.5),
per-adapter refit mechanics, and the e2e A/B as a third arm (§5.15). Optional extras:
`score_shared` prefix reuse, upstream MLX backend contribution (§5.14).

### Step 4 / HARNESS H1 (2026-10-06) — offline router replay: DONE, gate MARGINAL

**Built:** `decide/args_extractors.py` (deterministic v1: read path from task text / prior
results; bash = per-repo test-command template, fired only at verify moments — templates
learned for 11/15 repos) and `decide/router_replay.py`: all 2,306 points through the full
router rule (L2 3-class head + extractors), 5 folds grouped **by task** (no leakage), head
refit per fold (450 labels), tau grid + parameterized speedup model (L ∈ {5,8,12}s per
generated call, 0.3s router cost charged on non-routed turns too — conservative in both
directions; routed calls still execute the tool, so L=5 is the realistic figure).

**Head stability under k-fold:** OOF 0.81–0.85 across folds (≥ step-3's 0.80); layer
self-selection again picked cheap depths (16/16/19/19/22 of 32).

**Router table (n=2,306):**

| tau | auto | err\|auto | wrong-fab | no-extract leg | auto e/m/l |
|---|---|---|---|---|---|
| 0.60 | 51% | 7.5% | 6.6% | 33% | 31/42/90% |
| 0.70 | 47% | 6.0% | 5.4% | 30% | 26/37/89% |
| **0.80** | **42%** | **4.3%** | **4.0%** | 28% | 16/31/87% |
| 0.90 | 31% | 1.7% | 2.0% | 24% | 4/20/79% |

**Speedup (proxy shape):** tau 0.8 → **x1.56–1.65**; tau 0.7 → x1.71–1.82; tau 0.5 →
x1.92–2.05 (at 8% wrong-fab).

**Verdict vs the H2 gate (≥2× at ≤5% added wrong-action): NOT MET — marginal at both
ends.** tau 0.8 meets the error bar (4.0%) at x1.6; tau 0.5 meets the speed bar at 8%
error. The honest reading:

1. **The accelerator ceiling under the proxy shape is ~x1.6 at safe thresholds.** Real
   episode wall-clock speedup will be lower still (tool execution time is unsaved).
2. **The gap is in the no-extract leg (24–38% of confident points)** — reads whose path
   isn't visible in context, non-verify bash, and modify-class calls that correctly never
   route. Better extractors (grep-result ranking, rake/minitest repos) could cut the
   routable-miss share roughly in half; the file-shortlist head (H3) is the real fix.
3. **x2+ at safe error is an H4 story, not a proxy story** — owned-loop policy powers
   (termination, no final generation, speculative execution) sit on top of the same
   measured router curve.
4. Late-third routing is where the money already is: **87–90% of late-phase decisions
   auto-handled** at every tau.

**Decision:** proceed to H2 phase 0 (shadow — costs nothing, measures live agreement) with
tau = 0.8; run FIX-AB as planned and let real wall-clock (not the model) decide whether
the proxy shape ships. Treat the 2×-gate as an H4 target, not a proxy requirement.

### Step 5 / HARNESS H2-phase-0 (2026-10-06) — shadow proxy live: DONE, agreement in-family, conf shift flagged

**Built:** `decide/state_feed.py` (the one-pipeline bridge: OpenAI messages → canonical
calls → same render rules as v1; only v1→v2 delta = bash exit codes, genuinely invisible
live); `decide/fit_registry_head.py` → `decide/registry/…/base/next_move.json` (production
head on v2 states: LDA **layer 32/32**, T=6.5, OOF 0.84 — the exit-code info loss pushed
layer selection from 16–19 to full depth; router cost re-measured live at 0.27–2.1s
growing with history); `decide/proxy.py` (transparent pass-through :8090→:8080, shadow
decide on its own 4-bit instance concurrently with upstream generation, SSE/JSON
tool-call parsing, `shadow_log.jsonl` with full provenance); `opencode-shadow.json`.

**Live run:** base model through the proxy on `hashie-090e76` (restore → mutate → red →
`opencode run` → verify → restore): **fixed green in 97s**, 10 LLM turns, proxy added
decide latency only. Two prior runs on `money-ed0907` timed out at 420–480s (base-model
thrash at long history — matches historical base-arm behavior; shadow data still logged).

**Measured (successful session, n=10 turns):**
- **Live merged agreement 7/9 = 0.78** (2 non-tool turns excluded) — in-family with
  offline held-out 0.89 and H1's 4–6% err-given-auto. **No catastrophic renderer shift:
  the bridge holds.**
- **Confidence runs lower live than offline** (0.43–0.92, mean ≈ 0.6 vs offline
  right-answer mean 0.88): at tau 0.8 only 1–2/10 turns would route. Suspects: live actor
  = base model (heads fit on teacher trajectories), live histories longer/messier than
  teacher's. Action: accumulate shadow rows across more sessions before touching tau —
  the shift may be actor-specific, not renderer-specific.
- **Extractor verify-moment rule suppressed one correct high-conf route** (conf 0.92,
  pred=run, actual=bash at n=9 — an edit two calls back didn't count). The bash extractor
  is the conservative leg; loosening it is a measured decision, not a fix.
- One first-call miss (pred=run, actual=glob) — glob is rare in teacher traces (27/2306);
  expected tail.

**Status: H2 phase 0 works end-to-end** (render → decide → route-rule → log → verify
against actual). Next: (a) more shadow sessions across repos/tasks (cheap — the harness is
the e2e protocol with the shadow provider), (b) watch the conf distribution, (c) then
enable routing behind the circuit breaker and run FIX-AB. Both proxy + upstream were
stopped after the smoke; restart via the commands in proxy.py's docstring.

### Step 6 / FIX-AB batch 1 (2026-10-06) — first routing run FAILED (infra), postmortem done, fixed stack verified

**The failure (caught in review, batch aborted):** B-arm episodes degraded into empty event
files and silent 360s hangs. Root causes, in order of blame:

1. **GPU memory exhaustion, both MLX instances** — upstream log shows
   `kIOGPUCommandBufferCallbackErrorOutOfMemory` in mlx_lm's own generation path. Two 4-bit
   instances + long-context generation + decide churn, run past the documented 15–20
   session restart cadence with double footprint. The memory-pressure suspect:
   confirmed.
2. **A wedged decide held the Head lock forever** — route mode runs decide BEFORE
   forwarding; one stuck Metal forward (9.9s decides observed pre-wedge, layer-32 head on
   growing contexts) silenced every subsequent request (proven live: trivial request hung
   40s while /healthz still answered). This is the documented descriptor-cliff signature,
   hit by the decide instance.
3. **The fabrication contract was NOT at fault** — stub test (`decide/stub_server.py`,
   zero MLX): opencode parses our fabricated `tool_calls` (v1 shape, no preamble),
   executes the tool, and continues (1,797 executed calls until the 75s timeout — the stub
   always fabricates the same read). HARNESS §8.1 risk retired with evidence.

**Fixes (all implemented):**
- **Decide worker subprocess** (`proxy.py::_worker_main`): the extra MLX instance lives in
  a supervised spawn-process; parent has a 15s hard timeout; timeout/down → pass-through
  (never route, never block), worker auto-restarts. The request path can no longer wedge.
- **Cheap-depth head refit** (`LAYERS=[16,19,22]` in fit_registry_head.py): layer 16/32,
  OOF 0.82 (vs 0.84 full-depth — 2pts for half the compute). Post-refit live decides:
  **161–376ms** (vs 0.3–9.9s pre-fix).
- **Restart hygiene in the runner** (`fixab.py::restart_servers`): both servers restart
  every 5 episodes and on the wedge symptom (timeout + zero events) — the supervisor
  pattern from the runbook, applied to the batch.

**Verification mini-batch (2 tasks × 2 arms, fixed stack):** no wedge, no restart needed,
routing live:

| task | A (pass-through) | B (routed) |
|---|---|---|
| devise-64cb13 | green, 63.8s, 11 tools | **green, 57.5s, 5 tools, 2 fabricated** |
| aasm-1 | red (timeout), 34 tools | red (timeout), 11 tools, 3 fabricated |

Twin outcomes identical (the only gate that matters held); the devise twin shows the
accelerator signature — fewer tools, less wall-clock, first fabricated call was the
correct opening move (run the failing spec, conf 0.90). n=2 is anecdote, not evidence —
but the **mechanism is proven end-to-end live**: fabricated calls execute, the loop
continues, tasks go green.

**Remaining:** the full frozen 28-task FIX-AB (`decide/registry/fixab_tasks.json`, seed 0,
comparable on identical tasks), then the green-rate/speedup verdict. Servers stopped.

**Batch 1 INTERRUPTED at 21/56 (2026-10-06 evening, machine OOM → force reboot).**
State preserved (incremental `batch_results.json`); all repos restored; no stray
processes. Tallies at interruption: A 4/8 green (3 timeouts); B 6/13 green (7 timeouts),
12/13 B episodes with fabrications (47 calls); 4 twins complete — 3 same-outcome,
**1 flip: aasm-1 A-green/B-red, root-caused from logs**:

- **The flip mechanism (thrash amplification):** the base model entered an edit-loop (21
  edits); the router fabricated the identical verify-template bash after EVERY edit (20
  fabrications, conf 0.884 constant, ~15s/cycle) — a correct-per-call policy that
  tightened a failing loop by stripping the model's reflective generation turns. 45 tools
  → timeout, while A went green in 15 tools. Same mechanism implicated in crack-c6d63e
  (A red in 36s/3 tools — fast quit; B red in 360s/23 tools — kept grinding).
- **The head is exonerated in every recorded incident** — all fabrications were the
  sensible move; live agreement, conf calibration, and move accuracy all held.
- **Fix decided (v1.1, not yet applied to keep v1 measurement pure): repetition guard** —
  never fabricate the same exact (tool, args) ≥3 times per session; after that force
  generation (restores the thinking turn). Log-based analysis says it would have
  suppressed 18/20 aasm-1 fabrications.
- **The help term held the modeled ceiling:** factory_bot twin −15% wall at equal tools
  (2 fabs); crack-db057e green 79s with 3 fabs. 13% live routing rate × ~8s ⇒ the ~10–15%
  proxy ceiling, compressed by the conf shift as predicted.
- **OOM postmortem:** two MLX instances + 66-tool contexts + rspec children exceeded 32GB
  between restarts. Mitigations applied in code: `RESTART_EVERY` 5→3, `RUN_TIMEOUT`
  360→300, `fixab.py --resume` (skips done (task,arm) pairs, appends to results).
  Structural fix remains single-instance convergence (H3) / H4 owned loop.
- **Helper:** `decide/fixab_status.py` — live stats; `--stop` kills runner + restores
  repos + stops servers.

**Resume plan (next session):** `python decide/fixab.py --limit 28 --resume` after
`restart_servers()` → v1.1 guard → log-based reanalysis → targeted rerun of thrash tasks
(aasm-1, crack-c6d63e class) → verdict. Prediction on record: green parity-or-better,
wall −10–15% on uncensored greens, 1–2 more amplification losses before the guard lands.
