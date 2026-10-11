# RESULTS.md — every measured finding

The evidence archive: every benchmark, ablation, and latency measurement this
project has run, with the numbers to quote and the honest readings. Entry
point lives in `README.md`, orientation and vocabulary in `CONCEPTS.md`; the method's investigation record
is `DECIDE.md` + `HARNESS.md`. Artifacts are frozen in `results/` (paths
below are relative to the repo root; `decide/results/...` in older notes =
today's `results/...` — the tree was reorganized 2026-10-10).

Wording rule that governs every claim below: fidelity claims say "aligned
with the reference", never "more correct"; accuracy claims name the arena
and model size.

Level rule: every benchmark and eval below ran at label-free **L0**
(zero labels, zero fitting, frozen base) unless the section says L2. L2
appears only in the shadow/live and serving-layer measurements (the
`next_move` head); L1 has never been measured. Per-artifact levels:
`results/INDEX.md`.

**Contents**

- [Eval vs real Jev](#eval-vs-real-jev--fidelity-on-non-coding-domains-2026-10-07-openrouteraiapialphadecisions): fidelity on non-coding domains (n=300 plus the long-context batch)
  - [L0-vs-raw ablation](#l0-vs-raw-ablation-what-the-debiasing-stack-buys): what the debiasing stack buys
  - [Evaluation disclosure](#evaluation-disclosure): how the items were authored and what that limits
  - [Disagreement anatomy](#disagreement-anatomy): the 22 misses of the first 105 questions
- [BANKING77](#banking77--gold-label-accuracy-2026-10-07): gold-label accuracy
  - [Rotation budget](#rotation-budget--the-cheap-version-of-clefs-one-pass-trick): the cheap version of Clef's one-pass trick
  - [Full-77-way readout](#full-77-way-readout--the-26-option-unlock-2026-10-08): the >26-option unlock
  - [Full-77-way live Jev leg](#full-77-way-live-jev-leg-2026-10-08)
- [typed-decisions](#typed-decisions--the-zero-training-head-to-head-2026-10-08): the zero-training head-to-head
- [Decide vs generate](#decide-vs-generate--the-same-model-asked-two-ways-n300): the same model, asked two ways (n=300)
  - [Latency and GPU regimes](#latencygpu-regimes)
- [Serving-layer measurements](#serving-layer-measurements): state-level prefix sharing, prefix-KV, mass-read removal, 9B mechanics
- [Live re-read](#live-re-read-v3-renderer--promoted-head-2026-10-07): v3 renderer and promoted head
- [Answer-mass guard measurement](#answer-mass-guard-measurement-non-separating): non-separating

---

## Eval vs real Jev — fidelity on non-coding domains (2026-10-07, openrouter.ai/api/alpha/decisions)

Everything measured before this was coding/agent-shaped; this eval closes
the gap: `data/eval/questions.jsonl` (21 states x 5 = 105 questions, ZERO
coding — support tickets, IT ops, phishing, incident command, expense/
claims/fraud, moderation, contracts, HR, sales/CS), run against BOTH our
endpoint and real Jev (`~typesafe/jev-latest` → served `typesafe/
jev-1.13-20260917`, $0.0005 for the whole run). Harness:
`decide/bench/eval_vs_jev.py` (auto-translates the OpenRouter dialect — lowercase
types, choice `criteria` object defines options, score `criteria` = levels;
normalizes responses back). First batch (`results/jev_compare.json`):

| metric | ours (local, 4-bit 4B) | Jev (API) |
|---|---|---|
| exact-answer agreement (**fidelity to reference**, not accuracy) | **79.0% overall** (Choice 88%, Noul 84%, Score 60%) | — |
| Wilson 95% CI / chance-corrected kappa | 70.3–85.7% / **κ = 0.65** (Choice 0.84, Noul 0.67, Score 0.42; chance P_e = 0.40) | — |
| Score misses | all 10 are ONE-level boundary confusions, 0 far (25/25 within one level) | — |
| latency p50 / p95 | 1548 / 1795 ms | 671 / 780 ms |
| input tokens (sum) | 20,652 | 11,922 |
| resources | RSS 2.79GB, GPU 50% mean / 93% peak, CPU 6% of 1 core | — |

**Expansion to n=300** (2026-10-07, same day): a second authoring batch —
`data/eval/questions_v2.jsonl`, 39 states × 5 = 195 new items, non-coding
(shadow-IT, BEC/wire fraud, cold-chain pharma, ADA/facilities, ombudsman-
level claims, public-procurement rules, wage disputes, ransomware
containment, …), same schema/mix — run the same way
(`jev_compare.n300.json`, $0.001) and pooled with the first batch
(`decide/bench/pool_evals.py` → `jev_compare.pooled.json`):

| batch | n | agreement | Wilson 95% | κ | Choice / Noul / Score |
|---|---|---|---|---|---|
| v1 (first 105) | 105 | 79.0% | 70.3–85.7 | 0.65 | 88 / 84 / 60 |
| v2 (new 195) | 195 | 73.8% | 67.3–79.5 | 0.56 | 69 / 79 / 67 |
| **pooled** | **300** | **75.7%** | **70.5–80.2** | **0.59** | **75 / 81 / 65** |

Two honest readings. (1) The v2 items measured HARDER than v1 (73.8 vs
79.0) — a different authoring batch with more multi-factor judgment calls;
the pooled 75.7% (CI ±5) replaces the pilot's 79.0% (CI ±8) as the number
to quote. (2) The reference itself is nearly deterministic: **Jev
self-agreement = 98.7%** over 3 same-day repeats of the 105
(`decide/bench/jev_repeat.py` → `jev_repeat.json`; pairs 98.1–99.0%, 103/105
items stable, cost ~$0.0015). So ~99% is the practical ceiling any
deterministic system could reach against single Jev samples — our 75.7% is
a real fidelity gap, not reference noise.

**Long-context batch (n=50, separate regime — not pooled with the 300).**
`data/eval/questions_long.jsonl` (10 states × 5 questions, authored by
`decide/authoring/author_long.py`: operational case files — incident logs with
metric series, multi-party threads, fraud case history, reefer temp logs,
access-audit trails — states span ~0.9k–4.8k tokens, median ~2k; LLM-
assisted cores + template-generated structured bulk, values consistent
with each story). Run the same way → `jev_compare.long.json` ($0.0016):
**80.0% agreement (Wilson 67.0–88.8, κ 0.66; Choice 85 / Score 83 / Noul
76)** — agreement on long contexts is HIGHER than the short-context 75.7%
(more shared evidence per question), and greedy decoding COLLAPSES here
(70.0% overall, Choice 46% — see the decide-vs-generate section). Latency was
the honest weakness of the naive pipeline: L0 p50 18.8s per 5-question request
(each question re-prefilled the whole state once per rotation) vs Jev 815ms —
**the state-level prefix-sharing fix (landed the same day) brings the same
requests to ~3.7s mean = 5.3×, with byte-identical answers up to the
documented bf16 boundary** (see `state_share.json` and the latency regimes
table below; `--max-permutations 2` compounds it further).

**Blind external-judge leg (2026-10-08, `labels_long.scored.json`)**:
30 of the 50 long items (6 states, length-stratified 3.3k–17.6k chars)
labeled blind by Claude Opus — fresh session, sheet only, different
model family from both the base (Qwen) and Jev. This is
**judge-agreement, not human-verified accuracy**. Results: **ours
20/30 = 66.7%, Jev 25/30 = 83.3%** — the independent judge corroborates
the BANKING77 direction on judgment quality (the served reference
really is better than us); where the judge split our disagreements with
Jev, it sided with Jev 7 times, us 2. All 7 of our score misses are
within-one-level (3/7 exact). And the check we ran on our own headline:
**readout vs greedy generation against the judge = 20 vs 19 of 30 —
NOT separated at this n** (one discordant pair; the 80-vs-70
Jev-agreement gap does not replicate as a judge-labeled gap at n=30 —
the generation-collapse claim stands on the agreement evidence, not on
accuracy evidence yet). Reference stability in the LONG regime, measured
after the fact (3 fresh Jev passes, `jev_repeat.long.json`): **98.0%
self-agreement, 48/50 items stable** — same ceiling class as the short
regime's 98.7%, so the 80.0% long-agreement number sits under a ~98%
ceiling too. Notably both of Jev's unstable items are judge-leg
disagreement items (`carrier_liability` — where the judge and stored-Jev
counted us wrong — and `history_mitigates`, where fresh Jev runs flip
toward the answer we and the judge gave): those two scored rows are
reference noise as much as our error. Judge also flagged three dataset defects
(authoring artifacts, none changed its labels): a delivery log with
impossible dates, a qid that does not match its question text, one
internally inconsistent metric line — cleanup in the backlog (docs/ROADMAP.md).
The human-label item remains open, shrunk to adjudicating the ~10
disagreement items — that is the upgrade from judge-agreement to
accuracy.

Reading (first batch, still the fine-grained anatomy): a local 4-bit 4B
readout reaches ~79% exact agreement (100%
within-one-level on score) with a served reference model on non-coding
domains, at ~1.5s/5-question request. At n=105 the CI is ±8 points —
**pilot scale, not a benchmark** (accepted benchmarks run 10⁴ items); the
pooled n=300 tightens ±8 → ±5. Agreement measures shared judgment, not
correctness — treat it as fidelity. First full run measured 78.1% (82/105);
the stored re-run 79.0% — one-item run-to-run variance on the live-API
side, consistent with the 98.7% self-agreement. The confidence delta
(mean 0.128 on v1) and JSD (0.195) say our distributions are usable-but-
different; the token gap (20.6k vs 11.9k) is the permutation-
marginalization readout cost — prefix-KV already cut it, deeper cuts =
fewer permutations.

Re-run any comparison (server on :8095; key from `$OPENROUTER_API_KEY`):

```bash
python decide/bench/eval_vs_jev.py
```

**GPU stats, pure stdlib, no sudo** (2026-10-07): per-process GPU% is read
from the stock `ioreg` CLI (`ioreg -a -r -c AGXDeviceUserClient -l`): each
registry entry carries `IOUserClientCreator` ("pid N, name") and an
`AppUsage` array of `accumulatedGPUTime` (ns) — summed per pid, delta'd
over a monotonic interval = GPU% (same IORegistry source as Activity
Monitor; validated 96% GPU attribution under live MLX load). Nothing is
installed. Measured-broken paths, avoided: `ps -o pcpu` (decaying lifetime
average) and third-party per-pid cputime helpers (read 0 for non-self
pids).

### L0-vs-raw ablation (what the debiasing stack buys)

`decide/bench/l0_ablation.py` → `l0_ablation.json`: what the debiasing stack
buys over the naive single-prompt letter-logit read (the Mini-Jev-style
read) on the same 105 questions. raw 74.3% (κ 0.57) → L0 79.0% (κ 0.65):
**+4.7 points, all of it from permutation marginalization** — 7 answers
change, 5 flip to the reference's answer, 0 flip away; and the served
config's batch prior never activates on this eval (105/105
prior_method="none": each question layout appears once, min_prior_n=8).
Two decomposition facts: raw (74.3%) is numerically the same as
constrained greedy decoding (74.3%, next section) — so NONE of the
readout's edge over generation comes from the label-prompt constraint, all
of it is the debiasing; and the full-stack content-free prior (Zhao-style
probes, NOT what the endpoint serves) measures WORSE than marginalization
alone here (72.4%, Noul κ 0.67→0.49) — prior correction helps when a
question layout repeats across states, and hurts on single-shot layouts;
the served default is the right one. K-curve (choice): m=1 72% → m=2 80%
→ m=3 88% = full-K 88% — two rotations capture ~90% of the gain at half
the full-K cost (3-choice items reach their K at m=3); served latency
m=1/2/3 = 169/313/404 ms per question.

### Evaluation disclosure

The 350 items across three batches (v1 105 + v2 195 short-context; long 50
with authored cores + template-generated log/timeline bulk) were authored
with LLM assistance during development, carry **no human ground-truth
labels**, and this project's authors also built the system under
evaluation. Known validity limits of that setup: LLM-authored items skew
easier and stylistically toward the author; agreement can be inflated by
shared priors — and the v1→v2 agreement drop (79.0 → 73.8) shows the
number is batch-sensitive, which is itself a reason to quote the pooled CI
rather than a best batch. The items and the scoring harness ship in-repo
(`data/eval/questions{,_v2,_long}.jsonl`, `decide/bench/eval_vs_jev.py`,
which reports Wilson CIs + kappa; `decide/bench/jev_repeat.py` measures the
reference's own run-to-run noise at 98.7%) so the run is replicable end to
end; independent, human-labeled evaluation is explicitly invited
(CONTRIBUTING.md).

### Disagreement anatomy

(22 misses, raw answers logged in `jev_compare.json` rows):
- **Zero definite-vs-definite flips** (no question where both models are
  ≥0.8 confident in opposite answers) — but at n=22 the exact 95% upper
  bound on the flip rate among disagreements is **12.7%**
  (1−0.05^(1/22)); n is below the rule-of-three comfort zone (n≳30) and
  items share 21 states (correlated), so "the catastrophic mode is
  absent" is provisional, not established.
- Confidence drops 0.78→0.60 exactly on disagreements (Jev: 0.79→0.59) —
  the readout knows when it doesn't know; mean signed conf drift ~0.
- Score misses are all one-level and skewed: 7/10 we sit one notch LOWER
  (our "medium" vs their "high") — we grade risk calmer than Jev, a
  temperament/calibration difference, not error.
- Exactly ONE sharp overconfident miss per ~100 questions: a niche
  regulatory-fact Noul (UK manual-language rule) at 0.93 — the 4-bit base
  lacks the fact and uncalibrated L0 amplifies the gap into false
  certainty. Temperature calibration (L1) is the targeted fix.
- **τ=0.8 transfers to non-coding L0**: 42% auto-rate at 2.3% error
  (44/105 routed, 1 disagreement — the same 0.93 outlier; Wilson 95% CI
   on the routed error 0.4–11.8%; τ=0.9 does not evict it). Nearly
   identical to the seed-domain curve it was fit on (42% @ 4.3%); contrast the
  coding-domain live re-read where routed error was 37% — τ must be
  re-read per surface before routing.

---

## BANKING77 — gold-label accuracy (2026-10-07)

`decide/bench/bench_banking77.py` → `banking77.json`: the public
mteb/banking77 test split (one of the benchmarks in Cloudflare's Clef table),
400 items — top-10 intents × 40, identical options to both sides, gold labels
from the dataset. Unlike everything above, this measures **accuracy, not
agreement**:

| | ours (local 4-bit 4B) | Jev (API) |
|---|---|---|
| accuracy / macro-F1 | 87.5% / **0.876** (Wilson 84.2–90.2) | **95.5% / 0.954** (93.1–97.1) |
| pairwise agreement | 88.5% (85.0–91.3) | — |
| latency p50 | 1552 ms | 742 ms |

Reading, stated plainly: **on a neutral public benchmark the served
reference is better than us by 8 points** — same ballpark, different
class; the CIs do not overlap, so the gap is real, not sampling noise.
**The 9B base closes most of it (2026-10-08, `--model
mlx-community/Qwen3.5-9B-MLX-4bit`, L0-only, zero training:
`banking77.9b.json`): 91.25% / 0.912 macro-F1 (Wilson 88.1–93.6), p50
1973 ms** — +3.75 over the 4B, gap to Jev 8.0 → 4.25 points, CIs now
overlap by half a point (93.1–93.6). Capacity is the lever on this
zero-shot task, exactly as the size-axis hypothesis predicted.

**Resource profile (2026-10-08, same-day 60-item slice per model, the
stdlib ps+ioreg sampler; `banking77.{4b,9b}.resources.json`)**: RSS
2.80 → 5.34 GB (×1.9 — comfortable in 32 GB), CPU idle on both (6–8%
of one core), GPU saturated while serving on both (92.7% vs 83.1%
mean) — the 9B's cost is **pure GPU-time**: same-slice p50 1110 →
1983 ms = **×1.79** (the trustworthy within-session ratio; cross-session
absolute p50s vary ±40% on this machine). The honest trade, stated
plainly: on zero-shot tasks the 9B buys points with GPU-time (+3.75 for
×1.79); on train-friendly tasks (typed-decisions) the same ×1.79 buys
nothing — the crossover verdict again, measured in seconds. The
untested latency claw-back for the 9B is the m=2 × state-share
compound (backlog).

The 10-way restriction is NOT comparable to the published full-77-way
lines; the comparable pair there is the same-subset 0.876 vs 0.954. Our
errors cluster on semantically adjacent intents (apple/google pay →
automatic top-up ×15; beneficiary not allowed → cancel transfer ×14) —
fine-grained transfer/card vocabulary the 4-bit 4B base cannot separate.
Where the niche still holds: local/private, zero marginal cost, full
distributions — not raw classification quality.

### Rotation budget — the cheap version of Clef's one-pass trick

(2026-10-07, `decide/bench/rotation_budget.py` → `rotation_budget.json`):
Cloudflare trained order-robustness in so Clef scores all choices in ONE
prefill; our label-free substitute pays one prefill per rotation. Measured
on the same 400 BANKING77 items:

| config | prefills/decision | accuracy | macro-F1 | p50 |
|---|---|---|---|---|
| m=1 (+batch prior, active on repeated layouts) | 1 | 85.5% | 0.855 | 150 ms |
| **m=2** | **2** | **86.3%** | **0.863** | **407 ms** |
| m=3 | 3 | 87.0% | 0.870 | 536 ms |
| full K=10 (the served default) | 10 | 87.5% | 0.876 | 1552 ms |
| adaptive stop (anyjev default threshold) | 9.6 | 87.5% | 0.876 | 1987 ms |
| Jev (API) | — | 95.5% | 0.954 | 742 ms |

Capping at **m=2 costs 1.25 points and saves 3.8×** — and makes the local
endpoint *faster than Jev* on this task (407 vs 742 ms). The K-curve shape
replicates the v1-eval ablation (steep to m≈3, flat after) on an external
gold-label benchmark — two independent datasets agree. The certified
adaptive stopping reads 9.6/10 shifts at its stock threshold (too
conservative to help at K=10; per-question calibration exists but the
static cap is the practical lever). `--max-permutations` is a serving
flag on `decide/serve/systemone.py` (default: full K, unchanged behavior).

### Full-77-way readout — the >26-option unlock (2026-10-08)

`decide/bench/banking77_full77.py` → `banking77_full77.json` / `.9b.json`: 77
options exceed the letter readout (≤26-option Choice cap = an anyjev-0.2.0
single-token alphabet artifact), so options are scored by **multi-token
loglikelihood** (the lm-evaluation-harness standard; `MLXBackend.score_suffix_sum`:
one prompt forward + 77 sum-log-softmax continuation passes off copies of
the shared cache per rotation) with the same L0 math as the served pipeline
(m=2 cyclic rotations, batch prior 0.75, logmean marginalization), zero
training on the same 77 intents. Items: all 77 classes × 6–7 = **500 gold
rows** (deterministic stratified selection, Unsloth's cut size). Both legs
get the identical prompt; greedy decodes temp-0 and its typed answers are
parsed longest-first containment (1/500 and 0/500 parse failures at 4B/9B).

| full-77-way, 500 gold items | 4B readout | 4B greedy | 9B readout | 9B greedy |
|---|---|---|---|---|
| accuracy | 63.4% (59.1–67.5) | **66.8%** (62.6–70.8) | **66.4%** (62.1–70.4) | 60.8% (56.5–65.0) |
| macro-F1 | 0.617 | 0.649 | **0.647** | 0.582 |
| no-prior ablation | 57.0% | — | 64.8% | — |
| rotation flips | 127/500 | — | 60/500 | — |
| latency p50 | 5,446 ms | 601 ms | 9,295 ms | 1,127 ms |
| paired exact McNemar | 30 vs 47 **p = 0.068** (greedy ahead, n.s.) | | 57 vs 29 **p = 0.0034** (readout ahead) | |

Reading, both directions honestly: **at 4B the 77-way arena does not
separate the two (greedy nominally ahead 3.4 points, p = 0.068); at 9B the
zero-training readout BEATS generation on public gold — +5.6 points, exact
McNemar p = 0.0034 — the accuracy claim's first significant positive
leg.** Capacity moved the legs in opposite directions here: readout 63.4 →
66.4 (+3.0), typed-answer greedy 66.8 → 60.8 (−6.0; its misses are
semantically plausible adjacent intents, e.g. activate→get physical card,
not format failures). Same-stack findings: (a) the batch prior is worth
+6.4 at 4B but only +1.6 at 9B (57.0→63.4 vs 64.8→66.4) and rotation flips
halve (127→60/500) — the stronger base is less position-biased to begin
with; (b) speed: the readout pays ~9× p50 at 4B (5.4 s vs 0.6 s) and ~8×
at 9B (9.3 s vs 1.1 s) for the fixed 77-candidate scoring. Context, not
comparator: published full-77-way macro-F1 Clef 94.2 / Clef-flash 90.9 /
Jev 79.7 (full test split); Unsloth TRAINED 0.8B 74% / 2B 58% accuracy
(their 500-row cut, in-distribution — banking77 is in their training mix).
Our zero-training 9B readout (0.647 F1) sits between their two trained
models and far below the hosted reference on this arena.

Implementation facts measured, not guessed: bare option-text continuation
beats `" text"` by ~8 nats (chat template tail = `assistant\n<think>\n\n</think>\n\n`);
greedy emits `1. <text>` and the parser strips the list number; shared-vs-flat
boundary parity 0.0001–0.016 nats (the documented bf16 class). Gotcha that
bit during smoke: score continuations in PERMUTED (position) order — index
j must be listing line j, or `marginalize` scrambles the rotation-1 mapping.

### Full-77-way LIVE Jev leg (2026-10-08)

`decide/bench/banking77_full77_jev.py` → `banking77_full77.jev.json` — Jev
ran the SAME 500 items with the SAME 77 options through the decisions API
(raw message as state, same client/dialect as the 10-way leg; ~$0.003,
0 failures, p50 802 ms). **Jev 78.6% (74.8–82.0) / 0.770 macro-F1** — the
published 79.7 transfers to our harness almost exactly, so the
published-table gap is real on same-items ground:

| vs gold, 500 items | Jev | 9B readout | 9B greedy | 4B readout | 4B greedy |
|---|---|---|---|---|---|
| accuracy | **78.6%** | 66.4% | 60.8% | 63.4% | 66.8% |
| McNemar (only-Jev-right vs only-ours-right) | — | 71 vs 10, p<1e-5 | 99 vs 10, p<1e-5 | 89 vs 13, p<1e-5 | 70 vs 11, p<1e-5 |

Readings: (a) Jev decisively wins this arena at both sizes — we beat it on
only ~10/500 items it misses; the 77-way gap (−12.2 at 9B) is wider than
the 10-way same-subset gap (−4.3), consistent with fine-grained vocabulary
being a trained-model strength; (b) the readout is still the more
Jev-LIKE leg (9B Jev-agreement 0.778 readout vs 0.714 greedy) — even where
both trail on gold, the internal read tracks the reference better;
(c) Jev's top confusion is the SAME adjacent-intent pair as our 9B greedy
(order physical card → get physical card ×5) — those look-alike intents are
intrinsically hard, not a harness artifact; (d) the 77-option Choice
passes through the decisions API cleanly (0 unparsed answers) — confirming
the ≤26-option cap is our letter-verbalizer artifact, not the API's.

Row-level analysis notes (same artifacts): rotation instability is a
difficulty proxy — items whose argmax flips under rotation run 46.5% (4B)
/ 33.3% (9B) accuracy vs ~70% for stable items; conf is overconfident in
the mid-range (ECE 0.083 / 0.106, 9B mid bins claim 30–60% and deliver
24–38%; only the 0.9–1.0 bin ≈ 0.87–0.88 holds) — a ranking signal, not a
probability, until the L1 calibration item lands; when both our legs fail
they pick the same wrong answer ~60% of the time (shared knowledge gaps).

**The flip flag as a router gate was tested and rejected (R1 replay,
2026-10-08, n=1,000 pooled)**: adding rot_flip to conf moves AUC by
+0.0005/+0.0026 (4B/9B, bootstrap CIs straddle 0) and coverage@≤5%-error
not at all — flipped items ALREADY carry low conf (p50 0.39/0.52 vs
0.79/0.89 stable), and the conf ≥ 0.9 error tail contains almost no
flipped items (2/500 and 0/500), so the flag adds nothing exactly where
conf fails. rot_flip stays a logged difficulty diagnostic; conf remains
the only serving gate (τ re-reads unchanged). Also verified during R1:
**PACT is real** (arXiv:2609.35865, Yida Lin, 2026-09-26 — abstract
confirms 9.8 vs 13.8% relabelling flips, permutation-consistency term,
contextual temperature) — the trained-in cousin of our marginalization.
Research framing noted for later: semantic entropy (Farquhar et al.,
Nature 2024) is the general "disagreement across equivalent views" lineage
our m=2 rotation is the cheapest two-view case of.

---

## typed-decisions — the zero-training head-to-head (2026-10-08)

`LocalLLaMA/typed-decisions` (Apache-2.0, license verified) is the public
benchmark Unsloth's trained decision models report on: 4 workflows
(agent-trace observability / customer service / invoice processing /
security incidents) × 100 test states × 5 fixed questions (choice action,
noul needs_review, choice outcome, score risk, score urgency) = **2,000
decisions**. Gold = teacher-derived soft labels (each case labelled three
times by the teacher at temperature 0.7; gold = the mean of those
distributions — the dataset's own teacher self-agreement is 0.735; a
measured consensus, not oracle truth). Harness:
`decide/bench/bench_typed_decisions.py` (dialect translated
lowercase/criteria → canonical, option descriptions folded into
instructions; dataset cached via the HF datasets-server API, stdlib only)
→ `typed_decisions.json` (4B) / `typed_decisions.9b.json` (9B). Ours is a
**zero-training label-free L0 readout**; every reference line below was
TRAINED on this benchmark's train split (in-distribution by construction):

*Drift note (2026-10-11, independent re-verification): the frozen 4B artifact
predates a code change and re-measures at 61.85% (misses 763 vs 769; 8/2,000
answers differ; two reruns identical). The 9B and prior-strength artifacts
re-measure exactly. The table keeps the frozen numbers; see
`results/INDEX.md`.*

| system | accuracy (2,000 decisions) |
|---|---|
| chance (uniform over each question's options) | ~30% |
| Unsloth head-on-base, BEFORE training (0.8B / 2B) | 36% / 33% |
| **ours, frozen 4-bit 4B, zero training** | **61.55%** (59.4–63.7) |
| **ours, frozen 4-bit 9B, zero training** | **60.4%** (58.2–62.5) |
| Unsloth fine-tuned Qwen3.5-0.8B (LoRA + Clef head) | 73% |
| Unsloth fine-tuned Qwen3.5-2B | 78% |
| HF dataset leaderboard top (2026-10-08, Darwin-27B) | 74% |

Reading, two regimes and both honest: (1) **our untrained readout (61.5%)
sits far above the untrained baseline (36% ≈ chance 30%) — the L0 debiasing
stack buys ~+25 over the naive head-on-base read — but 12–17 points BELOW
trained small models.** On a benchmark built for fine-tuning (five FIXED
question layouts, repeated across 1,200 train states; majority-consensus
gold), training-in wins at every size we can afford. (2) **Base size does
not move this benchmark** (9B 60.4 vs 4B 61.6, CIs overlap; per-type choice
+3.8, score −5.2) — contrast banking77, where the same 9B gained +3.75 and
halved the Jev gap. The train-in vs read-out crossover (docs/PRIOR_ART.md)
is therefore **regime-dependent, not size-only**: zero-shot/unseen-layout
tasks scale with base capacity (read-out's home turf), while fixed-layout
in-distribution tasks reward fine-tuning at any size (typed-decisions).

Mechanism footnote (measured on the misses): the score-type deficit is
partly calibratable, not purely capability — score misses run
one-notch-HOT (4B +1×196/344, hot:cold 3.4:1; 9B +1×207/386, 4.3:1;
per-question signature: urgency hot, discrepancy_severity cold — both
models), score questions are `ordered=True` = never rotation-debiased
(prior-only, the weakest debias), and the bias survived an active
0.75-strength batch prior — while the 9B's choice accuracy IMPROVED
(57.5→61.3), so its score drop is plausibly threshold drift, not lost
capability. The prior_strength=1.0 replay MEASURED it (9B,
`typed_decisions.9b.ps1.json`): score accuracy unchanged at exactly 51.75
with the +1-hot misses persisting (hot:cold 4.3:1 → 2.4:1, no collapse)
while full correction cost choice −3.5 — the residual is **per-question
offset, not batch skew** — but see the R2 replay below: that offset is NOT
recoverable by a per-question constant, so the argmax-side fix is closed
negative and the L1 item survives only as confidence calibration
(ECE/tail), and the stock 0.75 prior default is the measured right choice.

**The per-question offset was tested and rejected (R2 replay, 2026-10-08,
5-fold grouped-by-state CV on the stored misses, both models, n=800 score
decisions each)**: the median per-question signed error fits to OFFSET 0
on every score question at 4B (delta exactly 0.00 points) and to 0
everywhere except 9B `risk` (−1, +5 on that qid = +0.62 pooled, n.s.) —
the +1-hot signature lives among the MISSES, but correct answers dominate
any per-question fit, so no serving-side constant recovers it: the score
deficit is capability/miss-anatomy, not a calibratable threshold. Choice
questions are structurally untouched (per-question temperature is
argmax-preserving for choice — the Well-Tempered-Classifier result,
arXiv:2602.14862 — and the discrete offset was applied to score qids
only). The one formally untested variant, per-question temperature on the
full level distributions (argmax CAN move there via E[level]), needs a
re-run with distribution logging — listed only if ever wanted.

Caveats that bound the comparison: their numbers are published (protocol =
their 3k-row test cut), ours is the dataset's official 400-state test
split; their before-training line is their head-on-base read, not ours.
Bonus the soft gold enables: mean JSD vs the gold DISTRIBUTIONS = 0.300
(4B) / 0.316 (9B) — the readouts track the consensus distributions, not
just argmax. Latency p50 1765 / 2357 ms per 5-question request. The 1.5B
floor test (where read-out should degrade toward the untrained line)
remains the open size-axis point.

---

## Decide vs generate — the same model, asked two ways (n=300)

`decide/bench/bench_vs_generate.py`: the same eval questions, the same prompt
text, the same base model — read out of logits (L0, what the endpoint
serves) vs greedily decoded (temp 0, 12 tokens; generation's best case).
Jev's answers replayed from the stored eval runs (no API calls). At n=105
(first batch, `bench_vs_generate.json`) and at n=300 (both batches,
`bench_vs_generate.n300.json` + `pool_evals.py`):

| | readout (L0) | greedy generation |
|---|---|---|
| agreement with real Jev, n=105 | **79.0%** (Choice 88 / Noul 84 / Score 60) | 74.3% (Choice 72 / Noul 82 / Score 60) |
| agreement with real Jev, **n=300** | **75.7%** | 71.0% |
| paired delta, n=300 | **+4.7 points** (17 only-readout-right vs 3 only-greedy-right; exact McNemar **p = 0.0026**) | — |
| latency (short ~150-tok prompts) | 308 / 336 ms | 200 / 221 ms |
| output | **full probability distribution**, deterministic | a string (12 tokens) |
| parse failures | none (structured) | 0 here; free-text at other temps is unbounded |

**The same experiment on the 9B (2026-10-08, `bench_vs_generate.9b{,.v2,.long}.json`,
`--resources` telemetry on)** — same items, same stored Jev references:

| | readout (L0) 9B | greedy 9B | (4B readout) |
|---|---|---|---|
| n=105 | 81.0% | 80.0% | 79.0% |
| **n=300 pooled** | **82.0%** | 75.7% | 75.7% |
| long n=50 | **84.0%** | 76.0% | 80.0% |
| paired delta, n=300 | **+6.3** (28 vs 9; McNemar **p = 0.0026**) | — | +4.7 |

Reading: two things move together. (1) **Capacity lifts fidelity more than
it lifts banking77**: the 9B readout reads 82.0% pooled Jev-agreement
(+6.3 over the 4B's 75.7 — the repo's best fidelity number), 84.0% on long
contexts. (2) **The readout-vs-generation edge GROWS with base size**:
+4.7 (4B) → +6.3 (9B) at n=300 — the bigger base improves greedy decoding
too (71.0 → 75.7) but improves the readout more. The +5–10-points-of-
judgment claim is now measured at two sizes. On short single batches the
two paths tie (81.0 vs 80.0, p = 1.0 at n=105); the edge concentrates
exactly on the harder multi-factor v2 batch (+9.3, p = 0.0009) and long
contexts (+8, p = 0.34 at n=50 — direction matches the 4B, not significant
alone). Long free-form decode on the 9B: mean 9,034 ms for ~119 tokens vs
readout p50 3,675 ms = **2.46× faster**, the same regime as the 4B.
Resources (in-process serving loop): RSS ~5.33 GB flat, GPU mean 74–81%
(one >100% ioreg peak = counter-straddle artifact, discarded), CPU 9–12%
of one core, repeatability 10/10 + 6/6.

**Accuracy-grade check on public gold (2026-10-08,
`decide/bench/banking77_greedy.py` → `banking77_greedy.{4b,9b}.json`)** —
the designated closer for the readout-vs-generation claim: greedy
generation on the same 400 gold-labeled BANKING77 items, same label-only
prompts, typed answers scored against gold. **4B: greedy 87.75% vs readout
87.5% (9 vs 10 discordant, McNemar p = 1.0). 9B: greedy 92.25% vs readout
91.25% (14 vs 10, p = 0.54 — greedy nominally ahead, not significant).**
Verdict, stated plainly: **on the 10-way restricted gold the readout TIES
generation at both sizes** — while greedy is ~8× faster on these short
one-word items (p50 199 vs 1552 ms full-K). The readout's +4.7/+6.3
fidelity edge is therefore not a 10-way accuracy win — the wording for
fidelity everywhere says "aligned with the reference", never "more
correct". The accuracy picture across arenas, completed by the full-77-way
unlock (above): 10-way ties at both sizes; 77-way 4B behind by 3.4 points,
n.s. (p = 0.068); **77-way 9B the readout significantly ahead (+5.6,
p = 0.0034) — the first positive accuracy leg** — with the blind-judge tie
(n=30) on the long batch as the third check. What the readout uniquely
sells is distributions, reference-grade agreement, determinism, and
long-context holding-power — at a speed that is a wash at worst.

Reading (4B, n=300): the readout is **+4.7 points more aligned with the
reference than decoding from the same model** — at n=105 the CIs
overlapped, at n=300 the paired test separates them (p = 0.003; the delta
is identical +4.6 on the harder v2 batch alone). The L0-vs-raw ablation
(previous section) decomposes WHERE the edge lives: the naive raw logit
read scores 74.3% — numerically the same as constrained greedy — so the
entire edge is permutation marginalization, not the label-prompt
constraint. Both paths are deterministic under greedy (10/10 repeats);
only the readout is deterministic AND probabilistic.

**On long contexts the gap WIDENS** (long batch, n=50, same harness +
`--gen-tokens-long` → `bench_vs_generate.long.json`): readout **80.0% vs
greedy 70.0%** — and per type, Choice **84.6% vs 46.2%**. Constrained
letter-decoding degrades badly as context grows (the label-token attention
dilutes), while the K-rotation readout holds. The readout's edge over
naive decoding is largest exactly where agents actually work — long
operational context.

### Latency/GPU regimes

(Measured 2026-10-07; the long-context rows are the n=50 long batch — per
question, prompt ~3.5k tok mean, split at 2k): the speed answer depends on
prompt and answer length —

| regime (measured, n) | readout | generation | verdict |
|---|---|---|---|
| short prompt (~150 tok), 12-tok answer (n=105) | L0 308ms | 200ms | generation 1.5x faster |
| short states, 5-q request, state-shared L0 (n=21) | **1071ms/req** | — | was 1649ms: 1.55x from sharing |
| long ~1.1k-tok states, 12-tok answer (n=25) | L0 1371ms | 817ms | generation 1.7x faster |
| long ~1.1k-tok states, free-form ~118-tok answer (n=25) | L0 1371ms | 3475ms | **readout 2.5x faster** |
| long ~5.8k-tok states, 12-tok answer (n=25) | L0 6651ms | 4010ms | generation 1.65x faster |
| long ~5.8k-tok states, free-form ~118-tok answer (n=25) | L0 6651ms | 6636ms | ~equal (readout wins on distribution + agreement) |
| **long states, 5-q request, state-shared L0 (n=10)** | **~3700ms/req (5.3x)** | — | the state-share fix; was 22.2s/req |
| long 3.4k-tok prompt, L2 (1 early-exit fwd @ block 16, head layout only) | **1085ms / 902ms GPU** | 2164ms / 1865ms GPU (140 tok) | **readout 2x faster, 2x less GPU** |
| vs thinking-model tool-call decode, 200-600 tok | ~1.1s | ~8s (historical, docs/DECIDE.md §5) | readout ~7x+ |

Reading this honestly: at **L0 with full-K rotations, long contexts are
compute-expensive** — every rotation re-prefills the whole prompt, so the
readout's latency scales ~K× the prompt while a 12-token greedy decode
scales 1×; generation beats it there (1.65–1.7×) until the answer itself
is long (free-form ≈118 tokens), where they meet or the readout wins. Two
levers close the gap, both landed and measured: **state-level prefix
sharing across the questions of a request** (5.3× on long requests,
default-on in the endpoint — turns ~15–20 full state reads into one) and
the **m=2 rotation budget** (`--max-permutations 2`: ~90% of the agreement
gain at half the rotations, BANKING77-measured). Compounded on a long
request that is ~10× fewer token-reads than the naive pipeline. The
historical ~7× figure holds only against thinking-model tool-call decodes
and the L2 head (one early-exit forward, no rotations) — L2 remains the
only mode that is both faster AND cheaper than generation on long inputs,
but it exists today only for the registry head's layout.

Detail worth knowing: on the constrained "answer with the label only"
prompt the model's natural answer is ONE token — so the free-form row uses
the same question WITHOUT the label-only constraint ("2-3 sentences of
reasoning, then 'Answer: <label>'", mean 118 tokens, cap 200): with the
constraint the model hits EOS after ~3 tokens and a 200-token cap never
engages (measured: capped and uncapped decode time are identical). Memory
is identical across rows (same weights); the saving is GPU-time and the
absence of a decode loop. The one honest loss case is short single-shot
Q&A, where generation is 1.5x faster. Why reading can beat decoding in
principle: prefill is compute-bound (many tokens per weight pass) while
autoregressive decode is memory-bandwidth-bound (one token per full weight
pass) — canonical statement in Sarathi-Serve (arXiv:2403.02310). mlx-lm's
own benchmark table (shipped in the pip package as mlx_lm/BENCHMARKS.md;
absent from the GitHub repo as of 2026-10-08) lists a 4-bit Qwen3-4B on
64GB M4 Max at 1,622 tok/s prompt processing vs 134.5 tok/s generation.
Note those are throughput numbers — per-request latency is what our tables
measure, and short prompts favor generation.

**Where the hosted players' 40ms comes from** (and why "same Qwen" is not
the puzzle it looks like): Clef-flash's 38.8ms median is (a) their LATENCY
PROTOCOL — the Decision Index measures a single request at a time on one
on-card GPU process (their p95 is 122ms, 3× the median), while Jev's own
524ms figure is measured over HTTPS and explicitly flagged "not
comparable" by the index; (b) ONE pass — trained-in order-robustness means
no rotations, no decode loop; total compute ≈ 2 × params × tokens ≈ a
couple of TFLOPs; (c) datacenter GPUs (~100–200 effective TFLOPs) →
~10–20ms of actual math plus serving overhead. On short single-question
inputs (our m=1 banking77 reads, ~150ms) the gap is a 4–8× that IS the
laptop-vs-datacenter hardware ratio. The model family was never the
variable; math speed = chip speed. What the local stack can and did copy
is the passes factor (m=2, state-sharing); the chip factor (~10×) is the
permanent, honest price of private/local.

**Summary line for the whole speed story**: after the two landed levers
(m=2 + state-share), local readout speed is a wash at worst versus a
generation stack given the same sharing optimization (they tie at one-word
answers — arithmetic estimate, not measured; on gold accuracy the picture
is per-arena: 10-way ties at both sizes, full-77-way 4B behind n.s. /
**9B ahead p = 0.0034**; they lose once answers run a sentence or more,
because the decode loop pays per word forever while a readout answer is
always one scoring pass) — and the readout is always the side returning
+5–10 points of AGREEMENT WITH THE REFERENCE MODEL (fidelity — measured at
n=300 on two base sizes, McNemar p = 0.003; fidelity, never a blanket
accuracy claim) with full probability distributions attached.

---

## Serving-layer measurements

**State-level prefix sharing** (`decide/core/state_shared.py` +
`score_shared_pairs` in the MLX backend; default ON in the endpoint,
`--no-state-shared` disables): ONE forward of the state block, then every
(question, rotation) of the request runs as a short suffix from a copy of
that prompt cache — instead of re-prefilling the whole state once per
question per rotation. The mechanism is the standard KV/prefix-cache reuse
every cloud serving stack has (vLLM's automatic block prefix caching;
OpenAI/Anthropic bill cached prompt tokens cheaper); what's ours is the
state-level prompt split that makes it work for the multi-question fan-out
on a local stack, where no server is doing it for you. Measured
(`decide/bench/bench_state_share.py` → `state_share.json`): **long-context
requests 22.2s → 3.7s mean = 5.3×** (up to 7.9× on the biggest states);
short 1.65s → 1.07s = 1.55×. Parity: 0/50 argmax flips on the long set;
2/105 on short, both Noul items sitting at exactly 0.50 ties (the
documented bf16 cache-boundary wiggle, conf delta ≤ 0.031). L2-routed
questions keep the ordinary early-exit path; any failure falls back to the
standard pipeline automatically.

**Prefix-KV (`score_shared`)**: `make_prompt_cache` + per-suffix cache
copy, HF-parity boundary check with flat fallback. Measured numeric
boundary: MLX's cache-incremental vs flat kernels differ in bf16
accumulation (conf delta ≤ 0.02, **0/12 argmax flips** on seed states) —
documented, not exact like the HF backend. Speed: 2.2× on a 368-token
prefix × 3 suffixes; the SystemOne endpoint's 4-question smoke went
2.3s → 0.8s.

**Mass-read removal** (`--no-mass` / `DECIDE_NO_MASS=1` on
`decide/serve/proxy.py`; `decide/bench/bench_mass_removal.py` →
`mass_removal.json`): the shadow worker's L2 decide ran the forward to full
depth so the answer-mass diagnostic could be read at the answer position;
with the guard measured NON-separating (MASS_TAU=0.0 log-only) the mass leg
is skippable: the forward early-exits at the head's block. Measured on 30
real observe states: **196 → 96 ms/decision = 2.05×** (p50 171 → 83),
parity 0/30 argmax flips, conf delta 0.0 (the head reads the same block
either way). The guard code stays, gated by the flag; `mass` logs as `None`
when off. The SystemOne endpoint is unaffected by construction (its L2
already early-exits; L0 mass falls out of the same forward for free).

**9B mechanics**: `mlx-community/Qwen3.5-9B-MLX-4bit` is an mlx-vlm
conversion of the VLM 9B but loads + reads fine through our backend (the
`_inner()` / `language_model` paths were already VLM-aware; n_layers=32).
9B serves L0-only (head=false, no registry artifact for it — expected,
fit-on-what-you-serve). 9B latency: typed-decisions p50 2357 vs 1765 ms
(×1.34); banking77 1973 vs 1552 ms (×1.27; ×1.79 within-session — see the
resource-profile note above). Model ~5.95 GB.

---

## Live re-read (v3 renderer + promoted head, 2026-10-07)

Batch: 6 opencode sessions through the shadow proxy on a python/unittest
sandbox (`data/repos/qjev-sandbox`, stdlib only — deliberately a FOREIGN
domain; the historical logs are seed-domain). 81 rows, all L2 (numbers via
`python decide/bench/shadow_stats.py results/shadow_log.v3.jsonl`):
agreement 0.548, ECE 0.238, conf p50 0.87, coverage@0.8 59% with routed
error 37% — vs the v2-era seed-domain logs' 0.47–0.66 agreement and 15–34% routed
error. Reading: the head is **overconfident on foreign live traffic** and
the seed-tuned tau curve does not transfer as-is. **Shadow-only stays
correct.** The remedy ran: the batch produced 73 real observe rows →
`decide/adapt/observe.py` refit (30/60 thresholds fired) →
`next_move_observed.json` (oof 0.683, in-domain hits 64/73 vs the seed's
40/73). With more foreign telemetry, mix gold + observed and re-read the
tau curve before any routing leg.

## Answer-mass guard measurement (NON-separating)

Measured on seed states (the seed-dependent script is not shipped): seed,
foreign and degenerate states all put 0.92–0.99 mass on the label alphabet
at full depth (the model complies with the readout format unconditionally),
and at the head's exit layer mass is ~1e-5 noise. Default `MASS_TAU=0.0` =
log-only, never gate. Kept gateable so a future model/artifact that DOES
separate can enable it without code change. The practical tripwires remain
tau-on-confidence and observe().
