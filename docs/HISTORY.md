# HISTORY.md — the session digest

Condensed at publish (2026-10-11) from the internal session log: one
entry per work session, outcomes only. No number changed. Where the
detail lives: measurements → `docs/RESULTS.md` (+ `results/INDEX.md` for
the artifacts); decisions and their evidence → `docs/DECISIONS.md`; the
pre-fork investigation and design records → `docs/DECIDE.md` +
`docs/HARNESS.md`; current status → `docs/ROADMAP.md` and `docs/TODO.md`.

**2026-10-06 — the investigation sprint (pre-fork).**
MLX backend for AnyJev 0.2.0 (raw/L0 smoke on the 4B); L0 agreement on
120 real trace states (merged 3-class 0.72 vs raw 0.61; late-third 0.95);
L2 linear heads (merged 0.89, exit blocks 16/19 of 32, conf separated
.88/.64); H1 offline router replay (tau 0.8 → 42% auto at 4.3% error,
proxy-shape ceiling ~1.6x); H2-phase-0 shadow proxy live (merged
agreement 7/9); FIX-AB batch 1 interrupted at 21/56 by an OOM reboot —
postmortem produced the supervised decide worker with a hard timeout and
the cheap-depth head refit. Full record: `docs/DECIDE.md` §8.

**2026-10-07 — the domain-general fork.**
Level machinery landed: answer-mass guard (measured non-separating →
log-only), `level="auto"` degradation, the observe() leg. Renderer v3 +
A/B refit (gold-only holdout 0.826, synthetic corpus neutral at 285) —
`next_move_v3.json` promoted. The SystemOne endpoint shipped
(`POST /v1/systemone`, L0 for arbitrary questions, L2 for the head
layout). Prefix-KV (`score_shared`, 2.2x). Live re-read: 6 opencode
sessions on the sandbox, 81 rows — foreign-domain overconfidence
(agreement 0.548, ECE 0.238, routed error 37% at 59% coverage) → the
proxy stays shadow-only. First eval vs the real Jev API: 78.1% agreement
on 105 questions ($0.0005). Decide-vs-generate on the same batch: 79.0%
vs 74.3%. Release prep: the gold-seed dataset removed (recorded work
sessions from another project don't belong in a public release), LICENSE,
THIRD_PARTY, .gitignore. A second pass hardened the README (prior art,
responsible use, trademark disclaimer, citations).

**2026-10-08 — the crossover measurement batch.**
typed-decisions head-to-head (4B 61.55 / 9B 60.4 vs trained 73/78 —
training-in wins on fixed layouts; base size does not move it);
banking77 on the 9B (91.25%, gap to Jev 8.0 → 4.25); resource profiles
(RSS 2.80/5.34 GB, the 9B's cost is GPU-time, within-session ratio
1.79x); score-miss anatomy (+1-hot per-question offset); the
prior-strength 1.0 replay (score flat, choice -3.5 → keep the 0.75
default); blind external-judge leg on 30 long items (ours 20/30, Jev
25/30, readout-vs-greedy 20 vs 19 — not separated at n=30);
decide-vs-generate on the 9B (pooled 82.0 vs 75.7, +6.3, p=0.0026);
banking77-greedy closer (ties at both sizes); long-regime Jev
self-agreement 98.0%. All numbers: `docs/RESULTS.md`.

**2026-10-08 #2 — the full-77-way unlock.**
Multi-token loglikelihood scoring past the 26-option letter cap
(`score_suffix_sum`, m=2 rotations, batch prior). 500 stratified gold
items: 4B readout 63.4 vs greedy 66.8 (p=0.068, n.s.); 9B readout 66.4
vs greedy 60.8 (p=0.0034) — the accuracy claim's first significant
positive leg. Implementation facts (continuation format, permuted-order
scoring gotcha, boundary parity) recorded with the results.

**2026-10-08 #3 — the full-77-way LIVE Jev leg.**
Jev ran the same 500 items ($0.003, 0 failures): 78.6 / 0.770 macro-F1 —
the published 79.7 transfers to this harness; Jev wins the arena at both
sizes (all McNemar p<1e-5); the readout stays the more Jev-like leg
(0.778 vs 0.714 agreement at 9B).

**2026-10-08 #4 — the pre-publish citation audit.**
Every external number the docs quote was re-verified against primary
sources: the Clef table (94.20 / 90.93 / 79.74 macro-F1 on the full
3,080-row split), Unsloth's number set (36/33 before training, 73/78
after; their banking77 74/58 on a 500-row cut), dataset licenses
(banking77 MIT; typed-decisions Apache-2.0; both MLX model cards
apache-2.0), anyjev (Apache-2.0; tech report arXiv:2610.00831; 0.3.0's
L1/L2 removal confirmed), mlx-lm (MIT), the OpenRouter terms revision
(restrict nothing this repo does), and every quoted arXiv ID. One
unverifiable citation was deleted rather than kept (a claimed NeurIPS
2023 paper absent from the proceedings listing; arXiv:1906.02530 carries
the claim alone). Corrections landed from the audit: typed-decisions
gold is teacher-derived (previously misdescribed as annotator-majority),
the Decision-Index latency framing replaced with the verified protocol
facts, four citations added. The same session closed the MLX early-exit
survey (exactly one static precedent — NeoJev; no dynamic-exit MLX
precedent) and adjudicated R1 (the rotation-flip router gate: no added
gating value; stays a diagnostic) and R2 (per-question offset fit:
closes negative; L1 survives as confidence calibration only) — both
recorded with their numbers in `docs/RESULTS.md`.

**2026-10-10 — publish review + rename.**
The pre-publish checklist (`docs/TODO.md`, PP-1..PP-20): home-path scrub
of the observe log, install-prereq corrections, the output policy
(`runio.py`: `runs/` by default, `--force` for frozen paths, fail-fast
preflights), docs reorganization into the README + `docs/` shape,
`docs/API.md` written, the test suite (`tests/`, 32 tests), run-from-
anywhere anchoring. Renamed to `take-my-layer` (the old name embedded
two third-party marks; candidates and sweep record in
`docs/ROADMAP.md`). A wording scrub removed the pre-fork substrate
references; `l0_smoke.py` fixtures rewritten to the v3 format.

**2026-10-11 — independent re-verification + friction fixes.**
An independent from-scratch verification (scratch copy, both Nix and pip
environments) reproduced every frozen artifact — most bit-identically —
and surfaced two real defects, both fixed: a save-time crash in
`banking77_greedy.py` (output-path variable shadowed by the generation
result; both greedy legs then reproduced exactly) and a stale
`typed_decisions.json` (4B artifact predates a code change; re-measures
61.85 vs the frozen 61.55 — annotated in `results/INDEX.md`; the 9B and
prior-strength artifacts reproduce exactly). Also landed from the same
pass: `decide/doctor.py` (stdlib preflight), install-doc fixes (the
`HF_HUB_CACHE` override, wall-time estimates), endpoint-limit pins in
the tests, and the typed-decisions drift note in `docs/RESULTS.md`.
