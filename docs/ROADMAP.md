# ROADMAP.md — status and backlog

Adapted from the AnyJev gap analysis (2026-10-07). Measured details for
every number quoted here live in docs/RESULTS.md. Open measurement work
(bench gaps, verification debts): `docs/TODO.md`.

## P0 — DONE (2026-10-07), with one measured negative result

- `observe()` on shadow telemetry: DONE — the proxy pairs every shadow row
  that has a mapped `actual_tool` with its FULL rendered state into
  `logs/observe_log.jsonl` (gitignored; the frozen 73-row copy ships as
  `results/observe_log.jsonl`); `decide/adapt/observe.py` replays it through
  `Decider.observe` (auto-solve at 30 labels, refit at 60/120/…) and writes
  `next_move_observed.json` (the seed is never overwritten; promote via
  `--head`). This is also the remedy for the live confidence shift
  (docs/DECIDE.md §8 step 5).
- `level="auto"` fallback: DONE — worker degrades L2 → L0 (broken head
  path) → pass-through (worker down/slow); only L2 answers carry
  would_route.
- Answer-mass guard: mechanism shipped (mass read in the same forward,
  logged per decision, gateable via `--mass-tau`) — but **measured
  NON-SEPARATING on this model** (seed-state measurement, not shipped;
  see docs/RESULTS.md): default `MASS_TAU=0.0` = log-only. The practical
  tripwires remain tau-on-confidence and observe().

## P1 — the domain-general build-out (landed 2026-10-07)

- Synthetic state generation + teacher labelling (policy §3): DONE —
  `decide/adapt/synth_states.py` (5 domains × 2 exemplars → N states at
  temp 1.0 with a validate/repair pipeline, L0 teacher labels; the
  2026-10-07 batch accepted 285/300 but is not shipped — regenerate on
  demand); `decide/adapt/fit_registry_head.py` runs the A/B protocol
  (gold-only vs gold+synth, grouped-by-task holdout) →
  `results/synth_gain.json`.
- Renderer generalization: DONE — v3 drops the repo-kind tag and the
  single-extension glob filter; refit landed in the same change. **Measured** (gold
  holdout grouped by task, 482 points): gold-only 0.826 (layer 16),
  gold+285-synth 0.824 — synthetic corpus NEUTRAL at n=285 on a
  same-domain holdout (the recipe's ≥300 gain bar was just missed, and a
  same-domain holdout cannot measure domain generality anyway). `next_move_v3.json` promoted →
  `next_move.json`; the v2 seed kept as `next_move_v2_seed.json`. The
  synth corpus's real test is foreign-domain agreement — regenerate the
  corpus and re-run the A/B at larger N if observe telemetry shows drift.
- SystemOne endpoint: DONE — `decide/serve/systemone.py` serves the
  `POST /v1/systemone` contract (state + Choice/Score/Noul questions →
  answers + probabilities + input_tokens) on the same Decider: L0 readout
  for arbitrary questions, L2 for the registry head layout.
- `noul` yes/no questions for termination / done-yet / needs_review:
  available via the endpoint (`type: "Noul"`), zero build cost.
- Generic call construction (extractors or file-shortlist head) to
  restore a routing leg — OPEN.
- Label-free adaptation (`adapt="routed"`) as the domain-shift test —
  targets reworded/relisted layouts, so measure before trusting it for
  domain shift — OPEN.

## P2

- `score_shared` prefix-KV in the MLX backend: DONE (2026-10-07) —
  measured 2.2× on a 368-token prefix × 3 suffixes; endpoint 4-question
  smoke 2.3s → 0.8s (numbers in docs/RESULTS.md).
- Mass-read removal (`--no-mass`): DONE + measured (2.05×, parity 0/30) —
  docs/RESULTS.md.
- State-level prefix sharing: DONE + measured (long 5.3×, short 1.55×,
  default ON, `--no-state-shared` disables) — docs/RESULTS.md.
- **Close the observe() loop (small build, not done)**: today the loop is
  self-logging but human-gated — the proxy banks observe rows
  automatically, but the refit (`decide/adapt/observe.py`) and the
  promotion (`--head`) are commands a human runs when curious. The last
  mile is a periodic refit with **canary-gated auto-promotion**:
  recompute the artifact on a schedule, swap it in ONLY if its holdout
  metric beats the incumbent's, else keep serving the old one; log every
  swap. Scope honesty, stated up front: if a deployment ever reaches the
  point of owning a rich labeled history of its exact decision surface,
  the crossover verdict (docs/PRIOR_ART.md) says the right next tool is a
  fine-tune or a classical trained classifier — the auto-loop is for the
  regime where labels arrive as a sparse trickle of your own traffic, not
  as a dataset.
- Truncated-checkpoint serving (blocks 0..b as a standalone MLX model) —
  OPEN.
- Upstream the MLX backend to AnyJev (their help-wanted slot) — OPEN.
- Bench metrics: ECE + coverage@tau + routed error now computed per batch
  by `decide/bench/shadow_stats.py` (flip-rate needs permutation
  diagnostics that L2 rows don't log — still open if wanted).

## Backlog (post-rename)

- **`git init` + GitHub push** (rename DONE 2026-10-10 → `take-my-layer`).
  Publish decision: GO. The rename was the last blocker ("qwen-jev" embedded
  two third-party marks = the risky pattern under nominative-fair-use
  doctrine; the -jev namespace is crowded — AnyJev, Mini-Jev, PocketJev,
  Simple Jev; candidates weighed: haruspex / prefillhead / auspex /
  take-my-layer, the last chosen for being mark-free and describing the
  mechanism). Swept: `README.md` title, `flake.nix` description,
  `decide/serve/systemone.py` (the response `model` is now the served model
  id), `decide/bench/eval_vs_jev.py` (X-OpenRouter-Title). Left on purpose:
  the sandbox name `qjev-sandbox` (a data fixture present in every logged
  state), `docs/HISTORY.md` (historical record), and file names that name
  the comparison target (`eval_vs_jev.py`, `jev_*.json`). Still manual:
  `mv` the repo folder, rebuild `.venv` (its console scripts embed the old
  absolute path), then `git init` + push (docs/TODO.md PP-19).
  "Jev"/"Qwen" stay in prose (nominative) + the README disclaimer
  section.
- Human ground-truth leg: adjudicate just the ~10 disagreement rows in
  `results/labels_long.scored.json` (~15 min, the highest-leverage
  version), or more judge-labeled items (the blind external judge worked
  well), or nothing — the judge leg stands as the labeled evidence,
  honestly named.
- Fix the 3 authoring defects the judge found in `questions_long.jsonl`
  (impossible dates, qid/text mismatch, metric inconsistency — then
  re-read whether the stored answers still hold).
- L1 temperature calibration for the 0.93 tail + continuous Score values
  (caveat: TS degrades under shift — tail fix only).
- More foreign observe telemetry → gold+observed mixed refit → τ re-read.
- Consider m=2–3 rotation budget as the L0 default (the K-curve says
  ~90% of the gain at half cost AND it is the main long-context latency
  lever — measure Jev-agreement impact first, it's one flag).
- Test the m=2 × state-share compound.
- Cross-model comparisons: the 9B legs are done (docs/RESULTS.md); a
  smaller 1.5B is the floor test. Parallel/batched serving is the
  alternative direction: GPU idles 10–70% between passes today, dynamic
  batching of concurrent requests' reads would buy ~1.5-2× throughput —
  only worth it if multi-agent serving ever matters.
- Zero-training head-to-heads: typed-decisions leg DONE; full-77-way
  banking77 DONE (docs/RESULTS.md).
- Close the observe() loop (P2 item above).
- Truncated-checkpoint serving; upstream the MLX backend + the
  state-share split to AnyJev.
