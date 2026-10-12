# TODO.md — status, backlog and open work

One checklist for the whole project. The former `ROADMAP.md` (build status and
backlog) was merged in here on 2026-10-11, so there is a single place to look.
`[x]` = done and checked, `[ ]` = open work or an idea nobody has started.
**Open** comes first; **Done** keeps the evidence (what was checked, how, with
which numbers). Dated results live in docs/RESULTS.md; the wording rules for any
new number are in its header (fidelity = "aligned with the reference", never
"more correct"; accuracy claims name the arena and model size). The raw session
log is docs/HISTORY.md. Item ids (`PP-n`, `#n`) are stable: other docs cite them.

## Open

### Measurement gaps (benches and proofs a README or RESULTS claim still lacks)

- [ ] **#1 Determinism proof.** README says re-running anything returns
   identical answers; `state_share.json` is the only equivalence artifact
   and `jev_repeat.py` covers the reference side only. Run any bench twice
   on the same inputs and byte-compare the stored answers. Cost: trivial.
   Partial 2026-10-11: the new default (batched passes) returned identical bytes on
   3 of 3 repeats of the README request over HTTP and on 8 requests in process
   (docs/RESULTS.md, "Speed/precision sweep"); a bench-level proof is still open.

- [ ] **#2 Same-arena ladder: L0 vs L1 vs L2 on one held-out task.** The levels
   table asserts what each level buys, but L2 has only the `next_move`
   holdout (0.826) and L1 nothing. Fit a temperature and a head for one
   benchmark question (e.g. a banking77 or Noul layout) and evaluate all
   three levels on the same holdout. Cost: one fit plus one eval run.

- [ ] **#3 L1 calibration bench.** The "0.93 knowledge-gap tail" is L1's stated
   reason to exist with no reliability curve behind it. Fit the temperature
   and report ECE / reliability on held-out L0 outputs, in-distribution.
   Covers the former ROADMAP item "L1 temperature calibration for the 0.93 tail +
   continuous Score values" (caveat: temperature scaling degrades under shift, so
   it is a tail fix only). Cost: small.

- [ ] **#4 Foreign-domain L2 generalization.** All 73 observe rows and 81 shadow
   rows are `qjev-sandbox`; the domain-general charter has no evidence
   outside it. The synth A/B was neutral, and its "real test is
   foreign-domain agreement" (former ROADMAP P1). Blocked on usage, not code.
   Next steps once there is foreign observe telemetry: a gold+observed mixed
   refit, then a τ re-read.

- [ ] **#5 AnyJev parity point.** Their published label-free L0 is BANKING77
   20-way on Qwen3-8B (0.803 on 300 items). Run the same config on this
   backend and models for a like-for-like cross-implementation number.
   Cost: one bench run (a 20-way option on `bench_banking77.py`).

- [ ] **#6 Length sweep.** Latency and accuracy exist for short and long buckets;
   a 0.5k / 1k / 2k / 4k / 8k curve would back "wins where agents actually
   work" with a shape instead of two dots. Cost: one GPU session.
   Partial 2026-10-11: prefill throughput is flat at about 1.6k tokens/s from
   128 to 4k tokens and for chunk sizes 256 to unchunked.

- [ ] **#7 Concurrency/throughput baseline.** Everything is single-request; the
   dynamic-batching idea (Build backlog) has no baseline number to beat.
   Cost: moderate (client plus server under load).

- [ ] **#8 CLINC150 as a second gold-label arena.** banking77 is the only
   gold-label accuracy bench; CLINC150 sits in the reference ecosystem's
   training mix. Lower priority. Cost: loader plus bench run.

- [ ] **#9b Real-label check of the L2 exit-layer candidates.** The 2026-10-11 sweep
  (#9, Done) found that a head fitted on few labels gains 14 points going from layer 16
  to layer 20 and nothing beyond layer 22, but with synthetic states labelled by the
  model's own readout. Re-run on real labelled traffic (the observe log once there is
  foreign telemetry, #4) before changing `LAYERS = [16, 19, 22]` in `observe.py` and
  `fit_registry_head.py` to about `[20, 22, 24]`. The shipped head (1,824 gold labels,
  layers 16-22 tied) is unchanged and cannot be refit without the withheld seed.

- [ ] **#10 9B with the shipped fast path.** `python decide/bench/bench_fast_path.py --model
  mlx-community/Qwen3.5-9B-MLX-4bit --out runs/fast_path.9b.json` (about 15 minutes), then
  promote to `results/`. The 9B numbers in RESULTS ("Speed/precision sweep") come from the
  prototype of the same algorithm, not from the shipped code.

- [ ] **#11 Validate `--early-stop` on fresh data before it can become a default.** tau
  (0.9 Choice / 0.8 yes/no) was chosen on the same 350 + 400 items it was scored on; it
  also needs a confidence recalibration (ECE rises 0.035 -> 0.046 on the 4B, and the share
  of answers at confidence >= 0.9 agree 93.0% instead of 96.3%). A fresh arena (CLINC150,
  #8, or new LLM-authored batches) and a temperature fix would settle it.

- [ ] **#12 1.5B floor test** (cross-model): where the readout degrades toward the
  untrained line.

- [ ] **#13 Human ground-truth leg and judge noise** (CONTRIBUTING, "the highest-value
  contribution"). Adjudicate the ~10 disagreement rows in `results/labels_long.scored.json`
  (about 15 minutes, the highest-leverage version), or run a second judge pass on the 30
  long items, or label more items with the blind external judge. The judge-leg scoring step
  was performed outside the repo (CONTRIBUTING says so); ship a script if one-command
  replication is wanted.

- [ ] **#14 Fix the 3 authoring defects the judge found in `questions_long.jsonl`**
  (impossible dates, qid/text mismatch, an inconsistent metric line), then re-read whether
  the stored answers still hold (`author_long.py --check` will report the difference).

### Build backlog and ideas

- [ ] **Close the observe() loop (small build)**: today the loop is
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

- [ ] Truncated-checkpoint serving (blocks 0..b as a standalone MLX model).

- [ ] Upstream the MLX backend to AnyJev (their help-wanted slot), together with the
  state-share split and the batched suffix passes.

- [ ] Generic call construction (extractors or file-shortlist head) to
  restore a routing leg.

- [ ] Label-free adaptation (`adapt="routed"`) as the domain-shift test —
  targets reworded/relisted layouts, so measure before trusting it for
  domain shift.

- [ ] Flip-rate diagnostics for L2 rows (needs permutation diagnostics that L2 rows do
  not log; wanted only if the router gate is revisited).

- [ ] Dynamic batching of concurrent requests' reads (about 1.5-2x throughput was the
  guess), only if multi-agent serving ever matters; baseline is #7. Note the premise
  "the GPU idles 10-70% between passes" did not hold for one request's suffix passes
  (2026-10-11: evaluating all passes lazily in one `mx.eval` gained 0.0%).

- [ ] Ideas from the 2026-10-11 sweep, none started:
  - `--max-permutations`: take the spread shifts (0, K/2, ...) instead of adjacent ones.
    BANKING77 K=10, m=2: 87.5% vs 85.75% (4B), 91.0% vs 89.5% (9B); mixed and not
    significant on the Jev set.
  - Length-aware model choice: the 9B for short and mid-length states, the 4B for long
    ones (prefill-bound; 9B long requests take 5.6 s against 2.9 s).
  - A cross-request prefix cache for repeated or growing long states (little for the
    proxy: 1.6x fewer prefill tokens on 340-token prompts, about 50 ms).
  - Sharing the question stem between rotations (-20% suffix tokens; per-pass cost is
    fixed-dominated, so no gain expected).

### Decisions waiting for the maintainer

- [ ] Make the proxy's `--no-mass` the default: 2.05x faster per shadow decision with
  0/30 answer flips, but the logged `mass` diagnostic disappears (the guard stays log-only
  either way).
- [ ] Make `--early-stop` a default: only after #11.

## Done

### Build history (the former ROADMAP)

- [x] `observe()` on shadow telemetry: DONE — the proxy pairs every shadow row
  that has a mapped `actual_tool` with its FULL rendered state into
  `logs/observe_log.jsonl` (gitignored; the frozen 73-row copy ships as
  `results/observe_log.jsonl`); `decide/adapt/observe.py` replays it through
  `Decider.observe` (auto-solve at 30 labels, refit at 60/120/…) and writes
  `next_move_observed.json` (the seed is never overwritten; promote via
  `--head`). This is also the remedy for the live confidence shift
  (docs/DECIDE.md §8 step 5).

- [x] `level="auto"` fallback: DONE — worker degrades L2 → L0 (broken head
  path) → pass-through (worker down/slow); only L2 answers carry
  would_route.

- [x] Answer-mass guard: mechanism shipped (mass read in the same forward,
  logged per decision, gateable via `--mass-tau`) — but **measured
  NON-SEPARATING on this model** (seed-state measurement, not shipped;
  see docs/RESULTS.md): default `MASS_TAU=0.0` = log-only. The practical
  tripwires remain tau-on-confidence and observe().

- [x] Synthetic state generation + teacher labelling (policy §3): DONE —
  `decide/adapt/synth_states.py` (5 domains × 2 exemplars → N states at
  temp 1.0 with a validate/repair pipeline, L0 teacher labels; the
  2026-10-07 batch accepted 285/300 but is not shipped — regenerate on
  demand); `decide/adapt/fit_registry_head.py` runs the A/B protocol
  (gold-only vs gold+synth, grouped-by-task holdout) →
  `results/synth_gain.json`.

- [x] Renderer generalization: DONE — v3 drops the repo-kind tag and the
  single-extension glob filter; refit landed in the same change. **Measured** (gold
  holdout grouped by task, 482 points): gold-only 0.826 (layer 16),
  gold+285-synth 0.824 — synthetic corpus NEUTRAL at n=285 on a
  same-domain holdout (the recipe's ≥300 gain bar was just missed, and a
  same-domain holdout cannot measure domain generality anyway). `next_move_v3.json` promoted →
  `next_move.json`; the v2 seed kept as `next_move_v2_seed.json`. The
  synth corpus's real test is foreign-domain agreement — regenerate the
  corpus and re-run the A/B at larger N if observe telemetry shows drift.

- [x] SystemOne endpoint: DONE — `decide/serve/systemone.py` serves the
  `POST /v1/systemone` contract (state + Choice/Score/Noul questions →
  answers + probabilities + input_tokens) on the same Decider: L0 readout
  for arbitrary questions, L2 for the registry head layout.

- [x] `noul` yes/no questions for termination / done-yet / needs_review:
  available via the endpoint (`type: "Noul"`), zero build cost.

- [x] `score_shared` prefix-KV in the MLX backend: DONE (2026-10-07) —
  measured 2.2× on a 368-token prefix × 3 suffixes; endpoint 4-question
  smoke 2.3s → 0.8s (numbers in docs/RESULTS.md).

- [x] Mass-read removal (`--no-mass`): DONE + measured (2.05×, parity 0/30) —
  docs/RESULTS.md.

- [x] State-level prefix sharing: DONE + measured (long 5.3×, short 1.55×,
  default ON, `--no-state-shared` disables) — docs/RESULTS.md.

- [x] Batched suffix passes (endpoint default) + opt-in `--early-stop`: DONE (2026-10-11)
  — short request 1036 -> 552 ms, agreement with Jev unchanged (76.0%); docs/RESULTS.md
  "Speed/precision sweep", `results/fast_path.json`, `decide/bench/bench_fast_path.py`,
  tests/test_fast_path.py. The open items that came out of it are listed above (#9b, #10,
  #11 and the maintainer decisions).

- [x] Bench metrics: ECE + coverage@tau + routed error are computed per batch by `decide/bench/shadow_stats.py`.

- [x] Cross-model comparisons: the 9B legs are done (docs/RESULTS.md). The 1.5B floor test is #12.

- [x] Zero-training head-to-heads: typed-decisions leg DONE; full-77-way
  banking77 DONE (docs/RESULTS.md).

- [x] Consider m=2–3 rotation budget as the L0 default (the K-curve says
  ~90% of the gain at half cost AND it is the main long-context latency
  lever — measure Jev-agreement impact first, it's one flag).
  MEASURED 2026-10-11: with 3-5 option Choices the flag costs accuracy and
  it never touches yes/no or Score; `--early-stop` is the budget knob that
  held (docs/RESULTS.md "Speed/precision sweep").

- [x] Test the m=2 × state-share compound. (Superseded by the same section.)

- [x] **Rename → `take-my-layer`, publish decision GO** (2026-10-10; `git init` and the
  first commit followed, PP-19a; the GitHub repo is PP-19b). The rename was the last
  blocker ("qwen-jev" embedded two third-party marks = the risky pattern under
  nominative-fair-use doctrine; the -jev namespace is crowded — AnyJev, Mini-Jev,
  PocketJev, Simple Jev; candidates weighed: haruspex / prefillhead / auspex /
  take-my-layer, the last chosen for being mark-free and describing the mechanism).
  Swept: `README.md` title, `flake.nix` description, `decide/serve/systemone.py` (the
  response `model` is now the served model id), `decide/bench/eval_vs_jev.py`
  (X-OpenRouter-Title). Left on purpose: the sandbox name `qjev-sandbox` (a data fixture
  present in every logged state), `docs/HISTORY.md` (historical record), and file names
  that name the comparison target (`eval_vs_jev.py`, `jev_*.json`). The manual steps
  (move the repo folder, rebuild `.venv`, whose console scripts embed the absolute path)
  are done. "Jev"/"Qwen" stay in prose (nominative) + the README disclaimer section.

### Speed/precision sweep (2026-10-11)

- [x] **#9 Layer-left sweep for the L2 head (speed, not RAM).** The exit block is
   repo-controlled, not an AnyJev property: `mlx_backend.py` owns the
   early-stop block loop, and the candidate list is the `LAYERS` constant in
   `observe.py` / `fit_registry_head.py` (today `[16, 19, 22]`; the shipped
   head landed on 16/32). A/B plan: set `LAYERS = [4, 8, 12, 16, 20, 24,
   28]`, run `observe.py` per point on the shipped 73-row replay (about a
   minute each; it prints layer, oof and in-domain hits), take ms/decision
   from the `bench_mass_removal.py` harness shape, then re-read τ and ECE
   via `shadow_stats.py` at the winning layer, and compound with
   `--max-permutations 2`. Guardrail: earlier exit saves compute only —
   weights stay resident, so RAM is a model-choice lever (4B 2.8 GB /
   9B 5.3 GB), not a layer lever. Quality n is the 73 replay rows (small);
   the seed-head holdout cannot be recomputed (seed withheld).
   **Measured 2026-10-11 (docs/RESULTS.md, "Speed/precision sweep"): do not exit
   earlier.** The 73-row replay cannot rank layers (OOF 0.58-0.66 at every layer;
   the logged tool agrees with the model's own readout on only 49% of those rows).
   With the model's own L0 readout as the target on 150 synthetic states
   (5 domains, leave-one-domain-out too), teacher agreement is 0.56 at layer 16,
   0.70 at layer 20, 0.73 at layer 22 and flat to layer 32 (layer 20 minus 16:
   +14.3 points, 95% CI +8.8 to +20.1); exit@20 costs 172 ms against 142 at
   layer 16 and 291 at full depth. The plan's "compound with `--max-permutations
   2`" does not apply: the L2 head is a single pass. Open: a real-label check of
   the candidate list `[20, 22, 24]` for `observe.py` / `fit_registry_head.py`
   (`LAYERS` still reads `[16, 19, 22]`; the shipped head, fitted on 1,824 gold
   labels where layers 16-22 tied, is unchanged).

- [x] Measured and rejected, with numbers in docs/RESULTS.md "Speed/precision sweep":
  pipelining the suffix passes (0.0%), the last-position output head alone (about 4%, not
  bit-exact), dequantizing to bf16 for prefill (no gain), prefill chunk size (within 5%),
  fixed one-phrasing yes/no reads (costs 3.3 points at 9B), L0 block skipping (accuracy
  swings of -12 to +4 points that flip sign between arenas), `--max-permutations 2` as a
  latency mode for 3-5 option Choices.

### Pre-publish checklist (2026-10-10, re-verified)

Every item was re-run or re-read against the tree in a scratch copy; the
review itself did not modify the repo (fixes are marked DONE). File:line
references are as of 2026-10-10. Tags: **[confirmed]** reproduced as stated,
**[partly]** true with the correction noted, **[new]** found during the
re-check. Order: irreversible once pushed > misleads or breaks a stranger's
first run > clarity > polish.

**Status:** everything is done, including PP-19b (the GitHub push, 2026-10-11
— see its entry under P2). The stored Jev
answers stay (PP-18, maintainer's decision). PP-5 removed the main way to lose evidence
by accident (scripts write to `runs/` and refuse to replace frozen files).

#### P0 — before the first push

- [x] **PP-1 Scrub the home path from `results/observe_log.jsonl`.** DONE
  2026-10-10. The review's first count (path only) was incomplete: the file
  held the dev-machine path 210× (`/Users/<user>/…/qwen-jev`) and the macOS
  username 32× more, in `ls -l` owner columns, in 5 of the same 44 rows (all
  inside `state`; no other tracked file had either). Now `/path/to/repo` and
  `user`. 73 rows, every non-`state` field identical, substitution exactly
  reversible. Tree-wide grep for the username and for any `/Users/<name>`
  path: 0 hits. `observe.load_rows` reads 73 rows from both
  versions with identical ts/repo/tool/label. Provenance recorded in
  `results/INDEX.md`: `results/mass_removal.json` (13 of its 30 states
  changed) and `registry/…/next_move_observed.json` were computed from the
  pre-scrub text. All 73 rows are `qjev-sandbox` (no private work).
- [x] **PP-2 Rename (the former ROADMAP's "last blocker").** DONE 2026-10-10 →
  `take-my-layer` (mark-free; "Shut up and take my layer" is the tagline).
  Swept: `README.md` title, `flake.nix` description,
  `decide/bench/eval_vs_jev.py` (`X-OpenRouter-Title: take-my-layer-eval`)
  and `decide/serve/systemone.py` (the response `model` is now the served
  model id instead of the hard-coded `qwen-jev/Qwen3.5-4B-MLX-4bit`; checked
  live: the 4B and 9B servers each report their own id, answers unchanged).
  The dev shell still builds with the new `flake.nix`; the roadmap and
  `AGENTS.md` pointers were updated. Left on purpose: the sandbox name
  `qjev-sandbox` (725 hits in the two logs plus `.gitignore` and
  `RESULTS.md`; a data fixture, and renaming it would change the state text
  the head reads), `docs/HISTORY.md` (historical record), and file names that
  name the comparison target (`eval_vs_jev.py`, `jev_*.json`). The manual
  steps are done too (folder renamed, `.venv` rebuilt, direnv reloaded): no
  `qwen-jev` path is left in `.venv/bin`, `pyvenv.cfg` or `.direnv`.

#### P1 — before linking it anywhere

- [x] **PP-3 Install prerequisites are wrong.** DONE 2026-10-10. [confirmed]
  README and the flake comment said Python ≥3.10, but mlx-lm 0.32.0 needs ≥3.11
  (3.10 silently resolved to 0.31.3, 3.9 failed), and
  `decide/core/mlx_backend.py` imports non-public mlx-lm internals. Now:
  `requirements.txt` pins `mlx-lm==0.32.0` with the reason; README "Running"
  states macOS 14+ (PyPI ships macosx_14/15/26 wheels, checked), Python 3.11+,
  the non-Nix install (`python3.12 -m venv .venv`, `pip install -r
  requirements.txt`, optional `export HF_HOME=$PWD/.hf`, since the project-local
  cache is a Nix-shell feature) and the cost of each model (4B: 2.9 GB download,
  2.8 GB resident; 9B: 5.6 GB, 5.3 GB; from `du` of the snapshots and
  `results/banking77.*.resources.json`). The flake comment, AGENTS.md and
  THIRD_PARTY.md match. Checked: a fresh venv from the Nix Python 3.12 with
  `pip install -r requirements.txt` resolves in 32 s to the same versions as the
  Nix shell (mlx-lm 0.32.0, anyjev 0.2.0, mlx 0.32.3, transformers 5.19.0),
  passes the test suite and serves the README example bit-identically
  (confidence 0.5955441762711209); `nix develop -c …` still builds with the pin.
- [x] **PP-4 README claims to fix.** DONE 2026-10-10. (a) The "4B reads about
  30% before training (its own caption)" sentence now says what Unsloth's guide
  shows: Qwen3.5-0.8B 36% → 73%, 2B 33% → 78%, no 4B figure. (b) "Where it
  loses" labels 63.4% (4B) and 66.4% (9B) against Jev's 78.6% and discloses that
  the 4B sits below greedy (66.8%, p = 0.068); the headline table row too
  (numbers re-read from `results/banking77_full77*.json`). (c) "Already using
  AnyJev?" says 0.3.0 (2026-10-06) removed L1/L2, heads, `observe` and
  `level="auto"` and that this repo pins 0.2.0 (checked against the upstream
  CHANGELOG; PyPI's latest is still 0.3.0). (d) The samples are captured
  responses with the `model` field; probabilities are shortened to four decimals
  and the text says so (the server returns full precision). Also fixed: "breaking
  the 20-option limit" (see PP-20).
- [x] **PP-5 Benchmarks overwrite frozen evidence and exit 0 on total
  failure.** DONE 2026-10-10. New `decide/runio.py`, used by every script that
  writes a result: `--out` defaults to `runs/<name>` (gitignored), `--force` is
  needed to replace an existing file in `results/`, `registry/` or `data/eval/`,
  the output path is validated before any work starts, and reports record
  repo-relative paths (`rel()`). The HTTP benches (`bench_banking77`,
  `bench_typed_decisions`, `eval_vs_jev`) check `/healthz` first (exit 2) and
  stop after three failed calls; the Jev legs (also `jev_repeat`,
  `banking77_full77_jev`) check the key first and stop on a failed first Jev
  call; a run in which every call failed exits 1 and writes nothing; partial
  failures print a warning. The six scripts without `--out` have one
  (`banking77_full77_jev`, `bench_mass_removal`, `bench_state_share`,
  `l0_ablation`, `pool_evals`, `rotation_budget`; `fit_registry_head` got
  `--report`), `bench_mass_removal.py` and `l0_smoke.py` got argparse (`--help`
  no longer loads the model and rewrites `results/mass_removal.json`), and
  `banking77_full77_jev` retries failed calls from its checkpoint instead of
  reusing them. Tested against a fake endpoint (success writes only `runs/`; a
  500-only endpoint exits 1 with nothing written; a dead Jev URL exits 1) and
  with no endpoint (exit 2); `results/`, `registry/` and `data/eval/` were
  byte-identical to the pre-change snapshot afterwards, and `pool_evals.py`
  regenerates `results/jev_compare.pooled.json` byte for byte. Not run end to
  end: the long model-loading benches (they only gained `--out`/`--force` and
  root-anchored paths). `observe.py` still writes
  `registry/…/next_move_observed.json` by default, on purpose (replay output,
  PP-14).
- [x] **PP-6 Live logs default into `results/`.** DONE 2026-10-10.
  [confirmed] `proxy.py` defaulted to `results/shadow_log.jsonl` and
  `results/observe_log.jsonl`, `systemone.py` to `results/systemone_log.jsonl`.
  `observe_log` holds the full rendered state and is both committed evidence
  and the replay input for refits, so someone's real sessions would have been
  appended to it. Now: the proxy writes `logs/shadow_log.jsonl` and
  `logs/observe_log.jsonl`, the endpoint `logs/systemone_log.jsonl`;
  `.gitignore` has `/logs/` and `.envrc.local` (sourced by `.envrc`; the file
  does not exist today). `observe.py` keeps the frozen
  `results/observe_log.jsonl` as its default, so a fresh clone can replay the
  shipped 73 rows; live traffic is `--log logs/observe_log.jsonl`
  (`bench_mass_removal.py` also keeps reading the frozen copy). Checked from
  a scratch cwd: the default write path creates only `logs/observe_log.jsonl`
  and `results/observe_log.jsonl` is byte-identical afterwards; `--help` of
  both scripts shows the new defaults. The one local telemetry row moved to
  `logs/`. Docs updated: README (layout, `observe.py` line), AGENTS.md,
  `results/INDEX.md`, the roadmap; DECIDE.md and HISTORY.md keep the old paths on
  purpose (historical record).
- [x] **PP-7 First screen, names, links.** DONE 2026-10-10. The README is now a
  front page of 88 lines (it was 485): the plain definition of "state" and
  "typed questions" first, the ticket-triage request and its real response,
  the benchmark table with the hardware named, how to run it in three commands,
  and an index list of every doc. Jev is named and linked (TypeSafe AI,
  typesafe.ai) at the top, AnyJev is linked, the research-prototype status line
  says the shipped head is not reproducible (gold seed withheld) and that the
  Jev-agreement numbers have no human labels, and the benchmark rows link
  BANKING77, typed-decisions and the Unsloth guide (all links checked: 200).
  The long text moved, nothing deleted: `docs/CONCEPTS.md` (orientation,
  "in plain words", the two surfaces, the readout levels, glossary; Clef,
  NeoJev, Unsloth and AnyJev now linked there), `docs/RUNNING.md` (the Running
  guide and the repository map), `docs/LIMITATIONS.md` (responsible use and
  measured limits); the trademark paragraph went to `THIRD_PARTY.md`. The
  duplicate "Headline results" / "Results at a glance" are merged. Two wording
  fixes on the way: the Unsloth numbers (73% and 78%) are no longer attributed
  to Clef, and "greedy collapses on long inputs (46% vs 85%)" now says it is
  the Choice questions (overall long-input agreement is 80% vs 70%).
  `tests/test_docs.py` keeps every relative link resolving, the README index
  complete and the README under 150 lines. Cross-references in AGENTS.md,
  CONTRIBUTING.md, DECISIONS.md, RESULTS.md, `results/INDEX.md` and `docs/API.md`
  point to the new homes.
- [x] **PP-8 API undocumented, limits unenforced.** DONE 2026-10-10. New
  `docs/API.md` (fields, the three question types with captured examples, errors
  with the exact messages, auth, limits, flags); the README shows the
  ticket-triage request (Choice + Noul + Score in one POST) next to the coding
  head example. The limits are enforced and listed by `GET /healthz`: state ≤
  32,000 characters (`--max-state-chars`), body ≤ 1 MiB judged from
  `Content-Length` before reading (413), a bad `Content-Length` is a 400, and
  rejected requests close the connection so an unread body cannot corrupt the
  next one. The old unused `MAX_STATE_CHARS` was 8,000, but 4 of the 10
  long-context eval states are longer (up to 17,621 characters), so enforcing it
  would have broken the long-context benchmarks. Checked on the live server
  started from `/tmp`: 32,000 characters 200 in 2.7 s, 32,001 → 400, the longest
  eval state 7.0 s, a 2,000,000-byte header → 413, `GET /v1/systemone` → 405
  with `Allow: POST`, bearer auth 401/200 (healthz open), two identical requests
  give identical responses. Found while documenting: Choice accepts up to 26
  options, not 20 (PP-20).
- [x] **PP-9 Copy-paste-clean commands.** DONE 2026-10-10. The 43-line block is
  split under headings (endpoint, shadow proxy, adaptation loop, benchmarks in
  three groups: HTTP clients of :8095, scripts that load their own model, so stop
  the server first, and CPU-only or API-only). The redundant `bash -c 'export
  HF_HOME=…'` wrappers are gone (`nix develop -c <cmd>` exports it; checked:
  `HF_HOME`, Python 3.12.15, mlx-lm 0.32.0, anyjev 0.2.0), readiness is stated
  (the `systemone on :8095 (head loaded: …)` line, `GET /healthz`) and the four
  Jev legs are named. `bench_mass_removal` and `banking77_greedy` were missing
  from the list and are in.
- [x] **PP-10 Run-from-anywhere trap.** DONE 2026-10-10. Every script derives
  `ROOT = Path(__file__).resolve().parents[N]`, puts it on `sys.path` and
  anchors its default paths to it (paths the user passes stay relative to the
  cwd). Checked from `/tmp`: `--help` of all 26 scripts exits 0 and writes
  nothing; `systemone.py` and `proxy.py` start from `/tmp`, find the head and
  write `logs/` under the repo root; `shadow_stats.py` and `pool_evals.py`
  reproduce their frozen numbers; both authoring scripts pass `--check`. One
  spot was missed at first and found later: `decide/core/trace_states.py` had
  `DATA = "data"`, and the renderer strips the repo path prefix computed from
  it, so the state text the head reads (and the rows logged for `observe.py`)
  depended on the directory the proxy was started from. It is anchored now;
  `tests/test_render.py` fails on the old behaviour (checked by putting it back),
  and a proxy started from `/tmp` logs relative tool paths in a real run.
- [x] **PP-11 Agent-harness path under-specified.** DONE 2026-10-10. README
  gives the three processes in order, a concrete prompt on the sandbox (make
  `Greeter.farewell()` include the name and update the test, which has a real
  failing step because the default name is "world"), the readiness check, the
  follow-up `shadow_stats.py logs/shadow_log.jsonl`, where each log goes, that
  opencode is installed separately, that `OPENCODE_CONFIG` is relative (run from
  the repo root) and that the proxy only shadows conversations that mention
  `data/repos/<name>/`. Verified end to end with the installed opencode 1.18.33
  on the documented ports, twice: in a scratch copy of the sandbox (61 s) and in
  the repo's own sandbox from the repo root with the proxy started from `/tmp`
  (50 s; sandbox restored byte-identical afterwards). Both: tests green, 7 shadow
  rows (6 labelled), 6 observe rows, `shadow_stats.py` reads the live log; in the
  second run the rendered tool paths are relative. Things found and now written
  in `docs/RUNNING.md`: the agent edits the sandbox files (restore them);
  `opencode run` hangs at startup when its stdin is an open pipe, which looked
  like a proxy stall for an hour of debugging (3 of 3 hangs with an open pipe,
  3 of 3 starts in 2 s with `< /dev/null`; not a terminal problem); opencode
  makes its own HTTPS connections at startup; the logs contain the agent's
  absolute paths, so scrub them before promoting one to `results/`.
- [x] **PP-12 No tests, no CI.** DONE 2026-10-10. `tests/` (stdlib `unittest`,
  26 tests, about 8 s, no model or GPU): the endpoint contract (validation,
  limits, auth, routes, healthz, through the real `Handler` on an ephemeral
  port), the `runio` policy, `--help` of every script from another directory,
  byte-for-byte regeneration of both eval sets and the pooled report, the
  `shadow_stats` numbers, the frozen-file guard and the fail-fast of the HTTP
  benches. `.github/workflows/ci.yml`: a portable job (ubuntu, `anyjev` +
  numpy; the 9 scripts that import MLX at module level are skipped) and an
  informational macOS job with the full requirements. The suite passes in the
  Nix venv, in a fresh pip venv, and with MLX made unimportable (9 skipped).
  The workflow itself has not run (no repo yet); only its YAML was parsed.

#### P2 — polish

- [x] **PP-13 `systemone.py` / `proxy.py` details.** DONE 2026-10-10. (The
  hard-coded response `model` was fixed with PP-2.) `GET /v1/systemone` is a 405
  with `Allow: POST` and a usage line, other paths a 404 that lists the routes;
  both parsers have a description and `--port` help; the proxy docstring and
  `--upstream` help say it takes a URL.
- [x] **PP-14 `registry/README.md`.** DONE 2026-10-10. Says which head is
  served (`next_move.json`; `next_move_v3.json` is the same bytes, SHA-256
  `e9268efc…`), that `v2_seed` is provenance and `observed` is replay output
  (fitted on 60 of the 73 shipped rows, out-of-fold 0.683, weaker than the served
  head), how to serve another head (`--head`) and promote one, and that the gold
  seed is not shipped.
- [x] **PP-15 Glossary.** DONE 2026-10-10. `docs/CONCEPTS.md` ends with a
  glossary: state, typed question, readout, label-free, head, Face 1 and 2,
  shadow proxy, flywheel, tau, answer mass, bright line, fidelity vs accuracy,
  fork ("at fork" is the split from the private prototype on 2026-10-07, not a
  GitHub fork), gold seed, frozen evidence, rotations, prefix sharing.
- [x] **PP-16 Docs housekeeping.** DONE 2026-10-10. `docs/RESULTS.md` has a
  contents block linking all 7 sections and their sub-sections (anchors checked
  by `tests/test_docs.py`, which also fails if a new H2 is not listed).
  `AGENTS.md` no longer duplicates the command list: it points to
  `docs/RUNNING.md` (now complete: `fit_registry_head`, `stub_server`, the
  authoring `--check` runs and the tests were added) and keeps only the rules an
  agent needs; the same test fails if RUNNING.md stops mentioning a script. The
  README docs index lists CONTRIBUTING, AGENTS and THIRD_PARTY (PP-7), the
  `.gitignore` comment is fixed (PP-6), `CONTRIBUTING.md` has a Development
  section and a "Questions, bugs, corrections" paragraph (the repository's
  issue tracker). Not done on purpose: moving DECIDE/HARNESS/HISTORY to
  `docs/history/` (16 files and 27 `§` references to touch, no gain).
- [x] **PP-17 Unguarded scripts.** DONE 2026-10-10. `author_v2.py` and
  `author_long.py` parse arguments before building anything (`--help` is safe),
  write `runs/questions_*.jsonl` by default and need `--out data/eval/… --force`
  to regenerate in place; `--check` compares the regenerated text with the
  shipped file. Both regenerate bit for bit, so the eval sets are reproducible
  (`--check` exits 1 once they differ, i.e. after the known `questions_long`
  fixes). `shadow_stats.py` and `stub_server.py` use argparse.
- [x] **PP-18 Output rights of the stored Jev answers.** Terms read
  2026-10-10 from the primary pages (raw text, not search summaries): TypeSafe
  MCA (rev. 2026-09-23), Terms of Use (2026-09-19), Acceptable Use Policy
  (2026-09-23), OpenRouter Terms (2026-08-31). Decision (maintainer,
  2026-10-10): **keep** the stored answers, for benchmark visibility, with
  the evaluation-only notice in `THIRD_PARTY.md`.
  - MCA §4.2: "As between Customer and TypeSafe and to the extent permitted by
    Laws, TypeSafe does not claim ownership of Input and TypeSafe disclaims
    ownership of Output. TypeSafe hereby assigns to Customer all of its right,
    title, and interest, if any, in the Output."
  - None of the four documents restricts benchmarking, comparing, or
    publishing results or outputs (grepped for benchmark, publish, compar,
    evaluat, performance; the hits are warranty, license and publicity
    boilerplate).
    TypeSafe's Confidential Information (MCA §14) covers pricing,
    credentials, Documentation and "other non-public information with respect
    to the Services"; the stored `jev_cost` fields are OpenRouter's public
    list price. MCA §16.4 grants no right to use TypeSafe's name, brand or
    logo: keep to plain-text references plus the not-affiliated line, no logo.
  - The restriction that matters, MCA §2.3(b): Customer will not "use the
    Services or any Output (defined below) to perform model distillation,
    train a model to imitate the output of the Services, or develop (or to
    facilitate the development of) a similar or competing product or
    service". This repo is Jev-style and scores itself against Jev, so
    "facilitate" is the open word.
    What sits on the other side: the POLICY §3 bright line (nothing fitted on
    another model's answers), small sets (350 typed questions, 400 + 500
    BANKING77 labels), and the evaluation-only notice now in `THIRD_PARTY.md`.
  - Which terms bind an OpenRouter user is itself unclear: OpenRouter makes the
    user responsible for each model's own terms and lists TypeSafe's website
    Terms of Use for Jev (silent on API output), while the API agreement is the
    MCA. Neither has a publication restriction. Not legal advice.
  - The first list of files was incomplete. Per-item Jev answers sit in 15
    files: `jev_compare{,.n300,.long}.json`, `jev_repeat{,.long}.json`,
    `banking77{,.9b}.json`, `banking77_full77.jev.json`,
    `bench_vs_generate{,.n300,.long,.9b,.9b.v2,.9b.long}.json`,
    `labels_long.scored.json`. Aggregate-only (Jev numbers, no per-item
    answers): `jev_compare.pooled.json`, `rotation_budget.json`,
    `l0_ablation.json`.
  - Had the call been "drop": stripping the per-item fields breaks the
    CPU-only recompute of the agreement numbers (`pool_evals.py` reads
    `jev_compare*.json`; `rotation_budget.py` and `banking77_greedy.py` read
    `banking77*.json`) and the "reproduces from frozen artifacts" claims in
    `results/INDEX.md`. Optional certainty: ask TypeSafe for a written OK
    (`support@typesafe.ai`, the contact in its AUP).

- [x] **PP-19a `git init` and the first commit.** Done: two commits (`Init env`, `Initial release`),
  107 tracked files including `flake.nix` and `flake.lock` (Nix refuses an untracked flake;
  that is why they are added before the first `nix develop`). The GitHub half is PP-19b.

- [x] **PP-19b GitHub repo.** DONE 2026-10-11: public at
  github.com/alexshagov/take-my-layer (remote `origin`, `main`; local HEAD =
  the public HEAD, confirmed via `git ls-remote`; four commits, 110 tracked
  files against the audit's 107 — net delta since the first release: the
  2026-10-11 fast-path batch (+`batching.py`, `bench_fast_path.py`,
  `test_fast_path.py`, `fast_path.json`) minus `docs/ROADMAP.md`, merged
  into this file). The README
  tagline is live ("Shut up and take my layer."). Two deviations from the
  plan, read off the live repo page 2026-10-12: the About line is the short
  "Zero-training decision layer for a local LLM on Apple Silicon", not the
  planned 241-char Jev-style line, and the page lists no topics (planned:
  mlx, apple-silicon, qwen, logprobs, local-llm).

- [x] **PP-20 Option-cap wording (20 vs 26).** DONE 2026-10-10. anyjev 0.2.0's
  letter readout caps Choice at 26 options (`MAX_OPTIONS`); the endpoint accepted
  21 and rejected 30 with "at most 26 options in letter readout". The wording is
  corrected in `docs/RESULTS.md` (the section title "the >26-option unlock" and
  both "≤26-option Choice cap" mentions), `decide/core/mlx_backend.py`,
  `decide/bench/banking77_full77.py`, the `systemone.py` docstring, the README
  sentence and `docs/API.md`. No number changed.

#### Open checks from the 2026-10-10 review (all done)

- [x] Cold-cache first run: done 2026-10-10 from a fresh pip venv with an empty
  `HF_HOME`: the endpoint downloaded the 4B model (2.9 GB) and was ready after
  72 s, and the README example came back bit-identical (confidence
  0.5955441762711209).
- [x] Clean non-Nix install: done 2026-10-10, see PP-3 (`pip install -r
  requirements.txt` in a fresh venv, 32 s, same versions as the Nix shell, test
  suite green).
- [x] Proxy + opencode end-to-end: done 2026-10-10, see PP-11.
- [x] A 9B server run: done 2026-10-10 in a scratch copy (`model` field
  correct; the README example was served at L0, since the 9B has no head).
- [x] TypeSafe's output terms (PP-18): read 2026-10-10, see PP-18.

#### Verified, no action

README example reproduces exactly (0.5955 / 0.4039 / 0.0006, 140 tokens; 3
runs byte-identical; ready in 1.6 s at 2.8 GB; a mixed Choice/Noul/Score
request took 0.69 s; bad requests return clear 400s). Headline numbers match
the frozen artifacts (75.7% with +4.7, p = 0.0026; 9B 82.0% / +6.3; 98.7%
ceiling; banking77 87.5 / 91.25 / 95.5; full-77 66.4 vs 60.8, p = 0.00335;
typed-decisions 61.6 / 60.4; 22.2 → 3.7 s; ECE 0.238, 37% routed error at
59% coverage). No secrets in the tree. Docs have no dangling references
except files removed on purpose. `results/INDEX.md` lists every artifact.
External citations hold: AnyJev (PyPI, GitHub, arXiv 2610.00831), the Clef
table, Unsloth 36/33 → 73/78, the OpenRouter ToS revision.

External sources re-checked 2026-10-10:
[Unsloth guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth),
[Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/),
[anyjev on PyPI](https://pypi.org/project/anyjev/),
[AnyJev Technical Report](https://arxiv.org/abs/2610.00831),
[AnyJev CHANGELOG](https://github.com/nokia-applied-research/AnyJev/blob/main/CHANGELOG.md),
[OpenRouter terms](https://openrouter.ai/terms).

- [x] The Unsloth 36/33 pre-training numbers (they exist only in the frozen
  `results/typed_decisions*.json` reference lines) were re-verified 2026-10-10: the guide
  confirms 0.8B 36% -> 73% and 2B 33% -> 78%; the README wording was fixed in PP-4.
