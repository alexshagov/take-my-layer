# TODO.md: open work

What is still open: measurement gaps, a build backlog and decisions to make. Finished
work and its evidence live where the result is reported: docs/RESULTS.md (numbers),
docs/DECISIONS.md (why), results/INDEX.md (artifacts). Item ids (`#n`) are stable:
other docs cite them. Wording rules for any new number are in docs/RESULTS.md
(fidelity = "aligned with the reference", never "more correct"; accuracy claims name
the arena and model size).

## Measurement gaps

Benches and proofs that a README or RESULTS claim still lacks.

- [ ] **#2 Same-arena ladder: L0 vs L1 vs L2 on one held-out task.** The levels
   table asserts what each level buys, but L2 has only the `next_move`
   holdout (0.826) and L1 nothing. Fit a temperature and a head for one
   benchmark question (e.g. a banking77 or Noul layout) and evaluate all
   three levels on the same holdout. Cost: one fit plus one eval run.

- [ ] **#3 L1 calibration bench.** The "0.93 knowledge-gap tail" is L1's stated
   reason to exist with no reliability curve behind it. Fit the temperature
   and report ECE / reliability on held-out L0 outputs, in-distribution. Also
   covers continuous Score values (caveat: temperature scaling degrades under
   shift, so it is a tail fix only). Cost: small.

- [ ] **#4 Foreign-domain L2 generalization.** All 73 observe rows and 81 shadow
   rows are `qjev-sandbox`; the domain-general charter has no evidence
   outside it. The synthetic-data A/B was neutral, and its real test is
   foreign-domain agreement. Blocked on usage, not code.
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
   dynamic-batching idea (build backlog) has no baseline number to beat.
   Cost: moderate (client plus server under load).

- [ ] **#8 CLINC150 as a second gold-label arena.** banking77 is the only
   gold-label accuracy bench; CLINC150 sits in the reference ecosystem's
   training mix. Lower priority. Cost: loader plus bench run.

- [ ] **#9b Real-label check of the L2 exit-layer candidates.** The 2026-10-11 sweep
  (docs/RESULTS.md, "Exit layer of an L2 head") found that a head fitted on few labels
  gains 14 points going from layer 16 to layer 20 and nothing beyond layer 22, but with
  synthetic states labelled by the model's own readout. Re-run on real labelled traffic
  (the observe log once there is foreign telemetry, #4) before changing
  `LAYERS = [16, 19, 22]` in `observe.py` and `fit_registry_head.py` to about
  `[20, 22, 24]`. The shipped head (1,824 gold labels, layers 16-22 tied) is unchanged
  and cannot be refit without the withheld seed.

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

- [ ] **#15 `--early-stop`, `--no-batch` and the L2 head under repeated requests.** The
  determinism check (docs/API.md, "Repeatability") covered the default path only. Replay
  the 70-request sequence on two fresh servers in each of those modes and byte-compare.
  Cost: small, and worth committing as a bench this time.

## Build backlog and ideas

- [ ] **Close the observe() loop (small build)**: today the loop is
  self-logging but human-gated: the proxy banks observe rows
  automatically, but the refit (`decide/adapt/observe.py`) and the
  promotion (`--head`) are commands a human runs when curious. The last
  mile is a periodic refit with **canary-gated auto-promotion**:
  recompute the artifact on a schedule, swap it in only if its holdout
  metric beats the incumbent's, else keep serving the old one; log every
  swap. Scope honesty, stated up front: if a deployment ever reaches the
  point of owning a rich labeled history of its exact decision surface,
  the crossover verdict (docs/PRIOR_ART.md) says the right next tool is a
  fine-tune or a classical trained classifier: the auto-loop is for the
  regime where labels arrive as a sparse trickle of your own traffic, not
  as a dataset.

- [ ] Truncated-checkpoint serving (blocks 0..b as a standalone MLX model).

- [ ] Upstream the MLX backend to AnyJev (their help-wanted slot), together with the
  state-share split and the batched suffix passes.

- [ ] Generic call construction (extractors or file-shortlist head) to
  restore a routing leg.

- [ ] Label-free adaptation (`adapt="routed"`) as the domain-shift test.
  It targets reworded/relisted layouts, so measure before trusting it for
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

## Decisions to make

- [ ] What the endpoint's running prior should do for a stream of repeated requests
  (found while checking determinism; docs/API.md, "Repeatability"): keep it and
  document it (done), expose a way to reset it without a restart, or require more
  distinct states before it applies. A code change; the docs only describe today's
  behaviour.

- [ ] Make the proxy's `--no-mass` the default: 2.05x faster per shadow decision with
  0/30 answer flips, but the logged `mass` diagnostic disappears (the guard stays log-only
  either way).

- [ ] Make `--early-stop` a default: only after #11.
