# POLICY.md: domain-general policy, provenance, and the calibration rule

The operating rules that keep the artifact honest. The charter: the layer
is domain-general: it must keep working when the repo is Python, the
tests are pytest, the task is sysadmin work, or the agent harness is not
opencode, not only on the single substrate it was born on.

## Domain-general policy

1. **Substrate: heads read the base model, never an SFT-tuned adapter.**
   SFT-tuned weights shift hidden states toward the tuned tool patterns
   (docs/DESIGN.md, 'The method and the backend'); the current registry head already honors this
   (`registry/mlx-community__Qwen3.5-4B-MLX-4bit/base/`). If a different
   adapter is ever served, refit on that exact artifact, no exceptions.

2. **Question vocabulary stays coarse and domain-free.** inspect / run /
   modify (3-class `next_move`, `decide/core/labels.py` is the one
   source) is the proven shape; AnyJev's 20 workflow questions (which
   tool, done yet, needs_review) are the template. Fine tool identity is
   label noise (docs/DESIGN.md, 'Coarse questions'): a second-stage
   question, not a bigger head.

3. **Calibration data: gold seed not shipped; synth + observe are the
   in-repo corpora.** The measured generalization recipe (AnyJev's
   `bench/synth_states` and its research log, upstream):
   - the 2,306 verifier-backed seed points were the seed for the shipped
     heads (removed from the repo at release prep: see Provenance below;
     their measured effect survives in the artifacts and
     `results/synth_gain.json`);
   - **supplement** with teacher-labelled synthetic states across domains
     (other languages, pytest/jest/go-test output shapes, sysadmin/file
     tasks, varied history lengths): ~3 exemplars → N states at temp 1.0,
     labels hard or soft, gain appears at ≥300 synthetic (ours measured
     neutral at 285 on a same-domain holdout);
   - **never synthetic-only**: AnyJev measured synthetic-only 6–9 points
     below 300 gold labels. (30–40 synthetic examples is not a viable
     replacement: 30 is `observe()`'s solve minimum, deep in the weak end
     of the label curve: 20 labels → 0.654 vs 300 → 0.772. Recorded here
     so the misremembered "replace gold labels with 30–40 synthetic"
     version does not resurface.)
   - `observe()` live telemetry from real sessions is the permanent
     adaptation engine (auto-solve at 30, refit at 60/120/…).
   - **Calibration rule for L1/L2 (the zero-training thesis):** artifacts
     are fitted only from the serving surface's own verified traffic
     (observe()), never from public benchmark train splits, and never
     against another model's answers (calibration targets are outcomes of
     what actually happened; a reference API is the ruler we measure with,
     not an ingredient, and matching it would be distillation). Benchmark
     numbers are quoted at zero-training L0; the trained-vs-readout gap on
     train-friendly benchmarks is reported as a finding (what fine-tuning
     buys), not something to calibrate away. Fitting per-question knobs on
     a benchmark's labels to climb it would be capacity-free fine-tuning,
     the classical-ML regime (labels → train a classifier) this project
     exists to avoid; on that turf use Unsloth. (Scoping note:
     `typed-decisions` as a "public fitting corpus" means seeding heads
     for new question surfaces, never serving answers on its own
     benchmark items, which is the calibration-away this rule forbids.)
   - Public-dataset option (license verified Apache-2.0, 2026-10-08):
     `LocalLLaMA/typed-decisions` (the Laya/Clef/Unsloth-format corpus;
     12 public sources incl. banking77 and clinc150) is usable as a
     public supplement/refit corpus now that the gold seed is gone, and
     is the common benchmark for the trained-vs-readout head-to-head.

4. **The fabrication leg was removed at fork (seed-domain-only by
   construction).** The router, when it eventually routes, acts on
   decisions only until a domain-general way to build the call exists:
   generic extractors or the file-shortlist second-stage head (docs/TODO.md,
   build backlog). The fabrication contract itself (fabricated `tool_calls` SSE
   shape that opencode parses) is proven; see
   `decide/research/stub_server.py` and docs/DESIGN.md, "Shadow proxy and the
   supervised worker".

5. **Unseen-domain safety comes from tripwires, not optimism:**
   answer-mass guard (low mass on the label alphabet → not a readable
   decision → fall back to generation; ours to build; AnyJev ships mass
   only as a diagnostic, not as a gate; measured non-separating on this
   model, `MASS_TAU=0.0` log-only; see docs/RESULTS.md) and
   `level="auto"` degradation (L2 → L0 → pass-through) when the decide
   worker is down/slow. On an unseen domain the head will read
   *something*; these are what catch it.

## Provenance

Forked from an unpublished single-domain prototype on 2026-10-07
(its steps 1–6 complete there: MLX backend, L0 bootstrap, L2
heads, router replay H1, shadow proxy H2-0, FIX-AB batch 1 interrupted at
21/56). docs/DESIGN.md condenses that investigation. The prototype keeps the full original (incl. all FIX-AB state);
none of it is part of this repository.

**Removed at fork (substrate-specific machinery):** `args_extractors.py`
(verify templates + path resolution for the seed domain's test
frameworks), `router_replay.py` (H1, which drove those extractors; findings
live in docs/DESIGN.md, 'Router replay'), `fixab.py` + `fixab_status.py` +
`registry/fixab_tasks.json` + `pipeline/` (the mutation/A-B e2e
protocol), the `data/repos` symlink, and the toolchain env plumbing in
flake.nix. The proxy is now **shadow-only** (no fabrication leg).

**Removed at GitHub-release prep (2026-10-07):** the gold-seed
dataset itself, `data/traces/` (595 recorded work sessions
→ 2,306 decision points), `data/tasks.json`, `data/repos.json`, and the
historical shadow logs (`shadow_log{,.run1,.step6}.jsonl`).
Recorded sessions from another project don't belong in a public release.
What ships instead: the head artifacts fit from that seed
(`registry/…/next_move.json`, lineage recorded in each artifact: holdout
0.826), the model-generated in-repo corpora (`data/eval/questions.jsonl`,
`results/observe_log.jsonl`), and our own v3 shadow log. The synthetic
states corpus is generated on demand by `decide/adapt/synth_states.py`
(not shipped; its measured A/B result is frozen in
`results/synth_gain.json`). Refitting without the seed =
`decide/adapt/synth_states.py` + the observe() loop
(`load_decision_points` raises a clear error; docs/DESIGN.md keeps the
historical numbers).

Release practice (2026 norm: fitted artifacts + documented lineage,
training data withheld: cf. open-weights releases with
data-construction blueprints): the heads are **not bit-reproducible and
do not claim to be**; each artifact records its base model, layer, label
vocabulary and holdout, and refits go through
`decide/adapt/synth_states.py` + `decide/adapt/observe.py`.

**Seed-domain residue status:** the v3 renderer (2026-10-07) dropped the
deliberate repo-kind tag and the single-extension glob filter, and the
release prep removed the seed dataset itself (see above). Residual
same-substrate things that remain by design: the `_render_event` compact
formats descend from the seed domain's test-runner output shapes, and the
shipped heads' lineage is the unpublished seed. The v3 artifact was promoted
and its live re-read ran (docs/RESULTS.md, "Live re-read").

**Methodology provenance note:** early design work read a non-public
decision server for methodology only; it is not a dependency and no code or
data from it is used.
