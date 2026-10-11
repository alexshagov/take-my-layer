# POLICY.md — domain-general policy, provenance, and the calibration bright line

The operating rules that keep the artifact honest. The charter: the layer
is DOMAIN-GENERAL — it must keep working when the repo is Python, the
tests are pytest, the task is sysadmin work, or the agent harness is not
opencode, not only on the single substrate it was born on.

## Domain-general policy

1. **Substrate: heads read the BASE model, never an SFT-tuned adapter.**
   SFT-tuned weights shift hidden states toward the tuned tool patterns
   (docs/DECIDE.md §5.8); the current registry head already honors this
   (`registry/mlx-community__Qwen3.5-4B-MLX-4bit/base/`). If a different
   adapter is ever served, refit on that exact artifact — no exceptions.

2. **Question vocabulary stays coarse and domain-free.** inspect / run /
   modify (3-class `next_move`, `decide/core/labels.py` is the one
   source) is the proven shape; AnyJev's 20 workflow questions (which
   tool, done yet, needs_review) are the template. Fine tool identity is
   label noise (docs/DECIDE.md §8 step 2/3 finding) — a second-stage
   question, not a bigger head.

3. **Calibration data: gold seed not shipped; synth + observe are the
   in-repo corpora.** The measured generalization recipe (AnyJev
   `bench/synth_states`, research_log:563):
   - the 2,306 verifier-backed seed points were the SEED for the shipped
      heads (removed from the repo at release prep — see Provenance below;
     their measured effect survives in the artifacts and
     `results/synth_gain.json`);
   - **supplement** with teacher-labelled synthetic states across domains
     — other languages, pytest/jest/go-test output shapes, sysadmin/file
     tasks, varied history lengths — ~3 exemplars → N states at temp 1.0,
     labels hard or soft, gain appears at ≥300 synthetic (ours measured
      NEUTRAL at 285 on a same-domain holdout);
   - **never synthetic-only**: AnyJev measured synthetic-only 6–9 points
     BELOW 300 gold labels. (30–40 synthetic examples is NOT a viable
     replacement — 30 is `observe()`'s solve minimum, deep in the weak end
     of the label curve: 20 labels → 0.654 vs 300 → 0.772. Recorded here
     so the misremembered "replace gold labels with 30–40 synthetic"
     version does not resurface.)
   - `observe()` live telemetry from real sessions is the permanent
     adaptation engine (auto-solve at 30, refit at 60/120/…).
   - **Bright line for L1/L2 (the zero-training thesis, stated as a
     rule):** calibration artifacts are fitted ONLY from the serving
     surface's own verified traffic (observe()) — never from public
     benchmark train splits, and never against ANOTHER MODEL's answers
     (calibration targets are outcomes of what actually happened; a
     reference API is the ruler we measure with, not an ingredient —
     matching it would be distillation). Benchmark numbers are quoted
     at zero-training L0; the trained-vs-readout gap on train-friendly
     benchmarks is reported as a finding (what fine-tuning buys), not
     something to calibrate away. Fitting per-question knobs on a
     benchmark's labels to climb it would be capacity-free fine-tuning
     — the classical-ML regime (labels → train a classifier) this
     project exists to avoid; on that turf use Unsloth. (Scoping note:
     `typed-decisions` as a "public fitting corpus" means seeding heads
     for NEW question surfaces, never serving answers on its own
     benchmark items — that would be the calibration-away this line
     forbids.)
   - Public-dataset option (license VERIFIED Apache-2.0, 2026-10-08):
     `LocalLLaMA/typed-decisions` — the Laya/Clef/Unsloth-format corpus
     (12 public sources incl. banking77 and clinc150) — usable as a
     public supplement/refit corpus now that the gold seed is gone, and
     the common benchmark for the trained-vs-readout head-to-head.

4. **The fabrication leg was REMOVED at fork (seed-domain-only by
   construction).** The router, when it eventually routes, acts on
   decisions only until a domain-general way to build the call exists:
   generic extractors or the file-shortlist second-stage head (docs/TODO.md,
   build backlog). The fabrication contract itself (fabricated `tool_calls` SSE
   shape that opencode parses) is proven — see
   `decide/research/stub_server.py` and the pre-fork step-6 record.

5. **Unseen-domain safety comes from tripwires, not optimism:**
   answer-mass guard (low mass on the label alphabet → not a readable
   decision → fall back to generation; ours to build — AnyJev ships mass
   only as a diagnostic, not as a gate; MEASURED NON-separating on this
   model, `MASS_TAU=0.0` log-only — see docs/RESULTS.md) and
   `level="auto"` degradation (L2 → L0 → pass-through) when the decide
   worker is down/slow. On an unseen domain the head will read
   *something*; these are what catch it.

## Provenance

Forked from a private single-domain prototype on 2026-10-07
(docs/DECIDE.md steps 1–6 complete there: MLX backend, L0 bootstrap, L2
heads, router replay H1, shadow proxy H2-0, FIX-AB batch 1 interrupted at
21/56). docs/DECIDE.md + docs/HARNESS.md are the historical investigation
records; the parent project retains the full original (incl. all FIX-AB
state).

**Removed at fork (substrate-specific machinery):** `args_extractors.py`
(verify templates + path resolution for the seed domain's test
frameworks), `router_replay.py` (H1 — drove those extractors; findings
live in docs/DECIDE.md §8 step 4), `fixab.py` + `fixab_status.py` +
`registry/fixab_tasks.json` + `pipeline/` (the mutation/A-B e2e
protocol), the `data/repos` symlink, and the toolchain env plumbing in
flake.nix. The proxy is now **shadow-only** (no fabrication leg).

**Removed at GitHub-release prep (2026-10-07):** the gold-seed
dataset itself — `data/traces/` (595 recorded work sessions
→ 2,306 decision points), `data/tasks.json`, `data/repos.json`, and the
historical shadow logs (`shadow_log{,.run1,.step6}.jsonl`) —
recorded sessions from another project don't belong in a public release.
What ships instead: the head artifacts fit from that seed
(`registry/…/next_move.json`, lineage recorded in each artifact — holdout
0.826), the model-generated in-repo corpora (`data/eval/questions.jsonl`,
`results/observe_log.jsonl`), and our own v3 shadow log. The synthetic
states corpus is generated on demand by `decide/adapt/synth_states.py`
(not shipped; its measured A/B result is frozen in
`results/synth_gain.json`). Refitting without the seed =
`decide/adapt/synth_states.py` + the observe() loop
(`load_decision_points` raises a clear error; docs/DECIDE.md keeps the
historical numbers).

Release practice (2026 norm: fitted artifacts + documented lineage,
training data withheld — cf. open-weights releases with
data-construction blueprints): the heads are **not bit-reproducible and
do not claim to be**; each artifact records its base model, layer, label
vocabulary and holdout, and refits go through
`decide/adapt/synth_states.py` + `decide/adapt/observe.py`.

**Seed-domain residue status:** the v3 renderer (2026-10-07) dropped the
deliberate repo-kind tag and the single-extension glob filter, and the
release prep removed the seed dataset itself (see above). Residual
same-substrate things that remain by design: the `_render_event` compact
formats descend from the seed domain's test-runner output shapes, and the
shipped heads' lineage is the private seed. The v3 artifact was promoted
and its live re-read ran (docs/RESULTS.md, "Live re-read").

**Methodology provenance note:** a private production decision-server
was read for methodology early on (never a dependency, no code used);
references were anonymized to "the reference server" at release prep.
