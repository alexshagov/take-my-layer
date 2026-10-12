# DECISIONS.md: the decision record

One entry per decision that shaped the artifact: what was chosen, why, and
where the evidence lives. Measured numbers: docs/RESULTS.md. Where the
design comes from: docs/DESIGN.md. Open work: docs/TODO.md.

## Method & scope

- **Read the decision out of the model instead of generating it.** The
  agent step was ~8 s of decoding a ~200-token tool call; a readout is one
  prefill. Evidence: DESIGN.md "The idea", "Feasibility on the prototype's data"; RESULTS "Decide vs
  generate".
- **Deploy AnyJev's method (`anyjev==0.2.0`), build the missing MLX
  backend ourselves.** AnyJev never shipped an MLX backend (their
  help-wanted slot); the backend here is an original implementation of
  their documented Backend protocol. The pin is load-bearing: 0.3.0
  removed the L1/L2 head machinery this repo serves. THIRD_PARTY.md;
  requirements.txt.
- **Domain-general charter.** The layer must keep working off its birth
  substrate (different languages, test frameworks, harnesses). This drives
  the coarse-label policy, the renderer-v3 generalization, and the
  seed-removal below. POLICY.md header.
- **Coarse question vocabulary: inspect / run / modify.** Fine tool
  identity measured as label noise twice: at L0 (read↔grep = 31/65
  errors) and at L2 (6-tool merged ≡ 3-class exact). Evidence:
  DESIGN.md "Coarse questions"; POLICY.md §2.
- **Two surfaces; benchmarks on the decision endpoint only.** The endpoint
  is deterministic and never refits (replaying a request sequence on a fresh
  server is bit-identical; its only state is the running batch prior); the proxy is
  shadow-only. Routing was measured unsafe on foreign live traffic
  (routed error 37% at 59% coverage): shadow-only stays correct until a
  domain-general call-construction leg exists and τ is re-read per
  surface. docs/CONCEPTS.md "The two surfaces"; RESULTS "Live re-read"; POLICY §4.
- **The fabrication leg was removed at fork** (it was seed-domain-only by
  construction). The contract itself is proven and kept as evidence
  (`decide/research/stub_server.py`). POLICY §4.

## Data & adaptation

- **The gold seed is not shipped.** Recorded work sessions from another
  project don't belong in a public release. The heads ship as fitted
  artifacts with lineage; refits go through synth_states.py + observe().
  POLICY.md "Provenance".
- **Calibration rule: L1/L2 fit only on the serving surface's own
  observe() outcomes**, never benchmark train splits, never another
  model's answers (that would be distillation, or capacity-free
  fine-tuning). Benchmarks are quoted at zero-training L0; the
  trained-vs-readout gap is reported as a finding. POLICY §3.
- **Never synthetic-only; supplement, don't replace.** AnyJev's recipe:
  gold stays the strong corpus, synthetic supplements at ≥300; ours
  measured neutral at 285 on a same-domain holdout. POLICY §3;
  `results/synth_gain.json`.
- **observe() fits a readout adapter, not a fine-tune.** No base weight is
  updated; nothing of ours enters the forward pass. A deployment owning a
  rich labeled history of its exact decision surface should graduate to
  real fine-tuning: the train-in vs read-out crossover verdict.
  docs/CONCEPTS.md "The two surfaces" (the observe() loop); PRIOR_ART.md.

## Serving

- **Fit on what you serve.** Heads read hidden states of the exact served
  artifact (quantization and truncation shift them). POLICY §1.
- **Answer-mass guard default OFF** (`MASS_TAU=0.0`, log-only): measured
  non-separating on this model (0.92–0.99 mass on every state class).
  Kept gateable for a future model that does separate. RESULTS
  "Answer-mass guard measurement".
- **Full-K rotations by default.** The 2026-10-07 plan was
  `--max-permutations 2` as the latency mode (3.8× on a 10-option Choice).
  Re-measured 2026-10-11 on the 350 Jev-agreement questions (mostly
  yes/no and 3-5 option Choices) it costs accuracy at 4B and never touches
  yes/no or Score, so full K stays the default and `--early-stop` is the
  latency mode. RESULTS "Rotation budget", "Speed/precision sweep".
- **Batched suffix passes default ON (2026-10-11); `--early-stop` opt-in.**
  The passes of a request run as one forward over the shared state cache,
  under a memory budget (`--kv-budget-gib`); 1.9× on short requests, the
  same 76.0% agreement with Jev, 4 of 350 answers differ at bf16 noise level.
  Early stopping saves a further 19% of passes but makes the reported
  confidences more optimistic (ECE 0.035 → 0.046), so it is off by default.
  RESULTS "Speed/precision sweep".
- **State-level prefix sharing default ON.** 5.3× on long requests with
  parity up to the documented bf16 cache-boundary class. RESULTS
  serving-layer.
- **The decide worker is a supervised subprocess with a hard timeout.**
  Born from the measured OOM/wedge postmortem (an in-process decide once
  held the Head lock and silenced every later request). DESIGN.md
  "Shadow proxy and the supervised worker".

## Claims & wording

- **Fidelity claims say "aligned with the reference", never "more
  correct"; accuracy claims name the arena and model size.** The
  readout-vs-generation edge is fidelity (+4.7 / +6.3 at n=300, McNemar
  p=0.003); accuracy is per-arena (10-way ties at both sizes; full-77-way
  9B ahead +5.6, p=0.0034). RESULTS header + "Decide vs generate".
- **External numbers are citation-audited before publish** (Clef table,
  Unsloth set, dataset licenses, arXiv IDs, OpenRouter terms), and
  unverifiable citations are deleted rather than kept.
- **A non-public decision server was read for methodology only:**
  never a dependency, no code used.
  POLICY.md provenance note.
