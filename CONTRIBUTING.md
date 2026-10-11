# Contributing

Apache-2.0; contributions under the same license.

## Questions, bugs, corrections

Use the repository's issue tracker for bugs, questions and corrections to a
quoted number (name the artifact in `results/` and the section of
`docs/RESULTS.md`). A pull request should keep `python -m unittest discover -s
tests` green; for a new number, follow "Changing measurement code" below.

## The rule that bounds everything

Calibration artifacts (L1/L2 heads, temperatures) are fitted ONLY from the
serving surface's own observe() telemetry — never on public benchmark
train splits, never against another model's answers (docs/POLICY.md §3).
Benchmarks are quoted at zero-training L0. If a task looks like "labels →
train a classifier", that is fine-tuning territory (cf. Unsloth), not this
project.

## The highest-value contribution: independent labels

The shipped evaluations carry no human ground truth (docs/RESULTS.md,
"Evaluation disclosure"): items were authored with LLM assistance and the
project's authors built the system under evaluation. The single most
useful contribution is an independent, human-labeled read — even
adjudicating just the ~10 disagreement rows in
`results/labels_long.scored.json`. The items and stored answers ship
in-repo and every headline bench number replicates with one command per
leg (docs/RUNNING.md); the blind-judge leg is the one exception — its
sheets and scored rows ship, but the scoring step was performed outside
the repo.

## Development

Set up as in docs/RUNNING.md (Nix, or Python 3.11+ with `pip install -r
requirements.txt` on an Apple Silicon Mac). The tests need no model and no
GPU:

```bash
python -m unittest discover -s tests -v     # about 10 s
```

They cover the endpoint contract (validation, limits, auth, routes), the
output policy of the scripts, `--help` of every script, and the byte-for-byte
regeneration of the eval sets and the pooled report. Scripts find the repo
root themselves and run from any directory. Anything a script produces goes
to `runs/` (gitignored); `results/`, `registry/` and `data/eval/` are shipped
evidence that no script replaces without `--force`.

## Changing measurement code

- Wording rules hold everywhere: fidelity = "aligned with the reference",
  never "more correct"; accuracy claims name the arena and model size
  (docs/RESULTS.md header).
- New numbers land in docs/RESULTS.md with n, CI, and the honest reading;
  the raw artifact in `results/` (copy it from `runs/`, add its row to
  `results/INDEX.md`; append logs and drafts don't go there).
- Heads are fitted on the exact artifact you serve (POLICY §1); head
  promotions are deliberate, never in-place overwrites of
  `next_move.json`.
- The word on style: measured facts, short sentences, no marketing.
