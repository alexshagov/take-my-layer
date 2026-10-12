# results/ (artifact index)

Frozen measurement artifacts behind the numbers quoted in `docs/RESULTS.md`.
One entry per file: what wrote it, at which readout level, what it measures,
where it is read. Every headline bench reproduces with one command per leg
(docs/RUNNING.md); the API legs need `$OPENROUTER_API_KEY`. A run writes to
the gitignored `runs/` (compare with `diff runs/X results/X`) and no script
replaces a file here without `--force`; the producers read these files as
their frozen inputs (stored Jev answers, stored readout runs).

Level rule: every benchmark and eval ran at label-free **L0** (zero labels,
zero fitting, frozen base) unless the level column says otherwise. L2
appears only in the shadow/live, mass, and head-fit rows; L1 was never
measured.

Notes on the two runtime logs: `shadow_log.v3.jsonl` and `observe_log.jsonl`
are proxy outputs kept as frozen evidence at the dates in RESULTS (both are
all-L2). The live proxy and the endpoint write to the gitignored `logs/`
(`shadow_log.jsonl`, `observe_log.jsonl`, `systemone_log.jsonl`), never
here; replay live traffic with `observe.py --log logs/observe_log.jsonl`,
and copy a log into `results/` only deliberately, as new frozen evidence, after
scrubbing it (the states hold full tool outputs and shell commands, including
absolute paths and anything secret in them).

`observe_log.jsonl` was path-scrubbed on 2026-10-10 before publication: the
dev-machine absolute path became `/path/to/repo` (210×) and the `ls -l`
owner name became `user` (32×). Nothing else changed (73 rows, every
non-`state` field identical). `mass_removal.json` (13 of its 30 states
changed) and `registry/…/next_move_observed.json` were computed from the
pre-scrub text, so a rerun on the scrubbed log can differ: measured
2026-10-11, `observe.py` on the shipped log refits to oof 0.650 with 60/73
in-domain hits (frozen: 0.683, 64/73; the seed head's 40/73 is unchanged),
a small drift, more than last digits.

One artifact predates the current code and does not re-measure exactly:
`typed_decisions.json` (4B, 2026-10-08) was produced by the pre-reorg
pipeline. A rerun on the 2026-10-11 code (same machine, same pinned deps,
two identical runs) gives **61.85%** (misses 763 vs 769; 8 of 2,000 answers
differ) and its `.gold` provenance string says "annotator-majority", the
wording used before the dataset's teacher-derived provenance was verified
(THIRD_PARTY.md). The quoted 61.6% stays as the frozen measurement; the 9B
and prior-strength artifacts re-measure exactly. Refresh deliberately:
run, compare with `diff runs/X results/X`, then `cp`.

| artifact | producer | level | measures | quoted in RESULTS.md |
|---|---|---|---|---|
| `jev_compare.json` | `decide/bench/eval_vs_jev.py` (n=105) | L0 | readout vs real Jev, agreement + latency/resources | "Eval vs real Jev" |
| `jev_compare.n300.json` | `eval_vs_jev.py --questions data/eval/questions_v2.jsonl` | L0 | second authoring batch (n=195) | same |
| `jev_compare.pooled.json` | `decide/bench/pool_evals.py` (CPU-only; reads the two above; regenerates this file byte for byte, checked 2026-10-10) | L0 | pooled n=300 + paired McNemar readout-vs-greedy | "Eval vs real Jev", "Decide vs generate" |
| `jev_compare.long.json` | `eval_vs_jev.py --questions data/eval/questions_long.jsonl` | L0 | long-context regime (n=50) | "Eval vs real Jev" |
| `jev_repeat.json` | `decide/bench/jev_repeat.py --runs 3` | reference only | Jev self-agreement ceiling (98.7%) | same |
| `jev_repeat.long.json` | `jev_repeat.py --questions … --stored results/jev_compare.long.json` | reference only | long-regime ceiling (98.0%); reference-noise check | "Blind external-judge leg" |
| `labels_long.scored.json` | blind external judge (Claude Opus) over `data/eval/labels_long*.sheet.md`; scoring step performed outside the repo | L0 items | judge-agreement on 30 long items | "Blind external-judge leg" |
| `bench_vs_generate.json` | `decide/bench/bench_vs_generate.py` | L0 vs greedy | 4B readout vs greedy, same model | "Decide vs generate" |
| `bench_vs_generate.n300.json` | same, n=300 | L0 vs greedy | readout-vs-greedy headline (+4.7) | same |
| `bench_vs_generate.long.json` | same, long items | L0 vs greedy | generation collapse on long contexts | same |
| `bench_vs_generate.9b.json`, `bench_vs_generate.9b.v2.json`, `bench_vs_generate.9b.long.json` | `bench_vs_generate.py --model …9B… --resources` | L0 vs greedy | 9B legs (v1 / v2 / long) | "Decide vs generate", "Latency/GPU regimes" |
| `banking77.json` | `decide/bench/bench_banking77.py` (4B endpoint :8095 + Jev API) | L0 | 10-way gold accuracy vs Jev on the same subset | "BANKING77" |
| `banking77.9b.json` | same against a 9B endpoint (`--out`) | L0 | 9B 10-way leg | same |
| `banking77.4b.resources.json`, `banking77.9b.resources.json` | ps+ioreg ResourceSampler summary of those runs (dumped ad hoc; no dedicated script) | L0 | RSS/GPU cost of the two runs | same |
| `rotation_budget.json` | `decide/bench/rotation_budget.py` | L0 | m=1/2/3/adaptive curve → m=2 default | "Rotation budget" |
| `banking77_full77.json` | `decide/bench/banking77_full77.py` | L0 | full-77-way readout, 4B (multi-token scoring) | "Full-77-way readout" |
| `banking77_full77.9b.json` | same, `--model …9B…` | L0 | 9B leg + prior-strength ablation | same |
| `banking77_full77.jev.json` | `decide/bench/banking77_full77_jev.py` | L0 readout, Jev side reference | live Jev leg on the same 500 items | "Full-77-way LIVE Jev leg" |
| `banking77_greedy.4b.json`, `banking77_greedy.9b.json` | `decide/bench/banking77_greedy.py --readout results/banking77[.9b].json` | greedy (generation) | typed-answer greedy legs (10-way) | "BANKING77" |
| `typed_decisions.json` | `decide/bench/bench_typed_decisions.py` (4B) | L0 | zero-training head-to-head vs trained models | "typed-decisions" |
| `typed_decisions.9b.json` | same against a 9B endpoint | L0 | size does not move this benchmark | same |
| `typed_decisions.9b.ps1.json` | same against an endpoint run with `--prior-strength 1.0` | L0 | prior-strength replay | same |
| `l0_ablation.json` | `decide/bench/l0_ablation.py` | raw vs L0 vs K | raw vs L0 vs rotation-budget K-curve | "L0-vs-raw ablation" |
| `mass_removal.json` | `decide/bench/bench_mass_removal.py` | L2 | `--no-mass` before/after (2.05× + parity) | "Serving-layer measurements" |
| `state_share.json` | `decide/bench/bench_state_share.py` | L0 | state-level prefix sharing parity + speed (5.3×) | "Serving-layer measurements" |
| `fast_path.json` | `decide/bench/bench_fast_path.py` | L0 | sequential vs batched suffix passes (the default) vs batched + early stop: Jev agreement on 350 questions, BANKING77 gold (200 items), ms, passes, peak memory | "Speed/precision sweep" |
| `synth_gain.json` | `decide/adapt/fit_registry_head.py --synth <regenerated corpus>` | L2 head fit | gold-only vs gold+synth A/B (neutral at n=285) | docs/POLICY.md §3 |
| `shadow_log.v3.jsonl` | proxy runtime (v3 renderer live re-read); read by `decide/bench/shadow_stats.py` | L2 | shadow agreement / router curve / ECE | "Live re-read" |
| `observe_log.jsonl` | proxy runtime observe() leg; replayed by `decide/adapt/observe.py` | L2 | labeled traffic bank for the adaptation loop | docs/POLICY.md §3 |
