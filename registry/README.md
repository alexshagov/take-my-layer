# registry/ (fitted head artifacts)

One directory per model, one subdirectory per adapter state: `registry/<model id with / as __>/base/` is the untouched base model. A head reads the hidden state of exactly that quantized artifact ("fit on what you serve"), so it must not be used with another quant, a truncated build or an SFT-tuned adapter.

All four files answer the same question, `next_move` ("What should the agent do next?", options `inspect`, `run`, `modify`). The head was built as an experiment for a coding-agent harness: the one place the fitted L2 level is demonstrated on real agent traffic (the shadow proxy); it is not a general capability of the endpoint. Each file is a linear map (method `head:lda`, solved in closed form, no gradient steps) from the last-position hidden state at block 16 of 32 (hidden size 2560) to the three options, plus a fitted temperature.

| file | what it is | served |
|---|---|---|
| `next_move.json` | The head both entry points load by default. Renderer v3, gold-only variant (`A_gold`): fitted on 1,824 states from the gold seed, tau 0.8. Holdout accuracy 0.826 (split by task), out-of-fold 0.837. The variant that added 285 synthetic states scored 0.824, neutral (`results/synth_gain.json`). | yes |
| `next_move_v3.json` | The name `decide/adapt/fit_registry_head.py` writes. Byte-identical to `next_move.json` (SHA-256 `e9268efc…`): the serving file is promoted by copying. | no (same bytes) |
| `next_move_v2_seed.json` | The earlier head, fitted on renderer-v2 states (450 states, 2026-10-06, out-of-fold 0.816). Kept as provenance of the v2 to v3 refit. | no |
| `next_move_observed.json` | Output of `decide/adapt/observe.py` replaying the shipped `results/observe_log.jsonl` (fitted on 60 of its 73 rows, out-of-fold 0.683, renderer v2). It shows the loop working; it is weaker than `next_move.json`, and every run of `observe.py` overwrites it. | no |

Serve another head with `--head <path>` on `systemone.py` or `proxy.py`. Promote a refit deliberately: copy it over `next_move.json` after re-reading shadow agreement (`decide/bench/shadow_stats.py`). No script writes `next_move.json` by default, and `fit_registry_head.py` refuses to replace an existing registry file without `--force`.

The gold seed behind `next_move.json` (recorded work sessions from another project) is not part of this repo, so the shipped head cannot be reproduced from the repo alone. New heads come from the synthetic-state pipeline and the observe loop (`decide/adapt/synth_states.py`, `observe.py`; docs/POLICY.md §3).
