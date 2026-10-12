# Third-party notices

This project is Apache-2.0 (see LICENSE). It redistributes no third-party
source code. Dependencies and model-lineage licenses:

## anyjev == 0.2.0 (PyPI dependency, unmodified)

- License: Apache-2.0 (Nokia Applied Research; verified 2026-10-08 on PyPI
  and github.com/nokia-applied-research/AnyJev, which also publishes a
  technical report, arXiv:2610.00831). Imported at runtime; no
  files from the package are copied or vendored into this repository.
- Used for: the readout machinery (Decider L0 marginalization/priors,
  Question types, LinearHead, prompt builders) and the adaptation engine
  (`observe()` / `fit_head()` / `level="auto"`). Pinned in
  `requirements.txt` because 0.3.0 removed the L1/L2 head machinery this
  repo is built on.
- This repo's `decide/core/mlx_backend.py` is an original implementation of
  anyjev's documented Backend protocol for Apple MLX (upstream
  contribution of it is on the roadmap).

## mlx-lm (PyPI dependency)

- License: MIT (mlx-community). Used to load and serve the quantized model
  and for generation in the synthetic-state pipeline. Pinned to 0.32.0 in
  `requirements.txt`: `decide/core/mlx_backend.py` uses non-public mlx-lm
  internals, so a newer release can break it.
- The transitive `mlx` package itself (Apple's MLX framework; MIT) is also a
  used dependency: measured with 0.32.3, per `requirements.txt`.

## Model lineage (covers shipped artifacts)

- Base model: `mlx-community/Qwen3.5-4B-MLX-4bit` (auto-downloaded to the
  local HF cache on first run; not redistributed in this repo). The base
  model's license governs the weights; check the model card on Hugging
  Face for its terms (the Qwen base-model family ships Apache-2.0; the
  4B and 9B MLX conversions' cards both declare apache-2.0, verified
  2026-10-08).
- Derived artifacts shipped here inherit the base model's license terms
  for model-generated content: the registry heads
  (`registry/…/next_move*.json`, small linear probes fit on hidden
  states), `data/eval/questions.jsonl` (LLM-authored eval set), and the
  observe telemetry logs.

## Evaluation datasets (cached in-repo; licenses permit redistribution)

- `mteb/banking77`: test split cached at `data/eval/banking77/test.jsonl`
  (3,080 rows) alongside the dataset card: MIT (verified 2026-10-08).
- `LocalLLaMA/typed-decisions`: cached at
  `data/eval/typed_decisions/test.jsonl`: Apache-2.0 (verified
  2026-10-08). Gold labels are teacher-derived (mean of three
  temperature-0.7 teacher passes), not human-annotator majorities.

## Jev API answers (TypeSafe AI): stored as evaluation references

- `results/` keeps the Jev API's answers (for typed questions also its
  confidences) on this repo's own eval items, fetched through OpenRouter
  (`~typesafe/jev-latest`): `jev_compare{,.n300,.long}.json` (350 typed
  questions), `jev_repeat{,.long}.json`, `banking77.json` and
  `banking77.9b.json` (400 items), `banking77_full77.jev.json` (500 items),
  `bench_vs_generate*.json`, `labels_long.scored.json`. They exist only so
  the agreement and accuracy numbers recompute without an API key.
- Terms read 2026-10-10: TypeSafe's Master Customer Agreement (rev. 2026-09-23,
  typesafe.ai/legal/mca), Terms of Use (2026-09-19, the document OpenRouter
  lists for Jev), Acceptable Use Policy (2026-09-23), and OpenRouter's Terms
  (2026-08-31, which defer to each model's own terms). TypeSafe disclaims
  ownership of Output and assigns its rights in it to the customer (MCA
  §4.2); none of the documents restricts benchmarking, comparing, or
  publishing results. The restriction that applies is MCA §2.3(b): do not use
  the Services or any Output to perform model distillation, train a model to
  imitate the Services, or develop a similar or competing product.
- Use here: evaluation references only (agreement and accuracy scoring). No
  head or other trained artifact in this repo is fitted or distilled on these
  answers (the calibration rule, docs/POLICY.md §3: heads are fitted
  only from the serving surface's own observe() telemetry), and they are not
  offered as training data. This is a reading of public terms, not legal
  advice.
- Independent project; not affiliated with TypeSafe AI.

## Trademarks and affiliation

Independent project, not affiliated with, endorsed by, or sponsored by
TypeSafe AI (Jev), Alibaba (Qwen), OpenRouter, AnyJev, Cloudflare (Clef), or
Unsloth. "Jev" and "Qwen" are used strictly nominatively: to identify the
commercial decision API whose request style this endpoint follows, and the
base model family whose internals are read. Cloudflare's published
Clef/Clef-flash figures and Unsloth's published training numbers are cited
for comparison only; no code, data, or model output from either is used or
redistributed. The comparison results against the Jev API were obtained
through OpenRouter endpoints; what the terms of each allow is recorded in the
Jev section above.

## Tooling

- Resource profiling in `decide/bench/eval_vs_jev.py` uses only stock macOS
  tools (`ps`, `ioreg`), no third-party monitoring code.
