# PRIOR_ART.md — prior art, credits, and the train-in vs read-out axis

The mechanisms are established techniques; what's ours is the composition
— one typed-question endpoint + a fixed early-exit head + the observe()
refit loop, all on one local 4-bit artifact — not any single part.
(Measured consequences live in docs/RESULTS.md.)

- **Label-logprob readout (L0)**: verbalizers / pattern-verbalizer pairs /
  cloze-style prompting — PET (Schick & Schütze, EACL 2021,
  arXiv:2001.07676) and "Small Language Models Are Also Few-Shot Learners"
  (NAACL 2021, arXiv:2009.07118, the small-model precedent); automatic
  label-word search — LM-BFF (Gao et al., ACL 2021, arXiv:2012.15723);
  abstractions codified in OpenPrompt (Ding et al., ACL 2022 demo
  paper, github.com/thunlp/OpenPrompt); loglikelihood scoring is the
  lm-evaluation-harness default for decoder-only models
  (github.com/EleutherAI/lm-evaluation-harness).
- **L0 debiasing**: prior correction — contextual calibration (Zhao et
  al., ICML 2021, arXiv:2102.09690); answer-option-order sensitivity —
  Pezeshkpour & Hruschka (2023, arXiv:2308.11483); few-shot
  demonstration-order sensitivity — Lu et al. (ACL 2022,
  arXiv:2104.08786); the cheap permutation+prior combination — PriDe
  (Zheng et al., ICLR 2024 spotlight, arXiv:2309.03882); the unified view —
  Batch Calibration (Zhou et al., ICLR 2024, arXiv:2309.17249). Permutation
  marginalization + prior correction is that lineage; PriDe is the cheaper
  cousin of our K-prefill scheme. A trained-in cousin exists: PACT
  (Lin, 2026, arXiv:2609.35865) adds a permutation-consistency training
  term to single-token typed decisions (answer flips under relabelling
  13.8% → 9.8%) where we marginalize label-free; closest published
  setting to ours. Measured here on the 105-question eval
  (`decide/bench/l0_ablation.py`): marginalization is worth **+4.7 points**
  over the naive single-prompt read (the letter-logit read Mini-Jev ships) —
  5 answers flip to the reference, 0 away; the K-curve shows m=2 of K
  rotations already captures ~90% of the gain; and the Zhao-style
  content-free prior HURT on this eval's single-shot layouts (−6.6 vs
  marginalization alone), which is why the served default keeps it off
  until a layout repeats (min 8 states).
- **Linear head on hidden states (L2)**: the probing literature (Hewitt &
  Liang 2019, arXiv:1909.03368; Belinkov, *Computational Linguistics*
  2022, arXiv:2102.12452) criticizes probes as *evidence about what a
  model understands* — a criticism of claims, not of architecture. A
  production head is judged by held-out task accuracy on the served
  artifact (ours: 0.826); we make no interpretability claims. Linear =
  high selectivity is the defensible capacity choice.
- **Early exit**: CALM (Schuster et al., NeurIPS 2022, arXiv:2207.07061)
  exits per token by confidence; our fixed half-depth exit for one head is
  the layer-pruning family, not adaptive exit. MLX precedents
  (re-surveyed 2026-10-08): exactly ONE static case — NeoJev
  (`dowands/NeoJev-Qwen3.8-27B-L56-MLX`, Apache-2.0) keeps 56/64 layers
  of a 4-bit Qwen3.8-27B with a self-distilled label-free readout head
  (0.727 mean accuracy, 310 ms p50 on an M5 Pro; a plain logit lens at
  layer 56 reads only 0.663 — same head-over-raw gap we measure for L2);
  no dynamic/CALM-style exit exists in MLX, and mlx-lm exposes no
  intermediate-layer hooks (issue #1053) — a hand-rolled layer loop like
  ours remains the only route, hence the upstream-the-MLX-backend
  backlog item.
- **Routing ≠ this**: RouteLLM (LMSYS, arXiv:2406.18665,
  github.com/lm-sys/RouteLLM) and semantic-router (Aurelio,
  github.com/aurelio-labs/semantic-router) put a *separate* model
  before/instead of the LLM and return a routing choice; here the served
  model's own distribution is the classifier. The
  decisions-without-generation ecosystem this interoperates with: the Jev
  API (TypeSafe AI; publicly described as trained via its RLCD method,
  weight-level details undisclosed), Cloudflare Clef
  (blog.cloudflare.com/clef-decision-models), Unsloth's train-your-own
  decision models (unsloth.ai/docs/basics/
  train-your-own-decision-model-with-unsloth), the OpenRouter/OpenAI
  decisions endpoints, and the local-MLX cluster (NeoJev's layer-56
  readout, mlx-decision serving Clef-class weights on Apple Silicon,
  laya-mlx's tiny full-depth decision models).
- **Train-in vs read-out — the axis this project sits on** (assessed
  2026-10-08): Unsloth's "Train your own Decision Model" is the
  fine-tune-to-decide route — LoRA plus a Clef-design head trained on the
  public `LocalLLaMA/typed-decisions` mix, and it is their mechanism at
  every model size (the 0.8B/2B cost table is the marketing; their code
  example is the 4B). Source-level (verified 2026-10-10): under LoRA the
  backbone is frozen and the trainables are the adapters plus a freshly
  initialized Clef-style decision head; the loss is label-smoothed
  cross-entropy plus a Brier calibration term and an ordinal term for
  score questions; the save step writes adapters, head and calibration,
  so the served model is not the frozen base. Fine-tuning *trains the
  decision interface and its calibration*, on top of the base's existing
  semantics: the untrained head decides at chance (BANKING77 7% → 74% on
  Qwen3.5-0.8B), and a tiny base suffices (0.8B ≈ 2B ≈ 78%), so base
  competence is not the binding substrate there. This repo *reads out*
  capability the frozen base already
  has — zero gradient steps — so base competence is the binding
  constraint, and model SIZE, not training, sits on our quality axis.
  Working hypothesis, MEASURED 2026-10-08 (see docs/RESULTS.md,
  typed-decisions): the train-in vs read-out crossover is
  **regime-dependent, not size-only**. On typed-decisions (fixed layouts,
  in-distribution gold) our zero-training readout reads 61.6% (4B) / 60.4%
  (9B) vs their trained 73–78% — fine-tuning wins at every size we can
  afford, and base size does not move the number. On zero-shot tasks
  (BANKING77, Jev-agreement) base capacity is the lever: the same 9B
  gained +3.75 points and halved the Jev gap — and on the full-77-way
  unlock the 9B readout (+3.0 over the 4B) BEATS greedy generation
  (+5.6, p=0.0034) while the 4B does not (p=0.068): capacity flips the
  readout-vs-generation accuracy comparison on the same task. Open
  size-axis point: the 1.5B floor test, where read-out should degrade
  toward the untrained line (their before-training reads sit at chance).
  Caveat that bounds every such comparison: banking77 is IN their
  training mix (test decontaminated but in-distribution), and their
  74%/58% (0.8B/2B) are 77-way accuracy on their 500-row cut — our
  matching 77-way run reads 63.4% (4B) / 66.4% (9B) zero-training on our
own 500-row selection (between their two trained lines; the row
selections differ — exact protocol in docs/RESULTS.md). As of
2026-10-10 no third party publishes a label-free logits readout on
typed-decisions: the dataset card's zero-shot table lists zero-shot
general models, and its only label-free reference points are the
input-ignoring prior (0.470) and uniform (0.308).

## Ecosystem watch (re-verify before publish)

Cloudflare Clef/Clef-flash (publish-sooner pressure; blog 2026-10-01:
Clef = Qwen3.8-27B, Clef-flash = Qwen3.5-9B, Apache-2.0 open weights,
Jev-API compatible, 64k context, median latency 209ms / flash 38.8ms vs
Jev 524ms; leads the "Jev Decision Index" public-benchmark subset — on the
index's blended full score Jev still leads, 60.11 vs 53.08), Unsloth
"Train your own Decision Model" (fourth player; trains-in capability so
small bases suffice), OpenDecider, Mini-Jev (github.com/deep-diver/mini-jev,
letter-logit read), and the local-MLX decision cluster (NeoJev 27B 4-bit
~15GB, mlx-decision on PyPI, laya-mlx 13ms, plus the awesomejev MLX
entries). Differentiators that survive all of them: the SMALL-memory end
(4B ≈ 3GB), the observe() adaptation loop (nobody ships fit-on-your-traffic
heads), zero-training deployment, and the measured fidelity/latency story.
"Only local readout" claims are outdated — do not use them.
