# CONCEPTS.md — what this is, how it differs, and the vocabulary

The README is the short version. This page holds the longer explanations: who this is for, how it relates to Jev, Unsloth, Cloudflare Clef and a plain classifier, the two ways to run it, the readout levels, and a glossary.

[Jev](https://typesafe.ai) is TypeSafe AI's hosted decision API, the reference this repo measures itself against. [AnyJev](https://github.com/nokia-applied-research/AnyJev) is Nokia Applied Research's open method and library for reading decisions out of an LLM's logits ([PyPI](https://pypi.org/project/anyjev/), [technical report](https://arxiv.org/abs/2610.00831)); this repo deploys it on Apple's MLX.

## Is this for you?

**What it is.** A decision layer for a local language model. Give it a
state and typed questions (pick one of these, yes or no, rate on a
scale), and it returns probabilities in a single pass; nothing is
generated as text. The model file is never modified, and nothing leaves
your machine.

**How is this different from [Jev](https://typesafe.ai)?** Same idea, local and free to run. We measure
how closely our answers track Jev's on a 300-question eval: 75.7% at 4B,
82.0% at 9B. If sending data to a hosted API is fine, use Jev.

**How is this different from [Unsloth's decision models](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
or [Cloudflare's Clef](https://blog.cloudflare.com/clef-decision-models/)?**
Those train a model on labelled rows. Unsloth's recipe is LoRA adapters plus a
gradient-trained head, and its 0.8B and 2B models read 73% and 78% on the
typed-decisions benchmark, where our zero-label readout reads 61.6%
(Cloudflare's Clef ships fine-tuned decision models the same way). You get zero
setup, arbitrary questions and full distributions; you give up
in-distribution accuracy on fixed label sets.

**How is this different from "just train a classifier"?** Nothing, for a
fixed label set with labels in hand; a trained classifier beats both
approaches there. This is for questions and options you cannot enumerate
ahead of time.

**Already using [AnyJev](https://github.com/nokia-applied-research/AnyJev)?** This repo is a deployment of it: the missing
MLX backend for Macs, one HTTP endpoint, the observe() loop, and the
measurements. If you do not need Mac or this packaging, start with the
library, with one caveat: AnyJev 0.3.0 (2026-10-06) removed the
label-trained levels (L1, L2, heads, `observe`, `level="auto"`). The L2
head and the observe() flywheel here need `anyjev==0.2.0`, which
`requirements.txt` pins; the current library covers the label-free L0
readout that every benchmark here uses.

**Do you need it?**
- Fixed labels and a training set: no, train a classifier; we lose there.
- Hosted is fine and setup should stay at zero: no, use Jev or a
  Clef-style model.
- Local, private, no labels, questions or options changing per request:
  yes.
- A coding agent whose next action you want shadow-measured: yes (the
  proxy).

**Where is the catch?** Accuracy on trained-model turf, confidence that
does not transfer across domains, and a small-model budget.

## What this is, in plain words

A language model normally *writes* an answer, one word at a time. This
project skips the writing: it shows the model the question and reads the
answer straight off the model's internal state, like reading the gauge on
a machine instead of waiting for a written report. One look returns the
answer AND how confident the model is about every option. The model itself
stays exactly as it downloaded.

**Why bother.** It is deterministic (same question → same answer, every
time), it returns full confidence distributions instead of a string, it
runs on a laptop with no data leaving the machine, and it improves from
its own traffic: while it shadows real work it quietly records "what I
predicted" next to "what actually happened", and those free outcomes
sharpen it (the observe() loop). Speed is a wash at worst against a
well-optimized generation stack, and it wins where agents actually work:
long contexts, or any answer longer than a word.

**Honest lineage: we did not invent the trick.**
- **[AnyJev](https://github.com/nokia-applied-research/AnyJev)** (Nokia Applied Research; [PyPI](https://pypi.org/project/anyjev/), [technical report](https://arxiv.org/abs/2610.00831)) built the reading method this repo
  uses: the toolbox of marginalization, priors, heads, and the adaptation
  engine. This repo deploys that method.
- **[NeoJev](https://huggingface.co/dowands/NeoJev-Qwen3.8-27B-L56-MLX)** proved a fixed-layer fast read can run on Apple Silicon (a
  static early-exit head on a 27B model). That one mechanism is theirs
  first; we are the small-memory, zero-training cousin.
- **What WE added:** the missing Apple-Silicon (MLX) backend for AnyJev's
  toolbox (theirs never shipped one); scoring more than 26 options with
  multi-token loglikelihood (the 77-way BANKING77 bench; the endpoint's
  letter readout stops at 26); the caching tricks that make long inputs
  fast; a local decision endpoint speaking the Jev API's dialect; and,
  above all, the *measurements*: a scorecard against the
  commercial reference (where we win: fidelity, privacy, cost,
  adaptability; where we lose: raw accuracy on trained-model turf). If
  this repo is useful to the community, that glue-and-measurement layer,
  not method novelty, is why.

**Charter: the layer is DOMAIN-GENERAL.** It must keep working when the
repo is Python, the tests are pytest, the task is sysadmin work, or the
agent harness is not opencode, and it must generalize off the substrate
it was born on.

## The two surfaces

One readout core, two faces. Almost every published number is Face 1.

1. **The decision layer** (`decide/serve/systemone.py`,
   `POST /v1/systemone`): state + typed questions in, answers + full
   probability distributions out, zero training, deterministic. Every
   benchmark (Jev-agreement, banking77, typed-decisions,
   decide-vs-generate) ran on this surface or an in-process replica of its
   math. It never adapts: re-running anything returns identical answers.
2. **The agent harness** (`decide/serve/proxy.py`, the project's origin):
   the same readout sitting *inside a real agent loop* (opencode → proxy →
   generation server), with the `next_move` L2 head (inspect/run/modify).
   Shadow-only: it routes nothing, it logs what the head would have done
   against what the model actually called. This is where the **observe()
   flywheel** lives: the proxy banks every real decision together with
   the tool that was *actually called next* (a free label, since the
   outcome is observable in agent traffic), and `decide/adapt/observe.py`
   refits the head from that bank on demand.

Scope note: **nothing improves "for
free" on benchmark or judgment questions**, because their truth is not
observable in traffic, re-running is bit-identical, and fitting heads on
benchmark labels is the docs/POLICY.md §3 bright line. The flywheel turns
only where reality grades you for free.

**What the flywheel is, and is not.** The observe() loop fits a **readout
adapter**: a linear map (e.g. 2560×3 ≈ 7.7k parameters) plus a
temperature, solved in closed form on CPU from extracted hidden states.
It is *not* fine-tuning and *not* a LoRA, on two structural grounds: **no
weight of the base is ever updated**, and **nothing of ours ever enters
the forward pass**. The head reads a hidden state the frozen model
already produces. Adapter spectrum, for scale: prompt < linear head (ours)
< LoRA < full fine-tune. Consequences: expressiveness is capped (a readout
interprets what the base already represents; it cannot inject capability;
measured bound in docs/RESULTS.md, typed-decisions), one frozen base
serves many hot-swappable heads, and every other behavior of the base
stays bit-identical. The concession: a deployment owning a rich
labeled history of its exact decision surface should graduate to real
fine-tuning (the crossover verdict, docs/PRIOR_ART.md); the flywheel is
the tool for everything short of that point.

## The levels (readout ladder)

In plain words, before the precise version: every level is a way of getting
the answer's probability out of the model without generating text and
without changing its weights. **raw** asks once and reads the letter
probabilities — but the model favors whichever option is listed first.
**L0** is "shuffle and average": list the options in several orders, ask
once per order, average; the position bias cancels and what survives is
genuine preference — zero labels, works for any question out of the box.
**L1** is L0 plus a thermostat: one fitted number that makes "80% sure"
mean 80% right (confidence only, never which answer wins). **L2** is a tiny
gauge fitted beside the model for one repeated question: it peeks at an
early internal layer, faster than a full pass and sharper on its home turf
— the coding `inspect/run/modify` head is the only one shipped, an
experiment for a coding-agent harness (the shadow proxy), not a general
capability. The
endpoint picks automatically: the coding head's exact options get L2,
everything else gets L0; a model without a head (the 9B) serves L0 only.

Why the coding head exists: speed, and precision on its one question.
Speed is measured: the forward stops at block 16 of 32 and no rotations
run, so it answers in about 0.2 s, while L0 pays one full prefill per
option order. Precision is proven only on the head's own holdout (0.826)
and by the observe refit beating the seed head in-domain (64/73 vs 40/73);
L0 vs L2 on the same held-out task has never been run (docs/TODO.md,
measurement item 2).

All levels read the BASE model's internals in prefills; no weights are
changed at any level. `level="auto"` = serve the best available per
question: L2 where a head routes, else L1, else L0 (the proxy/endpoint
degradation chain).

Positioning: **no fine-tuning, no training run, no
dataset to build, no reference API to depend on**. The frozen model works
the moment it downloads; it banks its own calibration data as it is used,
and one command turns that bank into a sharper head (L1/L2 are post-hoc
calibration, sidecar artifacts, never weight updates).

**"Zero training," precisely.** No gradient step runs anywhere in this
repo. L0 uses no labels and fits nothing: it reads the frozen model's own
token distribution, averages over option rotations, and divides out a
prior estimated without labels. L1/L2 add a tiny sidecar map (a few
thousand parameters) fitted in closed form from labels (a few hundred
suffice; it is a least-squares-style solve on extracted hidden states,
not gradient training); no base weight changes and nothing of ours enters
the forward pass. Both routes run the same kind of engine, a base-LLM
prefill with no text decoded. What differs is what that forward pass
contains. In the train-in route, a Clef-style decision model, it contains
trained parameters: LoRA adapters plus a freshly initialized decoder
head, both gradient-trained and both saved with the served model. Unsloth's
guide shows what that buys on typed-decisions: Qwen3.5-0.8B reads 36%
before training and 73% after, the 2B 33% and 78% (it publishes no 4B
figure). Here the forward pass is the stock model, unchanged; the readout
sits outside it, and the zero-label L0 leg already reads 61.6% on
typed-decisions with no head and no labels at all.

| level | what it is | cost / decision | needs | used here for |
|---|---|---|---|---|
| raw | next-token log-probs of the answer labels, one prompt, position bias uncorrected | 1 prefill | nothing | baseline only |
| **L0** | permutation marginalization (rotate the option listing, average) + prior correction, a label-free calibrated-ish readout | K prefills (K = #options); budgetable to m≈2 at ~1 point cost, 3.8× faster | **zero labels** | **every headline benchmark and eval** (Jev fidelity, banking77, typed-decisions, decide-vs-generate, all ablations); the endpoint's arbitrary Choice/Score/Noul questions |
| L1 | L0 + a fitted temperature (calibration artifact over the L0 probabilities) | same as L0 | 100 to 500 labels | not shipped yet; the targeted fix for the 0.93 knowledge-gap tail |
| **L2** | a tiny linear head (e.g. 2560×3) reading the last-position hidden state at an early block; forward stops at the head's block | ~0.68× of one forward | 100 to 300 labels + hidden states | the `next_move` coding head (`registry/…/next_move.json`), served with the answer-mass log + tau router gate |

Measured numbers per level: docs/RESULTS.md (L0 vs raw: +4.7 points
Jev-agreement at n=300, McNemar p=0.003; L2 holdout 0.826). Coverage by
level: **L0 carries the whole scorecard**
(all reference evals, accuracy benchmarks, and ablations); L2 is measured
by head holdout (0.826) and the shadow re-read; L1 is not shipped and has
never been measured. The ladder stops at L2 (no L3).

## Glossary

**State.** The situation you want a decision about: a string, or JSON that is rendered to text.

**Typed question.** One of `Choice` (pick one of 2 to 26 options), `Noul` (yes or no) or `Score` (an ordered scale of 2 to 10 levels). Details: API.md.

**Readout.** Reading probabilities for the answer options straight off the model's logits at the answer position, instead of letting the model generate text. `raw`, L0, L1 and L2 are the levels of it (see "The levels").

**Label-free, zero-training.** No labels are used and nothing is fitted; no gradient step runs anywhere in the repo. L0 is label-free. L2 heads are fitted in closed form from labels (still no gradient steps).

**Head.** A tiny linear map from a hidden state to the options, fitted in closed form (`head:lda`). The shipped one is the `next_move` coding head in `registry/` (see registry/README.md).

**Face 1 and Face 2.** The two ways this runs. Face 1 is the decision endpoint (`systemone.py`), which every benchmark uses. Face 2 is the shadow proxy for a coding agent (`proxy.py`).

**Shadow proxy, shadow-only.** The proxy sits in front of a generation server and passes every request through unchanged. It also asks the head what it would do next and logs that against what the model actually did. It routes nothing.

**Flywheel, `observe()`.** The loop that turns shadow traffic into labels: the proxy pairs each rendered state with the tool the model actually called next, and `decide/adapt/observe.py` refits the head from that bank (first solve at 30 labels, refit at 60, 120, ...). It only works where outcomes are observable for free, which benchmarks are not.

**Tau (τ).** The confidence threshold (0.8) above which the head's answer *would* be acted on (`would_route` in the shadow log). The value was set on the coding domain where the head was fitted; re-read it per surface before any routing use (LIMITATIONS.md).

**Answer mass.** The total probability the model puts on the answer labels at the answer position. Low mass would mean "not a readable decision". Measured non-separating on this model, so it is logged only (`MASS_TAU=0.0`).

**Bright line.** The rule in POLICY.md section 3: calibration artifacts are fitted only from the serving surface's own observed traffic, never from public benchmark training splits and never against another model's answers. Benchmarks are quoted at zero-training L0.

**Fidelity vs accuracy.** Agreement with Jev measures fidelity to a reference. Accuracy is agreement with gold labels (BANKING77, typed-decisions). The docs keep the two words apart.

**Fork, "at fork".** This repository was forked from a private single-domain prototype on 2026-10-07 (POLICY.md, "Provenance"). "Removed at fork" means removed in that split. It does not mean a GitHub fork.

**Gold seed.** The recorded coding sessions (2,306 decision points) the shipped head was fitted on. Withheld from the repo, so the head cannot be reproduced from it.

**Frozen evidence.** `results/`: the measurement artifacts behind every quoted number. Scripts write to `runs/` and never replace these without `--force`.

**Rotations, marginalization.** For a Choice with K options, the L0 readout re-asks the question with the option list rotated and averages the answers, which cancels position bias. `--max-permutations 2` caps it at two rotations.

**State-level prefix sharing.** The state is read once per request and every question (and rotation) continues from that cached read.
