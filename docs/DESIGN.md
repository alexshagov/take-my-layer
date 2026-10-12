# DESIGN.md: where the design comes from

This repository was split out of an unpublished coding-agent prototype on 2026-10-07
(docs/POLICY.md, "Provenance"). That prototype's investigation produced the design and
several of the decisions in docs/DECISIONS.md. This page keeps what those decisions rest
on. The numbers below were measured on the prototype's data (recorded coding-agent
sessions), which is withheld, so they cannot be reproduced from this repository. The
benchmarks that can be reproduced are in docs/RESULTS.md.

## The idea

An agent step used to cost about 8 s: decode a 100 to 300 token tool call at 25 to 30
tokens/s, then parse and validate it. For decisions that are really choices (which tool,
is the task done, which file next), generation can be replaced by a readout: one prefill,
then read the next-token logits or a hidden state at an intermediate block, apply a small
fitted head, and return a probability distribution. Nothing is generated and nothing is
parsed. The model stays exactly as downloaded.

## The method and the backend

The method is AnyJev's (`anyjev==0.2.0`, Nokia Applied Research, Apache-2.0). Its levels
are explained in docs/CONCEPTS.md. In short: raw reads one prefill; L0 averages over
rotations of the option listing and divides out a label prior, with no labels; L1 adds a
temperature fitted on labels; L2 is a linear head on a hidden state at an intermediate
block, and the forward pass stops at that block.

The backend contract is small. `next_token_logprobs(prompts, token_ids)` is required (the
full-vocabulary log-softmax at the last position, gathered at the requested ids) and
unlocks raw, L0 and L1. Optional, probed with `hasattr`: `score_shared` (prefix-KV reuse),
`hidden_states` and `hidden_states_to(..., max_layer)` (L2), and the attributes
`n_layers` and `hidden_size`. AnyJev ships no MLX backend, so `decide/core/mlx_backend.py`
is an original implementation of that contract.

Facts learned while building it:

- `qwen3_5` nests its config under `args.text_config`, and `mx` has no `log_softmax`
  primitive. The full vocabulary is 248,320 tokens.
- Qwen3.5-4B has 32 blocks, hidden size 2560, and full attention only at blocks 3, 7, ...,
  31; the rest are linear-attention (GDN) blocks. Layer numbers follow the Hugging Face
  convention (0 = embeddings, i = block i, n = after the final norm).
- The manual block loop matches the stock forward: max |logit difference| about 0.037
  (quantization-noise scale), argmax identical.
- AnyJev's shipped heads cover dense Qwen3 checkpoints only, so heads are fitted here
  (`Decider.fit_head` needs at least `max(8, 2K)` labels; 100 to 300 in practice).
  A head is keyed to the exact model id and `load_artifacts` refuses a mismatch.
- Fit on what you serve. Hidden states shift with quantization, with truncation and with
  fine-tuned weights, so a head is fitted on the artifact that will be deployed, never on
  another quant or on the base model when an adapter is served.

## Feasibility on the prototype's data

Setup: 2,306 decision points from 595 recorded sessions (next tool: read 855, bash 710,
grep 519, edit 194, glob 27, write 1). A state is the task text plus a compact action
history (paths, commands, exit codes, output tails, edit diffs). That is a lower bound on
what the actor saw: full outputs and the assistant's own text were left out. Evaluation:
120 states, 40 each from the early, middle and late third of a session, one seed. L2 heads
were fitted on 450 disjoint points. Truth is the actor's actual next tool.

| | raw | L0 | L2, 6 tools | L2, 3 classes |
|---|---|---|---|---|
| exact, own classes | 0.57 | 0.47 | 0.81 | 0.89 |
| merged to 3 classes | 0.61 | 0.77 | 0.89 | 0.89 |
| early / middle / late third | .78 / .53 / .53 | .72 / .50 / .95 | .95 / .72 / 1.00 | .95 / .72 / 1.00 |
| confidence when right / wrong | .58 / .54 | .40 / .36 | .89 / .68 | .88 / .64 |
| seconds per decision | 0.30 | 1.67 | 0.21 | 0.25 |
| exit depth | full | full | block 16 of 32 | block 19 of 32 |

What it showed:

- Raw is a degenerate "bash magnet": it predicts bash for 74 of 120 states, so its 0.57 is
  close to bash's base rate. L0's debiasing removes that.
- L0's confidence does not separate right from wrong answers (0.40 against 0.36), so it
  cannot be thresholded. An L2 head with an out-of-fold temperature can (0.88 against
  0.64).
- On a 3-class head, confidence of 0.8 or more handled 80% of decisions at 3.1% error and
  0.9 or more handled 61% at 0% error (n = 120, so the denominators at high thresholds are
  small).
- The signal sits in the GDN trunk. Layer selection chose blocks 16 and 19, before the
  last full-attention blocks, which halves the compute.
- Prefill on the M-series machine measured 1,015 to 1,059 tokens/s on short prompts (an
  earlier assumption from external numbers was 500 to 1,000), about 165 ms per prompt.

Caveats: n = 120, one seed, a base model rather than the prototype's fine-tuned adapter,
and the actor's choice is the truth, so interchangeable options count against the model.

## Coarse questions

Fine tool identity is label noise. 31 of L0's 65 errors were read against grep, which the
actor used interchangeably for the same state. At L2 the 6-tool head merged to 3 classes
(0.89) equals the 3-class head scored exactly (0.89). The policy that follows is to ask
coarse questions (inspect, run, modify) and to treat a finer split as its own second-stage
question rather than a bigger head. docs/POLICY.md section 2 states the rule.

## Router replay

Before any proxy existed, all 2,306 points were run through the full router rule (L2
3-class head plus argument extractors) with five folds grouped by task, so no task leaks
across folds, and the head refitted per fold.

| tau | decisions routed | error among routed |
|---|---|---|
| 0.60 | 51% | 7.5% |
| 0.70 | 47% | 6.0% |
| 0.80 | 42% | 4.3% |
| 0.90 | 31% | 1.7% |

The modeled speedup was 1.56x to 1.65x at tau 0.8. The gate was at least 2x at no more
than 5% added wrong actions, and it was not met. The ceiling under a proxy that can only
short-circuit one call is about 1.6x at safe thresholds, and the gap was mostly in
argument construction (reads whose path is not visible, non-verify bash). The decision
was to go on to shadow mode with tau = 0.8, which is the `TAU` in
`decide/adapt/fit_registry_head.py`, and to let a live measurement decide whether routing
ships.

## Shadow proxy and the supervised worker

`decide/serve/proxy.py` passes requests through unchanged and runs the decision on its own
model instance next to the upstream generation, logging what it would have routed beside
what the model actually did. It routes nothing.

- Live merged agreement was 7 of 9 (0.78) against 0.89 offline, so the renderer bridge held.
  Confidence ran lower live (mean about 0.6 against 0.88 for right answers offline), so at
  tau 0.8 only 1 or 2 of 10 turns would have routed. The proxy stayed shadow-only:
  routing was later measured unsafe on foreign traffic (routed error 37% at 59% coverage;
  docs/RESULTS.md, "Live re-read").
- The first routing batch failed for infrastructure reasons: two MLX instances exhausted
  GPU memory, and one wedged decision held the head lock and silenced every later request
  while the health check still answered. The fix is the design that remains: the decision
  runs in a supervised worker process with a 15 s hard timeout, and a timeout or a dead
  worker degrades to pass-through, so the request path cannot wedge. The head was also
  refitted at cheaper depths (`LAYERS = [16, 19, 22]`), which brought live decisions to
  161 to 376 ms.
- A fabricated tool call parses and executes in opencode, the client used (a stub server
  that always returned the same read ran 1,797 calls before its timeout), so the response
  shape was not the problem.
- The batch was interrupted at 21 of 56 episodes by a machine out-of-memory reboot. One
  twin pair flipped from green to red: the router fabricated the same verify command after
  every edit (20 times, confidence 0.884), which removed the model's reflective turns and
  tightened a failing loop. Every individual fabricated call was a sensible move. A
  repetition guard (never fabricate the same tool and arguments three times in a session)
  was decided and not applied. The routing, fabrication and episode-runner code did not
  come over to this repository.

## Renderer and head registry

- Calibration states and live traffic go through one rendering pipeline
  (`decide/core/state_feed.py`). Recorded traces are rebuilt into chat-message arrays and
  rendered by the same code as live requests, so an offline-to-live shift in the text
  cannot hide in two code paths. The only difference found was bash exit codes, which are
  invisible in live traffic. Dropping that information moved the selected layer from 16 or
  19 to 32 of 32 until the head was refitted at cheap depths.
- Heads live in `registry/<model id>/<base or adapter hash>/<question>.json`. An artifact
  records its base model, layer, label vocabulary and holdout, and AnyJev's
  `load_artifacts` refuses one whose model id does not match (registry/README.md).
- A refit re-extracts hidden states over the calibration states and solves in closed form:
  minutes, not hours (`decide/adapt/fit_registry_head.py`).

## Risks that were open at the start

- Agreement with the actor is not task correctness. A routed call must also be right for
  the episode, which is why shadow mode comes before routing.
- Probe readability depends on the question. Choice-type reads are the proven surface; do
  not promise that every question can be read from a hidden state.
- State length eats the win, so prefix reuse is not optional. State-level prefix sharing
  and batched suffix passes are the answer that shipped (docs/RESULTS.md, "Speed/precision
  sweep").
- AnyJev 0.2.0 was pre-alpha and its backend contract could shift; the pin in
  requirements.txt is what guards against that.

## Not pursued at the start

Attention-head readouts for "where to look" questions, heads for fine tool identity (label
noise, above) and constrained generation of arguments stay out of scope. An upstream
AnyJev contribution is in docs/TODO.md.
