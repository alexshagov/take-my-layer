# take-my-layer

*Shut up and take my layer.*

A decision layer for a local language model on Apple Silicon: zero training, plug-and-play. You send a situation and a few typed questions; you get the answer with the model's probability for every option, read straight off its logits. Nothing is generated, nothing leaves your machine.

It deploys the method of [AnyJev](https://github.com/nokia-applied-research/AnyJev) (Nokia Applied Research) on an original MLX backend, and measures itself against [Jev](https://typesafe.ai), TypeSafe AI's hosted decision API. Independent; not affiliated with either.

**Status: research prototype.**

- The Jev-agreement numbers use LLM-authored questions with no human labels. The BANKING77 and typed-decisions rows use public gold labels.
- The one fitted head we ship (for coding agents) cannot be reproduced from this repo, because its training seed is withheld.
- If you have a fixed label set and training data, train a classifier instead.

More in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Benchmarks

Local 4-bit Qwen3.5 4B (9B where noted), label-free, zero training, on an Apple M5 with 32 GB.

Each row is a frozen artifact in `results/` and reproduces with one command ([docs/RUNNING.md](docs/RUNNING.md)). One exception: the typed-decisions 4B run predates a code change and re-measures 0.3 points higher ([results/INDEX.md](results/INDEX.md)). Tables with confidence intervals are in [docs/RESULTS.md](docs/RESULTS.md).

| what was measured | result |
|---|---|
| Agreement with the real Jev API on 300 non-coding questions (fidelity to a reference, not accuracy) | 4B **75.7%**, 9B **82.0%**; Jev agrees with itself 98.7% of the time, which is the ceiling |
| Readout vs greedy decoding of the same model, same 300 questions | **+4.7** points (4B), **+6.3** (9B); McNemar p = 0.003. On long inputs greedy Choice answers collapse (46% vs 85% agreement with Jev) |
| [BANKING77](https://huggingface.co/datasets/mteb/banking77), 10 intents, gold labels | 4B 87.5%, 9B 91.25%, Jev 95.5%: a trained model wins on its home turf |
| BANKING77, all 77 intents | 9B 66.4% beats greedy 60.8% (p = 0.0034); 4B 63.4% sits below greedy 66.8% (p = 0.068); Jev 78.6% |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), scored with zero training | 61.6% (4B), 60.4% (9B), vs 73% and 78% for 0.8B and 2B models [trained on it](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) |
| Latency, long input, 5 questions in one request | 22.2 s down to **3.7 s** mean (5.3× mean per-request speedup; state read once); the coding head answers in about 0.2 s |
| Batched suffix passes (the default), 70 requests / 350 questions | short request 1036 → **552 ms**, long 3520 → 2989 ms, mean over all 70 requests 1422 → **942 ms (1.5×)**; agreement with Jev unchanged (76.0%) |

Opt-in `--early-stop` reads 19% fewer passes, at slightly more optimistic confidences ([RESULTS](docs/RESULTS.md#speedprecision-sweep-2026-10-11)).

The losses are expected: a trained model beats a zero-training readout whenever labels and a fixed label set exist. This is for questions and options you cannot list ahead of time, on a machine where the data cannot leave. Why the numbers look like this, and what "label-free" means: [docs/CONCEPTS.md](docs/CONCEPTS.md).

## What you get

A support ticket, three questions of three types, one request:

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Ticket #4821 from Dana (Pro plan, customer for 14 months): \"Since this morning every export to CSV fails with a 502. We run payroll from those exports at 3pm today. I retried from two browsers.\" Status page: all green. No other open tickets mention exports.",
 "questions": {
    "route": {"type": "Choice", "instructions": "Which team should take this first?", "options": ["billing", "platform-oncall", "account-management", "docs"]},
    "escalate_now": {"type": "Noul", "instructions": "Does this need the on-call engineer right now?"},
    "urgency": {"type": "Score", "instructions": "How urgent is this ticket?", "levels": ["low", "medium", "high", "critical"]}}}'
```

```text
-> {"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "answers": {
       "route": {"type": "Choice", "answer": "platform-oncall", "confidence": 0.8516, "probabilities": {"billing": 0.0744, "platform-oncall": 0.8516, "account-management": 0.0579, "docs": 0.0161}},
       "escalate_now": {"type": "Noul", "answer": true, "confidence": 0.637, "probabilities": {"Yes": 0.637, "No": 0.363}},
       "urgency": {"type": "Score", "answer": "critical", "confidence": 0.6651, "probabilities": {"low": 0.0051, "medium": 0.0156, "high": 0.3142, "critical": 0.6651}}},
    "usage": {"input_tokens": 526}}
```

Real output of the 4B model on an Apple M5 (probabilities shortened to four decimals; the server returns full precision). No labels, no fitting, no text generated.

A freshly started server returns the same bytes for the same request; later in a server's life the confidences can shift ([repeatability](docs/API.md#repeatability)). Question types, errors and limits: [docs/API.md](docs/API.md).

## Run it

Needs an Apple Silicon Mac on macOS 14 or newer and Python 3.11 or newer (install one or use Nix).

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # any Python >= 3.11 works
pip install -r requirements.txt
export HF_HOME=$PWD/.hf HF_HUB_CACHE=$PWD/.hf/hub   # keep the model (2.9 GB) in the project; export both (an exported HF_HUB_CACHE silently overrides HF_HOME)
python decide/serve/systemone.py --port 8095        # first start downloads the 4B model (2.9 GB)
python decide/doctor.py                            # optional preflight: versions, pins, cache, ports
```

Then run the `curl` above. With Nix, use `nix develop -c python decide/serve/systemone.py --port 8095` instead.

The server batches the passes of a request by default; `--early-stop` (fewer passes, slightly more optimistic confidences) and `--no-batch` are in [docs/API.md](docs/API.md#server-flags). The shadow proxy for a coding agent, the adaptation loop, every benchmark command and the repository map are in [docs/RUNNING.md](docs/RUNNING.md).

## Docs

**Start here**
- [docs/API.md](docs/API.md): the endpoint, request and response, errors, limits
- [docs/RUNNING.md](docs/RUNNING.md): install, proxy, adaptation loop, every benchmark command, repository map
- [docs/CONCEPTS.md](docs/CONCEPTS.md): what this is and is not, vs Jev, Unsloth, Clef and a plain classifier, the readout levels, glossary
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): intended use and measured limits

**Evidence**
- [docs/RESULTS.md](docs/RESULTS.md): every measured finding, with CIs
- [results/INDEX.md](results/INDEX.md): which script wrote each frozen artifact
- [registry/README.md](registry/README.md): the shipped head artifacts

**Decisions and background**
- [docs/DECISIONS.md](docs/DECISIONS.md): what was chosen and why
- [docs/PRIOR_ART.md](docs/PRIOR_ART.md): prior art, credits, train-in vs read-out
- [docs/POLICY.md](docs/POLICY.md): domain-general policy and the calibration rule
- [docs/TODO.md](docs/TODO.md): open work (measurement gaps, build backlog, decisions to make)
- [docs/DESIGN.md](docs/DESIGN.md): where the design comes from, with the measurements behind it

**Project**
- [CONTRIBUTING.md](CONTRIBUTING.md): the calibration rule, the most useful contribution, development setup
- [AGENTS.md](AGENTS.md): the short guide for coding agents
- [THIRD_PARTY.md](THIRD_PARTY.md): dependencies, licenses, model lineage, Jev answers, trademarks

## License

Apache-2.0 ([LICENSE](LICENSE)). Independent project, not affiliated with, endorsed by or sponsored by TypeSafe AI (Jev), Alibaba (Qwen), OpenRouter or AnyJev; see [THIRD_PARTY.md](THIRD_PARTY.md).
