# RUNNING.md — install, run, benchmark

The README has the short start. This page is the full guide: install, the decision endpoint, the shadow proxy for a coding agent, the adaptation loop, every benchmark command, and a map of the repository.

Requirements: an Apple Silicon Mac on macOS 14 or newer, and Python 3.11 or
newer (mlx-lm 0.32.0 needs it; the Nix shell uses 3.12 — install a newer
Python or use Nix). Memory, measured: the
4B model is a 2.9 GB download and about 2.8 GB resident, the 9B 5.6 GB and
5.3 GB; 32 GB is comfortable when a benchmark runs next to a server. The
4-bit model downloads on first start into the Hugging Face cache.

With Nix, `nix develop` builds `.venv`, installs `requirements.txt` and sets
`HF_HOME` to `.hf/`, so the model lands inside the project. Without Nix:

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # any Python >= 3.11
pip install -r requirements.txt   # mlx-lm==0.32.0, anyjev==0.2.0
export HF_HOME=$PWD/.hf HF_HUB_CACHE=$PWD/.hf/hub   # optional: keep the model cache in the project; export BOTH — an inherited HF_HUB_CACHE (direnv, CI) silently overrides HF_HOME
```

`python decide/doctor.py` is a stdlib preflight (no model load): Python
version, the two version pins, where the HF cache really points, whether the
models are cached, and whether the serving ports are already busy.

Prefix any command below with `nix develop -c` if you use Nix (the shell sets
`HF_HOME` and puts `.venv` on the path); otherwise run it inside the activated
venv. The scripts find the repo root themselves, so the working directory
does not matter.

## The decision endpoint

```bash
python decide/serve/systemone.py --port 8095   # ready when it prints "systemone on :8095 (head loaded: True; ...)"
curl -s localhost:8095/healthz                 # {"ok": true, ...} once the model is loaded
```

Then POST to `/v1/systemone` as in the README; copy-paste requests for every
question type, the errors and the limits are in `docs/API.md`. Serving flags, with the
measured tradeoffs in docs/RESULTS.md:

```bash
python decide/serve/systemone.py --early-stop           # fewer passes: one rotation first, the rest only where unsure
python decide/serve/systemone.py --no-batch             # one suffix pass at a time (the pre-2026-10-11 behaviour)
python decide/serve/systemone.py --max-permutations 2   # 10-option Choice: ~90% of the gain, 3.8x faster; fewer-option Choices lose accuracy
python decide/serve/systemone.py --model mlx-community/Qwen3.5-9B-MLX-4bit   # L0 only, no registry head
```

## The agent-harness face (shadow proxy)

Three processes: a generation server, the shadow proxy in front of it, and an
agent (opencode, installed separately; `requirements.txt` does not include it).
Run `opencode` from the repo root, because `OPENCODE_CONFIG` is a relative path:

```bash
python -m mlx_lm server --model mlx-community/Qwen3.5-4B-MLX-4bit --port 8080    # 1. generation server
python decide/serve/proxy.py --port 8090 --upstream http://127.0.0.1:8080        # 2. shadow proxy
curl -s localhost:8090/admin/health      # wait for "upstream_ok": true and "worker_ready": true
OPENCODE_CONFIG=decide/serve/opencode-shadow.json opencode run \
  --model mlx-shadow/mlx-community/Qwen3.5-4B-MLX-4bit \
  'In data/repos/qjev-sandbox, make Greeter.farewell() include the name: Greeter("Ada").farewell() should return "Goodbye, Ada!", and a blank name still returns "Goodbye!". Update the test, then run python3 -m unittest discover -s data/repos/qjev-sandbox -p "test_*.py"'
python decide/bench/shadow_stats.py logs/shadow_log.jsonl   # agreement, router curve and ECE of the head
```

The proxy routes nothing: the model's own answer always goes through. For every
model call it also asks the head what it would do next and logs that against the
tool the model actually called: `logs/shadow_log.jsonl` holds the prediction,
`logs/observe_log.jsonl` the rendered state plus that label (the replay input
for `observe.py`). It only shadows conversations that mention a
`data/repos/<name>/` path, which is how it finds the repo name, so the prompt
names the sandbox. Measured 2026-10-10 on the 4B model, twice: this task
finished in 50 and 61 s, each time with 7 shadow rows and 6 labelled observe
rows. `--no-mass` on the proxy skips the full-depth answer-mass read
(early-exit L2 decide, 2.05x).

Things to know about this flow:

- The agent edits the sandbox: the prompt changes `greeter/__init__.py` and
  `test_greeter.py` in `data/repos/qjev-sandbox`. Restore them afterwards
  (`git checkout -- data/repos/qjev-sandbox` once the repo is under git).
- Run `opencode` from a terminal. From a script or CI add `< /dev/null`: with
  stdin left open as a pipe, `opencode run` waits for input and never starts
  (opencode 1.18.33, controlled test: 3 of 3 attempts with an open pipe hung at
  the `init` log line, 3 of 3 with `< /dev/null` started in about 2 s).
- opencode itself opens HTTPS connections to external hosts when it starts (seen
  with `lsof`). "Nothing leaves your machine" describes this repo's endpoint and
  proxy; the model calls opencode makes here go to `127.0.0.1:8090`.
- The logs hold full tool outputs and the agent's shell commands, including
  absolute paths (a run of the prompt above recorded the agent's `cd
  /Users/…/data/repos/qjev-sandbox && python3 -m unittest …`). They are
  gitignored; scrub them before copying one into `results/` (the shipped
  `results/observe_log.jsonl` was scrubbed that way).

To exercise the proxy without a generation server, point it at the contract-test
stub instead: `python decide/research/stub_server.py 8091` answers every chat
completion with a fabricated `read` tool call (`STUB_VARIANT=v2` adds the
streaming preamble a real server sends), and `proxy.py --upstream
http://127.0.0.1:8091` shadows it. The proxy still loads its own 4B model for
the head.

## The adaptation loop (the flywheel)

What a fresh clone runs: label-free **L0** for every question — except a
Choice with exactly the options `inspect`, `run`, `modify`, which the
shipped L2 coding head answers (loaded by default; `--head` swaps it). The
loop below is how that head, not a new one, gets sharper on your traffic;
a head for a *different* question means writing your own fitting step on
the anyjev 0.2.0 machinery (`fit_head`, the backend's hidden states,
`decide/core/labels.py` as the pattern) — fitted only on your own serving
telemetry (docs/POLICY.md §3).

```bash
python decide/adapt/synth_states.py --per-domain 60          # synthetic corpus (300), not shipped
python decide/adapt/observe.py                               # replay the shipped observe log -> refit head
python decide/adapt/observe.py --log logs/observe_log.jsonl  # ... or your own live proxy traffic
python decide/adapt/fit_registry_head.py --synth data/synth/states.jsonl   # needs the withheld gold seed
```

`fit_registry_head.py` is the A/B fit behind the shipped head (gold only vs gold
plus synthetic states). Pass `--out runs/next_move_v3.json`: with the default
registry output path it stops earlier, at the frozen-artifact guard. It needs
the gold seed, which is not in the repo, so it stops with an explanation; it
stays as the record of how `next_move.json` was made (registry/README.md). Its
output goes to `registry/…/next_move_v3.json` when you pass that path
(replacing an existing file needs `--force`) and `runs/synth_gain.json`.

## Benchmarks

Each writes `runs/<name>.json`. Three kinds, by what they need running:

```bash
# HTTP clients of the endpoint on :8095: start it first (they check before they begin)
python decide/bench/bench_banking77.py [--no-jev]       # 10-way gold accuracy vs Jev (~8 min, 4B)
python decide/bench/bench_typed_decisions.py            # zero-training head-to-head (~10 min, 4B; ~16 min, 9B)
python decide/bench/eval_vs_jev.py [--no-jev]           # ours vs the real Jev API (local side <1 min)

# load their own model: stop the server first (one MLX instance at a time, 32 GB)
python decide/bench/bench_vs_generate.py                # readout vs greedy decoding (~1-8 min/leg; 9B long leg ~20 min)
python decide/bench/banking77_full77.py                 # full-77-way (~50 min on the 4B, ~80 on the 9B)
python decide/bench/banking77_greedy.py                 # greedy leg on the 400 BANKING77 items (~1.5 min 4B, ~2.5 min 9B)
python decide/bench/l0_ablation.py                      # raw vs L0 vs K-curve (~3 min)
python decide/bench/rotation_budget.py                  # rotation-budget curve (~20 min)
python decide/bench/bench_state_share.py                # state-share parity and speed (~5 min)
python decide/bench/bench_fast_path.py                  # sequential vs batched vs batched + early stop: Jev agreement, BANKING77 gold, ms, memory (~9 min, 4B)
python decide/bench/bench_mass_removal.py               # --no-mass before/after (<1 min)

# no model: CPU only, or the Jev API only
python decide/bench/pool_evals.py                       # pooled n=300 + McNemar (CPU, seconds)
python decide/bench/shadow_stats.py                     # re-read the shipped shadow log (CPU, seconds)
python decide/bench/jev_repeat.py --runs 3              # reference noise ceiling (Jev API)
python decide/bench/banking77_full77_jev.py             # live Jev leg on the 77-way items (Jev API)
```

Rough wall times measured 2026-10-11 on the M5 (32 GB), models cached; expect
network latency on top for the API legs. The 9B legs of the HTTP benches need
`--out runs/<name>.9b.json`: their default output filename does not include
the model, so a 9B run would otherwise overwrite the 4B result in `runs/`.

The Jev legs (`bench_banking77`, `eval_vs_jev`, `jev_repeat`,
`banking77_full77_jev`) need `$OPENROUTER_API_KEY`; `--no-jev` skips the
reference side where it exists.

Benchmarks write to `runs/` (gitignored). `results/` is the frozen evidence
behind every quoted number, and no script replaces a file there unless you
pass `--force`. To check a run against the frozen artifact, `diff
runs/X.json results/X.json`; to refresh the evidence on purpose, `cp` it.
The scripts that call a server or an API check it before doing any work,
and a run in which every call failed exits non-zero without writing.

Benchmarks needing datasets: banking77 (mteb/banking77, MIT) caches to
`data/eval/banking77/test.jsonl`; typed-decisions (Apache-2.0) caches to
`data/eval/typed_decisions/test.jsonl` via the HF datasets-server API.

## Eval sets and tests

```bash
python decide/authoring/author_v2.py --check      # rebuild the 195-question set, compare with data/eval/questions_v2.jsonl
python decide/authoring/author_long.py --check    # same for the 50 long-context questions
python decide/research/l0_smoke.py                # raw vs L0 distributions on three hand-written coding states (loads the 4B model)
python -m unittest discover -s tests -v           # about 10 s; no model, no GPU
```

The two authoring scripts rebuild the shipped eval sets bit for bit. Without
`--check` they write `runs/questions_*.jsonl`; replacing the shipped files needs
`--out data/eval/… --force`. The tests cover the endpoint contract, the output
policy of the scripts, `--help` of every script and the docs links.

## Repository layout

```
decide/                     the Python package (scripts find the repo root; run from anywhere)
├── core/                   readout engine
│   ├── mlx_backend.py        AnyJev Backend protocol on MLX (logprobs, early-stop
│   │                         hidden states, prefix-KV + multi-token scoring)
│   ├── state_shared.py       state-level prefix sharing (one state forward per request) + early stopping
│   ├── batching.py           byte budget / row chunks for batched suffix passes (no MLX)
│   ├── state_feed.py         renderer: live messages / trace points -> state text
│   ├── trace_states.py       trace -> decision-point dataset (seed loader; raises,
│   │                         the gold seed is not shipped)
│   └── labels.py             the coarse inspect/run/modify vocabulary (one source)
├── serve/                  ENTRY POINTS
│   ├── systemone.py          POST /v1/systemone decision endpoint (Face 1)
│   └── proxy.py              shadow proxy :8090 -> upstream :8080 (Face 2)
├── adapt/                  head fitting / the adaptation loop
│   ├── fit_registry_head.py  A/B head fit -> registry artifact
│   ├── synth_states.py       synthetic state generation + L0 teacher labels
│   └── observe.py            observe_log replay -> refit head
├── runio.py                output + preflight policy shared by bench/adapt/authoring
├── bench/                  benchmarks & evals (each writes runs/<name>.json)
│   ├── eval_vs_jev.py        ours vs the real Jev API (+ the shared client/stats utils)
│   ├── bench_vs_generate.py  readout vs greedy, same model
│   ├── bench_banking77.py    10-way gold benchmark (+ item loaders/macro-F1)
│   ├── banking77_full77.py   full-77-way unlock (multi-token option scoring)
│   ├── banking77_full77_jev.py / banking77_greedy.py   the live-Jev / greedy legs
│   ├── bench_typed_decisions.py  typed-decisions head-to-head
│   ├── rotation_budget.py    m=1/2/3/adaptive rotation-budget curve
│   ├── l0_ablation.py        raw vs L0 vs K-curve
│   ├── jev_repeat.py         Jev self-agreement ceiling
│   ├── pool_evals.py         pool eval batches + paired McNemar
│   ├── shadow_stats.py       shadow-log re-read (agreement/router/ECE)
│   ├── bench_state_share.py  state-share parity + speed
│   ├── bench_fast_path.py    sequential vs batched vs batched + early stop (agreement, gold, ms, memory)
│   └── bench_mass_removal.py --no-mass before/after
├── research/               runnable smoke scripts (l0_smoke.py) +
│   │                         stub_server.py (zero-MLX contract test)
└── authoring/              eval-corpus generators (one-off, LLM-assisted, disclosed)

registry/                   fitted head artifacts (lineage + holdout in each JSON)
results/                    frozen measurement artifacts (the numbers docs quote;
                            results/INDEX.md maps file -> producer -> doc section)
runs/                       fresh benchmark output (gitignored; promote to results/ on purpose)
logs/                       live proxy / endpoint logs (gitignored; created on first run)
data/                       eval datasets, authored question sets, sandbox repo
tests/                      unit tests (no model, no GPU): python -m unittest discover -s tests
docs/                       the long-form docs; the README lists them
```
