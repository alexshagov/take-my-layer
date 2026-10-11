# AGENTS.md — notes for coding agents

The short version of what an agent needs to work here without breaking
anything. Humans: [README.md](README.md) is the front page; the long-form
guides live in `docs/`.

**Setup** (details in [docs/RUNNING.md](docs/RUNNING.md)):
`nix develop`, or Python 3.11+ with `pip install -r requirements.txt`.
Models auto-download into `.hf/` on first run (~2.9 GB for the 4B).

**Before and after any change:**
```bash
python -m unittest discover -s tests   # ~10 s, no model, no GPU
python -m compileall -q decide tests
```

**Ground rules** (break these and the artifact silently degrades):
- `anyjev` stays pinned to 0.2.0 — 0.3.0 removed the L1/L2 machinery this
  repo serves (`requirements.txt` says why).
- Fit on what you serve: heads read hidden states of the exact served
  model artifact; never mix quants or adapters.
- Scripts write to `runs/` (gitignored). `results/`, `registry/` and
  `data/eval/` are frozen evidence; replacing a file there needs `--force`.
- One MLX process at a time on a 32 GB machine: stop the endpoint before
  running a benchmark that loads its own model, and kill what you start.
- The endpoint batches the suffix passes of a request by default (`--no-batch`
  is the reference path when checking numerics); `--early-stop` is opt-in
  because it changes the reported confidences (docs/RESULTS.md).
- The shadow proxy stays shadow-only (it routes nothing); the answer-mass
  guard stays log-only (`MASS_TAU=0.0`).
- New numbers follow the wording rules in `docs/RESULTS.md` (fidelity =
  "aligned with the reference", never "more correct"; accuracy claims name
  the arena and model size).

**Where things are:** commands and the repository map →
[docs/RUNNING.md](docs/RUNNING.md); endpoint reference →
[docs/API.md](docs/API.md); artifact index →
[results/INDEX.md](results/INDEX.md); status and open work →
[docs/TODO.md](docs/TODO.md).
