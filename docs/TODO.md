# TODO.md — open work: pre-publish checklist + measurement gaps

Aggregated gaps from the 2026-10-10 review. The first section is the
pre-publish checklist (re-verified the same day); the rest is open
measurement work: benches and proofs that a README or RESULTS claim still
lacks. Build/status backlog lives in `docs/ROADMAP.md`; where an item
overlaps, the cross-ref is noted. Wording rules for any new number:
docs/RESULTS.md header.

## Pre-publish checklist (2026-10-10, re-verified)

Every item was re-run or re-read against the tree in a scratch copy; the
review itself did not modify the repo (fixes are marked DONE). File:line
references are as of 2026-10-10. Tags: **[confirmed]** reproduced as stated,
**[partly]** true with the correction noted, **[new]** found during the
re-check. Order: irreversible once pushed > misleads or breaks a stranger's
first run > clarity > polish.

**Order of operations:** done: everything except PP-19, which is the maintainer's
last step. The stored Jev answers stay (PP-18, maintainer's decision). `git init`
is deferred by the maintainer; when it happens: `git init`, `git add flake.nix
flake.lock` (PP-19), `git add -A` (dry run against a git-dir outside the repo,
2026-10-10 after the last change: see the file count in PP-19's note; no
caches/venv/archive/pyc/logs, no home path, username, key or token pattern in
any of them; re-run it first), first commit, then PP-19. Until then nothing is
under version control; PP-5 removed the main way to lose evidence by accident
(scripts write to `runs/` and refuse to replace frozen files).

### P0 — before the first push

- [x] **PP-1 Scrub the home path from `results/observe_log.jsonl`.** DONE
  2026-10-10. The review's first count (path only) was incomplete: the file
  held the dev-machine path 210× (`/Users/<user>/…/qwen-jev`) and the macOS
  username 32× more, in `ls -l` owner columns, in 5 of the same 44 rows (all
  inside `state`; no other tracked file had either). Now `/path/to/repo` and
  `user`. 73 rows, every non-`state` field identical, substitution exactly
  reversible. Tree-wide grep for the username and for any `/Users/<name>`
  path: 0 hits. `observe.load_rows` reads 73 rows from both
  versions with identical ts/repo/tool/label. Provenance recorded in
  `results/INDEX.md`: `results/mass_removal.json` (13 of its 30 states
  changed) and `registry/…/next_move_observed.json` were computed from the
  pre-scrub text. All 73 rows are `qjev-sandbox` (no private work).
- [x] **PP-2 Rename (the ROADMAP's "last blocker").** DONE 2026-10-10 →
  `take-my-layer` (mark-free; "Shut up and take my layer" is the tagline).
  Swept: `README.md` title, `flake.nix` description,
  `decide/bench/eval_vs_jev.py` (`X-OpenRouter-Title: take-my-layer-eval`)
  and `decide/serve/systemone.py` (the response `model` is now the served
  model id instead of the hard-coded `qwen-jev/Qwen3.5-4B-MLX-4bit`; checked
  live: the 4B and 9B servers each report their own id, answers unchanged).
  The dev shell still builds with the new `flake.nix`; the ROADMAP and
  `AGENTS.md` pointers were updated. Left on purpose: the sandbox name
  `qjev-sandbox` (725 hits in the two logs plus `.gitignore` and
  `RESULTS.md`; a data fixture, and renaming it would change the state text
  the head reads), `docs/HISTORY.md` (historical record), and file names that
  name the comparison target (`eval_vs_jev.py`, `jev_*.json`). The manual
  steps are done too (folder renamed, `.venv` rebuilt, direnv reloaded): no
  `qwen-jev` path is left in `.venv/bin`, `pyvenv.cfg` or `.direnv`.

### P1 — before linking it anywhere

- [x] **PP-3 Install prerequisites are wrong.** DONE 2026-10-10. [confirmed]
  README and the flake comment said Python ≥3.10, but mlx-lm 0.32.0 needs ≥3.11
  (3.10 silently resolved to 0.31.3, 3.9 failed), and
  `decide/core/mlx_backend.py` imports non-public mlx-lm internals. Now:
  `requirements.txt` pins `mlx-lm==0.32.0` with the reason; README "Running"
  states macOS 14+ (PyPI ships macosx_14/15/26 wheels, checked), Python 3.11+,
  the non-Nix install (`python3.12 -m venv .venv`, `pip install -r
  requirements.txt`, optional `export HF_HOME=$PWD/.hf`, since the project-local
  cache is a Nix-shell feature) and the cost of each model (4B: 2.9 GB download,
  2.8 GB resident; 9B: 5.6 GB, 5.3 GB; from `du` of the snapshots and
  `results/banking77.*.resources.json`). The flake comment, AGENTS.md and
  THIRD_PARTY.md match. Checked: a fresh venv from the Nix Python 3.12 with
  `pip install -r requirements.txt` resolves in 32 s to the same versions as the
  Nix shell (mlx-lm 0.32.0, anyjev 0.2.0, mlx 0.32.3, transformers 5.19.0),
  passes the test suite and serves the README example bit-identically
  (confidence 0.5955441762711209); `nix develop -c …` still builds with the pin.
- [x] **PP-4 README claims to fix.** DONE 2026-10-10. (a) The "4B reads about
  30% before training (its own caption)" sentence now says what Unsloth's guide
  shows: Qwen3.5-0.8B 36% → 73%, 2B 33% → 78%, no 4B figure. (b) "Where it
  loses" labels 63.4% (4B) and 66.4% (9B) against Jev's 78.6% and discloses that
  the 4B sits below greedy (66.8%, p = 0.068); the headline table row too
  (numbers re-read from `results/banking77_full77*.json`). (c) "Already using
  AnyJev?" says 0.3.0 (2026-10-06) removed L1/L2, heads, `observe` and
  `level="auto"` and that this repo pins 0.2.0 (checked against the upstream
  CHANGELOG; PyPI's latest is still 0.3.0). (d) The samples are captured
  responses with the `model` field; probabilities are shortened to four decimals
  and the text says so (the server returns full precision). Also fixed: "breaking
  the 20-option limit" (see PP-20).
- [x] **PP-5 Benchmarks overwrite frozen evidence and exit 0 on total
  failure.** DONE 2026-10-10. New `decide/runio.py`, used by every script that
  writes a result: `--out` defaults to `runs/<name>` (gitignored), `--force` is
  needed to replace an existing file in `results/`, `registry/` or `data/eval/`,
  the output path is validated before any work starts, and reports record
  repo-relative paths (`rel()`). The HTTP benches (`bench_banking77`,
  `bench_typed_decisions`, `eval_vs_jev`) check `/healthz` first (exit 2) and
  stop after three failed calls; the Jev legs (also `jev_repeat`,
  `banking77_full77_jev`) check the key first and stop on a failed first Jev
  call; a run in which every call failed exits 1 and writes nothing; partial
  failures print a warning. The six scripts without `--out` have one
  (`banking77_full77_jev`, `bench_mass_removal`, `bench_state_share`,
  `l0_ablation`, `pool_evals`, `rotation_budget`; `fit_registry_head` got
  `--report`), `bench_mass_removal.py` and `l0_smoke.py` got argparse (`--help`
  no longer loads the model and rewrites `results/mass_removal.json`), and
  `banking77_full77_jev` retries failed calls from its checkpoint instead of
  reusing them. Tested against a fake endpoint (success writes only `runs/`; a
  500-only endpoint exits 1 with nothing written; a dead Jev URL exits 1) and
  with no endpoint (exit 2); `results/`, `registry/` and `data/eval/` were
  byte-identical to the pre-change snapshot afterwards, and `pool_evals.py`
  regenerates `results/jev_compare.pooled.json` byte for byte. Not run end to
  end: the long model-loading benches (they only gained `--out`/`--force` and
  root-anchored paths). `observe.py` still writes
  `registry/…/next_move_observed.json` by default, on purpose (replay output,
  PP-14).
- [x] **PP-6 Live logs default into `results/`.** DONE 2026-10-10.
  [confirmed] `proxy.py` defaulted to `results/shadow_log.jsonl` and
  `results/observe_log.jsonl`, `systemone.py` to `results/systemone_log.jsonl`.
  `observe_log` holds the full rendered state and is both committed evidence
  and the replay input for refits, so someone's real sessions would have been
  appended to it. Now: the proxy writes `logs/shadow_log.jsonl` and
  `logs/observe_log.jsonl`, the endpoint `logs/systemone_log.jsonl`;
  `.gitignore` has `/logs/` and `.envrc.local` (sourced by `.envrc`; the file
  does not exist today). `observe.py` keeps the frozen
  `results/observe_log.jsonl` as its default, so a fresh clone can replay the
  shipped 73 rows; live traffic is `--log logs/observe_log.jsonl`
  (`bench_mass_removal.py` also keeps reading the frozen copy). Checked from
  a scratch cwd: the default write path creates only `logs/observe_log.jsonl`
  and `results/observe_log.jsonl` is byte-identical afterwards; `--help` of
  both scripts shows the new defaults. The one local telemetry row moved to
  `logs/`. Docs updated: README (layout, `observe.py` line), AGENTS.md,
  `results/INDEX.md`, ROADMAP; DECIDE.md and HISTORY.md keep the old paths on
  purpose (historical record).
- [x] **PP-7 First screen, names, links.** DONE 2026-10-10. The README is now a
  front page of 88 lines (it was 485): the plain definition of "state" and
  "typed questions" first, the ticket-triage request and its real response,
  the benchmark table with the hardware named, how to run it in three commands,
  and an index list of every doc. Jev is named and linked (TypeSafe AI,
  typesafe.ai) at the top, AnyJev is linked, the research-prototype status line
  says the shipped head is not reproducible (gold seed withheld) and that the
  Jev-agreement numbers have no human labels, and the benchmark rows link
  BANKING77, typed-decisions and the Unsloth guide (all links checked: 200).
  The long text moved, nothing deleted: `docs/CONCEPTS.md` (orientation,
  "in plain words", the two surfaces, the readout levels, glossary; Clef,
  NeoJev, Unsloth and AnyJev now linked there), `docs/RUNNING.md` (the Running
  guide and the repository map), `docs/LIMITATIONS.md` (responsible use and
  measured limits); the trademark paragraph went to `THIRD_PARTY.md`. The
  duplicate "Headline results" / "Results at a glance" are merged. Two wording
  fixes on the way: the Unsloth numbers (73% and 78%) are no longer attributed
  to Clef, and "greedy collapses on long inputs (46% vs 85%)" now says it is
  the Choice questions (overall long-input agreement is 80% vs 70%).
  `tests/test_docs.py` keeps every relative link resolving, the README index
  complete and the README under 150 lines. Cross-references in AGENTS.md,
  CONTRIBUTING.md, DECISIONS.md, RESULTS.md, `results/INDEX.md` and `docs/API.md`
  point to the new homes.
- [x] **PP-8 API undocumented, limits unenforced.** DONE 2026-10-10. New
  `docs/API.md` (fields, the three question types with captured examples, errors
  with the exact messages, auth, limits, flags); the README shows the
  ticket-triage request (Choice + Noul + Score in one POST) next to the coding
  head example. The limits are enforced and listed by `GET /healthz`: state ≤
  32,000 characters (`--max-state-chars`), body ≤ 1 MiB judged from
  `Content-Length` before reading (413), a bad `Content-Length` is a 400, and
  rejected requests close the connection so an unread body cannot corrupt the
  next one. The old unused `MAX_STATE_CHARS` was 8,000, but 4 of the 10
  long-context eval states are longer (up to 17,621 characters), so enforcing it
  would have broken the long-context benchmarks. Checked on the live server
  started from `/tmp`: 32,000 characters 200 in 2.7 s, 32,001 → 400, the longest
  eval state 7.0 s, a 2,000,000-byte header → 413, `GET /v1/systemone` → 405
  with `Allow: POST`, bearer auth 401/200 (healthz open), two identical requests
  give identical responses. Found while documenting: Choice accepts up to 26
  options, not 20 (PP-20).
- [x] **PP-9 Copy-paste-clean commands.** DONE 2026-10-10. The 43-line block is
  split under headings (endpoint, shadow proxy, adaptation loop, benchmarks in
  three groups: HTTP clients of :8095, scripts that load their own model, so stop
  the server first, and CPU-only or API-only). The redundant `bash -c 'export
  HF_HOME=…'` wrappers are gone (`nix develop -c <cmd>` exports it; checked:
  `HF_HOME`, Python 3.12.15, mlx-lm 0.32.0, anyjev 0.2.0), readiness is stated
  (the `systemone on :8095 (head loaded: …)` line, `GET /healthz`) and the four
  Jev legs are named. `bench_mass_removal` and `banking77_greedy` were missing
  from the list and are in.
- [x] **PP-10 Run-from-anywhere trap.** DONE 2026-10-10. Every script derives
  `ROOT = Path(__file__).resolve().parents[N]`, puts it on `sys.path` and
  anchors its default paths to it (paths the user passes stay relative to the
  cwd). Checked from `/tmp`: `--help` of all 26 scripts exits 0 and writes
  nothing; `systemone.py` and `proxy.py` start from `/tmp`, find the head and
  write `logs/` under the repo root; `shadow_stats.py` and `pool_evals.py`
  reproduce their frozen numbers; both authoring scripts pass `--check`. One
  spot was missed at first and found later: `decide/core/trace_states.py` had
  `DATA = "data"`, and the renderer strips the repo path prefix computed from
  it, so the state text the head reads (and the rows logged for `observe.py`)
  depended on the directory the proxy was started from. It is anchored now;
  `tests/test_render.py` fails on the old behaviour (checked by putting it back),
  and a proxy started from `/tmp` logs relative tool paths in a real run.
- [x] **PP-11 Agent-harness path under-specified.** DONE 2026-10-10. README
  gives the three processes in order, a concrete prompt on the sandbox (make
  `Greeter.farewell()` include the name and update the test, which has a real
  failing step because the default name is "world"), the readiness check, the
  follow-up `shadow_stats.py logs/shadow_log.jsonl`, where each log goes, that
  opencode is installed separately, that `OPENCODE_CONFIG` is relative (run from
  the repo root) and that the proxy only shadows conversations that mention
  `data/repos/<name>/`. Verified end to end with the installed opencode 1.18.33
  on the documented ports, twice: in a scratch copy of the sandbox (61 s) and in
  the repo's own sandbox from the repo root with the proxy started from `/tmp`
  (50 s; sandbox restored byte-identical afterwards). Both: tests green, 7 shadow
  rows (6 labelled), 6 observe rows, `shadow_stats.py` reads the live log; in the
  second run the rendered tool paths are relative. Things found and now written
  in `docs/RUNNING.md`: the agent edits the sandbox files (restore them);
  `opencode run` hangs at startup when its stdin is an open pipe, which looked
  like a proxy stall for an hour of debugging (3 of 3 hangs with an open pipe,
  3 of 3 starts in 2 s with `< /dev/null`; not a terminal problem); opencode
  makes its own HTTPS connections at startup; the logs contain the agent's
  absolute paths, so scrub them before promoting one to `results/`.
- [x] **PP-12 No tests, no CI.** DONE 2026-10-10. `tests/` (stdlib `unittest`,
  26 tests, about 8 s, no model or GPU): the endpoint contract (validation,
  limits, auth, routes, healthz, through the real `Handler` on an ephemeral
  port), the `runio` policy, `--help` of every script from another directory,
  byte-for-byte regeneration of both eval sets and the pooled report, the
  `shadow_stats` numbers, the frozen-file guard and the fail-fast of the HTTP
  benches. `.github/workflows/ci.yml`: a portable job (ubuntu, `anyjev` +
  numpy; the 9 scripts that import MLX at module level are skipped) and an
  informational macOS job with the full requirements. The suite passes in the
  Nix venv, in a fresh pip venv, and with MLX made unimportable (9 skipped).
  The workflow itself has not run (no repo yet); only its YAML was parsed.

### P2 — polish

- [x] **PP-13 `systemone.py` / `proxy.py` details.** DONE 2026-10-10. (The
  hard-coded response `model` was fixed with PP-2.) `GET /v1/systemone` is a 405
  with `Allow: POST` and a usage line, other paths a 404 that lists the routes;
  both parsers have a description and `--port` help; the proxy docstring and
  `--upstream` help say it takes a URL.
- [x] **PP-14 `registry/README.md`.** DONE 2026-10-10. Says which head is
  served (`next_move.json`; `next_move_v3.json` is the same bytes, SHA-256
  `e9268efc…`), that `v2_seed` is provenance and `observed` is replay output
  (fitted on 60 of the 73 shipped rows, out-of-fold 0.683, weaker than the served
  head), how to serve another head (`--head`) and promote one, and that the gold
  seed is not shipped.
- [x] **PP-15 Glossary.** DONE 2026-10-10. `docs/CONCEPTS.md` ends with a
  glossary: state, typed question, readout, label-free, head, Face 1 and 2,
  shadow proxy, flywheel, tau, answer mass, bright line, fidelity vs accuracy,
  fork ("at fork" is the split from the private prototype on 2026-10-07, not a
  GitHub fork), gold seed, frozen evidence, rotations, prefix sharing.
- [x] **PP-16 Docs housekeeping.** DONE 2026-10-10. `docs/RESULTS.md` has a
  contents block linking all 7 sections and their sub-sections (anchors checked
  by `tests/test_docs.py`, which also fails if a new H2 is not listed).
  `AGENTS.md` no longer duplicates the command list: it points to
  `docs/RUNNING.md` (now complete: `fit_registry_head`, `stub_server`, the
  authoring `--check` runs and the tests were added) and keeps only the rules an
  agent needs; the same test fails if RUNNING.md stops mentioning a script. The
  README docs index lists CONTRIBUTING, AGENTS and THIRD_PARTY (PP-7), the
  `.gitignore` comment is fixed (PP-6), `CONTRIBUTING.md` has a Development
  section and a "Questions, bugs, corrections" paragraph (the repository's
  issue tracker). Not done on purpose: moving DECIDE/HARNESS/HISTORY to
  `docs/history/` (16 files and 27 `§` references to touch, no gain).
- [x] **PP-17 Unguarded scripts.** DONE 2026-10-10. `author_v2.py` and
  `author_long.py` parse arguments before building anything (`--help` is safe),
  write `runs/questions_*.jsonl` by default and need `--out data/eval/… --force`
  to regenerate in place; `--check` compares the regenerated text with the
  shipped file. Both regenerate bit for bit, so the eval sets are reproducible
  (`--check` exits 1 once they differ, i.e. after the known `questions_long`
  fixes). `shadow_stats.py` and `stub_server.py` use argparse.
- [x] **PP-18 Output rights of the stored Jev answers.** Terms read
  2026-10-10 from the primary pages (raw text, not search summaries): TypeSafe
  MCA (rev. 2026-09-23), Terms of Use (2026-09-19), Acceptable Use Policy
  (2026-09-23), OpenRouter Terms (2026-08-31). Decision (maintainer,
  2026-10-10): **keep** the stored answers, for benchmark visibility, with
  the evaluation-only notice in `THIRD_PARTY.md`.
  - MCA §4.2: "As between Customer and TypeSafe and to the extent permitted by
    Laws, TypeSafe does not claim ownership of Input and TypeSafe disclaims
    ownership of Output. TypeSafe hereby assigns to Customer all of its right,
    title, and interest, if any, in the Output."
  - None of the four documents restricts benchmarking, comparing, or
    publishing results or outputs (grepped for benchmark, publish, compar,
    evaluat, performance; the hits are warranty, license and publicity
    boilerplate).
    TypeSafe's Confidential Information (MCA §14) covers pricing,
    credentials, Documentation and "other non-public information with respect
    to the Services"; the stored `jev_cost` fields are OpenRouter's public
    list price. MCA §16.4 grants no right to use TypeSafe's name, brand or
    logo: keep to plain-text references plus the not-affiliated line, no logo.
  - The restriction that matters, MCA §2.3(b): Customer will not "use the
    Services or any Output (defined below) to perform model distillation,
    train a model to imitate the output of the Services, or develop (or to
    facilitate the development of) a similar or competing product or
    service". This repo is Jev-style and scores itself against Jev, so
    "facilitate" is the open word.
    What sits on the other side: the POLICY §3 bright line (nothing fitted on
    another model's answers), small sets (350 typed questions, 400 + 500
    BANKING77 labels), and the evaluation-only notice now in `THIRD_PARTY.md`.
  - Which terms bind an OpenRouter user is itself unclear: OpenRouter makes the
    user responsible for each model's own terms and lists TypeSafe's website
    Terms of Use for Jev (silent on API output), while the API agreement is the
    MCA. Neither has a publication restriction. Not legal advice.
  - The first list of files was incomplete. Per-item Jev answers sit in 15
    files: `jev_compare{,.n300,.long}.json`, `jev_repeat{,.long}.json`,
    `banking77{,.9b}.json`, `banking77_full77.jev.json`,
    `bench_vs_generate{,.n300,.long,.9b,.9b.v2,.9b.long}.json`,
    `labels_long.scored.json`. Aggregate-only (Jev numbers, no per-item
    answers): `jev_compare.pooled.json`, `rotation_budget.json`,
    `l0_ablation.json`.
  - Had the call been "drop": stripping the per-item fields breaks the
    CPU-only recompute of the agreement numbers (`pool_evals.py` reads
    `jev_compare*.json`; `rotation_budget.py` and `banking77_greedy.py` read
    `banking77*.json`) and the "reproduces from frozen artifacts" claims in
    `results/INDEX.md`. Optional certainty: ask TypeSafe for a written OK
    (`support@typesafe.ai`, the contact in its AUP).
- [ ] **PP-19 GitHub setup.** Repo name `take-my-layer`. About line (241
  chars, limit 350): "Jev-style typed decisions (choice, yes/no, score +
  probabilities) read off a local Qwen's own logits on Apple Silicon. No text
  generated. Research prototype, benchmarked against the real Jev API.
  Independent; not affiliated with TypeSafe AI." Topics: mlx, apple-silicon,
  qwen, logprobs, local-llm. README tagline: "Shut up and take my layer."
  (in the README since PP-7, and the README no longer says "one prefill" or
  "one forward pass"). Keep "one forward pass" out of public copy: it is exact only for the L2
  head (an L0 Choice question with K options is K short passes sharing one
  state prefill); "read off the model's own logits, no text generated" holds
  at every level. On the first `git init`, run `git add flake.nix
  flake.lock` before `nix develop` / direnv (reproduced: Nix errors "not
  tracked by Git" and prints the exact command; negligible). The dry run of
  `git add -A` (git-dir outside the repo) on 2026-10-10, after the last change:
  107 files, 5.17 MB, no caches/venv/archive/pyc/logs/runs, no home path,
  username, key or token pattern in any of them. Order at publish time: dry run
  again, `git init`, `git add flake.nix flake.lock`, `git add -A`, first commit,
  then create the GitHub repo with the About line and topics above.
- [x] **PP-20 Option-cap wording (20 vs 26).** DONE 2026-10-10. anyjev 0.2.0's
  letter readout caps Choice at 26 options (`MAX_OPTIONS`); the endpoint accepted
  21 and rejected 30 with "at most 26 options in letter readout". The wording is
  corrected in `docs/RESULTS.md` (the section title "the >26-option unlock" and
  both "≤26-option Choice cap" mentions), `decide/core/mlx_backend.py`,
  `decide/bench/banking77_full77.py`, the `systemone.py` docstring, the README
  sentence and `docs/API.md`. No number changed.

### Open checks from the 2026-10-10 review (all done)

- [x] Cold-cache first run: done 2026-10-10 from a fresh pip venv with an empty
  `HF_HOME`: the endpoint downloaded the 4B model (2.9 GB) and was ready after
  72 s, and the README example came back bit-identical (confidence
  0.5955441762711209).
- [x] Clean non-Nix install: done 2026-10-10, see PP-3 (`pip install -r
  requirements.txt` in a fresh venv, 32 s, same versions as the Nix shell, test
  suite green).
- [x] Proxy + opencode end-to-end: done 2026-10-10, see PP-11.
- [x] A 9B server run: done 2026-10-10 in a scratch copy (`model` field
  correct; the README example was served at L0, since the 9B has no head).
- [x] TypeSafe's output terms (PP-18): read 2026-10-10, see PP-18.

### Verified, no action

README example reproduces exactly (0.5955 / 0.4039 / 0.0006, 140 tokens; 3
runs byte-identical; ready in 1.6 s at 2.8 GB; a mixed Choice/Noul/Score
request took 0.69 s; bad requests return clear 400s). Headline numbers match
the frozen artifacts (75.7% with +4.7, p = 0.0026; 9B 82.0% / +6.3; 98.7%
ceiling; banking77 87.5 / 91.25 / 95.5; full-77 66.4 vs 60.8, p = 0.00335;
typed-decisions 61.6 / 60.4; 22.2 → 3.7 s; ECE 0.238, 37% routed error at
59% coverage). No secrets in the tree. Docs have no dangling references
except files removed on purpose. `results/INDEX.md` lists every artifact.
External citations hold: AnyJev (PyPI, GitHub, arXiv 2610.00831), the Clef
table, Unsloth 36/33 → 73/78, the OpenRouter ToS revision.

External sources re-checked 2026-10-10:
[Unsloth guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth),
[Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/),
[anyjev on PyPI](https://pypi.org/project/anyjev/),
[AnyJev Technical Report](https://arxiv.org/abs/2610.00831),
[AnyJev CHANGELOG](https://github.com/nokia-applied-research/AnyJev/blob/main/CHANGELOG.md),
[OpenRouter terms](https://openrouter.ai/terms).

## Highest value, lowest cost

1. **Determinism proof.** README says re-running anything returns
   identical answers; `state_share.json` is the only equivalence artifact
   and `jev_repeat.py` covers the reference side only. Run any bench twice
   on the same inputs and byte-compare the stored answers. Cost: trivial.
2. **Same-arena ladder: L0 vs L1 vs L2 on one held-out task.** The levels
   table asserts what each level buys, but L2 has only the `next_move`
   holdout (0.826) and L1 nothing. Fit a temperature and a head for one
   benchmark question (e.g. a banking77 or Noul layout) and evaluate all
   three levels on the same holdout. Cost: one fit plus one eval run.
3. **L1 calibration bench.** The "0.93 knowledge-gap tail" is L1's stated
   reason to exist with no reliability curve behind it. Fit the temperature
   and report ECE / reliability on held-out L0 outputs, in-distribution.
   Also on the ROADMAP backlog ("L1 temperature calibration"). Cost: small.

## Measurement coverage

4. **Foreign-domain L2 generalization.** All 73 observe rows and 81 shadow
   rows are `qjev-sandbox`; the domain-general charter has no evidence
   outside it. The synth A/B was neutral, and its "real test is
   foreign-domain agreement" (ROADMAP P1). Blocked on usage, not code.
5. **AnyJev parity point.** Their published label-free L0 is BANKING77
   20-way on Qwen3-8B (0.803 on 300 items). Run the same config on this
   backend and models for a like-for-like cross-implementation number.
   Cost: one bench run (a 20-way option on `bench_banking77.py`).
6. **Length sweep.** Latency and accuracy exist for short and long buckets;
   a 0.5k / 1k / 2k / 4k / 8k curve would back "wins where agents actually
   work" with a shape instead of two dots. Cost: one GPU session.
7. **Concurrency/throughput baseline.** Everything is single-request; the
   ROADMAP dynamic-batching direction has no baseline number to beat.
   Cost: moderate (client plus server under load).
8. **CLINC150 as a second gold-label arena.** banking77 is the only
   gold-label accuracy bench; CLINC150 sits in the reference ecosystem's
   training mix. Lower priority. Cost: loader plus bench run.
9. **Layer-left sweep for the L2 head (speed, not RAM).** The exit block is
   repo-controlled, not an AnyJev property: `mlx_backend.py` owns the
   early-stop block loop, and the candidate list is the `LAYERS` constant in
   `observe.py` / `fit_registry_head.py` (today `[16, 19, 22]`; the shipped
   head landed on 16/32). A/B plan: set `LAYERS = [4, 8, 12, 16, 20, 24,
   28]`, run `observe.py` per point on the shipped 73-row replay (about a
   minute each; it prints layer, oof and in-domain hits), take ms/decision
   from the `bench_mass_removal.py` harness shape, then re-read τ and ECE
   via `shadow_stats.py` at the winning layer, and compound with
   `--max-permutations 2`. Guardrail: earlier exit saves compute only —
   weights stay resident, so RAM is a model-choice lever (4B 2.8 GB /
   9B 5.3 GB), not a layer lever. Quality n is the 73 replay rows (small);
   the seed-head holdout cannot be recomputed (seed withheld).

## Already tracked elsewhere

- 1.5B floor test (ROADMAP, cross-model comparisons): where the readout
  degrades toward the untrained line.
- Human-label leg and judge-noise repeat (CONTRIBUTING, "the highest-value
  contribution"): a second judge pass on the 30 long items or the ~10-row
  human adjudication.
- The 36/33 pre-training numbers exist only in the frozen
  `results/typed_decisions*.json` reference lines; re-verify the Unsloth
  guide before quoting them anywhere public. Re-verified 2026-10-10: the
  guide confirms 0.8B 36% → 73% and 2B 33% → 78%; what still needs fixing is
  the README wording (PP-4).
- The judge-leg scoring step was performed outside the repo (CONTRIBUTING
  says so now); ship a script if one-command replication is wanted.
