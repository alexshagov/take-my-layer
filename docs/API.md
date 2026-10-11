# API.md — the decision endpoint

`decide/serve/systemone.py` serves one POST. You send a **state** (what is going on) and up to eight **typed questions**. You get an answer and the full probability distribution for each question, read off the model's own logits. No text is generated.

```bash
python decide/serve/systemone.py --port 8095     # first start downloads the model (about 3 GB)
curl -s localhost:8095/healthz                   # answers once the model is loaded
```

Under Nix, prefix the command with `nix develop -c` (docs/RUNNING.md). The server binds `127.0.0.1` only and has no TLS. Set `SYSTEMONE_API_KEY` to require `Authorization: Bearer <key>` on `POST`; `/healthz` stays open.

## Request

`POST /v1/systemone` with a JSON body.

| field | type | notes |
|---|---|---|
| `state` | string, object or array | the situation to decide about. Objects and arrays are rendered to text. At most 32,000 characters (`--max-state-chars`) |
| `questions` | object, 1 to 8 entries | the key is the question name (letters, digits, `_`; up to 64 characters) |

Each question is an object with a `type`:

| type | fields | the answer is |
|---|---|---|
| `Choice` | `options`: 2 to 26 distinct strings. `instructions`: optional, up to 1,800 characters | one of the options |
| `Noul` | `instructions`: a yes/no question | `true` or `false` |
| `Score` | `levels`: 2 to 10 strings, ordered. `instructions`, and optional `criteria` (up to 2,000 characters) | one of the levels |

## Response

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "<question name>": {"type": "Choice", "answer": "...", "confidence": 0.84, "probabilities": {"...": 0.84}}},
 "usage": {"input_tokens": 526}}
```

| field | meaning |
|---|---|
| `model` | the model that was read (`--model`) |
| `answers.<name>.answer` | the most probable option or level; `true` or `false` for `Noul` |
| `answers.<name>.confidence` | the probability of that answer |
| `answers.<name>.probabilities` | every option or level (`Yes` and `No` for `Noul`); they sum to 1 |
| `usage.input_tokens` | tokens of the state plus one readout prompt per question: the prefill the request cost, not generated tokens |

The same request gives the same response, bit for bit. The examples below are real output from the 4B model on an Apple M5, with probabilities rounded to four decimals for display; the server returns full precision.

## Examples

### Several questions in one request (ticket triage)

One request, three question types, one pass over the state. This runs at L0: no labels, no head.

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Ticket #4821 from Dana (Pro plan, customer for 14 months): \"Since this morning every export to CSV fails with a 502. We run payroll from those exports at 3pm today. I retried from two browsers.\" Status page: all green. No other open tickets mention exports.",
 "questions": {
    "route": {"type": "Choice", "instructions": "Which team should take this first?", "options": ["billing", "platform-oncall", "account-management", "docs"]},
    "escalate_now": {"type": "Noul", "instructions": "Does this need the on-call engineer right now?"},
    "urgency": {"type": "Score", "instructions": "How urgent is this ticket?", "levels": ["low", "medium", "high", "critical"]}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "route": {"type": "Choice", "answer": "platform-oncall", "confidence": 0.8425, "probabilities": {"billing": 0.0809, "platform-oncall": 0.8425, "account-management": 0.0592, "docs": 0.0175}},
    "escalate_now": {"type": "Noul", "answer": true, "confidence": 0.637, "probabilities": {"Yes": 0.637, "No": 0.363}},
    "urgency": {"type": "Score", "answer": "critical", "confidence": 0.6637, "probabilities": {"low": 0.0051, "medium": 0.0177, "high": 0.3135, "critical": 0.6637}}},
 "usage": {"input_tokens": 526}}
```

### Choice

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Customer message: \"I was charged twice for the March invoice and nobody has answered my first email.\"",
 "questions": {
    "category": {"type": "Choice", "instructions": "What is this message about?", "options": ["billing", "bug-report", "account-access", "feature-request"]}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "category": {"type": "Choice", "answer": "billing", "confidence": 0.9853, "probabilities": {"billing": 0.9853, "bug-report": 0.0128, "account-access": 0.001, "feature-request": 0.0009}}},
 "usage": {"input_tokens": 130}}
```

### Noul (yes/no)

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Customer message: \"I was charged twice for the March invoice and nobody has answered my first email.\"",
 "questions": {
    "duplicate_charge": {"type": "Noul", "instructions": "Does the customer report a duplicate charge?"}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "duplicate_charge": {"type": "Noul", "answer": true, "confidence": 0.9944, "probabilities": {"Yes": 0.9944, "No": 0.0056}}},
 "usage": {"input_tokens": 109}}
```

### Score (ordered levels)

`criteria` is optional extra guidance appended to the question.

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Customer message: \"I was charged twice for the March invoice and nobody has answered my first email.\"",
 "questions": {
    "frustration": {"type": "Score", "instructions": "How frustrated is the customer?", "criteria": "Judge the tone and whether an earlier email went unanswered.", "levels": ["calm", "annoyed", "angry"]}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "frustration": {"type": "Score", "answer": "angry", "confidence": 0.6488, "probabilities": {"calm": 0.0039, "annoyed": 0.3473, "angry": 0.6488}}},
 "usage": {"input_tokens": 146}}
```

### State as an object

An object or array state is rendered to text before reading.

```bash
curl -s localhost:8095/v1/systemone -d '{"state": {"ticket": {"id": 4821, "plan": "Pro", "text": "I was charged twice for the March invoice and nobody has answered my first email."}, "history": ["email sent 3 days ago"]},
 "questions": {
    "category": {"type": "Choice", "instructions": "What is this message about?", "options": ["billing", "bug-report", "account-access", "feature-request"]}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "category": {"type": "Choice", "answer": "billing", "confidence": 0.98, "probabilities": {"billing": 0.98, "bug-report": 0.0179, "account-access": 0.0011, "feature-request": 0.001}}},
 "usage": {"input_tokens": 234}}
```

### The coding head (L2)

A Choice with exactly the options `inspect`, `run`, `modify` is served by the shipped coding head instead of the label-free readout. The head exists as an experiment for a coding-agent harness (the shadow proxy, docs/RUNNING.md): the fitted L2 level demonstrated on real agent traffic. It is not a capability of the endpoint in general — every other question, on every other option set, is label-free L0.

```bash
curl -s localhost:8095/v1/systemone -d '{"state": "Repo: demo. Task: fix the failing test in tests/test_x.py. Actions so far: 1. read tests/test_x.py",
 "questions": {
    "next": {"type": "Choice", "instructions": "What should the agent do next?", "options": ["inspect", "run", "modify"]}}}'
```

```json
{"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
 "answers": {
    "next": {"type": "Choice", "answer": "inspect", "confidence": 0.5955, "probabilities": {"inspect": 0.5955, "run": 0.4039, "modify": 0.0006}}},
 "usage": {"input_tokens": 140}}
```

## Which readout answers

Every question is read at L0: no labels, nothing fitted. The one exception is a `Choice` whose options are exactly `inspect`, `run`, `modify`: the shipped head (`registry/…/next_move.json`, fitted on coding-agent traffic) answers it at L2. The response does not say which level answered; `logs/systemone_log.jsonl` records the level, confidence and answer mass of every question (not the state text). A `--model` without a head artifact serves L0 only. The levels are explained in docs/CONCEPTS.md ("The levels").

Choice is limited to 26 options because the letter readout uses one token per option (A to Z). The 77-way BANKING77 numbers come from an offline multi-token scoring script (`decide/bench/banking77_full77.py`), not from this endpoint.

## Errors

Errors are `{"error": {"type": "...", "message": "..."}}`.

| status | type | when |
|---|---|---|
| 400 | `invalid_request` | bad JSON, missing `state`, empty or too many questions, unknown type, bad options or levels, state over the limit, bad `Content-Length` |
| 401 | `auth` | `SYSTEMONE_API_KEY` is set and the bearer token is missing or wrong |
| 404 | `not_found` | any other path |
| 405 | `method_not_allowed` | `GET /v1/systemone` (the response carries `Allow: POST`) |
| 413 | `payload_too_large` | body over 1 MiB, judged from `Content-Length` before it is read |
| 500 | `decide_failed` | the readout itself failed |

Messages seen from the running server:

| request | status | message |
|---|---|---|
| invalid JSON | 400 | `request body is not valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)` |
| no `state` | 400 | `'state' is required` |
| 9 questions | 400 | `at most 8 questions per request` |
| 32,001-character state | 400 | `'state' is 32001 characters; the limit is 32000 (server flag --max-state-chars)` |
| 2,000,000-byte body | 413 | `request body is 2000000 bytes; the limit is 1048576` |
| `Choice` with one option | 400 | `question 'q': choice needs at least 2 options` |

## Cost and limits

- Requests are served one at a time: one MLX instance sits behind a lock, so concurrent requests queue.
- Latency grows with the state, not with the number of questions. On the M5 the short examples above took 0.2 to 0.8 s (measured before batching; a short 5-question request now takes about 0.55 s against 1.04 s one pass at a time, docs/RESULTS.md "Speed/precision sweep"), a 32,000-character state with one question took 2.7 s, and the longest evaluation state (17,621 characters, 5 questions) took 7.0 s. The state is read once per request (state-level prefix sharing); `--no-state-shared` turns that off.
- Memory: about 2.8 GB resident for the 4B model, 5.3 GB for the 9B (measured, `results/banking77.*.resources.json`).
- Within a request the suffix passes (one per question and option rotation) run as one batched forward over the shared state cache, capped by a memory budget (`--kv-budget-gib`, default 1.5); a state too long to fit two rows in the budget, or any batching error, falls back to one pass at a time. Batching is the default since 2026-10-11 and `--no-batch` restores the one-at-a-time behaviour (same answers up to bf16 kernel noise; docs/RESULTS.md, "Speed/precision sweep").
- `GET /healthz` returns the limits in force (`max_questions`, `max_state_chars`, `max_body_bytes`) and how reads are served (`serving`: `state_shared`, `batched_suffixes`, `kv_budget_gib`, `early_stop`).

## Server flags

| flag | effect |
|---|---|
| `--port` | TCP port on 127.0.0.1 (default 8095) |
| `--model` | any MLX model id or path; a model without a registry head serves L0 only |
| `--head` | registry head artifact to load (default: derived from `--model`) |
| `--max-permutations` | cap the option rotations per Choice (default: all). `2` keeps most of the accuracy at about 3.8× less latency on a 10-option Choice (docs/RESULTS.md, "Rotation budget"); with the usual 3-5 options it costs accuracy, and it never affects yes/no (always both phrasings) or Score (one pass). `--early-stop` is the better budget knob |
| `--max-state-chars` | reject longer states (default 32,000) |
| `--no-state-shared` | read the state once per question instead of once per request |
| `--no-batch` | run the suffix passes of a request one after another instead of as one batched forward (the behaviour before 2026-10-11) |
| `--kv-budget-gib` | memory cap for one batched forward (default 1.5); a prefix that leaves room for fewer than two rows is read sequentially |
| `--early-stop` | off by default. Read one rotation per Choice/Noul question first and the remaining rotations only where the confidence is below `--early-stop-choice-tau` (default 0.9) / `--early-stop-noul-tau` (default 0.8). Fewer passes, answers essentially unchanged, but the reported confidences are a little more optimistic (docs/RESULTS.md) |
| `--prior-strength` | batch-prior correction exponent (default 0.75) |

## Compatibility

The request shape follows the TypeSafe SystemOne style (capitalised types, `options` and `levels` lists). The Jev API that OpenRouter hosts uses a different dialect (lowercase types, a `criteria` object); `decide/bench/eval_vs_jev.py` translates between the two for the comparison benchmarks. This repo is independent and not affiliated with TypeSafe AI.
