"""Decide-vs-generate: the headline benchmark for the same base model.

Same eval questions (default data/eval/questions.jsonl), same prompt text:
  - decide: the L0 readout (logits at the answer position, exactly what the
    SystemOne endpoint serves) — in-process, same Decider config;
  - generate: greedy decoding (temp 0) of up to 12 tokens from the same
    prompt — the naive alternative ("just ask the model").

Reported: latency, generation parse failures, agreement with real Jev
(replayed from a stored jev_compare*.json — no API calls), and
repeatability (3x on a sample: readout full-distribution equality vs
generation answer flips).

Run:  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
        python decide/bench/bench_vs_generate.py'
      # n~300 variant (replays the stored Jev answers in results/jev_compare.n300.json):
      python decide/bench/bench_vs_generate.py \
        --questions data/eval/questions_v2.jsonl \
        --jev results/jev_compare.n300.json \
        --out runs/bench_vs_generate.n300.json
-> runs/bench_vs_generate.json by default (results/ is frozen; see
decide/runio.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev import Decider, Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, render_chat_parts, resolve_labels
from anyjev.state import render_state

from decide.runio import add_out, check_out
from decide.core.mlx_backend import MLXBackend

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
MAX_TOKENS = 12


def build_q(spec: dict, qid: str) -> Question:
    from anyjev.question import QuestionError
    t = spec["type"]
    text = str(spec.get("instructions") or "").strip()
    if t == "Choice":
        return Question.choice(text or "Which option applies?", spec["options"], name=qid)
    if t == "Noul":
        return Question.noul(text or "Is this true?", name=qid)
    crit = str(spec.get("criteria") or "").strip()
    return Question.score((text + ("\nCriteria: " + crit if crit else "")).strip()
                          or "Rate this.", levels=spec["levels"], name=qid)


def parse_generated(text: str, q: Question) -> str | None:
    t = text.strip().strip(".,:;`*#\"'")
    low = t.lower()
    if q.kind == "noul":
        if low.startswith("yes"):
            return "Yes"
        if low.startswith("no"):
            return "No"
        return None
    if q.kind == "score":
        for i, lv in enumerate(q.options):
            if low.startswith(str(i + 1)) or low.startswith(lv.lower()):
                return lv
        return None
    first = low.split()[0] if low.split() else ""
    for i, (letter, opt) in enumerate(zip("ABCDEFGH", q.options)):
        if first == letter.lower() or low.startswith(opt.lower()):
            return opt
    return None


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--questions", default=str(ROOT / "data/eval/questions.jsonl"))
    ap.add_argument("--jev", default=str(ROOT / "results/jev_compare.json"),
                    help="stored eval run to replay Jev's answers from (row i "
                         "must match the questions file order)")
    add_out(ap, "bench_vs_generate.json")
    ap.add_argument("--gen-tokens-long", type=int, default=0,
                    help="also time a second greedy decode of this many tokens "
                         "per question (the realistic tool-call regime)")
    ap.add_argument("--resources", action="store_true",
                    help="sample this process's RSS/CPU/GPU around the main "
                         "loop (stdlib ps+ioreg sampler; the model is "
                         "in-process, so the pid covers it)")
    args = ap.parse_args()
    MODEL = args.model
    QUESTIONS, JEV = Path(args.questions), Path(args.jev)
    OUT = check_out(ap, args.out, args.force)

    reqs = [json.loads(l) for l in QUESTIONS.open() if l.strip()]
    jev_rows = {}
    for r in json.load(JEV.open())["rows"]:
        m = {x["qid"]: x["answer_j"] for x in r.get("questions", [])}
        # noul refs are bools in jev_compare.json; unify with Yes/No answers
        for k, v in m.items():
            if isinstance(v, bool):
                m[k] = "Yes" if v else "No"
        jev_rows[r["i"]] = m

    be = MLXBackend(MODEL)
    dec = Decider(be, level="auto", shared_prefix=True)
    reg_path = ROOT / "registry" / MODEL.replace("/", "__") / \
        "base" / "next_move.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        from anyjev import Question as Q
        head_q = Q.choice(reg["question"]["text"], reg["question"]["options"],
                          name=reg["question"]["name"])
        dec.load_artifact(head_q, reg["artifact"])
    # no registry artifact for this model -> L0-only serving
    # (fit-on-what-you-serve; same pattern as systemone.py)

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    greedy = make_sampler(temp=0.0)

    sampler = None
    if args.resources:
        import os
        from decide.bench.eval_vs_jev import ResourceSampler
        sampler = ResourceSampler(os.getpid())
        sampler.__enter__()

    t_dec = t_gen = 0.0
    dec_hits = gen_hits = gen_parse_fail = n = 0
    per_q = []
    for i, body in enumerate(reqs, 1):
        state = render_state(body["state"])
        jmap = jev_rows.get(i, {})
        for qid, spec in body["questions"].items():
            q = build_q(spec, qid)
            labels, ids = resolve_labels(be.tokenizer, q)
            prompt_spec = build_prompt(state, q, list(range(q.k)),
                                       DEFAULT_SYSTEM, labels)
            pre, suf = render_chat_parts(be.tokenizer, prompt_spec)
            prompt = pre + suf
            ref = jmap.get(qid)

            t0 = time.perf_counter()
            d = dec.decide(state, [q])[q.name]
            ms_dec = (time.perf_counter() - t0) * 1000
            t_dec += ms_dec / 1000

            t0 = time.perf_counter()
            out = mlx_lm.generate(be.model, be.tokenizer, prompt=prompt,
                                  max_tokens=MAX_TOKENS, sampler=greedy, verbose=False)
            ms_gen = (time.perf_counter() - t0) * 1000
            t_gen += ms_gen / 1000
            ms_gen_long = None
            gen_long_tokens = None
            if args.gen_tokens_long:
                # realistic long-answer regime: the SAME question asked without
                # the label-only constraint — reasoning + final answer, like a
                # real tool-call/justification decode. (With the label-only
                # constraint the model hits EOS after ~3 tokens and the cap
                # never engages — measured, hence this variant.)
                spec_long = build_prompt(
                    state, q, list(range(q.k)), system=(
                        "You are an assistant answering a question about the "
                        "state. Give 2-3 sentences of reasoning, then a final "
                        "line of the form 'Answer: <label>'."), labels=labels)
                prompt_long = "".join(render_chat_parts(be.tokenizer, spec_long))
                t0 = time.perf_counter()
                out_long = mlx_lm.generate(be.model, be.tokenizer,
                                           prompt=prompt_long,
                                           max_tokens=args.gen_tokens_long,
                                           sampler=greedy, verbose=False)
                ms_gen_long = (time.perf_counter() - t0) * 1000
                gen_long_tokens = len(be.tokenizer.encode(
                    out_long, add_special_tokens=False))
            ans_g = parse_generated(out, q)
            if ans_g is None:
                gen_parse_fail += 1
            n += 1
            dh = int(d.argmax == ref) if ref is not None else -1
            gh = int(ans_g == ref) if (ref is not None and ans_g is not None) else -1
            dec_hits += max(dh, 0)
            gen_hits += max(gh, 0)
            per_q.append(dict(i=i, qid=qid, type=q.kind, dec_ans=d.argmax,
                              gen_ans=ans_g, jev=ref, dec_ok=dh, gen_ok=gh,
                              dec_conf=round(float(d.confidence), 3),
                              prompt_tokens=len(be.tokenizer.encode(
                                  prompt, add_special_tokens=False)),
                              ms_decide=round(ms_dec, 1), ms_gen=round(ms_gen, 1),
                              **({"ms_gen_long": round(ms_gen_long, 1),
                                  "gen_long_tokens": gen_long_tokens}
                                 if ms_gen_long is not None else {})))
            print(f"  {i} {qid:<24} {q.kind:<7} ptok={per_q[-1]['prompt_tokens']:<5}"
                  f" dec={ms_dec:>7.0f}ms gen={ms_gen:>6.0f}ms"
                  + (f" gen-freeform={ms_gen_long:>7.0f}ms/{gen_long_tokens}tok"
                     if ms_gen_long is not None else ""), flush=True)

    # repeatability: 5 states, 3 reps, full-distribution equality vs answer flips
    if sampler is not None:
        sampler.__exit__()
        report_resources = sampler.summary()
    else:
        report_resources = None
    rep_states = reqs[::4][:5]
    dec_same = gen_same = reps = 0
    for body in rep_states:
        state = render_state(body["state"])
        for qid, spec in list(body["questions"].items())[:2]:
            q = build_q(spec, qid)
            labels, ids = resolve_labels(be.tokenizer, q)
            prompt = render_chat_parts(be.tokenizer, build_prompt(
                state, q, list(range(q.k)), DEFAULT_SYSTEM, labels))
            prompt = prompt[0] + prompt[1]
            probs0 = dec.decide(state, [q])[q.name].probs.tobytes()
            gen0 = parse_generated(mlx_lm.generate(
                be.model, be.tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                sampler=greedy, verbose=False), q)
            same_d = same_g = True
            for _ in range(2):
                if dec.decide(state, [q])[q.name].probs.tobytes() != probs0:
                    same_d = False
                g = parse_generated(mlx_lm.generate(
                    be.model, be.tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                    sampler=greedy, verbose=False), q)
                if g != gen0:
                    same_g = False
            reps += 1
            dec_same += int(same_d)
            gen_same += int(same_g)

    report = dict(
        n_questions=n,
        latency_ms=dict(decide=round(t_dec / n * 1000),
                        generate=round(t_gen / n * 1000),
                        speedup=round(t_gen / t_dec, 1),
                        note="short prompts (~150 tok), 12-token greedy answers — "
                             "generation's BEST case; long contexts/200-token tool "
                             "calls are the historical 8s vs 0.2s case"),
        agreement_with_jev=dict(
            decide=round(dec_hits / n, 3),
            generate=round(gen_hits / n, 3),
            gen_parse_fail=gen_parse_fail,
            decide_by_type={t: round(sum(max(q["dec_ok"], 0) for q in per_q
                                          if q["type"] == t) /
                                      sum(1 for q in per_q if q["type"] == t), 3)
                            for t in ("choice", "noul", "score")},
            generate_by_type={t: round(sum(max(q["gen_ok"], 0) for q in per_q
                                            if q["type"] == t) /
                                        sum(1 for q in per_q if q["type"] == t), 3)
                              for t in ("choice", "noul", "score")}),
        repeatability=dict(readout_dist_identical=f"{dec_same}/{reps}",
                           greedy_answer_identical=f"{gen_same}/{reps}",
                           note="readout also returns a full probability "
                                "distribution; generation does not"),
        **({"resources_serving_loop": report_resources}
           if report_resources else {}),
    )
    # per-question timing (present when --per-question timings were recorded)
    if per_q and "ms_decide" in per_q[0]:
        import numpy as np

        def split(rows):
            if not rows:
                return None
            out = {k: dict(mean=round(float(np.mean([r[k] for r in rows])), 1),
                           p50=round(float(np.median([r[k] for r in rows])), 1))
                   for k in ("ms_decide", "ms_gen") + (("ms_gen_long",)
                       if "ms_gen_long" in rows[0] else ())}
            out["prompt_tokens_mean"] = round(float(np.mean(
                [r["prompt_tokens"] for r in rows])), 0)
            return out
        med = float(np.median([r["prompt_tokens"] for r in per_q]))
        report["per_question_latency"] = dict(
            all=split(per_q),
            shorter=dict(n=sum(1 for r in per_q if r["prompt_tokens"] <= med),
                         **split([r for r in per_q if r["prompt_tokens"] <= med])),
            longer=dict(n=sum(1 for r in per_q if r["prompt_tokens"] > med),
                        **split([r for r in per_q if r["prompt_tokens"] > med])),
            split_at_prompt_tokens=round(med))
    OUT.write_text(json.dumps(dict(report, per_q=per_q), indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
