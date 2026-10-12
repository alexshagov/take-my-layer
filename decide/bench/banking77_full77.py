"""BANKING77 full-77-way readout, past the 26-option cap.

The 10-way bench (bench_banking77.py) tied readout vs greedy on accuracy
(4B 87.5/87.75, 9B 91.25/92.25 — banking77_greedy artifacts). This is the
harder second arena: all 77 intents, ~500 gold items (Unsloth's cut size).

77 options exceed the letter readout (<=26-option Choice cap, an
anyjev-0.2.0 single-token alphabet artifact), so options are scored by
MULTI-TOKEN LOGLIKELIHOOD (the lm-evaluation-harness standard):
  prompt  = state + question + full numbered listing + "Answer with the
            exact option text." (rendered through the chat template);
  each option scored as the assistant continuation from a copy of the
  prompt's KV cache (MLXBackend.score_suffix_sum — one prompt forward,
  77 short suffix passes);
  per rotation: softmax over the 77 option sums -> the served L0 math
  from the same building blocks (batch prior at 0.75 strength in position
  space, min_prior_n=8, cyclic-shift marginalization, logmean combine).

Greedy leg: SAME items, SAME prompt (identity listing order), temp-0
decode, answers parsed longest-first containment vs the 77 labels — parse
failures are counted (that asymmetry IS the test: generation must emit and
parse a typed answer, the readout scores fixed candidates).

Claim discipline (docs/POLICY.md §3): our side is zero-training L0; nothing
is fit on banking77. If the readout beats greedy here the accuracy claim
gets its first positive leg; if not, "aligned with the reference" stays
the only claim. Published full-77-way context (Cloudflare Clef table):
macro-F1 Clef 94.2 / Clef-flash 90.9 / Jev 79.7; Unsloth TRAINED 0.8B 74%
/ 2B 58% accuracy (in-distribution — banking77 is in their training mix).

Run (one model per process — memory discipline):
  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
    python decide/bench/banking77_full77.py \
      --model mlx-community/Qwen3.5-4B-MLX-4bit'
-> runs/banking77_full77.json (results/ is frozen; see decide/runio.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev.calibrate.contextual import apply_contextual, batch_prior
from anyjev.calibrate.permute import cyclic_shifts, marginalize
from anyjev.readout import PromptSpec, render_chat
from anyjev.state import render_state

from decide.bench.bench_banking77 import load_items_full, macro_f1
from decide.runio import add_out, check_out
from decide.core.mlx_backend import MLXBackend

SYSTEM = ("You are a decision function. You will be given a state and one "
          "question. Reply with the exact option text only: no words before "
          "or after it, no explanation.")
QUESTION = "Which customer intent does this message express?"
MAX_TOKENS = 24                 # labels run 2-8 words; 24 tokens never truncates


def build_user(state: str, options: list[str], perm) -> str:
    listing = "\n".join(f"{j + 1}. {options[i]}" for j, i in enumerate(perm))
    return (f"State:\n{state}\n\nQuestion: {QUESTION}\nOptions:\n"
            f"{listing}\n\nAnswer with the exact option text.")


def render_prompt(tokenizer, state: str, options: list[str], perm) -> str:
    return render_chat(tokenizer, PromptSpec(SYSTEM, build_user(state, options, perm)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def parse_answer(text: str, options: list[str]) -> str | None:
    """Longest-first containment vs the 77 labels; tolerant of list numbers,
    an 'Answer:' prefix, and surrounding punctuation. None = parse failure."""
    t = text.strip()
    t = re.sub(r"^(?:answer|option|intent)\s*:\s*", "", t, flags=re.I).strip()
    t = t.strip("`*#\"'.,:;()")
    t = re.sub(r"^\d+[.)]\s*", "", t).strip()
    low = t.lower()
    for opt in sorted(options, key=len, reverse=True):
        if low.startswith(opt.lower()):
            return opt
    for opt in sorted(options, key=len, reverse=True):
        if opt.lower() in low:
            return opt
    return None


def wilson(hits: int, n: int):
    from decide.bench.eval_vs_jev import wilson as w
    return w(hits, n)


def readout_leg(be, items, options, m, *, prior, strength, min_prior_n,
                leading_space):
    """Primary = served-mirror L0 (batch prior 0.75, position space, min_n=8).
    Also returns the prior-OFF ablation preds from the same pass (zero extra
    compute): the sum-loglik distributions are very peaked, so the prior's
    effect is itself a measured question on this task."""
    perms = cyclic_shifts(len(options), m)
    cont = (lambda o: " " + o) if leading_space else (lambda o: o)
    run_sums = np.zeros((len(perms), len(options)))     # Decider._running mirror
    n_run = 0
    preds, preds_off, confs, flips, lat = [], [], [], [], []
    prior_used = False
    for i, (text, gold) in enumerate(items, 1):
        state = render_state(text)
        t0 = time.perf_counter()
        p_rows = []
        for perm in perms:
            prompt = render_prompt(be.tokenizer, state, options, perm)
            # score in POSITION order (index j <-> listing line j), so
            # marginalize() maps position j back to option perm[j]
            sums = np.asarray(
                be.score_suffix_sum(prompt, [cont(options[i]) for i in perm]))
            p_rows.append(_softmax(sums))
        p_pos_raw = np.stack(p_rows)                     # [P, K] position space
        lat.append((time.perf_counter() - t0) * 1000)
        run_sums += p_pos_raw                            # accumulate BEFORE use
        n_run += 1                                        # (served parity)
        probs_off = marginalize(p_pos_raw, perms, "logmean")
        if prior and n_run >= min_prior_n:
            pi = batch_prior((run_sums / n_run)[None])   # [P, K]
            p_pos = apply_contextual(p_pos_raw, np.power(pi, strength))
            prior_used = True
        else:
            p_pos = p_pos_raw
        probs = marginalize(p_pos, perms, "logmean")     # option space [K]
        j = int(np.argmax(probs))
        preds.append(options[j])
        confs.append(float(probs[j]))
        preds_off.append(options[int(np.argmax(probs_off))])
        winners = [perm[int(np.argmax(p_rows[s]))] for s, perm in enumerate(perms)]
        flips.append(len(set(winners)) > 1)
        if i % 25 == 0 or i == len(items):
            acc = sum(p == g for p, (_, g) in zip(preds, items)) / i
            acc_off = sum(p == g for p, (_, g) in zip(preds_off, items)) / i
            print(f"  readout {i}/{len(items)}  acc={acc:.3f} (no-prior "
                  f"{acc_off:.3f})  ms/item={lat[-1]:.0f}", flush=True)
    return preds, preds_off, confs, flips, lat, prior_used


def greedy_leg(be, items, options):
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    greedy = make_sampler(temp=0.0)
    prompt0 = None
    preds, raws, lat = [], [], []
    parse_fail = 0
    for i, (text, gold) in enumerate(items, 1):
        state = render_state(text)
        prompt = render_prompt(be.tokenizer, state, options,
                               list(range(len(options))))
        if prompt0 is None:
            prompt0 = prompt
        t0 = time.perf_counter()
        out = mlx_lm.generate(be.model, be.tokenizer, prompt=prompt,
                              max_tokens=MAX_TOKENS, sampler=greedy,
                              verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
        a = parse_answer(out, options)
        if a is None:
            parse_fail += 1
        preds.append(a)
        raws.append(out)
        if i % 50 == 0 or i == len(items):
            acc = sum(p == g for p, (_, g) in zip(preds, items)) / i
            print(f"  greedy  {i}/{len(items)}  acc={acc:.3f}  "
                  f"parse_fail={parse_fail}", flush=True)
    return preds, raws, lat, parse_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--max-permutations", type=int, default=2,
                    help="cyclic rotations per item (m; full-K=77 is the "
                         "accuracy-max default on served surfaces)")
    ap.add_argument("--no-prior", action="store_true",
                    help="disable the batch-prior mirror (ablation)")
    ap.add_argument("--prior-strength", type=float, default=0.75)
    ap.add_argument("--min-prior-n", type=int, default=8)
    ap.add_argument("--leading-space", action="store_true",
                    help="score continuations as ' <option>' instead of the "
                         "bare option text (chat templates end the prompt "
                         "with a newline; default = bare)")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-readout", action="store_true",
                    help="greedy leg only (readout answers replayed from "
                         "--stored)")
    ap.add_argument("--stored", default=None,
                    help="replay readout preds from a previous artifact "
                         "(rows[].readout) instead of recomputing")
    add_out(ap, "banking77_full77.json")
    args = ap.parse_args()
    out = check_out(ap, args.out, args.force)
    if (args.skip_readout or args.skip_greedy) and not args.stored:
        ap.error("--skip-readout/--skip-greedy replay from --stored; pass one")

    classes, options, items = load_items_full(args.n_target)
    print(f"{len(items)} items, {len(classes)} classes, model {args.model}, "
          f"m={args.max_permutations}", flush=True)

    be = MLXBackend(args.model) if not (args.skip_readout and args.stored) else None
    if args.skip_readout:
        stored = json.load(open(args.stored))
        r_preds = [r["readout"] for r in stored["rows"]]
        r_preds_off = [r.get("readout_no_prior") for r in stored["rows"]]
        assert len(r_preds) == len(items)
        r_confs = [r.get("conf") for r in stored["rows"]]
        r_flips = [r.get("rot_flip") for r in stored["rows"]]
        r_lat = [np.nan]
        prior_used = stored.get("readout", {}).get("prior_used")
    else:
        (r_preds, r_preds_off, r_confs, r_flips, r_lat,
         prior_used) = readout_leg(
            be, items, options, args.max_permutations,
            prior=not args.no_prior, strength=args.prior_strength,
            min_prior_n=args.min_prior_n, leading_space=args.leading_space)

    if args.skip_greedy:
        g_preds, g_raws, g_lat, parse_fail = [], [], [np.nan], 0
        stored = json.load(open(args.stored))
        g_preds = [r["greedy"] for r in stored["rows"]]
        parse_fail = stored["greedy"]["parse_failures"]
    else:
        if be is None:
            be = MLXBackend(args.model)
        g_preds, g_raws, g_lat, parse_fail = greedy_leg(be, items, options)

    n = len(items)
    golds = [g for _, g in items]
    r_acc = sum(p == g for p, g in zip(r_preds, golds))
    r_acc_off = sum(p == g for p, g in zip(r_preds_off, golds))
    g_acc = sum(p == g for p, g in zip(g_preds, golds))
    r_f1, r_tab = macro_f1(r_preds, golds)
    r_f1_off, _ = macro_f1(r_preds_off, golds)
    g_f1, g_tab = macro_f1(g_preds, golds)
    b = sum(1 for p, r, g in zip(g_preds, r_preds, golds) if r == g and p != g)
    c = sum(1 for p, r, g in zip(g_preds, r_preds, golds) if p == g and r != g)
    nd = b + c
    p_mcn = (sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd * 2
             if nd else 1.0)

    # length-bias diagnostic: sum-loglik favors short options — are the
    # predictions systematically shorter than gold?
    tok = be.tokenizer if be is not None else None
    if tok is not None:
        tlen = {o: len(tok.encode(o, add_special_tokens=False)) for o in options}
        d_pred = np.mean([tlen.get(p, np.nan) for p in r_preds if p])
        d_gold = np.mean([tlen[g] for g in golds])
        d_greedy = np.mean([tlen.get(p, np.nan) for p in g_preds if p])
        length_bias = dict(mean_tokens=dict(readout=round(float(d_pred), 2),
                                            greedy=round(float(d_greedy), 2),
                                            gold=round(float(d_gold), 2)))
    else:
        length_bias = None

    def pcts(v):
        v = sorted(x for x in v if x == x)              # drop NaN
        return dict(p50=round(v[len(v) // 2]), p95=round(v[int(len(v) * .95)]))

    report = dict(
        benchmark="BANKING77 full-77-way (mteb/banking77 test split) — all 77 "
                  f"intents, {n} gold items (~6-7/class, Unsloth's cut size), "
                  "multi-token loglikelihood option scoring",
        model=args.model,
        method=dict(readout="zero-training L0 readout: per-option sum "
                            "log-softmax continuations off one prompt-cache "
                            "copy per rotation (score_suffix_sum), softmax "
                            "over the 77 sums, batch prior (0.75, position "
                            "space, min_n=8) + cyclic-shift marginalization "
                            "(logmean) — same building blocks as the served "
                            "pipeline; nothing fit on banking77",
                    greedy="temp-0 decode, same prompt (identity listing "
                           "order), answers parsed longest-first containment "
                           "vs the 77 labels",
                    rotations=args.max_permutations,
                    continuation="leading-space" if args.leading_space else "bare",
                    prior_used=bool(prior_used)),
        readout=dict(accuracy=round(r_acc / n, 4),
                     accuracy_wilson95=[round(x, 3) for x in wilson(r_acc, n)],
                     macro_f1=round(r_f1, 4),
                     rotation_flips=f"{sum(r_flips)}/{n}",
                     latency_ms=pcts(r_lat)),
        readout_ablation_no_prior=dict(accuracy=round(r_acc_off / n, 4),
                                       accuracy_wilson95=[
                                           round(x, 3) for x in
                                           wilson(r_acc_off, n)],
                                       macro_f1=round(r_f1_off, 4)),
        greedy=dict(accuracy=round(g_acc / n, 4),
                    accuracy_wilson95=[round(x, 3) for x in wilson(g_acc, n)],
                    macro_f1=round(g_f1, 4),
                    parse_failures=parse_fail,
                    latency_ms=pcts(g_lat)),
        paired=dict(only_readout_right=b, only_greedy_right=c,
                    mcnemar_exact_p=round(min(p_mcn, 1.0), 5)),
        chance=round(1 / 77, 4),
        context="published full-77-way macro-F1 (Cloudflare Clef table): Clef "
                "94.2 / Clef-flash 90.9 / Jev 79.7; Unsloth trained 0.8B 74% "
                "/ 2B 58% accuracy — TRAINED, in-distribution (banking77 is "
                "in their training mix); ours is zero-training frozen 4-bit",
        length_bias_diagnostic=length_bias,
        max_tokens=MAX_TOKENS,
    )
    out.write_text(json.dumps(
        dict(report, rows=[dict(text=t, gold=g, readout=rp, greedy=gp,
                                readout_no_prior=ro,
                                conf=round(cf, 4) if cf is not None else None,
                                rot_flip=fl)
                           for (t, g), rp, ro, gp, cf, fl
                           in zip(items, r_preds, r_preds_off, g_preds,
                                  r_confs, r_flips)]),
        indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
