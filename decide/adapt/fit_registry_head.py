"""Fit the production head and write it to the registry.

See docs/DESIGN.md, 'Renderer and head registry'.

NOTE (release prep): the gold seed (data/traces/) is no longer shipped —
running this requires regenerating a labeled corpus first (synth_states.py
and/or observe_log.jsonl via observe.py). The shipped next_move.json is the
artifact of the last run on the seed; its A/B measurement lives in
results/synth_gain.json.

Renderer v3 + head refit = the ONE atomic change (docs/POLICY.md). Protocol:

  gold = the verifier-backed seed corpus (the strong corpus, policy §3);
  split gold by TASK into train/holdout (no leakage across folds);
  A = fit_head(gold-train)                — the gold-only head;
  B = fit_head(gold-train + synthetic)    — the supplemented head (if --synth);
  evaluate A and B on the gold holdout    — the honest synthetic-gain number;
  write B (or A without --synth) to --out (next_move_v3.json; the serving
  next_move.json is promoted deliberately, never overwritten here). An
  existing registry file needs --force; the A/B report goes to
  runs/synth_gain.json (results/synth_gain.json is the frozen copy).

Run:  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; \
        python decide/adapt/fit_registry_head.py --synth data/synth/states.jsonl'
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from anyjev import Decider, Question
from anyjev.heads import LinearHead
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels

from decide.core.labels import CLASSES, MERGE
from decide.core.mlx_backend import MLXBackend
from decide.core.state_feed import calls_from_point, render
from decide.core.trace_states import load_decision_points
from decide.runio import RUNS, check_out

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
SEED = 0
HOLDOUT_FRAC = 0.2
TAU = 0.8                       # from the H1 router replay (docs/DESIGN.md, 'Router replay')
# cheap-depth candidates only (the layer-32 pick costs 0.3-10s/decide and
# churns Metal; folds at 16-22 held OOF 0.81-0.85)
LAYERS = [16, 19, 22]

REG = ROOT / "registry" / MODEL.replace("/", "__") / "base"


def load_synth(path: Path) -> list[tuple[str, int]]:
    rows = []
    seen = set()
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") not in CLASSES or not r.get("state"):
                continue
            key = r["state"][:200]
            if key in seen:
                continue
            seen.add(key)
            rows.append((r["state"], CLASSES.index(r["label"])))
    return rows


def holdout_eval(dec: Decider, head: LinearHead, layer: int, q: Question,
                 labels, states: list[str], y: list[int]) -> float:
    hits = 0
    for st, yi in zip(states, y):
        spec = build_prompt(st, q, list(range(q.k)), dec.system, labels)
        pre, suf = render_chat_parts(dec.backend.tokenizer, spec)
        feats, _, _, _ = dec.backend.hidden_states_to([pre + suf], [layer],
                                                      max_layer=layer)
        pred = int(head.probs(feats[:, 0])[0].argmax())
        hits += int(pred == yi)
    return hits / len(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=None,
                    help="synthetic states jsonl to supplement gold (policy §3)")
    ap.add_argument("--out", default=str(REG / "next_move_v3.json"),
                    help="head artifact to write (an existing registry file "
                         "needs --force)")
    ap.add_argument("--report", default=str(RUNS / "synth_gain.json"),
                    help="A/B report (default runs/synth_gain.json)")
    ap.add_argument("--force", action="store_true",
                    help="allow --out/--report to replace shipped artifacts")
    ap.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    args = ap.parse_args()
    path = check_out(ap, args.out, args.force)
    report_path = check_out(ap, args.report, args.force)

    q = Question.choice("What should the agent do next?", CLASSES, name="next_move")
    pts = load_decision_points()
    rng = random.Random(SEED)
    tasks = sorted({p.task_id for p in pts})
    rng.shuffle(tasks)
    n_hold = max(1, int(len(tasks) * args.holdout_frac))
    hold_ids = set(tasks[:n_hold])
    train = [p for p in pts if p.task_id not in hold_ids]
    hold = [p for p in pts if p.task_id in hold_ids]
    st_tr = [render(p.task_text, p.repo, calls_from_point(p)) for p in train]
    y_tr = [CLASSES.index(MERGE[p.label]) for p in train]
    st_hd = [render(p.task_text, p.repo, calls_from_point(p)) for p in hold]
    y_hd = [CLASSES.index(MERGE[p.label]) for p in hold]
    synth = load_synth(Path(args.synth)) if args.synth else []
    print(f"gold train {len(st_tr)} / holdout {len(st_hd)} (tasks "
          f"{len(tasks) - n_hold}/{n_hold}) / synth {len(synth)}")

    be = MLXBackend(MODEL)
    dec = Decider(be, level="auto")
    labels, _ = resolve_labels(be.tokenizer, q)

    fits = {}
    t0 = time.perf_counter()
    art_a = dec.fit_head(q, st_tr, y_tr, layers=LAYERS)
    fits["A_gold"] = (LinearHead.from_dict(art_a), int(art_a["layer_abs"]), art_a)
    if synth:
        art_b = dec.fit_head(q, st_tr + [s for s, _ in synth],
                             y_tr + [y for _, y in synth], layers=LAYERS)
        fits["B_gold+synth"] = (LinearHead.from_dict(art_b), int(art_b["layer_abs"]), art_b)
    print(f"fit(s) done in {time.perf_counter() - t0:.0f}s")

    report = {}
    for name, (head, layer, art) in fits.items():
        oof = art["cv"].get("oof_acc", float("nan"))
        ho = holdout_eval(dec, head, layer, q, labels, st_hd, y_hd)
        report[name] = dict(layer=layer, oof_acc=oof, holdout_acc=ho,
                            n_fit=art.get("n_calib", 0))
        print(f"{name}: layer {layer} oof {oof:.3f} holdout {ho:.3f} "
              f"(n={art.get('n_calib', 0)})")

    best_name = max(report, key=lambda k: report[k]["holdout_acc"])
    head, layer, art = fits[best_name]
    out = {
        "model": MODEL, "adapter": None, "renderer": "v3",
        "question": {"kind": "choice", "name": q.name, "text": q.text,
                     "options": CLASSES},
        "tau": TAU, "seed": SEED, "variant": best_name,
        "n_synth": len(synth), "report": report,
        "date": time.strftime("%Y-%m-%d"),
        "artifact": art,
    }
    path.write_text(json.dumps(out, indent=1))
    report_path.write_text(json.dumps(report, indent=1))
    print(f"saved {path} (variant {best_name}); measurement -> "
          f"{report_path}")


if __name__ == "__main__":
    main()
