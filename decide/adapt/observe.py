"""observe() replay — the adaptation engine on shadow telemetry.

Replays an observe log (the proxy writes one row after every shadow
decision: the FULL rendered state + the coarse label of the tool the model
actually called) through Decider.observe(): the first head is solved at
30 labels and re-solved at 60/120/... (refit_factor=2). The default is the
frozen results/observe_log.jsonl; live proxy traffic lands in the gitignored
logs/observe_log.jsonl (pass it via --log). Each refit is reported
and the last head is written to the registry as next_move_observed.json — the
seed next_move.json is never overwritten; promote deliberately (cp, or run the
proxy with --head next_move_observed.json).

Memory discipline (docs/POLICY.md — 32GB): this is a supervised, short-lived
process — the heavy refit NEVER runs inside the live proxy worker.

Run:  nix develop -c bash -c 'export HF_HOME=$PWD/.hf; python decide/adapt/observe.py'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anyjev import Decider, Question
from anyjev.heads import LinearHead
from anyjev.readout import build_prompt, render_chat_parts, resolve_labels

from decide.core.labels import CLASSES
from decide.core.mlx_backend import MLXBackend

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
REGDIR = ROOT / "registry" / MODEL.replace("/", "__") / "base"
SEED = REGDIR / "next_move.json"
OBSERVE_LOG = ROOT / "results/observe_log.jsonl"
# cheap-depth candidates only, same policy as fit_registry_head.py
LAYERS = [16, 19, 22]
FIT_AT = 30


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows, seen = [], set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label") not in CLASSES or not r.get("state"):
                continue
            key = (r.get("ts"), r.get("repo"), r.get("tool"))
            if key in seen:                      # proxy retries/replays dedupe
                continue
            seen.add(key)
            rows.append(r)
    rows.sort(key=lambda r: r.get("ts") or 0.0)   # replay in arrival order
    return rows[:limit] if limit else rows


def _seed_oof(seed: dict) -> float:
    """v2 registry files carry 'oof'; v3 carries 'report' (A/B protocol)."""
    if "oof" in seed:
        return float(seed["oof"].get("oof_acc", float("nan")))
    rep = seed.get("report") or {}
    return float(rep.get(seed.get("variant", ""), {}).get("holdout_acc",
                                                          float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="replay only the first N rows (chrono order)")
    ap.add_argument("--log", default=str(OBSERVE_LOG),
                    help="observe log to replay (default: the frozen results/ copy; "
                         "live proxy traffic is logs/observe_log.jsonl)")
    ap.add_argument("--out", default=str(REGDIR / "next_move_observed.json"),
                    help="where to write the refit head")
    args = ap.parse_args()

    rows = load_rows(Path(args.log), args.limit)
    n = len(rows)
    if n < FIT_AT:
        print(f"{n} usable observe rows (< {FIT_AT}) — nothing to solve yet; "
              f"keep the shadow proxy running.")
        return

    seed = json.loads(SEED.read_text())
    q = Question.choice(seed["question"]["text"], seed["question"]["options"],
                        name=seed["question"]["name"])
    dec = Decider(MLXBackend(MODEL), level="auto")
    dec.load_artifact(q, seed["artifact"])        # baseline head, for comparison
    seed_head = LinearHead.from_dict(seed["artifact"])
    layer_seed = int(seed["artifact"]["layer_abs"])
    labels, label_ids = resolve_labels(dec.backend.tokenizer, q)

    art = None
    for i, r in enumerate(rows, 1):
        new = dec.observe(q, r["state"], CLASSES.index(r["label"]),
                          fit_at=FIT_AT, refit_factor=2.0, layers=LAYERS)
        if new is not None:
            art = new
            print(f"refit at n={i}: layer {new['layer_abs']} "
                  f"oof_acc={new['cv'].get('oof_acc', float('nan')):.3f}")

    out = {
        "model": MODEL, "adapter": None, "renderer": "v2",
        "question": seed["question"],
        "tau": seed["tau"], "source": "observe",
        "n_observed": n, "seed": SEED.name,
        "oof": art["cv"], "date": time.strftime("%Y-%m-%d"),
        "artifact": art,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"observed head: layer {art['layer_abs']}/{dec.backend.n_layers} "
          f"oof_acc={art['cv'].get('oof_acc', float('nan')):.3f} "
          f"(seed {_seed_oof(seed):.3f}) "
          f"n_calib={art.get('n_calib', n)}")
    print(f"saved {out_path}")

    # telemetry sanity, NOT the OOF number above: hits of both heads on the
    # last <=100 observed rows (in-sample for the observed head). The observed
    # head serves from the decider at its own layer; the seed head is read at
    # its layer through the same single-forward path the proxy guard uses.
    sample = rows[-100:]
    new_hits = seed_hits = 0
    for r in sample:
        y = r["label"]
        d = dec.decide(r["state"], [q], level="L2")[q.name]
        new_hits += int(d.argmax == y)
        spec = build_prompt(r["state"], q, list(range(q.k)), dec.system, labels)
        pre, suf = render_chat_parts(dec.backend.tokenizer, spec)
        feats, _, _, _ = dec.backend.hidden_states_to([pre + suf], [layer_seed],
                                                      max_layer=layer_seed)
        seed_hits += int(CLASSES[int(seed_head.probs(feats[:, 0])[0].argmax())] == y)
    print(f"hits on last {len(sample)} observed rows: observed head "
          f"{new_hits}, seed head {seed_hits}")


if __name__ == "__main__":
    main()
