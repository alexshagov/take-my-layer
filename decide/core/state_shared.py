"""State-level prefix sharing for L0 readout (the long-context latency fix —
docs/RESULTS.md).

anyjev's L0 splits each question's prompt at [state+question | options], so
the LONG state block is re-prefilled once per question per rotation — on a
5-question request that is ~15-20 full state reads. This module splits at
[state | question+options] instead: ONE state forward (prompt cache), then
every (question, rotation) runs as a short suffix from a copy of that cache
(MLXBackend.score_shared_pairs).

Same L0 math as anyjev's Decider._run L0 branch, assembled from the same
building blocks (softmax -> batch prior at the same accumulation points ->
log-mean marginalization), so answers track the served pipeline; the only
numeric difference is the bf16 kernel boundary at the new split point (the
same class of difference the existing score_shared boundary documents:
0/12 argmax flips, conf delta <= 0.02). L2-routed questions (the registry
head layout) do not pass through here — the caller sends them to the
ordinary Decider path (hidden-state early exit, no rotations).

Private-attr note: touches dec._running / dec._labels_for (anyjev 0.2.0,
pinned) so the batch prior sees EXACTLY the history the old path would
have accumulated.

Two serving options sit on top (docs/RESULTS.md, "Speed/precision sweep"):
  * batched suffix passes: done inside the backend (PrefixScorer), default ON,
    the answers are the same up to bf16 kernel noise;
  * early stopping (`early_stop=EarlyStop(...)`, endpoint flag --early-stop,
    default OFF): read ONE rotation per Choice/Noul question first and the
    remaining rotations only for the questions whose prior-corrected confidence
    is below tau — two scoring rounds at most. It keeps its own per-rotation
    prior (a rotation only contributes when it was actually read), so it never
    touches the dec._running layout the full path uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyjev.calibrate.contextual import apply_contextual, batch_prior
from anyjev.calibrate.permute import cyclic_shifts, marginalize, spread_order
from anyjev.readout import PromptSpec, build_prompt, label_ids_for_perm, render_chat_parts
from anyjev.result import Decision
from anyjev.state import render_state


def _softmax(lp: np.ndarray) -> np.ndarray:
    z = np.asarray(lp, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _perms(q, max_permutations):
    """Same rotation sets Decider._perms(q, 'L0') produces."""
    if q.ordered:
        return [list(range(q.k))]
    if q.kind == "noul":
        return [[0, 1], [1, 0]]
    return cyclic_shifts(q.k, max_permutations)


def _split_at_state(spec: PromptSpec, state_text: str):
    """Candidate split indices right after the state block, best-first:
    prefer the suffix to start at 'Question:' (the state's blank separator
    stays in the prefix), then at the blank line, then at the newline."""
    base = len("State:\n" + state_text)
    return [base + 2, base + 1, base]


@dataclass(frozen=True)
class EarlyStop:
    """Per-type confidence thresholds for reading the remaining rotations. Defaults are the
    pair that left every answer of the 2 x 350 Jev-agreement questions unchanged offline
    (docs/RESULTS.md). Lower = fewer passes, more answers can move."""
    choice_tau: float = 0.9
    noul_tau: float = 0.8

    def __post_init__(self):
        for name in ("choice_tau", "noul_tau"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


def _plan(dec, be, state_text, questions):
    """Shared prefix text + one plan per question: its rotations and, per rotation, the
    (suffix_text, label_token_ids) pair — the same split and ids the old path used."""
    prefix = None
    plans = []
    for q in questions:
        labels, ids = dec._labels_for(q)            # noqa: SLF001 (0.2.0 pinned)
        perms = _perms(q, dec.max_permutations)     # noqa: SLF001
        cand = pre = None
        spec0 = build_prompt(state_text, q, perms[0], dec.system, labels)
        for c in _split_at_state(spec0, state_text):
            if c <= 0 or c >= len(spec0.user):
                continue
            parts = render_chat_parts(be.tokenizer,
                                      PromptSpec(spec0.system, spec0.user, c))
            if parts[1]:                            # sentinel survived the template
                cand, pre = c, parts[0]
                break
        if pre is None:
            raise ValueError(f"state split failed for question {q.id!r}")
        if prefix is None:
            prefix = pre
        pairs = []
        for perm in perms:
            s = build_prompt(state_text, q, perm, dec.system, labels)
            parts = render_chat_parts(be.tokenizer,
                                      PromptSpec(s.system, s.user, cand))
            pairs.append((parts[1], label_ids_for_perm(q, ids, perm)))
        plans.append(dict(q=q, perms=perms, pairs=pairs))
    return prefix, plans


def decide_state_shared(dec, be, state, questions, *, early_stop=None) -> dict:
    """{question.id: Decision} for every question, L0, one shared state
    forward. Falls back by raising — the caller should retry through the
    ordinary Decider path. `early_stop`: an EarlyStop, or None for every rotation."""
    if dec.prior not in ("batch", "none"):
        raise ValueError(f"state-share serves prior='batch' (got {dec.prior!r})")
    state_text = render_state(state)
    prefix, plans = _plan(dec, be, state_text, questions)
    if early_stop is not None and callable(getattr(be, "prefix_scorer", None)):
        return _decide_early_stop(dec, be, prefix, plans, early_stop)
    return _decide_full(dec, be, prefix, plans)


def _decide_full(dec, be, prefix, plans) -> dict:
    all_pairs = [pair for pl in plans for pair in pl["pairs"]]
    lps = be.score_shared_pairs(prefix, all_pairs)
    out: dict = {}
    pos = 0
    for pl in plans:
        q, perms = pl["q"], pl["perms"]
        by_q = lps[pos:pos + len(perms)]
        pos += len(perms)
        p_pos_raw = np.stack([_softmax(lp) for lp in by_q])         # [P, K]
        stack = p_pos_raw[None]                                     # [N=1, P, K]
        sums, n = dec._running.get(q.key, (np.zeros(stack.shape[1:]), 0))
        dec._running[q.key] = (sums + stack.sum(axis=0), n + 1)     # noqa: SLF001
        prior = dec.running_prior(q) if dec.prior == "batch" else None
        strength = dec.strength()
        p_pos = (apply_contextual(p_pos_raw, np.power(prior, strength))
                 if prior is not None else p_pos_raw)
        probs = marginalize(p_pos, perms, dec.combine)
        mass = float(np.exp(np.stack(by_q)).sum(axis=1).mean())
        diag = {"answer_mass": mass, "permutations": len(perms), "perms": perms,
                "p_pos_raw": p_pos_raw,
                "prior_method": dec.prior if prior is not None else "none",
                "prior_strength": strength if prior is not None else 0.0,
                "state_shared": True}
        out[q.id] = Decision(q, np.asarray(probs), "L0", diag)
    return out


def _read_order(q, n_rot):
    """Rotation indices in reading order. Choice: shift 0, K/2, K/4, ... (anyjev's spread
    order, so any prefix of it is a well-separated subset); Noul: the two phrasings; Score
    has one pass."""
    if q.ordered or n_rot == 1:
        return list(range(n_rot))
    if q.kind == "noul":
        return list(range(n_rot))
    return spread_order(n_rot)


def _decide_early_stop(dec, be, prefix, plans, es: EarlyStop) -> dict:
    scorer = be.prefix_scorer(prefix)
    running = dec.__dict__.setdefault("_early_running", {})      # (q.key, rotation) -> (sum p, n)
    strength = dec.strength()
    lp: list[dict] = [dict() for _ in plans]                     # rotation -> label log-probs
    raw: list[dict] = [dict() for _ in plans]                    # rotation -> softmax over labels
    order = [_read_order(pl["q"], len(pl["perms"])) for pl in plans]
    first = [1 if (pl["q"].kind in ("choice", "noul") and not pl["q"].ordered and len(o) > 1)
             else len(o) for pl, o in zip(plans, order)]

    def score_round(todo):
        for (qi, ri), v in zip(todo, scorer.score([plans[qi]["pairs"][ri] for qi, ri in todo])):
            lp[qi][ri] = v
            raw[qi][ri] = _softmax(v)

    def posterior(qi, reads):
        """Prior-corrected, marginalized distribution over `reads`. As in the full path the
        item's own raw read is part of the running prior it is corrected with."""
        q = plans[qi]["q"]
        pp, used = [], False
        for ri in reads:
            s, n = running.get((q.key, ri), (0.0, 0))
            s, n = s + raw[qi][ri], n + 1
            if dec.prior == "batch" and n >= dec.min_prior_n:
                pp.append(apply_contextual(raw[qi][ri], np.power(batch_prior((s / n)[None]), strength)))
                used = True
            else:
                pp.append(raw[qi][ri])
        return marginalize(np.array(pp), [plans[qi]["perms"][ri] for ri in reads], dec.combine), used

    score_round([(qi, ri) for qi in range(len(plans)) for ri in order[qi][:first[qi]]])
    reads = [list(order[qi][:first[qi]]) for qi in range(len(plans))]
    todo = []
    for qi, pl in enumerate(plans):
        if first[qi] < len(order[qi]):
            tau = es.noul_tau if pl["q"].kind == "noul" else es.choice_tau
            if float(posterior(qi, reads[qi])[0].max()) < tau:
                todo += [(qi, ri) for ri in order[qi][first[qi]:]]
                reads[qi] = list(order[qi])
    if todo:
        score_round(todo)

    out: dict = {}
    for qi, pl in enumerate(plans):
        q = pl["q"]
        probs, used = posterior(qi, reads[qi])
        for ri in reads[qi]:                                     # the item joins the prior once
            s, n = running.get((q.key, ri), (0.0, 0))
            running[(q.key, ri)] = (s + raw[qi][ri], n + 1)
        mass = float(np.mean([np.exp(lp[qi][ri]).sum() for ri in reads[qi]]))
        diag = {"answer_mass": mass, "permutations": len(reads[qi]),
                "rotations_available": len(pl["perms"]),
                "perms": [pl["perms"][ri] for ri in reads[qi]],
                "p_pos_raw": np.stack([raw[qi][ri] for ri in reads[qi]]),
                "prior_method": dec.prior if used else "none",
                "prior_strength": strength if used else 0.0,
                "state_shared": True, "early_stop": True}
        out[q.id] = Decision(q, np.asarray(probs), "L0", diag)
    return out
