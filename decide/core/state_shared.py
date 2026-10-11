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
"""
from __future__ import annotations

import numpy as np

from anyjev.calibrate.contextual import apply_contextual, batch_prior
from anyjev.calibrate.permute import cyclic_shifts, marginalize
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


def decide_state_shared(dec, be, state, questions) -> dict:
    """{question.id: Decision} for every question, L0, one shared state
    forward. Falls back by raising — the caller should retry through the
    ordinary Decider path."""
    if dec.prior not in ("batch", "none"):
        raise ValueError(f"state-share serves prior='batch' (got {dec.prior!r})")
    state_text = render_state(state)

    prefix = None
    all_pairs: list = []   # (suffix_text, label_token_ids) across ALL questions
    plan: list = []        # (qi, perm_idx) in the same order as all_pairs
    for qi, q in enumerate(questions):
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
        for pi, perm in enumerate(perms):
            s = build_prompt(state_text, q, perm, dec.system, labels)
            parts = render_chat_parts(be.tokenizer,
                                      PromptSpec(s.system, s.user, cand))
            all_pairs.append((parts[1], label_ids_for_perm(q, ids, perm)))
            plan.append((qi, pi))

    lps = be.score_shared_pairs(prefix, all_pairs)
    by_q: list = [[None] * len(_perms(q, dec.max_permutations))
                  for q in questions]
    for (qi, pi), lp in zip(plan, lps):
        by_q[qi][pi] = lp

    out: dict = {}
    for qi, q in enumerate(questions):
        p_pos_raw = np.stack([_softmax(lp) for lp in by_q[qi]])     # [P, K]
        perms = _perms(q, dec.max_permutations)
        stack = p_pos_raw[None]                                     # [N=1, P, K]
        sums, n = dec._running.get(q.key, (np.zeros(stack.shape[1:]), 0))
        dec._running[q.key] = (sums + stack.sum(axis=0), n + 1)     # noqa: SLF001
        prior = dec.running_prior(q) if dec.prior == "batch" else None
        strength = dec.strength()
        p_pos = (apply_contextual(p_pos_raw, np.power(prior, strength))
                 if prior is not None else p_pos_raw)
        probs = marginalize(p_pos, perms, dec.combine)
        mass = float(np.exp(np.stack(by_q[qi])).sum(axis=1).mean())
        diag = {"answer_mass": mass, "permutations": len(perms), "perms": perms,
                "p_pos_raw": p_pos_raw,
                "prior_method": dec.prior if prior is not None else "none",
                "prior_strength": strength if prior is not None else 0.0,
                "state_shared": True}
        out[q.id] = Decision(q, np.asarray(probs), "L0", diag)
    return out
