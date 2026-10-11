"""Batched suffix passes and early stopping, without a model.

The byte-budget arithmetic is pure; the decision logic runs through the real
anyjev Decider and decide_state_shared against a stub backend whose label
log-probs are a known function of the prompt, so what is checked is the reading
policy (how many rotations are read, in how many rounds, with which prior), not
a language model. The batched forward itself is checked against the sequential
one on the real model by decide/bench/bench_fast_path.py.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anyjev import Decider, Question  # noqa: E402
from anyjev.backends.fake import FakeTokenizer  # noqa: E402

from decide.core import batching  # noqa: E402
from decide.core.state_shared import EarlyStop, decide_state_shared  # noqa: E402
from decide.serve import systemone as S  # noqa: E402


class Budget(unittest.TestCase):
    def test_row_bytes_counts_kv_with_slack_and_state_twice(self):
        # 8 attention layers x 2 (k,v) x 4 heads x 256 dim x 2 bytes = 32 KiB per token
        kv = 32 * 1024
        self.assertEqual(batching.per_row_bytes(kv, 1000, 256, 5_000_000),
                         kv * (1000 + 512) + 10_000_000)

    def test_rows_respect_the_budget_the_pair_count_and_the_cap(self):
        row = 100 * 2 ** 20
        self.assertEqual(batching.rows_per_forward(1 * 2 ** 30, row, 50), 10)     # budget-bound
        self.assertEqual(batching.rows_per_forward(1 * 2 ** 30, row, 4), 4)       # pair-bound
        self.assertEqual(batching.rows_per_forward(10 ** 15, 1, 500), batching.MAX_ROWS)

    def test_no_batching_when_two_rows_do_not_fit_or_there_is_one_pair(self):
        row = 800 * 2 ** 20
        self.assertEqual(batching.rows_per_forward(1 * 2 ** 30, row, 11), 0)      # 1 row fits
        self.assertEqual(batching.rows_per_forward(10 ** 12, 1, 1), 0)
        self.assertEqual(batching.rows_per_forward(10 ** 12, 0, 5), 0)

    def test_chunks_are_balanced_and_cover_everything(self):
        for n, rows in [(10, 9), (11, 11), (64, 8), (5, 2), (1, 1)]:
            sizes = batching.chunk_sizes(n, rows)
            self.assertEqual(sum(sizes), n)
            self.assertLessEqual(max(sizes), rows)
            self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(batching.chunk_sizes(10, 9), [5, 5])
        self.assertEqual(batching.chunk_sizes(0, 4), [])


class StubBackend:
    """Label log-probs as a known function of the suffix text: an option whose text starts
    with 'sure' gets logit 4, a yes/no question containing 'SURE' gets Yes logit 3, everything
    else 0; the first listed label always gets +0.5 (position bias); answer mass is 0.905."""
    name = "stub"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.rounds: list[int] = []          # size of each scoring call

    def _lp(self, suffix, ids):
        labels = [self.tokenizer.id_to_label(i) for i in ids]
        if "Answer Yes or No." in suffix or "Answer No or Yes." in suffix:
            logits = [3.0 if ("SURE" in suffix and lab == "Yes") else 0.0 for lab in labels]
        else:
            opts = re.findall(r"(?m)^(?:[A-Z]|\d)\. (.*)$", suffix)
            logits = [4.0 if t.startswith("sure") else 0.0 for t in opts]
        logits[0] += 0.5
        z = np.array(logits)
        return z - np.log(np.exp(z).sum()) - 0.1

    def score_shared_pairs(self, prefix, pairs):
        self.rounds.append(len(pairs))
        return [self._lp(s, ids) for s, ids in pairs]

    def prefix_scorer(self, prefix):
        be = self

        class Scorer:
            def score(self, pairs):
                return be.score_shared_pairs(prefix, pairs)
        return Scorer()


class NoScorerBackend(StubBackend):
    prefix_scorer = None                     # an older backend: only score_shared_pairs


def questions():
    return [Question.choice("Which?", ["sure-a", "b", "c"], name="easy"),
            Question.choice("Which one?", ["x", "y", "z"], name="hard"),
            Question.noul("SURE it is?", name="yes_sure"),
            Question.noul("Is it?", name="yes_unsure"),
            Question.score("How bad?", levels=["low", "mid", "high"], name="level")]


def decider(be, **kw):
    return Decider(be, level="L0", shared_prefix=True, **kw)


class FullPath(unittest.TestCase):
    def test_one_round_with_every_rotation(self):
        be = StubBackend()
        out = decide_state_shared(decider(be), be, "the state", questions())
        self.assertEqual(be.rounds, [3 + 3 + 2 + 2 + 1])
        self.assertEqual({k: d.diagnostics["permutations"] for k, d in out.items()},
                         {"easy": 3, "hard": 3, "yes_sure": 2, "yes_unsure": 2, "level": 1})
        self.assertEqual(out["easy"].argmax, "sure-a")
        self.assertTrue(out["yes_sure"].answer)
        self.assertAlmostEqual(out["easy"].diagnostics["answer_mass"], np.exp(-0.1), places=9)

    def test_early_stop_is_ignored_by_a_backend_without_a_scorer(self):
        be = NoScorerBackend()
        out = decide_state_shared(decider(be), be, "s", questions(), early_stop=EarlyStop())
        self.assertEqual(be.rounds, [11])
        self.assertTrue(all("early_stop" not in d.diagnostics for d in out.values()))


class EarlyStopPath(unittest.TestCase):
    def test_confident_questions_read_once_unsure_ones_read_everything(self):
        be = StubBackend()
        out = decide_state_shared(decider(be), be, "the state", questions(), early_stop=EarlyStop())
        reads = {k: d.diagnostics["permutations"] for k, d in out.items()}
        self.assertEqual(reads, {"easy": 1, "hard": 3, "yes_sure": 1, "yes_unsure": 2, "level": 1})
        # round 1: one read per question; round 2: the missing rotations of the unsure ones
        self.assertEqual(be.rounds, [5, 2 + 1])
        self.assertTrue(all(d.diagnostics["early_stop"] for d in out.values()))
        self.assertEqual(out["hard"].diagnostics["rotations_available"], 3)

    def test_a_request_where_everything_is_confident_takes_a_single_round(self):
        be = StubBackend()
        qs = [Question.choice("Which?", ["sure-a", "b", "c"], name="a"),
              Question.noul("SURE it is?", name="n")]
        decide_state_shared(decider(be), be, "s", qs, early_stop=EarlyStop())
        self.assertEqual(be.rounds, [2])

    def test_thresholds_decide_who_reads_more(self):
        qs = lambda: [Question.choice("Which?", ["sure-a", "b", "c"], name="a")]      # top prob 0.94
        be = StubBackend()
        decide_state_shared(decider(be), be, "s", qs(), early_stop=EarlyStop(choice_tau=0.9))
        self.assertEqual(be.rounds, [1])
        be = StubBackend()
        decide_state_shared(decider(be), be, "s", qs(), early_stop=EarlyStop(choice_tau=0.99))
        self.assertEqual(be.rounds, [1, 2])

    def test_answers_match_the_full_path_where_the_read_is_unambiguous(self):
        be1, be2 = StubBackend(), StubBackend()
        full = decide_state_shared(decider(be1), be1, "s", questions())
        fast = decide_state_shared(decider(be2), be2, "s", questions(), early_stop=EarlyStop())
        for k in full:
            self.assertEqual(full[k].argmax, fast[k].argmax, k)
            self.assertAlmostEqual(sum(fast[k].distribution.values()), 1.0)

    def test_the_prior_activates_per_rotation_after_enough_requests(self):
        be = StubBackend()
        dec = decider(be)
        methods = []
        for _ in range(10):
            out = decide_state_shared(dec, be, "s", [Question.choice("Which one?", ["x", "y", "z"], name="hard")],
                                      early_stop=EarlyStop())
            methods.append(out["hard"].diagnostics["prior_method"])
        self.assertEqual(methods[:7], ["none"] * 7)            # min_prior_n = 8, the 8th request joins it
        self.assertEqual(methods[-1], "batch")
        # it never touches the layout the full path keeps
        self.assertEqual(dec._running, {})

    def test_the_full_path_still_keeps_the_original_prior_layout(self):
        be = StubBackend()
        dec = decider(be)
        q = Question.choice("Which one?", ["x", "y", "z"], name="hard")
        for _ in range(3):
            decide_state_shared(dec, be, "s", [q])
        sums, n = dec._running[q.key]
        self.assertEqual((sums.shape, n), ((3, 3), 3))

    def test_thresholds_are_validated(self):
        for kw in ({"choice_tau": 0.0}, {"noul_tau": 1.5}, {"choice_tau": -1}):
            with self.assertRaises(ValueError):
                EarlyStop(**kw)
        EarlyStop(choice_tau=1.0)                              # 1.0 = never stop early


class EndpointFlags(unittest.TestCase):
    def test_defaults_batch_on_early_stop_off(self):
        args = S.make_parser().parse_args([])
        self.assertFalse(args.no_batch)
        self.assertFalse(args.early_stop)
        self.assertEqual(args.kv_budget_gib, 1.5)
        self.assertEqual((args.early_stop_choice_tau, args.early_stop_noul_tau), (0.9, 0.8))
        self.assertIsNone(S.EARLY_STOP)
        self.assertTrue(S.BATCH_SUFFIXES)
        self.assertEqual(S.serving_config(), {"state_shared": True, "batched_suffixes": True,
                                              "kv_budget_gib": 1.5, "early_stop": None})

    def test_flags_parse(self):
        args = S.make_parser().parse_args(["--no-batch", "--early-stop", "--kv-budget-gib", "3",
                                           "--early-stop-noul-tau", "0.7"])
        self.assertTrue(args.no_batch and args.early_stop)
        self.assertEqual((args.kv_budget_gib, args.early_stop_noul_tau), (3.0, 0.7))


if __name__ == "__main__":
    unittest.main()
