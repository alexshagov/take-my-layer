"""Row budgeting for batched suffix passes (no MLX import, so it is unit-testable anywhere).

A batched pass broadcasts the shared prefix cache to B rows: every full-attention layer's
K/V is materialised B times (its capacity is the prefix rounded up to the cache step, plus
one more step when the suffix overflows it) and every recurrent layer's state is copied B
times (input and output). That is the only memory the batching adds, so the number of rows
per forward is capped by a byte budget instead of a fixed constant.

Measured (4B, docs/RESULTS.md "Speed/precision sweep"): a 9k-token prefix over 11-14 rows
peaks at 6.8-7.4 GiB against 3.9 GiB for the sequential path, and gains only ~6% there, so a
prefix that leaves room for fewer than MIN_ROWS rows is scored sequentially.
"""
from __future__ import annotations

MIN_ROWS = 2            # below this a batched forward has nothing to amortise
MAX_ROWS = 64           # also bounds the [rows, vocab] float32 logits of the last positions
DEFAULT_BUDGET_BYTES = int(1.5 * 2 ** 30)


def per_row_bytes(kv_bytes_per_token: int, prefix_tokens: int, cache_step: int,
                  state_bytes: int) -> int:
    """Bytes one extra row costs: K/V for (prefix + two cache steps of slack) tokens, and the
    recurrent state twice (the broadcast input and the updated output)."""
    return kv_bytes_per_token * (prefix_tokens + 2 * cache_step) + 2 * state_bytes


def rows_per_forward(budget_bytes: int, row_bytes: int, n_pairs: int) -> int:
    """How many suffixes to run in one batched forward. 0 means 'do not batch' (the budget
    cannot hold MIN_ROWS rows, or there are fewer than MIN_ROWS suffixes)."""
    if n_pairs < MIN_ROWS or row_bytes <= 0:
        return 0
    rows = min(MAX_ROWS, n_pairs, int(budget_bytes // row_bytes))
    return rows if rows >= MIN_ROWS else 0


def chunk_sizes(n_pairs: int, rows: int) -> list[int]:
    """Split n_pairs into near-equal chunks of at most `rows` (a 9-row budget over 10 pairs
    runs 5 + 5, not 9 + 1: the lone row would pay a whole forward's fixed cost)."""
    if rows <= 0 or n_pairs <= 0:
        return []
    n_chunks = -(-n_pairs // rows)
    base, extra = divmod(n_pairs, n_chunks)
    return [base + (1 if i < extra else 0) for i in range(n_chunks)]
