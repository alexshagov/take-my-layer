"""MLX backend for AnyJev (docs/DECIDE.md §5.1).

Implements the Backend protocol (anyjev/backends/base.py): a backend does
exactly one thing — next-token log-probs of requested token ids from the
full-vocabulary log-softmax at the last position.

raw / L0 / L1 via next_token_logprobs. L2 via hidden_states / hidden_states_to
(the early-stop block loop): we own the layer loop in MLX, so the GDN hybrid
is just layers in a list (AnyJev's HF block-loop balks at exotic
architectures; here there is no such constraint).

Layer indexing follows the HF backend convention exactly (anyjev/backends/hf.py):
0 = embedding output, i (1..n-1) = output of block i (0-indexed block i-1),
n_layers = after the FINAL NORM (applied even when stopped early, logit-lens
style, so artifacts stay portable across backends). Negative indices convert
as n_blocks + 1 + i.

score_shared (prefix-KV fast path) IS implemented for the full-model path
(make_prompt_cache + per-suffix cache copy): the L0 permutations of one
question share the state+question prefix; the hidden-state paths
(hidden_states / hidden_states_to) do not share.

Parity reference: anyjev/backends/hf.py::HFBackend.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


class MLXBackend:
    def __init__(self, model_path: str, *, adapter_path: str | None = None):
        self.model, self.tokenizer = mlx_load(model_path, adapter_path=adapter_path)
        self.name = model_path
        # mlx_lm model classes carry the HF-style config on `.args` (dict-like
        # ModelArgs; hybrids like qwen3_5 nest the text config) and expose
        # decoder blocks directly as `.layers`.
        cfg = getattr(self.model, "args", {}) or {}
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        self.n_layers = int(get("num_hidden_layers") or len(self.model.layers))
        self.hidden_size = int(get("hidden_size") or 0)
        # prefix-KV fast-path counters (mirrors anyjev's HFBackend)
        self.shared_groups = 0
        self.shared_prompts = 0
        self.shared_fallbacks = 0

    # ---- shared internals -------------------------------------------------

    def _inner(self):
        """embed_tokens / layers / final norm (qwen3_5: Model.language_model.model)."""
        m = self.model
        for attr in ("language_model", "model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                if hasattr(m, "model"):
                    m = m.model
                break
        return m

    def _forward_blocks(self, prompt_ids: Sequence[int], blocks: int,
                        capture: Sequence[int]):
        """Embeddings -> first `blocks` decoder blocks; return last-position
        hidden states for the requested absolute layer indices (HF convention).
        `blocks` runs 0-indexed blocks 0..blocks-1; layer n_layers applies the
        final norm on top of whatever the last executed block produced."""
        inner = self._inner()
        h = inner.embed_tokens(mx.array(list(prompt_ids))[None])    # [1, T, H]
        want = set(int(x) for x in capture)
        out: dict[int, mx.array] = {}
        if 0 in want:
            out[0] = h[:, -1, :].astype(mx.float32)
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)
        for i in range(min(blocks, len(inner.layers))):
            layer = inner.layers[i]
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=None)
            if (i + 1) in want:
                out[i + 1] = h[:, -1, :].astype(mx.float32)
        if self.n_layers in want:
            out[self.n_layers] = inner.norm(h)[:, -1, :].astype(mx.float32)
        missing = want - set(out)
        if missing:
            raise ValueError(f"layers {sorted(missing)} beyond executed blocks "
                             f"(ran 0..{blocks - 1})")
        return out

    def _logits_from_last_hidden(self, last_hidden_1h: mx.array) -> mx.array:
        """Project a [1, H] last-position hidden state through the output head."""
        lm = self.model.language_model
        if getattr(lm.args, "tie_word_embeddings", False):
            return lm.model.embed_tokens.as_linear(last_hidden_1h)
        return lm.lm_head(last_hidden_1h)

    # ---- protocol: raw / L0 / L1 ------------------------------------------

    def _last_position_logprobs(self, prompt_ids: Sequence[int]) -> np.ndarray:
        logits = self.model(mx.array(list(prompt_ids))[None])       # [1, T, V]
        last = logits[:, -1, :].astype(mx.float32)[0]               # [V]
        lsm = last - mx.logsumexp(last)                             # full-vocab log-softmax
        return np.asarray(lsm, dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for text, ids in zip(prompts, token_ids):
            prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)
            lp = self._last_position_logprobs(prompt_ids)
            out.append(lp[np.asarray(list(ids), dtype=np.int64)])
        return out

    # ---- protocol: prefix-KV fast path (score_shared) ----------------------

    def score_shared_pairs(self, prefix: str, pairs):
        """State-level prefix sharing across QUESTIONS (and their rotations):
        ONE forward of `prefix` (the state block + template head), then each
        (suffix_text, token_ids) pair runs from a copy of that cache — the
        multi-question fan-out fast path (docs/RESULTS.md, state-share). Same
        boundary-exactness contract and flat fallback as score_shared, per
        pair; returns one label-logprob array per pair, in order."""
        import copy as _copy

        from mlx_lm.models.cache import make_prompt_cache

        tok = self.tokenizer
        p_ids = tok.encode(prefix, add_special_tokens=False)
        sfx = [(s, tok.encode(s, add_special_tokens=False), list(ids))
               for s, ids in pairs]
        fallback = [i for i, (s, s_ids, _) in enumerate(sfx)
                    if not s_ids or tok.encode(prefix + s, add_special_tokens=False)
                    != p_ids + s_ids]
        results: list = [None] * len(sfx)
        if len(fallback) < len(sfx):
            cache = make_prompt_cache(self.model)
            self.model(mx.array(p_ids)[None], cache=cache)
            for i, (s, s_ids, ids) in enumerate(sfx):
                if i in fallback:
                    continue
                c = _copy.deepcopy(cache)     # arrays are immutable; refs only
                logits = self.model(mx.array(s_ids)[None], cache=c)
                last = logits[:, -1, :].astype(mx.float32)[0]
                lsm = np.asarray(last - mx.logsumexp(last), dtype=np.float64)
                results[i] = lsm[np.asarray(ids, dtype=np.int64)]
            self.shared_groups += 1
            self.shared_prompts += len(sfx) - len(fallback)
        if fallback:
            lps = self.next_token_logprobs(
                [prefix + sfx[i][0] for i in fallback],
                [sfx[i][2] for i in fallback])
            for i, lp in zip(fallback, lps):
                results[i] = lp
            self.shared_fallbacks += len(fallback)
        return results

    def score_suffix_sum(self, prefix: str, suffixes):
        """Multi-token loglikelihood scoring on a shared prefix (the
        lm-evaluation-harness standard, and the >26-option unlock): ONE
        forward of `prefix`, then each suffix runs from a copy of that cache
        and we return SUM log-softmax over EVERY suffix token, i.e.
        log p(suffix | prefix). The FIRST suffix token's logprob comes from
        the prefix forward's last-position logits (which score_shared_pairs
        reads labels from but otherwise discards); tokens t >= 2 come from
        the suffix forward's row t-2. Same boundary-exactness check and
        flat fallback as score_shared_pairs. Returns one float per suffix."""
        import copy as _copy

        from mlx_lm.models.cache import make_prompt_cache

        tok = self.tokenizer
        p_ids = tok.encode(prefix, add_special_tokens=False)
        sfx = [(s, tok.encode(s, add_special_tokens=False)) for s in suffixes]
        fallback = [i for i, (s, s_ids) in enumerate(sfx)
                    if not s_ids or tok.encode(prefix + s, add_special_tokens=False)
                    != p_ids + s_ids]
        results: list = [None] * len(sfx)
        if len(fallback) < len(sfx):
            cache = make_prompt_cache(self.model)
            logits = self.model(mx.array(p_ids)[None], cache=cache)
            last = logits[:, -1, :].astype(mx.float32)[0]           # [V]
            lsm_first = np.asarray(last - mx.logsumexp(last), dtype=np.float64)
            for i, (s, s_ids) in enumerate(sfx):
                if i in fallback:
                    continue
                c = _copy.deepcopy(cache)     # arrays are immutable; refs only
                lg = self.model(mx.array(s_ids)[None], cache=c)[0].astype(mx.float32)
                if len(s_ids) > 1:
                    rows = lg[:-1]                                 # [T-1, V]
                    lsm = np.asarray(rows - mx.logsumexp(rows, axis=-1)[:, None],
                                     dtype=np.float64)
                    pick = lsm[np.arange(len(s_ids) - 1), s_ids[1:]]
                    results[i] = float(lsm_first[s_ids[0]]) + float(pick.sum())
                else:
                    results[i] = float(lsm_first[s_ids[0]])
            self.shared_groups += 1
            self.shared_prompts += len(sfx) - len(fallback)
        if fallback:
            for i in fallback:
                s, s_ids = sfx[i]
                all_ids = p_ids + s_ids
                lg = self.model(mx.array(all_ids)[None])[0].astype(mx.float32)
                rows = lg[len(p_ids) - 1: -1]                       # [T, V]
                lsm = np.asarray(rows - mx.logsumexp(rows, axis=-1)[:, None],
                                 dtype=np.float64)
                results[i] = float(lsm[np.arange(len(s_ids)), s_ids].sum())
            self.shared_fallbacks += len(fallback)
        return results

    def score_shared(self, groups, token_ids):
        """For each (prefix, [suffixes]) group: log p(t | prefix + suffix_j)
        for the group's token ids, running the prefix ONCE through the model
        with a prompt cache and each suffix from a copy of that cache (the
        multi-question fan-out fast path). Boundary-exactness is
        checked exactly like the HF backend (encode(prefix+suffix) ==
        encode(prefix)+encode(suffix)); groups that fail it — or have empty
        suffixes — fall back to flat scoring (counted in shared_fallbacks)."""
        import copy as _copy

        from mlx_lm.models.cache import make_prompt_cache

        tok = self.tokenizer
        results: List[List[Optional[np.ndarray]]] = [
            [None] * len(sfx) for _, sfx in groups]
        fallback: List[tuple] = []
        for gi, (prefix, suffixes) in enumerate(groups):
            p_ids = tok.encode(prefix, add_special_tokens=False)
            s_ids_list = [tok.encode(s, add_special_tokens=False)
                          for s in suffixes]
            exact = all(
                tok.encode(prefix + s, add_special_tokens=False) == p_ids + s_ids
                for s, s_ids in zip(suffixes, s_ids_list)) and s_ids_list
            if not exact or any(not s for s in s_ids_list):
                self.shared_fallbacks += 1
                fallback.extend((gi, si, prefix + s)
                                for si, s in enumerate(suffixes))
                continue
            cache = make_prompt_cache(self.model)
            self.model(mx.array(p_ids)[None], cache=cache)
            for si, s_ids in enumerate(s_ids_list):
                c = _copy.deepcopy(cache)     # arrays are immutable; refs only
                logits = self.model(mx.array(s_ids)[None], cache=c)
                last = logits[:, -1, :].astype(mx.float32)[0]
                lsm = np.asarray(last - mx.logsumexp(last), dtype=np.float64)
                results[gi][si] = lsm[
                    np.asarray(list(token_ids[gi]), dtype=np.int64)]
            self.shared_groups += 1
            self.shared_prompts += len(s_ids_list)
        if fallback:
            lps = self.next_token_logprobs([t for _, _, t in fallback],
                                           [token_ids[gi] for gi, _, _ in fallback])
            for (gi, si, _), lp in zip(fallback, lps):
                results[gi][si] = lp
        return results

    # ---- protocol: L2 ------------------------------------------------------

    def _norm_layers(self, layers: Optional[Sequence[int]]) -> List[int]:
        ls = list(range(self.n_layers + 1)) if layers is None else \
            [int(x) if x >= 0 else self.n_layers + 1 + int(x) for x in layers or []]
        return sorted(set(ls))

    def hidden_states_to(self, prompts: Sequence[str], layers: Sequence[int],
                         token_ids: Optional[Sequence[Sequence[int]]] = None,
                         positions: Optional[Sequence[int]] = None,
                         max_layer: Optional[int] = None,
                         lens_ids: Optional[Sequence[int]] = None,
                         ) -> Tuple[np.ndarray, List[Optional[np.ndarray]], None, None]:
        """Early-stop block loop. feats float32 [N, len(layers), H] at the last
        position; lps = label log-probs when token_ids given. positions /
        lens_ids unsupported here (bench-only features)."""
        if positions is not None or lens_ids is not None:
            raise NotImplementedError("positions/lens_ids not supported by MLX backend")
        ls = self._norm_layers(layers)
        stop = int(max_layer) if max_layer is not None else max(ls)
        feats, lps = [], []
        for k, text in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)
            cap = self._forward_blocks(prompt_ids, blocks=stop, capture=ls)
            feats.append(np.stack([np.asarray(cap[l])[0] for l in ls]))   # [L, H]
            if token_ids is not None:
                hid = mx.array(cap[ls[-1]])                          # [1, H]
                logits = self._logits_from_last_hidden(hid).astype(mx.float32)
                lp = np.asarray(logits[0] - mx.logsumexp(logits[0]),
                                dtype=np.float64)
                lps.append(lp[np.asarray(list(token_ids[k]), dtype=np.int64)])
            else:
                lps.append(None)
        return np.asarray(feats, dtype=np.float32), lps, None, None

    def hidden_states(self, prompts: Sequence[str], layers: Sequence[int],
                      token_ids: Optional[Sequence[Sequence[int]]] = None,
                      positions: Optional[Sequence[int]] = None,
                      ) -> Tuple[np.ndarray, List[Optional[np.ndarray]], None]:
        """Full forward (all blocks). Same contract as hidden_states_to."""
        if positions is not None:
            raise NotImplementedError("positions not supported by MLX backend")
        ls = self._norm_layers(layers)
        feats, lps = [], []
        for k, text in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)
            cap = self._forward_blocks(prompt_ids, blocks=self.n_layers, capture=ls)
            feats.append(np.stack([np.asarray(cap[l])[0] for l in ls]))
            if token_ids is not None:
                hid = mx.array(cap[ls[-1]])
                logits = self._logits_from_last_hidden(hid).astype(mx.float32)
                lp = np.asarray(logits[0] - mx.logsumexp(logits[0]),
                                dtype=np.float64)
                lps.append(lp[np.asarray(list(token_ids[k]), dtype=np.int64)])
            else:
                lps.append(None)
        return np.asarray(feats, dtype=np.float32), lps, None
