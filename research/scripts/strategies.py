"""
strategies.py — KV cache perplexity measurement strategies.

Three run functions, each taking lib as the first argument:
  - run_chunk_token_by_token   (Strategy B)
  - run_chunk_batch_prefill    (Strategy C / D)
  - run_structured             (structured JSONL mode)

Import lib and pointer types from llama_bindings; apply_kv_hook from parse_state.

K and V are quantized on independent schedules:
  - Per-channel quants (e.g. int4_ch): k_group_size=128 — scales computed over
    a group of tokens, so multiple tokens must accumulate before quantizing.
  - Per-token quants (e.g. int4_tok): v_group_size=1 — each token quantized
    immediately after decode for correct autoregressive error propagation.
  - Mixed (e.g. int4_ch:int4_tok): K every 128 tokens, V every 1 token.

Return values
-------------
All run_* functions return a RunResult namedtuple:
  log_probs  : list[float]          — log-prob of correct next token (for PPL)
  log_dists  : np.ndarray or None   — [n_decode, n_vocab] float16, full log-softmax
                                       distributions (only when return_log_dists=True)
  kl_divs    : list[float] or None  — per-token KL(base ∥ this) in nats
                                       (only when base_log_dists is provided)

KL divergence
-------------
Pass base_log_dists (np.ndarray [n_decode, n_vocab] float16 from a fp16 run) to
compute per-token KL(fp16 ∥ quant) on the fly without storing the quantized
distributions. Mean KL directly quantifies distribution shift from quantization.

KL(P ∥ Q) = Σ_v P(v) · (logP(v) − logQ(v))
where P = fp16 distribution, Q = quantized distribution.
"""

import ctypes
from collections import namedtuple
from typing import Optional

import numpy as np

import llama_bindings as llama


RunResult = namedtuple("RunResult", ["log_probs", "log_dists", "kl_divs", "top1s", "diags"])
RunResult.__new__.__defaults__ = (None, None, None, None)  # log_dists, kl_divs, top1s, diags default to None


def _kl_div(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL(P ∥ Q) in nats. Both inputs are log-softmax vectors (float32)."""
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _fire_hook(kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size):
    """Fire the hook when K or V (or both) have accumulated enough tokens.
    Returns updated (n_pending_k, n_pending_v).

    Pass 0 (not None) for the non-firing side: None means "quantize all cells"
    in apply_kv_hook (prefill semantics), while 0 means "skip".
    """
    do_k = n_pending_k >= k_group_size
    do_v = n_pending_v >= v_group_size
    if do_k or do_v:
        kv_hook(ctx,
                n_new_k=n_pending_k if do_k else 0,
                n_new_v=n_pending_v if do_v else 0)
        if do_k: n_pending_k = 0
        if do_v: n_pending_v = 0
    return n_pending_k, n_pending_v


def _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v):
    """Flush any remaining pending tokens at end of chunk.

    Pass 0 (not None) for the non-pending side: None means "quantize all cells"
    in apply_kv_hook (prefill semantics), while 0 means "skip".
    """
    do_k = n_pending_k > 0
    do_v = n_pending_v > 0
    if do_k or do_v:
        kv_hook(ctx,
                n_new_k=n_pending_k if do_k else 0,
                n_new_v=n_pending_v if do_v else 0)


def _collect_logits(lib, ctx, n_vocab, next_token,
                    t,                  # decode step index (into log_dists / base_log_dists)
                    log_probs,
                    log_dists_list,
                    kl_divs,
                    top1s_list,
                    base_log_dists,
                    return_log_dists,
                    diag_lists=None,    # dict{"H","p_max","self_surp"} or None
                    prev_top1=None):    # top1 from previous step, for self_surp
    """Read logits from context, compute log-softmax, update all collectors.

    Returns current top1 (int or None) so caller can pass it as prev_top1 next step.

    diag_lists: if not None, appends per-step diagnostics that need no ground truth:
      H         — output entropy (nats); low = confident, high = confused
      p_max     — probability of top-1 prediction; proxy for model confidence
      self_surp — log-prob of prev_top1 under this step's distribution; measures
                  whether the model's own previous prediction is still coherent
                  (NaN for the first decode step)
    """
    ptr = lib.llama_get_logits_ith(ctx, 0)
    logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
    log_q = llama.log_softmax(logits.astype(np.float32))

    log_probs.append(float(log_q[next_token]))

    if return_log_dists:
        log_dists_list.append(log_q.astype(np.float16))

    if base_log_dists is not None:
        log_p = base_log_dists[t].astype(np.float32)
        kl_divs.append(_kl_div(log_p, log_q))

    top1 = None
    if top1s_list is not None or diag_lists is not None:
        top1 = int(np.argmax(log_q))
        if top1s_list is not None:
            top1s_list.append(top1)

    if diag_lists is not None:
        p     = np.exp(log_q)
        H     = -float(np.sum(p * log_q))
        p_max = float(p[top1])
        ss    = float(log_q[prev_top1]) if prev_top1 is not None else float("nan")
        diag_lists["H"].append(H)
        diag_lists["p_max"].append(p_max)
        diag_lists["self_surp"].append(ss)

    return top1


# ── KV state save / restore ──────────────────────────────────────────────────

def save_kv_state(lib, ctx, seq_id=0) -> bytes:
    """Snapshot the KV cache for seq_id into a bytes blob."""
    size = lib.llama_state_seq_get_size(ctx, seq_id)
    buf  = ctypes.create_string_buffer(size)
    written = lib.llama_state_seq_get_data(ctx, buf, size, seq_id)
    return bytes(buf.raw[:written])


def restore_kv_state(lib, ctx, blob: bytes, seq_id=0):
    """Restore KV cache for seq_id from a bytes blob produced by save_kv_state."""
    buf = ctypes.create_string_buffer(blob)
    lib.llama_state_seq_set_data(ctx, buf, len(blob), seq_id)


def trim_kv_seq(lib, ctx, pos, seq_id=0):
    """Remove all KV cells at positions >= pos from seq_id (metadata-only, no PCIe).

    Could avoid blob restore after a draft if the draft hook only wrote new cells and
    left 0..pos-1 fp16 intact; then prior fp16 would still be valid after trim.
    run_adaptive_gen does not use this: each quant rollout window calls draft_hook with
    n_new_k/v=None (bulk quant of the full cache), which overwrites the prompt too, so
    restore_kv_state is required.

    p1=-1 means "all positions from pos to end".
    """
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_seq_rm(mem, seq_id, pos, -1)


def adaptive_verify_accept(logits: np.ndarray, token_id: int,
                           top_k: Optional[int], top_p: Optional[float]) -> bool:
    """Whether token_id passes verifier acceptance under greedy, top-k, or top-p.

    If top_p is set (0<p<=1), uses nucleus top-p. Else if top_k is set (>=1), uses top-k.
    If both are None, uses greedy argmax match.
    """
    if top_p is not None and 0.0 < top_p <= 1.0:
        m = float(np.max(logits))
        pr = np.exp(logits - m)
        pr = pr / np.sum(pr)
        order = np.argsort(-pr)
        cum = 0.0
        nucleus = set()
        for i in order:
            cum += float(pr[i])
            nucleus.add(int(i))
            if cum >= top_p - 1e-12:
                break
        return int(token_id) in nucleus
    if top_k is not None and top_k >= 1:
        k = min(int(top_k), logits.size)
        part = np.argpartition(logits, -k)[-k:]
        return int(token_id) in part
    return int(np.argmax(logits)) == int(token_id)


def _single_decode(lib, ctx, token: int, pos: int) -> int:
    """Run one decode step (token at pos). Logits available at index 0. Returns ret code."""
    batch = lib.llama_batch_init(1, 0, 1)
    batch.n_tokens     = 1
    batch.token[0]     = token
    batch.pos[0]       = pos
    batch.n_seq_id[0]  = 1
    batch.seq_id[0][0] = 0
    batch.logits[0]    = 1
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    return ret


def _verify_window_batched_fp16(lib, ctx, n_vocab, window_tokens, pos_start,
                                return_logits: bool = False):
    """Teacher-forced prefill: one llama_decode for all tokens (no KV hook).

    Equivalent to sequential _single_decode per token for causal models; faster
    when verifying under fp16 (kv_hook=None in verify_window).
    If return_logits, returns (greedy, list of length-L full-vocab logits rows).
    """
    L = len(window_tokens)
    if L == 0:
        return ([], []) if return_logits else []
    batch = lib.llama_batch_init(L, 0, 1)
    for i in range(L):
        batch.token[i]     = window_tokens[i]
        batch.pos[i]       = pos_start + i
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]    = 1
    batch.n_tokens = L
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"_verify_window_batched_fp16: llama_decode failed: {ret}")
    greedy = []
    logits_rows = [] if return_logits else None
    for i in range(L):
        ptr = lib.llama_get_logits_ith(ctx, i)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        greedy.append(int(np.argmax(logits)))
        if return_logits:
            logits_rows.append(logits)
    if return_logits:
        return greedy, logits_rows
    return greedy


class StopStringState:
    """Mutable state for check_stop_strings (reasoning blocks span multiple tail windows)."""

    __slots__ = ("in_think_block",)

    def __init__(self):
        self.in_think_block = False


def check_stop_strings(lib, vocab, gen_ids: list, stop_strings, state: StopStringState,
                       min_len: int = 4) -> bool:
    """True if any stop string appears in the detokenized tail of gen_ids.

    Mirrors run_sweep.py / run_generate: skip matching inside <redacted_thinking> /
    analysis channel until the closing / final marker (same logic as strategies.run_generate).
    """
    if not stop_strings or len(gen_ids) < min_len:
        return False
    _stop_check_len = max((len(s) for s in stop_strings), default=0) * 3 + 32
    tail = gen_ids[-_stop_check_len:] if _stop_check_len else gen_ids
    tail_text = llama.detokenize(lib, vocab, tail, remove_special=False)
    if "<redacted_thinking>" in tail_text or "<|channel|>analysis" in tail_text:
        state.in_think_block = True
    if "</redacted_thinking>" in tail_text or "<|channel|>final" in tail_text:
        state.in_think_block = False
    if state.in_think_block:
        return False
    return any(s in tail_text for s in stop_strings)


def check_answer_regex(lib, vocab, gen_ids: list, answer_re, state: StopStringState,
                       tail_tokens: int = 256) -> bool:
    """True if answer_re matches the detokenized suffix (outside think/analysis blocks)."""
    if answer_re is None or not gen_ids:
        return False
    tail = gen_ids[-tail_tokens:] if tail_tokens > 0 else gen_ids
    tail_text = llama.detokenize(lib, vocab, tail, remove_special=False)
    if "<think>" in tail_text or "<|channel|>analysis" in tail_text:
        state.in_think_block = True
    if "</think>" in tail_text or "<|channel|>final" in tail_text:
        state.in_think_block = False
    if state.in_think_block:
        return False
    return answer_re.search(tail_text) is not None


def generate_window(lib, ctx, n_vocab, first_token, pos_start, W,
                    kv_hook=None, k_group_size=64, v_group_size=64,
                    stop_fn=None,
                    stop_strings=None, vocab=None, prefix_out_ids=None,
                    stop_state: Optional[StopStringState] = None,
                    answer_re=None):
    """Generate up to W tokens autoregressively starting from first_token.

    In run_adaptive_gen: used for the fp16 bootstrap window (kv_hook=None) and for
    each draft-quant quant rollout window (kv_hook=draft_hook).

    first_token is the first output token; it is decoded at pos_start to
    produce the second token's logits. The last returned token is NOT decoded
    (its KV is not written) — it serves as the prime token for the next window.

    Returns list of up to W tokens (includes first_token).
    After return: KV has positions pos_start .. pos_start+len(output)-2 written.

    If stop_strings is set, pass vocab and optional prefix_out_ids (tokens emitted
    before this window). Pass a shared stop_state across windows so reasoning-block
    tracking stays consistent.
    """
    prefix_out_ids = prefix_out_ids or []
    need_state = stop_strings is not None or answer_re is not None
    st = stop_state if stop_state is not None else (StopStringState() if need_state else None)

    tokens = [first_token]
    token  = first_token
    n_pending_k = n_pending_v = 0

    for _ in range(W - 1):
        if stop_fn and stop_fn(token):
            break
        ret = _single_decode(lib, ctx, token, pos_start + len(tokens) - 1)
        if ret != 0:
            break
        if kv_hook:
            n_pending_k += 1
            n_pending_v += 1
            n_pending_k, n_pending_v = _fire_hook(
                kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size)
        ptr    = lib.llama_get_logits_ith(ctx, 0)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        token  = int(np.argmax(logits))
        tokens.append(token)
        if st is not None and vocab is not None:
            combined = prefix_out_ids + tokens
            if stop_strings is not None and check_stop_strings(lib, vocab, combined, stop_strings, st):
                break
            if answer_re is not None and check_answer_regex(lib, vocab, combined, answer_re, st):
                break

    if kv_hook:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)

    return tokens


def verify_window(lib, ctx, n_vocab, window_tokens, pos_start,
                  kv_hook=None, k_group_size=64, v_group_size=64,
                  return_logits: bool = False):
    """Feed window_tokens (starting at pos_start) and return greedy next-token preds.

    When kv_hook is None (fp16 verifier): one batched prefill (_verify_window_batched_fp16)
    — same semantics as sequential decode, fewer llama_decode calls.

    When kv_hook is set (verifier quant): sequential decode only. Each step must run
    decode → hook (quantize new KV) → read logits so later positions attend to a
    quantized prefix. A single batched prefill produces logits under fp16 KV for the
    whole window; post-hoc quantization of cells cannot fix those logits, so there
    is no drop-in batched equivalent without a different acceptance rule or backend
    support for interleaved quant in the graph.

    If return_logits is True, returns (greedy, logits_rows) where logits_rows[i]
    is the full vocab logits after feeding window_tokens[i].

    Returns greedy[i] = argmax after feeding window_tokens[i], i.e. the predicted
    token for position pos_start+i+1.  Caller checks:
      greedy[i] == window_tokens[i+1]   for i in 0..len-2   (intra-window)
      greedy[-1] == <first token of next window>             (cross-window, optional)
    """
    if kv_hook is None:
        return _verify_window_batched_fp16(
            lib, ctx, n_vocab, window_tokens, pos_start, return_logits=return_logits)

    greedy = []
    logits_rows = [] if return_logits else None
    n_pending_k = n_pending_v = 0
    for i, tok in enumerate(window_tokens):
        ret = _single_decode(lib, ctx, tok, pos_start + i)
        if ret != 0:
            raise RuntimeError(f"verify_window decode failed at step {i}: {ret}")
        if kv_hook:
            n_pending_k += 1
            n_pending_v += 1
            n_pending_k, n_pending_v = _fire_hook(
                kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size)
        ptr = lib.llama_get_logits_ith(ctx, 0)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        greedy.append(int(np.argmax(logits)))
        if return_logits:
            logits_rows.append(logits)
    if kv_hook:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)
    if return_logits:
        return greedy, logits_rows
    return greedy


# ── end KV state helpers ──────────────────────────────────────────────────────

def run_chunk_token_by_token(lib, ctx, tokens, n_vocab, kv_hook=None, n_prompt=0,
                             k_group_size=64, v_group_size=64,
                             base_log_dists=None, return_log_dists=False,
                             return_top1=False, return_diagnostics=False):
    """
    Compute log-probs for a chunk using token-by-token strategy.

    K is quantized every k_group_size tokens (scales computed over the group).
    V is quantized every v_group_size tokens (1 for per-token quants).
    No re-quantization of earlier groups.

    n_prompt: leading tokens whose log-probs are discarded (treated as context).

    Returns RunResult(log_probs, log_dists, kl_divs, top1s, diags).
    diags: dict{"H", "p_max", "self_surp"} per decode step (or None).
    """
    N = len(tokens)
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    log_probs      = []
    log_dists_list = []
    kl_divs        = [] if base_log_dists is not None else None
    top1s_list     = [] if return_top1 else None
    diag_lists     = {"H": [], "p_max": [], "self_surp": []} if return_diagnostics else None
    t          = 0   # decode step counter (indexes into base_log_dists)
    prev_top1  = None

    n_pending_k = 0
    n_pending_v = 0
    for i in range(N - 1):
        batch = lib.llama_batch_init(1, 0, 1)
        batch.n_tokens = 1
        batch.token[0] = tokens[i]
        batch.pos[0] = i
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        ret = lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")

        if kv_hook:
            n_pending_k += 1
            n_pending_v += 1
            n_pending_k, n_pending_v = _fire_hook(
                kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size)

        if i < n_prompt:
            continue

        prev_top1 = _collect_logits(lib, ctx, n_vocab, tokens[i + 1], t,
                                    log_probs, log_dists_list, kl_divs, top1s_list,
                                    base_log_dists, return_log_dists,
                                    diag_lists=diag_lists, prev_top1=prev_top1)
        t += 1

    if kv_hook:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)

    log_dists = np.stack(log_dists_list) if log_dists_list else None
    return RunResult(log_probs, log_dists, kl_divs, top1s_list, diag_lists)


def run_chunk_batch_prefill(lib, ctx, tokens, n_vocab, kv_hook=None, n_prompt=0,
                            _quantize_prompt_only=False,
                            k_group_size=64, v_group_size=64,
                            base_log_dists=None, return_log_dists=False,
                            return_top1=False, return_diagnostics=False):
    """
    Batch-prefill the prompt, then decode continuation token-by-token.

    Default (always-on compression):
      - Prompt KV quantized once after batch prefill (all cells, n_new_k/v=None)
      - K quantized every k_group_size decode tokens
      - V quantized every v_group_size decode tokens (1 for per-token quants)
      - No re-quantization of earlier groups — each cell quantized exactly once

    With _quantize_prompt_only=True (Strategy D):
      - Prompt KV quantized once; decode KV stays FP16

    n_prompt must be >= 1 when using this function.

    Returns RunResult(log_probs, log_dists, kl_divs, top1s, diags).
    diags: dict{"H", "p_max", "self_surp"} per decode step (or None).
    """
    N = len(tokens)
    assert n_prompt >= 1, "batch-prefill mode requires --n-prompt >= 1"
    assert n_prompt < N - 1, f"n_prompt ({n_prompt}) must leave at least 1 token to decode"

    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    # 1. Batch prefill — no logits needed for prompt tokens
    batch = lib.llama_batch_init(n_prompt, 0, 1)
    batch.n_tokens = n_prompt
    for i in range(n_prompt):
        batch.token[i]     = tokens[i]
        batch.pos[i]       = i
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]    = 0
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"prefill decode failed: {ret}")

    # Quantize all prompt KV cells in one shot (n_new_k/v=None → all cells).
    # Pass n_prompt so sink-aware wrappers can protect the first N tokens.
    if kv_hook:
        kv_hook(ctx, n_new_k=None, n_new_v=None, n_prompt=n_prompt)

    # 2. Continuation: token-by-token decode, collect log-probs
    log_probs      = []
    log_dists_list = []
    kl_divs        = [] if base_log_dists is not None else None
    top1s_list     = [] if return_top1 else None
    diag_lists     = {"H": [], "p_max": [], "self_surp": []} if return_diagnostics else None
    t          = 0   # decode step counter
    prev_top1  = None

    n_pending_k = 0
    n_pending_v = 0
    for i in range(n_prompt, N - 1):
        batch = lib.llama_batch_init(1, 0, 1)
        batch.n_tokens     = 1
        batch.token[0]     = tokens[i]
        batch.pos[0]       = i
        batch.n_seq_id[0]  = 1
        batch.seq_id[0][0] = 0
        batch.logits[0]    = 1
        ret = lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")

        # Always-on: quantize decode tokens per schedule; never re-quantize old groups
        # Strategy D (quantize_prompt_only): skip — decode KV stays FP16
        if kv_hook and not _quantize_prompt_only:
            n_pending_k += 1
            n_pending_v += 1
            n_pending_k, n_pending_v = _fire_hook(
                kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size)

        prev_top1 = _collect_logits(lib, ctx, n_vocab, tokens[i + 1], t,
                                    log_probs, log_dists_list, kl_divs, top1s_list,
                                    base_log_dists, return_log_dists,
                                    diag_lists=diag_lists, prev_top1=prev_top1)
        t += 1

    if kv_hook and not _quantize_prompt_only:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)

    log_dists = np.stack(log_dists_list) if log_dists_list else None
    return RunResult(log_probs, log_dists, kl_divs, top1s_list, diag_lists)


def run_structured(lib, ctx, prompt_tokens, completion_tokens, n_vocab, kv_hook=None,
                   quantize_prompt_only=False, k_group_size=64, v_group_size=64,
                   base_log_dists=None, return_log_dists=False,
                   return_top1=False, return_diagnostics=False):
    """
    One structured example: batch-prefill the prompt, measure PPL of completion.
    Always uses batch-prefill — prompt is real context, completion is target.

    Returns RunResult(log_probs, log_dists, kl_divs, top1s, diags).
    """
    tokens = prompt_tokens + completion_tokens
    n_prompt = len(prompt_tokens)
    return run_chunk_batch_prefill(lib, ctx, tokens, n_vocab, kv_hook=kv_hook,
                                   n_prompt=n_prompt,
                                   _quantize_prompt_only=quantize_prompt_only,
                                   k_group_size=k_group_size,
                                   v_group_size=v_group_size,
                                   base_log_dists=base_log_dists,
                                   return_log_dists=return_log_dists,
                                   return_top1=return_top1,
                                   return_diagnostics=return_diagnostics)


def _rep_rate(token_ids, window=20, ngram=3):
    """Fraction of n-grams in the last `window` tokens that are duplicates.

    Returns 0.0 when there is not enough history.  A value > 0.5 is a strong
    signal the model has entered a repetition loop.
    """
    if len(token_ids) < ngram + 1:
        return 0.0
    recent = token_ids[-window:]
    grams  = [tuple(recent[i:i + ngram]) for i in range(len(recent) - ngram + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def run_generate(lib, ctx, vocab, prompt_tokens, n_vocab, kv_hook=None,
                 max_new_tokens=512, is_eog=None,
                 stop_strings=None,
                 k_group_size=64, v_group_size=64,
                 return_diagnostics=False,
                 stream_text=False,
                 stream_text_stride=1,
                 rep_rate_threshold=0.0,
                 rep_rate_consecutive=5):
    """Greedy autoregressive generation after batch-prefilling the prompt.

    Applies the same KV hook as the PPL pass so the quantization conditions
    match exactly.  Returns (generated, diags) where generated is a list of
    token ids and diags is a dict{"H", "p_max", "self_surp", "rep_rate"} of
    per-step lists (or None when return_diagnostics=False).
    Stops at max_new_tokens, when is_eog(token) returns True, or when any
    string in stop_strings appears in the decoded suffix of the output.
    is_eog: callable(token_id) -> bool, typically lib.llama_vocab_is_eog(vocab, id).

    When stream_text is True, prints each detokenized delta to stdout as tokens
    are generated (full-string detokenize each step; avoids duplicate output
    at the caller). stream_text_stride>1 prints every N tokens plus a final
    flush (faster; slightly less "live").

    When return_diagnostics is False, greedy decoding uses argmax on logits
    directly (no log_softmax over the vocab) for speed.

    rep_rate_threshold (0.0 = disabled): if the 3-gram repetition rate in the
    last 20 tokens exceeds this value for rep_rate_consecutive steps in a row,
    generation stops early (repetition loop detected).
    """
    n_prompt = len(prompt_tokens)
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    diag_lists = ({"H": [], "p_max": [], "self_surp": [], "rep_rate": []}
                  if return_diagnostics else None)
    _rep_consec = 0   # consecutive steps above rep_rate_threshold

    # Batch prefill — request logits only for the last prompt token
    batch = lib.llama_batch_init(n_prompt, 0, 1)
    batch.n_tokens = n_prompt
    for i in range(n_prompt):
        batch.token[i]     = prompt_tokens[i]
        batch.pos[i]       = i
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]    = 0
    batch.logits[n_prompt - 1] = 1
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"prefill failed: {ret}")

    if kv_hook:
        kv_hook(ctx, n_new_k=None, n_new_v=None, n_prompt=len(prompt_tokens))

    # First generated token comes from the last prompt token's logits (index n_prompt-1).
    # Subsequent tokens come from single-token decode batches where index 0 is correct.
    ptr    = lib.llama_get_logits_ith(ctx, n_prompt - 1)
    logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
    if diag_lists is not None:
        log_q  = llama.log_softmax(logits.astype(np.float32))
        token  = int(np.argmax(log_q))
    else:
        log_q  = None  # unused; greedy uses raw logits
        token  = int(np.argmax(logits.astype(np.float32)))
    generated   = [token]
    prev_top1   = token
    n_pending_k = 0
    n_pending_v = 0
    pos = n_prompt
    prev_stream = ""
    if stream_text:
        full = llama.detokenize(lib, vocab, generated, remove_special=False)
        print(full[len(prev_stream):], end="", flush=True)
        prev_stream = full
    # For stop-string detection: keep a small rolling decode buffer
    _stop_check_len = max((len(s) for s in stop_strings), default=0) * 3 + 32 if stop_strings else 0
    _in_think_block = False

    if diag_lists is not None:
        p     = np.exp(log_q)
        H     = -float(np.sum(p * log_q))
        diag_lists["H"].append(H)
        diag_lists["p_max"].append(float(p[token]))
        diag_lists["self_surp"].append(float("nan"))   # no previous step
        diag_lists["rep_rate"].append(0.0)

    for _ in range(max_new_tokens - 1):
        if is_eog is not None and is_eog(token):
            break

        batch = lib.llama_batch_init(1, 0, 1)
        batch.n_tokens     = 1
        batch.token[0]     = token
        batch.pos[0]       = pos
        batch.n_seq_id[0]  = 1
        batch.seq_id[0][0] = 0
        batch.logits[0]    = 1
        ret = lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")

        if kv_hook:
            n_pending_k += 1
            n_pending_v += 1
            n_pending_k, n_pending_v = _fire_hook(
                kv_hook, ctx, n_pending_k, n_pending_v, k_group_size, v_group_size)

        pos += 1

        ptr    = lib.llama_get_logits_ith(ctx, 0)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        if diag_lists is not None:
            log_q = llama.log_softmax(logits.astype(np.float32))
            token = int(np.argmax(log_q))
        else:
            log_q = None
            token = int(np.argmax(logits.astype(np.float32)))
        generated.append(token)

        if stream_text:
            n = len(generated)
            if stream_text_stride <= 1 or n % stream_text_stride == 0:
                full = llama.detokenize(lib, vocab, generated, remove_special=False)
                print(full[len(prev_stream):], end="", flush=True)
                prev_stream = full

        if diag_lists is not None:
            p     = np.exp(log_q)
            H     = -float(np.sum(p * log_q))
            ss    = float(log_q[prev_top1])
            diag_lists["H"].append(H)
            diag_lists["p_max"].append(float(p[token]))
            diag_lists["self_surp"].append(ss)
            diag_lists["rep_rate"].append(_rep_rate(generated))
        prev_top1 = token

        # Repetition early stop (works even without --save-diags).
        if rep_rate_threshold > 0.0:
            rr = _rep_rate(generated)
            if rr >= rep_rate_threshold:
                _rep_consec += 1
                if _rep_consec >= rep_rate_consecutive:
                    import sys
                    print(f"\n[rep-stop] rep_rate={rr:.2f} for {_rep_consec} steps "
                          f"at token {len(generated)} — stopping early", file=sys.stderr)
                    break
            else:
                _rep_consec = 0

        if stop_strings and len(generated) >= 4:
            tail = generated[-_stop_check_len:] if _stop_check_len else generated
            tail_text = llama.detokenize(lib, vocab, tail, remove_special=False)
            # Track reasoning blocks: <think> (Qwen3, DeepSeek-R1),
            # <|channel|>analysis (GPT-OSS). Skip stop strings inside them.
            if "<think>" in tail_text or "<|channel|>analysis" in tail_text:
                _in_think_block = True
            if "</think>" in tail_text or "<|channel|>final" in tail_text:
                _in_think_block = False
            if not _in_think_block and any(s in tail_text for s in stop_strings):
                break

    if kv_hook:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)

    if stream_text and stream_text_stride > 1 and len(generated) % stream_text_stride != 0:
        full = llama.detokenize(lib, vocab, generated, remove_special=False)
        print(full[len(prev_stream):], end="", flush=True)

    return generated, diag_lists
