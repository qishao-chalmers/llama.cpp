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

from collections import namedtuple

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
    Returns updated (n_pending_k, n_pending_v)."""
    do_k = n_pending_k >= k_group_size
    do_v = n_pending_v >= v_group_size
    if do_k or do_v:
        kv_hook(ctx,
                n_new_k=n_pending_k if do_k else None,
                n_new_v=n_pending_v if do_v else None)
        if do_k: n_pending_k = 0
        if do_v: n_pending_v = 0
    return n_pending_k, n_pending_v


def _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v):
    """Flush any remaining pending tokens at end of chunk."""
    do_k = n_pending_k > 0
    do_v = n_pending_v > 0
    if do_k or do_v:
        kv_hook(ctx,
                n_new_k=n_pending_k if do_k else None,
                n_new_v=n_pending_v if do_v else None)


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


def run_chunk_token_by_token(lib, ctx, tokens, n_vocab, kv_hook=None, n_prompt=0,
                             k_group_size=128, v_group_size=128,
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
                            k_group_size=128, v_group_size=128,
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
                   quantize_prompt_only=False, k_group_size=128, v_group_size=128,
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
                 max_new_tokens=512, eos_token_id=None,
                 stop_strings=None,
                 k_group_size=128, v_group_size=128,
                 return_diagnostics=False):
    """Greedy autoregressive generation after batch-prefilling the prompt.

    Applies the same KV hook as the PPL pass so the quantization conditions
    match exactly.  Returns (generated, diags) where generated is a list of
    token ids and diags is a dict{"H", "p_max", "self_surp", "rep_rate"} of
    per-step lists (or None when return_diagnostics=False).
    Stops at max_new_tokens, when eos_token_id is produced, or when any
    string in stop_strings appears in the decoded suffix of the output.
    """
    n_prompt = len(prompt_tokens)
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    diag_lists = ({"H": [], "p_max": [], "self_surp": [], "rep_rate": []}
                  if return_diagnostics else None)

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
    log_q  = llama.log_softmax(logits.astype(np.float32))
    token  = int(np.argmax(log_q))
    generated   = [token]
    prev_top1   = token
    n_pending_k = 0
    n_pending_v = 0
    pos = n_prompt
    # For stop-string detection: keep a small rolling decode buffer
    _stop_check_len = max((len(s) for s in stop_strings), default=0) * 3 + 32 if stop_strings else 0

    if diag_lists is not None:
        p     = np.exp(log_q)
        H     = -float(np.sum(p * log_q))
        diag_lists["H"].append(H)
        diag_lists["p_max"].append(float(p[token]))
        diag_lists["self_surp"].append(float("nan"))   # no previous step
        diag_lists["rep_rate"].append(0.0)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and token == eos_token_id:
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
        log_q  = llama.log_softmax(logits.astype(np.float32))
        token  = int(np.argmax(log_q))
        generated.append(token)

        if diag_lists is not None:
            p     = np.exp(log_q)
            H     = -float(np.sum(p * log_q))
            ss    = float(log_q[prev_top1])
            diag_lists["H"].append(H)
            diag_lists["p_max"].append(float(p[token]))
            diag_lists["self_surp"].append(ss)
            diag_lists["rep_rate"].append(_rep_rate(generated))
        prev_top1 = token

        if stop_strings and len(generated) >= 4:
            tail = generated[-_stop_check_len:] if _stop_check_len else generated
            tail_text = llama.detokenize(lib, vocab, tail, remove_special=False)
            if any(s in tail_text for s in stop_strings):
                break

    if kv_hook:
        _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)

    return generated, diag_lists
