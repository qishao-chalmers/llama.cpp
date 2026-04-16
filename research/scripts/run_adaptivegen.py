#!/usr/bin/env python3
"""
run_adaptivegen.py — split2 adaptive generation benchmark.

Three modes (--mode):
  verify : single context, full Q8_0 reconstruction (split2_mode=0). Baseline.
  draft  : single context, nibble-only GEMV (split2_mode=1). Upper-bound speedup.
  switch : one model, two contexts. Bootstrap with verifier to estimate draft
           agree-rate, then rollout: draft generates a window, verifier
           teacher-forces it and accepts the prefix that matches. After commit,
           draft_ctx KV is copied from verifier so the next draft step reuses
           verifier-quality cache. Measures real adaptive-gen throughput.

Load the Q8_0 model in split2 layout by setting:
  LLAMA_Q8_0_SPLIT2=1  python3 run_adaptivegen.py model_q8_0.gguf data.jsonl ...

The model is loaded once; both verifier_ctx and draft_ctx share the same weight
buffer. llama_set_split2_mode(ctx, 0|1) switches which GEMV path each context uses.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import llama_bindings as llama
import strategies


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(lib, vocab, text: str) -> list[int]:
    return llama.tokenize(lib, vocab, text, buf_size=max(256, len(text) * 4))


def _load_examples(path: str, lib, vocab, args) -> list[list[int]]:
    """Load prompt token lists from either a .txt or .jsonl file.

    .txt  (flat mode): tokenize the whole file, split into --limit non-overlapping
          chunks of --n-ctx tokens (strided across the corpus), same as run_sweep.py.

    .jsonl (structured mode): each line must have a 'prompt' field (or 'context'+'input'
          for LongBench format). Returns tokenized prompt+prefix+suffix per example.
    """
    limit = args.limit

    if not path.endswith(".jsonl"):
        # Flat .txt mode: tokenize once, stride into chunks
        text = open(path, encoding="utf-8").read()
        all_tokens = _tokenize(lib, vocab, text)
        print(f"Flat mode: {len(all_tokens)} tokens total from {path}", flush=True)

        # A chunk of length n_ctx fills the KV cache; the next decode has no free cell
        # (llama_kv_cache::find_slot → FAILED_PREPARE). Reserve space for bootstrap + rollout.
        reserve = max(8, args.bootstrap_window + args.max_gen_tokens + args.adaptive_window + 8)
        chunk_size = max(1, args.n_ctx - reserve)
        n_available = max(1, (len(all_tokens) - chunk_size) // chunk_size + 1)
        n_want = min(limit, n_available) if limit else n_available
        stride = max(chunk_size, len(all_tokens) // n_want) if n_want > 1 else chunk_size

        examples = []
        for c in range(n_want):
            start = c * stride
            end   = start + chunk_size
            if end > len(all_tokens):
                break
            examples.append(all_tokens[start:end])
        print(f"  {len(examples)} chunk(s) of {chunk_size} tokens (n_ctx={args.n_ctx}, reserved {reserve} for gen)", flush=True)
        return examples

    # Structured .jsonl mode
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    examples = []
    skipped  = 0
    for rec in records:
        # Support: {"prompt":...}, {"context":...,"input":...} (LongBench)
        if "prompt" in rec:
            prompt_text = rec["prompt"]
        elif "context" in rec and "input" in rec:
            prompt_text = rec["context"] + "\n\n" + rec["input"]
        else:
            prompt_text = rec.get("input", "")

        tokens = _tokenize(lib, vocab,
                           args.prompt_prefix + prompt_text + args.prompt_suffix)
        # Same headroom as flat .txt chunks (KV must not be full after prefill).
        reserve = max(8, args.bootstrap_window + args.max_gen_tokens + args.adaptive_window + 8)
        max_prompt = max(1, args.n_ctx - reserve)
        if len(tokens) > max_prompt:
            # Truncate from the left (keep most recent context), leave room for generation
            tokens = tokens[-max_prompt:]
        if not tokens:
            skipped += 1
            continue
        examples.append(tokens)

    print(f"Structured mode: {len(examples)} examples from {path}"
          + (f" ({skipped} skipped)" if skipped else ""), flush=True)
    return examples


def _make_ctx(lib, model, args, split2_mode_init: int = 0):
    cparams = lib.llama_context_default_params()
    cparams.n_ctx          = args.n_ctx
    cparams.n_batch        = args.n_ctx           # must fit the full prompt in one submit
    cparams.n_ubatch       = min(args.n_batch, args.n_ctx)  # internal micro-batch for compute
    cparams.n_threads      = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.flash_attn_type = 1 if args.flash_attn else 0
    cparams.split2_mode_init   = int(split2_mode_init)
    cparams.q4_k_res_mode_init = int(split2_mode_init)  # 0=verify (base+res), 1=draft (base only)
    ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        raise RuntimeError("llama_init_from_model failed")
    return ctx


def _set_mode(lib, ctx, mode: int):
    """Set draft/verify mode: 0=full verify, 1=draft (base only).
    Calls both split2 and q4_k_res APIs — whichever applies to the loaded model."""
    if hasattr(lib, "llama_set_split2_mode"):
        lib.llama_set_split2_mode(ctx, mode)
        if hasattr(lib, "llama_get_split2_mode"):
            got = int(lib.llama_get_split2_mode(ctx))
            if got != mode:
                print(
                    f"WARNING: llama_get_split2_mode()={got} after llama_set_split2_mode({mode})",
                    file=sys.stderr,
                )
    if hasattr(lib, "llama_set_q4_k_res_mode"):
        lib.llama_set_q4_k_res_mode(ctx, mode)
        if hasattr(lib, "llama_get_q4_k_res_mode"):
            got = int(lib.llama_get_q4_k_res_mode(ctx))
            if got != mode:
                print(
                    f"WARNING: llama_get_q4_k_res_mode()={got} after llama_set_q4_k_res_mode({mode})",
                    file=sys.stderr,
                )


def _prefill(lib, ctx, n_vocab: int, prompt_tokens: list[int]) -> np.ndarray:
    """Clear KV, batch-prefill prompt, return logits after the last prompt token."""
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)
    _, logits_rows = strategies.verify_window(
        lib, ctx, n_vocab, prompt_tokens, pos_start=0,
        kv_hook=None, return_logits=True,
    )
    return logits_rows[-1]


def _accept_prefix(v_logits_rows, draft_tokens, top_k, top_p) -> int:
    """Return length of accepted prefix of draft_tokens under verifier logits."""
    if len(draft_tokens) <= 1:
        return 0
    L = min(len(v_logits_rows), len(draft_tokens) - 1)
    for i in range(L):
        if not strategies.adaptive_verify_accept(
                v_logits_rows[i], draft_tokens[i + 1], top_k=top_k, top_p=top_p):
            return i
    return L


def _sync_draft_kv_from_verifier(lib, verifier_ctx, draft_ctx, seq_id: int = 0) -> None:
    """Copy sequence KV (and bundled seq state) from verifier into draft.

    After verifier commits tokens, draft must not rebuild KV via the draft GEMV path;
    the next draft window should continue from verifier-quality KV."""
    blob = strategies.save_kv_state(lib, verifier_ctx, seq_id=seq_id)
    strategies.restore_kv_state(lib, draft_ctx, blob, seq_id=seq_id)


# ── mode: verify or draft (single context) ────────────────────────────────────

def run_single(lib, model, vocab, n_vocab: int, args,
               prompt_tokens: list[int], split2_mode: int) -> dict:
    """
    Greedy generation with one context fixed to split2_mode 0 (verify) or 1 (draft).
    Reports tok/s over the decode phase only (excludes prefill).
    """
    ctx = _make_ctx(lib, model, args, split2_mode_init=split2_mode)
    _set_mode(lib, ctx, split2_mode)
    is_eog = lambda tok: bool(lib.llama_vocab_is_eog(vocab, tok))

    t0 = time.perf_counter()
    logits = _prefill(lib, ctx, n_vocab, prompt_tokens)
    t_prefill = time.perf_counter() - t0

    first_token = int(np.argmax(logits))
    generated = [first_token]
    token = first_token
    pos   = len(prompt_tokens)

    t_dec = time.perf_counter()
    while len(generated) < args.max_gen_tokens:
        if is_eog(token):
            break
        ret = strategies._single_decode(lib, ctx, token, pos)
        if ret != 0:
            break
        ptr    = lib.llama_get_logits_ith(ctx, 0)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        token  = int(np.argmax(logits))
        generated.append(token)
        pos += 1
    t_decode = time.perf_counter() - t_dec

    lib.llama_free(ctx)

    n_gen  = len(generated)
    tok_s  = n_gen / t_decode if t_decode > 0 else 0.0
    mode_name = "verify" if split2_mode == 0 else "draft"
    return {
        "mode":        mode_name,
        "n_prompt":    len(prompt_tokens),
        "n_generated": n_gen,
        "t_prefill_s": round(t_prefill, 4),
        "t_decode_s":  round(t_decode,  4),
        "tok_s":       round(tok_s,     2),
        "generated_text": llama.detokenize(lib, vocab, generated, remove_special=False),
    }


# ── mode: switch (bootstrap + adaptive draft/verify rollout) ──────────────────

def run_switch(lib, model, vocab, n_vocab: int, args,
               prompt_tokens: list[int]) -> dict:
    """
    One model, two contexts:
      verifier_ctx: split2_mode=0 (full Q8_0)
      draft_ctx:    split2_mode=1 (nibble path)

    Phase 1 — Bootstrap (Wb tokens from verifier):
      Estimate draft agree-rate on the bootstrap window to decide whether to
      use the draft path at all during rollout.

    Phase 2 — Rollout:
      If agree_rate >= min_agree_rate:
        - draft_ctx generates adaptive_window tokens
        - verifier_ctx teacher-forces them, returns per-step logits
        - accept the longest prefix that passes adaptive_verify_accept
        - commit by replaying on verifier only, then copy KV verifier → draft
      Else:
        - verifier-only greedy decode (one token at a time); sync draft KV from verifier
    """
    verifier_ctx = _make_ctx(lib, model, args, split2_mode_init=0)
    draft_ctx    = _make_ctx(lib, model, args, split2_mode_init=1)
    _set_mode(lib, verifier_ctx, 0)
    _set_mode(lib, draft_ctx,    1)

    is_eog = lambda tok: bool(lib.llama_vocab_is_eog(vocab, tok))

    # Prefill both contexts
    v_last = _prefill(lib, verifier_ctx, n_vocab, prompt_tokens)
    _prefill(lib, draft_ctx, n_vocab, prompt_tokens)

    first_token = int(np.argmax(v_last))
    pos0 = len(prompt_tokens)

    # Snapshot KV after prefill
    v_kv0 = strategies.save_kv_state(lib, verifier_ctx)
    d_kv0 = strategies.save_kv_state(lib, draft_ctx)

    # Phase 1: bootstrap with verifier
    t0 = time.perf_counter()
    boot_tokens = strategies.generate_window(
        lib, verifier_ctx, n_vocab,
        first_token=first_token, pos_start=pos0,
        W=args.bootstrap_window, kv_hook=None,
    )
    t_boot = time.perf_counter() - t0

    # Estimate agree-rate: teacher-force boot_tokens through draft via MMVQ path.
    # Must use _single_decode (n_tokens=1) so the CUDA dispatch goes through MMVQ
    # and respects split2_draft=True. verify_window is batched (cuBLAS) and always
    # dequantizes full Q8_0 regardless of split2_draft, giving a misleading 100% rate.
    strategies.restore_kv_state(lib, draft_ctx, d_kv0)
    denom = max(0, len(boot_tokens) - 1)
    agree = 0
    tok = boot_tokens[0]
    for i in range(denom):
        ret = strategies._single_decode(lib, draft_ctx, tok, pos0 + i)
        if ret != 0:
            break
        ptr    = lib.llama_get_logits_ith(draft_ctx, 0)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        pred   = int(np.argmax(logits))
        if pred == boot_tokens[i + 1]:
            agree += 1
        tok = boot_tokens[i + 1]  # teacher-force: feed correct token as next input
    agree_rate = agree / denom if denom > 0 else 0.0

    # Reset both to post-prefill KV for rollout
    strategies.restore_kv_state(lib, verifier_ctx, v_kv0)
    strategies.restore_kv_state(lib, draft_ctx,    d_kv0)

    generated        = []
    n_accept         = 0
    n_draft_steps    = 0
    n_verify_steps   = 0
    n_fallback_steps = 0

    token = first_token
    pos   = pos0

    t_roll = time.perf_counter()
    while len(generated) < args.max_gen_tokens:
        if is_eog(token):
            break

        if agree_rate < args.min_agree_rate:
            # Verifier-only fallback: one greedy step; draft mirrors verifier KV
            ret = strategies._single_decode(lib, verifier_ctx, token, pos)
            if ret != 0:
                break
            _sync_draft_kv_from_verifier(lib, verifier_ctx, draft_ctx)
            ptr    = lib.llama_get_logits_ith(verifier_ctx, 0)
            logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
            token  = int(np.argmax(logits))
            generated.append(token)
            pos += 1
            n_fallback_steps += 1
            continue

        # Snapshot before draft window
        v_blob = strategies.save_kv_state(lib, verifier_ctx)
        d_blob = strategies.save_kv_state(lib, draft_ctx)

        # Draft generates a window
        draft_tokens = strategies.generate_window(
            lib, draft_ctx, n_vocab,
            first_token=token, pos_start=pos,
            W=args.adaptive_window, kv_hook=None,
        )
        n_draft_steps += len(draft_tokens)

        # Verifier teacher-forces the draft window
        strategies.restore_kv_state(lib, verifier_ctx, v_blob)
        _, v_logits_rows = strategies.verify_window(
            lib, verifier_ctx, n_vocab, draft_tokens, pos_start=pos,
            kv_hook=None, return_logits=True,
        )
        n_verify_steps += len(draft_tokens)

        acc_len = _accept_prefix(v_logits_rows, draft_tokens,
                                 top_k=args.verify_top_k, top_p=args.verify_top_p)

        if acc_len == 0:
            # Full rejection: commit `token` at `pos` on verifier, then mirror KV to draft.
            # v_logits_rows[0] = logits after feeding draft_tokens[0] (==token) at pos,
            # so argmax gives the correct next token from the verifier's perspective.
            next_tok = int(np.argmax(v_logits_rows[0]))
            strategies.restore_kv_state(lib, verifier_ctx, v_blob)
            strategies.restore_kv_state(lib, draft_ctx,    d_blob)
            strategies._single_decode(lib, verifier_ctx, token, pos)
            _sync_draft_kv_from_verifier(lib, verifier_ctx, draft_ctx)
            token = next_tok
            generated.append(token)
            pos += 1
            continue

        # Commit accepted tokens on verifier, then copy KV to draft once (no draft-path replay).
        accepted = draft_tokens[1:1 + acc_len]
        strategies.restore_kv_state(lib, verifier_ctx, v_blob)
        strategies.restore_kv_state(lib, draft_ctx,    d_blob)
        for t_acc in accepted:
            strategies._single_decode(lib, verifier_ctx, token, pos)
            token = int(t_acc)
            generated.append(token)
            pos += 1
        _sync_draft_kv_from_verifier(lib, verifier_ctx, draft_ctx)
        n_accept += acc_len

    t_rollout = time.perf_counter() - t_roll

    lib.llama_free(verifier_ctx)
    lib.llama_free(draft_ctx)

    n_gen       = len(generated)
    tok_s       = n_gen / t_rollout if t_rollout > 0 else 0.0
    accept_rate = n_accept / n_verify_steps if n_verify_steps > 0 else 0.0
    return {
        "mode":                  "switch",
        "n_prompt":              len(prompt_tokens),
        "n_generated":           n_gen,
        "agree_rate_bootstrap":  round(agree_rate,   4),
        "accept_rate_rollout":   round(accept_rate,  4),
        "bootstrap_window":      args.bootstrap_window,
        "adaptive_window":       args.adaptive_window,
        "n_draft_steps":         n_draft_steps,
        "n_verify_steps":        n_verify_steps,
        "n_fallback_steps":      n_fallback_steps,
        "t_boot_s":              round(t_boot,    4),
        "t_rollout_s":           round(t_rollout, 4),
        "tok_s":                 round(tok_s,     2),
        "generated_text": llama.detokenize(lib, vocab, generated, remove_special=False),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Adaptive-gen benchmark: draft / verify / switch modes. "
                    "Supports split2 (Q8_0 GGUF + LLAMA_Q8_0_SPLIT2=1) and "
                    "Q4_K_RES (q4k_res GGUF, no env var needed).")
    ap.add_argument("model",   help="GGUF model path. For split2: Q8_0 GGUF + LLAMA_Q8_0_SPLIT2=1. "
                                    "For Q4_K_RES: pass the _q4k_res.gguf directly.")
    ap.add_argument("dataset", help=".jsonl (structured: {'prompt':...}) or .txt (flat: chunked).")
    ap.add_argument("--mode", choices=["draft", "verify", "switch"], default="switch",
                    help="draft=nibble-only, verify=full Q8_0, switch=adaptive.")
    ap.add_argument("--out", default=None, help="JSON output path (default: stdout).")

    ap.add_argument("--n-ctx",          type=int,   default=4096)
    ap.add_argument("--n-batch",        type=int,   default=512)
    ap.add_argument("--n-threads",      type=int,   default=8)
    ap.add_argument("--n-gpu-layers",   type=int,   default=0)
    ap.add_argument("--flash-attn",     action="store_true")

    ap.add_argument("--prompt-prefix",  default="")
    ap.add_argument("--prompt-suffix",  default="")
    ap.add_argument("--limit",          type=int,   default=1)
    ap.add_argument("--max-gen-tokens", type=int,   default=256)

    # switch-mode parameters
    ap.add_argument("--bootstrap-window", type=int,   default=32,
                    help="Tokens generated by verifier for agree-rate estimation.")
    ap.add_argument("--adaptive-window",  type=int,   default=8,
                    help="Draft tokens per rollout window.")
    ap.add_argument("--min-agree-rate",   type=float, default=0.7,
                    help="Fall back to verifier-only if bootstrap agree-rate is below this.")
    ap.add_argument("--verify-top-k",     type=int,   default=None)
    ap.add_argument("--verify-top-p",     type=float, default=None)

    args = ap.parse_args()

    lib = llama.load_lib()
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    if not model:
        sys.exit(f"failed to load model: {args.model}")

    vocab  = lib.llama_model_get_vocab(model)
    n_vocab = int(lib.llama_vocab_n_tokens(vocab))

    examples = _load_examples(args.dataset, lib, vocab, args)
    if not examples:
        sys.exit("no examples loaded")

    if args.mode in ("draft", "verify", "switch"):
        has_split2   = hasattr(lib, "llama_set_split2_mode")
        has_q4k_res  = hasattr(lib, "llama_set_q4_k_res_mode")
        print(
            "[adaptivegen] "
            f"n_gpu_layers={args.n_gpu_layers}  "
            f"split2={'ok' if has_split2 else 'missing'}  "
            f"q4_k_res={'ok' if has_q4k_res else 'missing (rebuild libllama)'}  "
            f"LLAMA_Q8_0_SPLIT2={os.environ.get('LLAMA_Q8_0_SPLIT2', '')!r}",
            file=sys.stderr,
        )
        if not has_split2 and not has_q4k_res:
            print(
                "[adaptivegen] WARNING: neither llama_set_split2_mode nor llama_set_q4_k_res_mode "
                "found — draft/verify switching will be a no-op. Rebuild libllama.",
                file=sys.stderr,
            )
        if args.n_gpu_layers <= 0:
            print(
                "[adaptivegen] WARNING: n_gpu_layers=0 — draft/verify CUDA MMVQ paths only "
                "apply to GPU-offloaded weights; CPU will use generic paths.",
                file=sys.stderr,
            )

    results = []
    for i, prompt_tokens in enumerate(examples):
        if args.mode == "verify":
            r = run_single(lib, model, vocab, n_vocab, args, prompt_tokens, split2_mode=0)
        elif args.mode == "draft":
            r = run_single(lib, model, vocab, n_vocab, args, prompt_tokens, split2_mode=1)
        else:
            r = run_switch(lib, model, vocab, n_vocab, args, prompt_tokens)

        results.append(r)
        extra = ""
        if args.mode == "switch":
            extra = (f"  agree={r['agree_rate_bootstrap']:.2f}"
                     f"  accept={r['accept_rate_rollout']:.2f}"
                     f"  n_draft={r['n_draft_steps']}")
        print(f"[{r['mode']}] tok/s={r['tok_s']:.1f}"
              f"  n_gen={r['n_generated']}{extra}", file=sys.stderr)

    lib.llama_model_free(model)

    out_str = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_str)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
