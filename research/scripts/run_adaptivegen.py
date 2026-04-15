#!/usr/bin/env python3
"""
run_adaptivegen.py — single-script adaptive generation runner.

Goal:
  - Keep run_sweep.py unchanged.
  - Provide a minimal bootstrap+rollout adaptive-gen harness that can be used to
    compare split2 verify vs split2 draft (or two separate models).

This script uses the ctypes bindings (llama_bindings.py) and the window helpers
in strategies.py (generate_window / verify_window / save_kv_state / restore_kv_state).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import llama_bindings as llama
import strategies


def _read_jsonl(path: str, limit: int | None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def _tokenize_text(lib, vocab, text: str) -> list[int]:
    return llama.tokenize(lib, vocab, text, buf_size=max(256, len(text) * 4))


def _prefill_and_get_next_logits(lib, ctx, n_vocab: int, prompt_tokens: list[int]):
    """Prefill prompt and return logits after the last prompt token."""
    if len(prompt_tokens) == 0:
        raise ValueError("empty prompt_tokens")
    # Clear KV, then feed prompt tokens.
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)
    _, logits_rows = strategies.verify_window(
        lib, ctx, n_vocab, prompt_tokens, pos_start=0, kv_hook=None, return_logits=True
    )
    return logits_rows[-1]


def _accept_prefix_len(verifier_logits_rows: list[np.ndarray], draft_tokens: list[int],
                       top_k: int | None, top_p: float | None) -> int:
    """Return number of accepted tokens in draft_tokens (prefix length)."""
    # verifier_logits_rows[i] is logits after feeding draft_tokens[i].
    # accept checks whether draft_tokens[i+1] passes verifier.
    if len(draft_tokens) <= 1:
        return 0
    L = min(len(verifier_logits_rows), len(draft_tokens) - 1)
    acc = 0
    for i in range(L):
        if strategies.adaptive_verify_accept(verifier_logits_rows[i], draft_tokens[i + 1], top_k=top_k, top_p=top_p):
            acc += 1
        else:
            break
    return acc


def run_one_example(lib, model_path: str, draft_model_path: str | None, args, ex) -> dict:
    """Run adaptive-gen on a single JSONL example: {"prompt":..., "completion":...}."""
    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers

    # Load verifier model (full).
    verifier_model = lib.llama_model_load_from_file(model_path.encode(), mparams)
    if not verifier_model:
        raise RuntimeError(f"failed to load verifier model: {model_path}")
    verifier_vocab = lib.llama_model_get_vocab(verifier_model)
    n_vocab = int(lib.llama_vocab_n_tokens(verifier_vocab))

    # Load draft model (optional separate path; may be the same file).
    # If user wants split2-draft on the same GGUF, they can set env before running python.
    dm_path = draft_model_path or model_path
    draft_model = lib.llama_model_load_from_file(dm_path.encode(), mparams)
    if not draft_model:
        raise RuntimeError(f"failed to load draft model: {dm_path}")
    draft_vocab = lib.llama_model_get_vocab(draft_model)

    cparams = lib.llama_context_default_params()
    cparams.n_ctx = args.n_ctx
    cparams.n_batch = min(args.n_ctx, max(1, args.n_batch))
    cparams.n_ubatch = cparams.n_batch
    cparams.n_threads = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.flash_attn_type = 1 if args.flash_attn else 0

    verifier_ctx = lib.llama_init_from_model(verifier_model, cparams)
    draft_ctx = lib.llama_init_from_model(draft_model, cparams)
    if not verifier_ctx or not draft_ctx:
        raise RuntimeError("failed to create contexts")

    prompt = args.prompt_prefix + ex["prompt"] + args.prompt_suffix
    prompt_tokens = _tokenize_text(lib, verifier_vocab, prompt)

    # Prefill both contexts with the same prompt.
    v_last_logits = _prefill_and_get_next_logits(lib, verifier_ctx, n_vocab, prompt_tokens)
    _ = _prefill_and_get_next_logits(lib, draft_ctx, n_vocab, prompt_tokens)

    first_token = int(np.argmax(v_last_logits))
    pos0 = len(prompt_tokens)

    # Bootstrap: generate Wb tokens with verifier (fp16 KV).
    v_kv0 = strategies.save_kv_state(lib, verifier_ctx)
    d_kv0 = strategies.save_kv_state(lib, draft_ctx)

    t0 = time.time()
    boot_tokens = strategies.generate_window(
        lib, verifier_ctx, n_vocab,
        first_token=first_token,
        pos_start=pos0,
        W=args.bootstrap_window,
        kv_hook=None,
        stop_fn=None,
    )
    t_boot = time.time() - t0

    # Estimate draft agreement on bootstrap region (teacher-forced):
    strategies.restore_kv_state(lib, draft_ctx, d_kv0)
    d_preds = strategies.verify_window(lib, draft_ctx, n_vocab, boot_tokens, pos_start=pos0, kv_hook=None, return_logits=False)
    agree = 0
    denom = max(0, len(boot_tokens) - 1)
    for i in range(denom):
        if int(d_preds[i]) == int(boot_tokens[i + 1]):
            agree += 1
    agree_rate = (agree / denom) if denom > 0 else 0.0

    # Reset to pre-bootstrap KV for rollout.
    strategies.restore_kv_state(lib, verifier_ctx, v_kv0)
    strategies.restore_kv_state(lib, draft_ctx, d_kv0)

    generated = []
    n_accept = 0
    n_verify_steps = 0

    token = first_token
    pos = pos0

    while len(generated) < args.max_gen_tokens:
        if agree_rate < args.min_agree_rate:
            # Skip draft path: verifier-only greedy decode one token.
            ret = strategies._single_decode(lib, verifier_ctx, token, pos)
            if ret != 0:
                break
            logits = np.ctypeslib.as_array(lib.llama_get_logits_ith(verifier_ctx, 0), shape=(n_vocab,)).copy()
            token = int(np.argmax(logits))
            generated.append(token)
            pos += 1
            continue

        # Snapshot before rollout window.
        v_blob = strategies.save_kv_state(lib, verifier_ctx)
        d_blob = strategies.save_kv_state(lib, draft_ctx)

        # Draft rollout window.
        draft_tokens = strategies.generate_window(
            lib, draft_ctx, n_vocab,
            first_token=token,
            pos_start=pos,
            W=args.adaptive_window,
            kv_hook=None,
            stop_fn=None,
        )

        # Verify draft tokens under verifier (teacher-forced) to get logits.
        strategies.restore_kv_state(lib, verifier_ctx, v_blob)
        _, v_logits_rows = strategies.verify_window(
            lib, verifier_ctx, n_vocab, draft_tokens, pos_start=pos, kv_hook=None, return_logits=True
        )
        n_verify_steps += len(draft_tokens)

        acc_len = _accept_prefix_len(v_logits_rows, draft_tokens, top_k=args.verify_top_k, top_p=args.verify_top_p)

        if acc_len == 0:
            # Reject immediately: take verifier greedy at this position (next token after feeding token).
            # We already have logits after feeding draft_tokens[0] (which is current token).
            next_tok = int(np.argmax(v_logits_rows[0]))
            strategies.restore_kv_state(lib, verifier_ctx, v_blob)
            strategies.restore_kv_state(lib, draft_ctx, d_blob)
            token = next_tok
            generated.append(token)
            pos += 1
            continue

        # Accept acc_len tokens: commit by replaying accepted tokens into both contexts.
        accepted = draft_tokens[1:1 + acc_len]
        strategies.restore_kv_state(lib, verifier_ctx, v_blob)
        strategies.restore_kv_state(lib, draft_ctx, d_blob)
        for t_acc in accepted:
            # feed token at current pos to advance logits for next
            _ = strategies._single_decode(lib, verifier_ctx, token, pos)
            _ = strategies._single_decode(lib, draft_ctx, token, pos)
            token = int(t_acc)
            generated.append(token)
            pos += 1
        n_accept += acc_len

    out = {
        "agree_rate_bootstrap": agree_rate,
        "bootstrap_window": args.bootstrap_window,
        "adaptive_window": args.adaptive_window,
        "max_gen_tokens": args.max_gen_tokens,
        "n_generated": len(generated),
        "n_accept": n_accept,
        "verify_steps": n_verify_steps,
        "bootstrap_seconds": t_boot,
        "generated_text": llama.detokenize(lib, verifier_vocab, generated, remove_special=False),
    }

    lib.llama_free(verifier_ctx)
    lib.llama_free(draft_ctx)
    lib.llama_model_free(verifier_model)
    lib.llama_model_free(draft_model)

    return out


def main():
    ap = argparse.ArgumentParser(description="Adaptive-gen runner (bootstrap + rollout windows)")
    ap.add_argument("model", help="GGUF model path (verifier).")
    ap.add_argument("jsonl", help="JSONL with {'prompt':..., ...}.")
    ap.add_argument("--draft-model", default=None, help="Optional draft model GGUF path. If omitted, reuses --model.")
    ap.add_argument("--out", default=None, help="Write JSON results to this file.")

    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--flash-attn", action="store_true")

    ap.add_argument("--prompt-prefix", default="")
    ap.add_argument("--prompt-suffix", default="")

    ap.add_argument("--limit", type=int, default=1, help="Number of JSONL examples to run.")
    ap.add_argument("--bootstrap-window", type=int, default=32)
    ap.add_argument("--adaptive-window", type=int, default=8)
    ap.add_argument("--max-gen-tokens", type=int, default=256)

    ap.add_argument("--min-agree-rate", type=float, default=0.7, help="Skip draft if bootstrap agree rate below this.")
    ap.add_argument("--verify-top-k", type=int, default=None)
    ap.add_argument("--verify-top-p", type=float, default=None)

    args = ap.parse_args()

    lib = llama.load_lib()
    lib.llama_backend_init()

    examples = _read_jsonl(args.jsonl, limit=args.limit)
    if not examples:
        sys.exit("no examples loaded")

    results = []
    for ex in examples:
        if "prompt" not in ex:
            raise ValueError("JSONL entries must contain 'prompt'")
        results.append(run_one_example(lib, args.model, args.draft_model, args, ex))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

