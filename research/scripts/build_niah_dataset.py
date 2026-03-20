#!/usr/bin/env python3
"""build_niah_dataset.py — Build a Needle-In-A-Haystack JSONL for lost-in-the-middle evaluation.

Each example inserts a unique needle fact (a passphrase) at a controlled relative position
within a haystack of text passages, then asks the model to retrieve it.  Running this
dataset through run_sweep.py --eval-accuracy with different quant types reveals whether
KV cache quantization amplifies the U-shaped accuracy degradation at middle positions.

Output JSONL fields:
  prompt      : haystack passages + question (text)
  completion  : expected answer (the passphrase word)
  needle_pos  : relative position of needle in [0.0, 1.0]  (0 = very start, 1 = very end)
  needle_idx  : replicate index within this position
  id          : unique example identifier
  dataset     : "niah"

Usage:
    python3 build_niah_dataset.py /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
        research/data/c4_val.txt \
        --out research/data/niah_4096.jsonl \
        --target-ctx 4096 \
        --n-positions 11 \
        --n-needles 10 \
        --n-threads 4

Then run:
    python3 run_sweep.py models/Qwen3-8B-Q5_K_M.gguf research/data/niah_4096.jsonl \\
        --corpus-mode structured --eval-accuracy --eval-metric f1 \\
        --quants fp16 int8_ch int4_ch int3_ch int2_ch \\
        --n-gpu-layers 99 --flash-attn \\
        --save-per-example research/results/niah_per_ex.json \\
        --out research/results/niah_results.json
"""

import argparse
import json
import math
import os
import random
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import llama_bindings as llama


# ── Needle templates ──────────────────────────────────────────────────────────
# Each needle has a unique task_id so the question is unambiguous.

NEEDLE_TEMPLATE = (
    "Important note: the secret passphrase for task '{task_id}' is '{answer}'."
)
QUESTION_TEMPLATE = (
    "\n\nBased on the text above, what is the secret passphrase for task '{task_id}'?"
    " Answer with only the passphrase word.\nAnswer:"
)
ANSWER_PREFIX = " "   # single space before the answer token

# Unique passphrase words — short, alphanumeric, clearly unambiguous
_ADJECTIVES = [
    "crimson", "azure", "golden", "silver", "cobalt", "amber", "violet",
    "scarlet", "ivory", "onyx", "jade", "coral", "teal", "indigo", "russet",
]
_NOUNS = [
    "falcon", "anchor", "prism", "cipher", "beacon", "delta", "vector",
    "vortex", "nexus", "matrix", "zenith", "summit", "canyon", "glacier",
    "comet", "pulsar", "quasar", "nebula", "photon", "proton",
]


def _make_passphrases(n: int, seed: int) -> list:
    rng = random.Random(seed + 7)
    pool = [f"{a}{n_}" for a in _ADJECTIVES for n_ in _NOUNS]
    rng.shuffle(pool)
    if n > len(pool):
        # extend with numbered variants
        pool += [f"{p}{i}" for i, p in enumerate(pool)]
    return pool[:n]


def _split_passages(tokens: list, doc_tokens: int) -> list:
    """Split flat token list into fixed-size passages."""
    return [tokens[i:i + doc_tokens]
            for i in range(0, len(tokens) - doc_tokens + 1, doc_tokens)]


def _build_prompt_text(lib, vocab, haystack_passages: list, needle_text: str,
                       insert_idx: int) -> str:
    """Detokenize haystack passages with needle inserted at insert_idx, return prompt string."""
    parts = []
    for i, passage_tokens in enumerate(haystack_passages):
        if i == insert_idx:
            parts.append(needle_text)
        parts.append(llama.detokenize(lib, vocab, passage_tokens, remove_special=False))
    if insert_idx >= len(haystack_passages):
        parts.append(needle_text)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model",        help="GGUF model path (used for tokenization only)")
    ap.add_argument("haystack_file", help="Text file to use as haystack (wikitext etc.)")
    ap.add_argument("--out",        required=True, help="Output JSONL file")
    ap.add_argument("--target-ctx", type=int, default=4096,
                    help="Target total prompt+completion tokens per example (default 4096)")
    ap.add_argument("--n-positions", type=int, default=11,
                    help="Number of needle positions between 0.0 and 1.0 inclusive (default 11)")
    ap.add_argument("--n-needles",  type=int, default=10,
                    help="Number of needle replicates per position (default 10)")
    ap.add_argument("--doc-tokens", type=int, default=None,
                    help="Tokens per haystack document (default: auto from --target-ctx)")
    ap.add_argument("--n-threads",  type=int, default=4)
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="GPU layers for model load (0 = CPU only; tokenization only so 0 is fine)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--shuffle-haystack", action="store_true",
                    help="Shuffle haystack passage order across examples (default: strided)")
    args = ap.parse_args()

    # ── Load model for tokenization ──────────────────────────────────────────
    print(f"Loading model for tokenization: {args.model}", flush=True)
    lib = llama.load_lib()
    lib.llama_backend_init()
    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    vocab = lib.llama_model_get_vocab(model)

    # ── Load and tokenize haystack corpus ───────────────────────────────────
    print(f"Loading haystack: {args.haystack_file}", flush=True)
    with open(args.haystack_file, encoding="utf-8") as f:
        raw_text = f.read()
    all_tokens = llama.tokenize(lib, vocab, raw_text, buf_size=2_000_000)
    print(f"  Haystack: {len(all_tokens):,} tokens", flush=True)

    # ── Determine passage size and number of passages per example ────────────
    # Needle text + question adds ~30–50 tokens; completion ~2–4 tokens.
    # We want: n_passages * doc_tokens + needle_tokens + question_tokens ≈ target_ctx
    needle_sample = NEEDLE_TEMPLATE.format(task_id="TASKID00", answer="crimsonfalcon")
    question_sample = QUESTION_TEMPLATE.format(task_id="TASKID00")
    needle_q_tokens = len(llama.tokenize(lib, vocab, needle_sample + question_sample))
    completion_tokens_est = 4   # passphrase is ~1-2 tokens

    usable_ctx = args.target_ctx - needle_q_tokens - completion_tokens_est - 10

    if args.doc_tokens is None:
        # Auto: use 20 documents, each ~usable_ctx/20 tokens
        n_docs = 20
        doc_tokens = max(64, usable_ctx // n_docs)
    else:
        doc_tokens = args.doc_tokens
        n_docs = max(1, usable_ctx // doc_tokens)

    print(f"  doc_tokens={doc_tokens}, n_docs={n_docs}, "
          f"estimated prompt≈{n_docs*doc_tokens + needle_q_tokens} tokens", flush=True)

    # ── Split haystack into passages ─────────────────────────────────────────
    passages = _split_passages(all_tokens, doc_tokens)
    print(f"  {len(passages)} haystack passages available", flush=True)

    n_total = args.n_positions * args.n_needles
    if len(passages) < n_docs:
        print(f"WARNING: only {len(passages)} passages available, need {n_docs} per example. "
              f"Passages will be reused.", flush=True)

    # ── Generate passphrases ─────────────────────────────────────────────────
    passphrases = _make_passphrases(n_total + 50, args.seed)
    rng = random.Random(args.seed)

    # ── Build positions ───────────────────────────────────────────────────────
    if args.n_positions == 1:
        positions = [0.5]
    else:
        positions = [i / (args.n_positions - 1) for i in range(args.n_positions)]

    # ── Build examples ────────────────────────────────────────────────────────
    records = []
    phrase_idx = 0

    for pos_i, needle_pos in enumerate(positions):
        for ni in range(args.n_needles):
            passphrase = passphrases[phrase_idx]; phrase_idx += 1
            task_id    = f"T{pos_i:02d}N{ni:02d}"

            needle_text   = NEEDLE_TEMPLATE.format(task_id=task_id, answer=passphrase)
            question_text = QUESTION_TEMPLATE.format(task_id=task_id)

            # Select n_docs haystack passages (strided across corpus for diversity)
            base = (pos_i * args.n_needles + ni) * n_docs
            hay = [passages[(base + j) % len(passages)] for j in range(n_docs)]
            if args.shuffle_haystack:
                rng.shuffle(hay)

            # Insert needle at fractional position
            insert_idx = min(int(needle_pos * n_docs), n_docs - 1)
            # Round-trip: build prompt text
            prompt = _build_prompt_text(lib, vocab, hay, needle_text, insert_idx)
            prompt += question_text
            completion = ANSWER_PREFIX + passphrase

            records.append({
                "prompt":     prompt,
                "completion": completion,
                "answers":    [passphrase],   # for --eval-metric f1
                "needle_pos": round(needle_pos, 4),
                "needle_idx": ni,
                "task_id":    task_id,
                "id":         len(records),
                "dataset":    "niah",
            })

    # ── Sanity check token lengths ────────────────────────────────────────────
    print(f"Checking token lengths for {len(records)} examples...", flush=True)
    lengths = []
    for rec in records[:min(5, len(records))]:
        pt = llama.tokenize(lib, vocab, rec["prompt"])
        ct = llama.tokenize(lib, vocab, rec["completion"])
        lengths.append(len(pt) + len(ct))
    if lengths:
        print(f"  Sample prompt+completion lengths: {lengths}", flush=True)
        print(f"  Target: {args.target_ctx} tokens", flush=True)

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nWrote {len(records)} examples to {args.out}")
    print(f"  Positions: {[round(p, 2) for p in positions]}")
    print(f"  Replicates per position: {args.n_needles}")
    print(f"\nRun evaluation:")
    print(f"  python3 run_sweep.py <model> {args.out} \\")
    print(f"    --corpus-mode structured --eval-accuracy --eval-metric f1 \\")
    print(f"    --quants fp16 int8_ch int4_ch int3_ch int2_ch \\")
    print(f"    --n-ctx {args.target_ctx + 64} --n-gpu-layers 99 --flash-attn \\")
    print(f"    --save-per-example research/results/niah_per_ex.json \\")
    print(f"    --out research/results/niah_results.json")

    lib.llama_model_free(model)
    lib.llama_backend_free()


if __name__ == "__main__":
    main()
