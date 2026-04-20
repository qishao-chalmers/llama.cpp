#!/usr/bin/env python3
"""
run_adaptivegen.py — split2 adaptive generation benchmark.

By default, generation stops only when the model emits an EOG/EOS token
(`llama_vocab_is_eog`) or when --max-gen-tokens is reached — same core rule as
`strategies.run_generate` when no stop strings are configured.

Optional substring stops (--stop-strings, --task-stops, --auto-chat-stop) match
`run_sweep.py` when you explicitly enable them (run_sweep often adds stops for
accuracy eval; this benchmark leaves them off unless you opt in).

For ``.jsonl``, ``--eval-accuracy`` (default: on) loads gold from ``completion`` /
``answers`` / ``answer`` and scores like ``run_sweep.py`` (``--eval-metric exact|f1``).

``--answer-regex`` is used both to extract gold from reference text and to score generations.

Use ``--no-answer-extract`` to omit ``extracted_answer`` in JSON (scoring still uses the regex).

**Question selection (deterministic, not random):** JSONL examples are read in file order;
``--limit N`` takes the first ``N`` records after any ``--offset`` skip. Flat ``.txt`` mode uses
fixed-size chunks in order. Same file + same CLI flags + same model tokenizer → same prompts.

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

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from dataclasses import dataclass


class _Tee:
    """Write to two streams simultaneously (e.g. stderr + log file)."""
    def __init__(self, primary, secondary):
        self._p = primary
        self._s = secondary

    def write(self, data):
        self._p.write(data)
        self._s.write(data)

    def flush(self):
        self._p.flush()
        self._s.flush()

    def fileno(self):
        return self._p.fileno()

    def isatty(self):
        return self._p.isatty()

import numpy as np

import llama_bindings as llama
import strategies


# Task-specific stop strings (same as run_sweep.py): inferred from corpus filename.
_TASK_STOP_RULES = [
    (["aime"],                      ["\n\nQuestion:"]),
    (["gsm8k"],                     ["\n\nQuestion:"]),
    (["niah"],                      ["\n"]),
    (["humaneval", "human_eval"],   ["\n\ndef ", "\n\nif __name__"]),
]

_LAST_NUM_RE = re.compile(r"[\$]?\s*\*{0,2}([\d,]+)\*{0,2}")

_HEDGE_RE = re.compile(
    r"\b(wait|actually|hold on|let me (?:re)?check|let me recalculate|"
    r"but (?:wait|actually)|hmm+|i need to (?:re)?check|that\'?s not right|"
    r"let me redo|i made an error|correction|i(?:\'m| am) not sure|"
    r"let me (?:try|re-?do|go back|re-?compute|re-?calculate))\b",
    re.IGNORECASE,
)


def _f1_score(prediction: str, gold_list: list[str]) -> float:
    """Token-overlap F1 vs best-matching gold (LongBench-style; same as run_sweep)."""

    def _toks(s: str):
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        return s.split()

    pred_toks = _toks(prediction)
    if not pred_toks:
        return 0.0
    best = 0.0
    for gold in gold_list:
        gold_toks = _toks(gold)
        if not gold_toks:
            continue
        common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(pred_toks)
        recall = n_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def _example_label(rec: dict, idx: int) -> str:
    if "repo" in rec and "file" in rec:
        return f"{rec['repo']}/{rec['file']}"
    if "dataset" in rec:
        return f"{rec['dataset']}#{rec.get('id', idx)}"
    return str(rec.get("id", idx))


def _completion_text(rec: dict) -> str:
    if "completion" in rec:
        return rec["completion"]
    if "answers" in rec and rec["answers"]:
        a = rec["answers"]
        return a[0] if isinstance(a, list) else a
    return ""


def _extract_gold_exact(rec: dict, ans_re: re.Pattern) -> str | None:
    """Gold string for exact match (run_sweep: regex on completion, or rec['answer'])."""
    if rec.get("answer") is not None and str(rec.get("answer")).strip() != "":
        return str(rec["answer"]).strip().replace(",", "")
    text = _completion_text(rec)
    if not text:
        return None
    m = ans_re.search(text)
    raw = (m.group(1) or m.group(2)) if m else None
    if raw is None:
        fb = list(_LAST_NUM_RE.finditer(text))
        raw = fb[-1].group(1) if fb else None
    return raw.replace(",", "") if raw else None


def _gold_f1_answers(rec: dict, completion_text: str) -> list[str]:
    raw = rec.get("answers", [])
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    if raw and not isinstance(raw, list):
        return [str(raw)]
    return [completion_text] if completion_text else []


@dataclass
class AdaptiveDataset:
    """Loaded prompts + optional gold for --eval-accuracy (JSONL only)."""

    examples: list[list[int]]
    labels: list[str]
    eval_mode: str | None  # None | "exact" | "f1"
    gold_exact: list[str | None] | None
    gold_f1: list[list[str]] | None


def _infer_task_stops(filename: str) -> list:
    name = filename.lower()
    for keywords, stops in _TASK_STOP_RULES:
        if any(kw in name for kw in keywords):
            return stops
    return []


def _build_stop_strings(args, lib, vocab, corpus_basename: str) -> list:
    """Merge opt-in substring stops (empty by default = EOG/EOS only)."""
    out = [] if args.stop_strings is None else list(args.stop_strings)
    if getattr(args, "task_stops", False):
        for s in _infer_task_stops(corpus_basename):
            if s not in out:
                out.append(s)
    if (getattr(args, "auto_chat_stop", False)
            and not (args.prompt_prefix or "").strip()
            and not (args.prompt_suffix or "").strip()):
        try:
            _ap, _as, auto_stop, _fmt = llama.detect_chat_format(lib, vocab)
        except Exception:
            auto_stop = None
        if auto_stop and auto_stop not in out:
            out.append(auto_stop)
    return out


def _strip_reasoning_for_answer(gen_text: str) -> str:
    """Same post-processing as run_sweep eval_accuracy_pass (exact-match path)."""
    t = gen_text.strip()
    t = re.sub(r"<redacted_thinking>.*?</redacted_thinking>\s*", "", t, flags=re.DOTALL)
    t = re.sub(r"<redacted_thinking>.*", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>\s*",
               "", t, flags=re.DOTALL)
    m_final = re.search(r"<\|channel\|>final<\|message\|>(.*)", t, flags=re.DOTALL)
    if m_final:
        t = m_final.group(1).strip()
    return t


def _extract_answer(gen_text: str, ans_re: re.Pattern) -> str | None:
    """Last regex match, else last number (run_sweep semantics)."""
    t = _strip_reasoning_for_answer(gen_text)
    matches = list(ans_re.finditer(t))
    m = matches[-1] if matches else None
    raw = (m.group(1) or m.group(2)) if m else None
    if raw is None:
        fb = list(_LAST_NUM_RE.finditer(t))
        raw = fb[-1].group(1) if fb else None
    return raw.replace(",", "") if raw else None


def _attach_eval_scores(r: dict, ds: AdaptiveDataset, i: int, args, ans_re: re.Pattern) -> None:
    """Add gold / pred / accuracy_score / label (run_sweep eval_accuracy_pass semantics)."""
    if ds.eval_mode is None:
        return
    r["label"] = ds.labels[i]
    gen_text = r["generated_text"]
    gen_stripped = _strip_reasoning_for_answer(gen_text)
    gen_len = int(r["n_generated"])
    truncated = gen_len >= args.max_gen_tokens

    if ds.eval_mode == "f1":
        assert ds.gold_f1 is not None
        gold_list = ds.gold_f1[i]
        score = _f1_score(gen_stripped, gold_list)
        r["gold"] = gold_list
        r["accuracy_score"] = round(score, 4)
        return

    assert ds.gold_exact is not None
    gold = ds.gold_exact[i]
    pred = _extract_answer(gen_text, ans_re)
    if gold is None:
        r["gold"] = None
        r["pred"] = pred
        r["correct"] = None
        r["accuracy_score"] = None
        r["truncated"] = truncated
        r["inconclusive"] = False
        return
    all_matches = list(ans_re.finditer(gen_stripped))
    all_hedge_ms = list(_HEDGE_RE.finditer(gen_stripped))
    m = all_matches[-1] if all_matches else None
    inconclusive = (
        truncated and m is not None and all_hedge_ms and
        all_hedge_ms[-1].start() > m.start()
    )
    if inconclusive:
        pred_out = f"{pred}?" if pred is not None else "None"
        r["pred"] = pred_out
        r["gold"] = gold
        r["correct"] = False
        r["accuracy_score"] = 0.0
        r["inconclusive"] = True
        r["truncated"] = truncated
        return

    r["pred"] = str(pred) if pred is not None else None
    r["gold"] = gold
    r["inconclusive"] = False
    r["truncated"] = truncated
    ok = gold is not None and pred is not None and pred == gold
    r["correct"] = ok if gold is not None else None
    r["accuracy_score"] = 1.0 if ok else 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(lib, vocab, text: str) -> list[int]:
    return llama.tokenize(lib, vocab, text, buf_size=max(256, len(text) * 4))


def _prompt_reserve_tokens(args) -> int:
    """How many context slots to reserve for generation.

    In this script, bootstrap / adaptive windows do not all accumulate on top of the
    final generated sequence length: switch-mode bootstrap is rolled back before
    rollout, and draft windows are just the mechanism used to produce the same final
    generated suffix. The hard context budget is therefore dominated by:

      prompt_tokens + generated_tokens + a small safety margin <= n_ctx

    not by bootstrap_window + adaptive_window + max_gen_tokens.
    """
    return max(8, int(args.max_gen_tokens) + 8)


def _load_dataset(path: str, lib, vocab, args, ans_re: re.Pattern) -> AdaptiveDataset:
    """Load prompt token lists from a .txt or .jsonl file.

    .txt: non-overlapping chunks (same as before). No gold labels.

    .jsonl: ``prompt`` / ``context``+``input`` / ``input``; optional gold for
    ``--eval-accuracy`` (same fields as run_sweep: ``completion``, ``answers``, ``answer``).
    """
    limit = args.limit
    eval_on = bool(getattr(args, "eval_accuracy", False))
    eval_metric = getattr(args, "eval_metric", "exact")

    if not path.endswith(".jsonl"):
        text = open(path, encoding="utf-8").read()
        all_tokens = _tokenize(lib, vocab, text)
        print(f"Flat mode: {len(all_tokens)} tokens total from {path}", flush=True)

        reserve = _prompt_reserve_tokens(args)
        chunk_size = max(1, args.n_ctx - reserve)
        n_available = max(1, (len(all_tokens) - chunk_size) // chunk_size + 1)
        n_want = min(limit, n_available) if limit else n_available
        stride = max(chunk_size, len(all_tokens) // n_want) if n_want > 1 else chunk_size

        examples = []
        for c in range(n_want):
            start = c * stride
            end = start + chunk_size
            if end > len(all_tokens):
                break
            examples.append(all_tokens[start:end])
        print(
            f"  {len(examples)} chunk(s) of {chunk_size} tokens "
            f"(n_ctx={args.n_ctx}, reserved {reserve} for gen)",
            flush=True,
        )
        if chunk_size <= 16:
            print(
                f"[adaptivegen] WARNING: only {chunk_size} prompt token(s) fit with "
                f"n_ctx={args.n_ctx} and max_gen_tokens={args.max_gen_tokens}. "
                "Increase --n-ctx or lower --max-gen-tokens.",
                flush=True,
            )
        labels = [f"chunk{i}" for i in range(len(examples))]
        if eval_on:
            print("[adaptivegen] WARNING: --eval-accuracy applies to JSONL only; flat .txt has no gold.", flush=True)
        return AdaptiveDataset(
            examples=examples,
            labels=labels,
            eval_mode=None,
            gold_exact=None,
            gold_f1=None,
        )

    offset = int(getattr(args, "offset", 0) or 0)
    records = []
    skipped_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped_lines < offset:
                skipped_lines += 1
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    if offset:
        print(f"  jsonl: skipped first {offset} non-empty line(s), then loaded {len(records)} record(s)", flush=True)

    examples = []
    labels = []
    skipped = 0
    gold_exact: list[str | None] = []
    gold_f1: list[list[str]] = []

    for idx, rec in enumerate(records):
        if "prompt" in rec:
            prompt_text = rec["prompt"]
        elif "context" in rec and "input" in rec:
            prompt_text = rec["context"] + "\n\n" + rec["input"]
        else:
            prompt_text = rec.get("input", "")

        tokens = _tokenize(lib, vocab, args.prompt_prefix + prompt_text + args.prompt_suffix)
        reserve = _prompt_reserve_tokens(args)
        max_prompt = max(1, args.n_ctx - reserve)
        orig_len = len(tokens)
        if len(tokens) > max_prompt:
            tokens = tokens[-max_prompt:]
            print(
                f"[adaptivegen] WARNING: prompt for {_example_label(rec, idx)} truncated "
                f"from {orig_len} to {len(tokens)} token(s) because n_ctx={args.n_ctx} "
                f"and max_gen_tokens={args.max_gen_tokens} leave only {max_prompt} prompt slots.",
                flush=True,
            )
        if not tokens:
            skipped += 1
            continue

        examples.append(tokens)
        labels.append(_example_label(rec, idx))
        if eval_on:
            ctext = _completion_text(rec)
            if eval_metric == "f1":
                gold_f1.append(_gold_f1_answers(rec, ctext))
            else:
                gold_exact.append(_extract_gold_exact(rec, ans_re))

    print(f"Structured mode: {len(examples)} examples from {path}"
          + (f" ({skipped} skipped)" if skipped else ""), flush=True)

    if eval_on:
        if eval_metric == "f1":
            n_gold = sum(1 for g in gold_f1 if g)
            print(f"  eval_accuracy (f1): {len(examples)} examples, {n_gold} non-empty gold lists", flush=True)
            return AdaptiveDataset(
                examples=examples,
                labels=labels,
                eval_mode="f1",
                gold_exact=None,
                gold_f1=gold_f1,
            )
        n_gold = sum(1 for g in gold_exact if g is not None)
        print(f"  eval_accuracy (exact): gold found {n_gold}/{len(examples)}", flush=True)
        return AdaptiveDataset(
            examples=examples,
            labels=labels,
            eval_mode="exact",
            gold_exact=gold_exact,
            gold_f1=None,
        )

    return AdaptiveDataset(
        examples=examples,
        labels=labels,
        eval_mode=None,
        gold_exact=None,
        gold_f1=None,
    )


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
               prompt_tokens: list[int], split2_mode: int,
               stop_strings: list | None) -> dict:
    """
    Greedy generation with one context fixed to split2_mode 0 (verify) or 1 (draft).
    Reports tok/s over the decode phase only (excludes prefill).

    Stops at EOG/EOS (`llama_vocab_is_eog`), max_gen_tokens, and optionally when
    stop_strings is non-empty and a substring matches the detokenized tail.
    """
    ctx = _make_ctx(lib, model, args, split2_mode_init=split2_mode)
    _set_mode(lib, ctx, split2_mode)
    is_eog = lambda tok: bool(lib.llama_vocab_is_eog(vocab, tok))
    stop_state = strategies.StopStringState() if (stop_strings or args.stop_on_answer) else None

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
        if stop_strings and stop_state is not None:
            if strategies.check_stop_strings(lib, vocab, generated, stop_strings, stop_state):
                break
        if args.stop_on_answer and stop_state is not None:
            if strategies.check_answer_regex(lib, vocab, generated, args._ans_re, stop_state):
                break
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
               prompt_tokens: list[int], stop_strings: list | None) -> dict:
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

    stop_strings: optional list; if empty/None, only EOG/EOS and max_gen apply.
    When non-empty, shared StopStringState across bootstrap + rollout.
    """
    verifier_ctx = _make_ctx(lib, model, args, split2_mode_init=0)
    draft_ctx    = _make_ctx(lib, model, args, split2_mode_init=1)
    _set_mode(lib, verifier_ctx, 0)
    _set_mode(lib, draft_ctx,    1)

    is_eog = lambda tok: bool(lib.llama_vocab_is_eog(vocab, tok))
    stop_state = strategies.StopStringState() if (stop_strings or args.stop_on_answer) else None

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
    if stop_strings or args.stop_on_answer:
        boot_tokens = strategies.generate_window(
            lib, verifier_ctx, n_vocab,
            first_token=first_token, pos_start=pos0,
            W=args.bootstrap_window, kv_hook=None,
            stop_fn=is_eog,
            stop_strings=stop_strings, vocab=vocab, prefix_out_ids=[],
            stop_state=stop_state,
            answer_re=(args._ans_re if args.stop_on_answer else None),
        )
    else:
        boot_tokens = strategies.generate_window(
            lib, verifier_ctx, n_vocab,
            first_token=first_token, pos_start=pos0,
            W=args.bootstrap_window, kv_hook=None,
            stop_fn=is_eog,
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
        if stop_strings and stop_state is not None:
            if strategies.check_stop_strings(lib, vocab, generated, stop_strings, stop_state):
                break
        if args.stop_on_answer and stop_state is not None:
            if strategies.check_answer_regex(lib, vocab, generated, args._ans_re, stop_state):
                break
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
            if stop_strings and stop_state is not None:
                if strategies.check_stop_strings(lib, vocab, generated, stop_strings, stop_state):
                    break
            if args.stop_on_answer and stop_state is not None:
                if strategies.check_answer_regex(lib, vocab, generated, args._ans_re, stop_state):
                    break
            continue

        # Snapshot before draft window
        v_blob = strategies.save_kv_state(lib, verifier_ctx)
        d_blob = strategies.save_kv_state(lib, draft_ctx)

        # Draft generates a window. For stop-string detokenize, avoid duplicating `token`
        # when it is already generated[-1] (same as run_sweep prefix + window semantics).
        pre_ids = (
            generated[:-1]
            if len(generated) > 0 and token == generated[-1]
            else generated
        )
        if stop_strings or args.stop_on_answer:
            draft_tokens = strategies.generate_window(
                lib, draft_ctx, n_vocab,
                first_token=token, pos_start=pos,
                W=args.adaptive_window, kv_hook=None,
                stop_fn=is_eog,
                stop_strings=stop_strings, vocab=vocab, prefix_out_ids=pre_ids,
                stop_state=stop_state,
                answer_re=(args._ans_re if args.stop_on_answer else None),
            )
        else:
            draft_tokens = strategies.generate_window(
                lib, draft_ctx, n_vocab,
                first_token=token, pos_start=pos,
                W=args.adaptive_window, kv_hook=None,
                stop_fn=is_eog,
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
            if stop_strings and stop_state is not None:
                if strategies.check_stop_strings(lib, vocab, generated, stop_strings, stop_state):
                    break
            if args.stop_on_answer and stop_state is not None:
                if strategies.check_answer_regex(lib, vocab, generated, args._ans_re, stop_state):
                    break
            continue

        # Commit accepted tokens on verifier, then copy KV to draft once (no draft-path replay).
        accepted = draft_tokens[1:1 + acc_len]
        strategies.restore_kv_state(lib, verifier_ctx, v_blob)
        strategies.restore_kv_state(lib, draft_ctx,    d_blob)
        hit_stop_commit = False
        for t_acc in accepted:
            strategies._single_decode(lib, verifier_ctx, token, pos)
            token = int(t_acc)
            generated.append(token)
            pos += 1
            if stop_strings and stop_state is not None:
                if strategies.check_stop_strings(lib, vocab, generated, stop_strings, stop_state):
                    hit_stop_commit = True
                    break
            if args.stop_on_answer and stop_state is not None:
                if strategies.check_answer_regex(lib, vocab, generated, args._ans_re, stop_state):
                    hit_stop_commit = True
                    break
        _sync_draft_kv_from_verifier(lib, verifier_ctx, draft_ctx)
        n_accept += acc_len
        if hit_stop_commit:
            break

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
    ap.add_argument("--out", default=None, help="JSON results output path (default: stdout).")
    ap.add_argument("--log", default=None,
                    help="Log file path. Default: <out>.log when --out is set, else no log file.")

    ap.add_argument("--n-ctx",          type=int,   default=4096)
    ap.add_argument("--n-batch",        type=int,   default=512)
    ap.add_argument("--n-threads",      type=int,   default=8)
    ap.add_argument("--n-gpu-layers",   type=int,   default=0)
    ap.add_argument("--flash-attn",     action="store_true")

    ap.add_argument("--prompt-prefix",  default=None,
                    help="Prepended to every prompt. If omitted, auto-detected from model vocabulary "
                         "when --eval-accuracy is set (same as run_sweep). Pass '' to suppress.")
    ap.add_argument("--prompt-suffix",  default=None,
                    help="Appended to every prompt. Auto-detected when --eval-accuracy and prefix omitted.")
    ap.add_argument("--limit",          type=int,   default=1,
                    help="Max examples: JSONL = first N lines after --offset (0=all). "
                         "Flat .txt = up to N chunks.")
    ap.add_argument("--n-chunks",       type=int,   default=None,
                    help="Alias for --limit (run_sweep style: 0=all examples).")
    ap.add_argument("--offset",         type=int,   default=0,
                    help="JSONL only: skip this many non-empty lines before taking --limit (deterministic).")
    ap.add_argument("--max-gen-tokens", type=int,   default=256)

    ap.add_argument("--stop-strings", nargs="*", default=None,
                    help="Optional: stop when any substring appears in decoded output. "
                         "Default: do not use substring stops — stop only at EOG/EOS or --max-gen-tokens.")
    ap.add_argument("--task-stops", action="store_true",
                    help="Add task-specific stops from the corpus filename (gsm8k, niah, …), "
                         "like run_sweep with --eval-accuracy.")
    ap.add_argument("--auto-chat-stop", action="store_true",
                    help="Append chat-template stop from detect_chat_format when prefix/suffix are empty.")
    ap.add_argument("--eval-accuracy", action=argparse.BooleanOptionalAction, default=None,
                    help="Score predictions vs JSONL gold (default: on for .jsonl, off for .txt).")
    ap.add_argument("--eval-metric", choices=["exact", "f1"], default="exact",
                    help="exact: regex/numeric match; f1: token F1 vs answers (LongBench).")
    ap.add_argument("--answer-regex",
                    default=r"(?:####|[Tt]he answer is)\s*\$?\s*\*{0,2}\s*([\d,]+)\s*\*{0,2}|\\boxed\{([\d,]+)\}",
                    help="Regex for gold extraction and scoring (same default as run_sweep).")
    ap.add_argument("--stop-on-answer", action="store_true",
                    help="Stop generation as soon as --answer-regex matches in decoded output "
                         "(outside <think>/analysis blocks). Useful to avoid greedy repetition loops.")
    ap.add_argument("--no-answer-extract", action="store_true",
                    help="Omit extracted_answer field (accuracy still uses --answer-regex).")

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

    # ── log file setup (same style as run_sweep.py) ───────────────────────────
    _log_path = args.log
    if _log_path is None and args.out:
        base = args.out[:-5] if args.out.endswith(".json") else args.out
        _log_path = base + ".log"
    if _log_path:
        _log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)

        import atexit
        def _close_log():
            print(f"\nfinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log_file.close()
        atexit.register(_close_log)

    print(f"command : {' '.join(sys.argv)}", flush=True)
    print(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if _log_path:
        print(f"log     : {_log_path}", flush=True)

    lib = llama.load_lib()
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    if not model:
        sys.exit(f"failed to load model: {args.model}")

    vocab  = lib.llama_model_get_vocab(model)
    n_vocab = int(lib.llama_vocab_n_tokens(vocab))

    # --n-chunks is an alias for --limit (run_sweep style, 0=all).
    if args.n_chunks is not None:
        args.limit = args.n_chunks

    if args.eval_accuracy is None:
        args.eval_accuracy = args.dataset.endswith(".jsonl")

    # Auto-detect chat prefix/suffix from model vocabulary (same as run_sweep).
    # Only fires when both are None (i.e. not explicitly passed by the user).
    _fmt_name = "unknown"
    if args.eval_accuracy and args.prompt_prefix is None and args.prompt_suffix is None:
        auto_prefix, auto_suffix, auto_stop, _fmt_name = llama.detect_chat_format(lib, vocab)
        if auto_prefix is not None:
            args.prompt_prefix = auto_prefix
            args.prompt_suffix = auto_suffix
            if auto_stop:
                args.stop_strings = list(args.stop_strings or [])
                if auto_stop not in args.stop_strings:
                    args.stop_strings.append(auto_stop)
            print(f"[auto] detected chat format: {_fmt_name}", file=sys.stderr)
            print(f"[auto] prompt-prefix: {args.prompt_prefix!r}", file=sys.stderr)
            print(f"[auto] prompt-suffix: {args.prompt_suffix!r}", file=sys.stderr)
            if auto_stop:
                print(f"[auto] stop-string added: {auto_stop!r}", file=sys.stderr)
        else:
            print("[auto] chat format: unknown — prompt-prefix/suffix left empty", file=sys.stderr)

    # Normalize None → "" for downstream tokenization.
    if args.prompt_prefix is None:
        args.prompt_prefix = ""
    if args.prompt_suffix is None:
        args.prompt_suffix = ""

    # Qwen3 thinking suppression: inject empty <think> block so \n stop doesn't fire mid-think.
    if (_fmt_name == "qwen/chatml"
            and args.stop_strings and "\n" in args.stop_strings
            and llama._probe_special(lib, vocab, "<think>")):
        args.prompt_suffix = args.prompt_suffix.rstrip("\n") + "<think>\n\n</think>\n"
        print("[auto] qwen3 thinking suppressed: empty think block injected into suffix", file=sys.stderr)

    ans_re = re.compile(args.answer_regex)
    args._ans_re = ans_re
    ds = _load_dataset(args.dataset, lib, vocab, args, ans_re)
    if not ds.examples:
        sys.exit("no examples loaded")

    corpus_base = os.path.basename(args.dataset)
    stop_list = _build_stop_strings(args, lib, vocab, corpus_base)
    if stop_list:
        print(f"[adaptivegen] substring stop_strings (opt-in): {stop_list!r}", file=sys.stderr)
    if args.eval_accuracy:
        print(f"[adaptivegen] eval_accuracy={args.eval_accuracy}  metric={args.eval_metric}", file=sys.stderr)

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
    acc_sum = 0.0
    acc_n = 0
    for i, prompt_tokens in enumerate(ds.examples):
        ss = stop_list or None
        if args.mode == "verify":
            r = run_single(lib, model, vocab, n_vocab, args, prompt_tokens, 0, ss)
        elif args.mode == "draft":
            r = run_single(lib, model, vocab, n_vocab, args, prompt_tokens, 1, ss)
        else:
            r = run_switch(lib, model, vocab, n_vocab, args, prompt_tokens, ss)

        if not args.no_answer_extract:
            r["extracted_answer"] = _extract_answer(r["generated_text"], ans_re)
        if ds.eval_mode:
            _attach_eval_scores(r, ds, i, args, ans_re)
            if r.get("accuracy_score") is not None:
                acc_sum += float(r["accuracy_score"])
                acc_n += 1
        results.append(r)
        extra = ""
        if args.mode == "switch":
            extra = (f"  agree={r['agree_rate_bootstrap']:.2f}"
                     f"  accept={r['accept_rate_rollout']:.2f}"
                     f"  n_draft={r['n_draft_steps']}")
        ev = ""
        if ds.eval_mode and r.get("accuracy_score") is not None:
            ev = f"  acc={r['accuracy_score']}"
        print(f"[{r['mode']}] tok/s={r['tok_s']:.1f}"
              f"  n_gen={r['n_generated']}{extra}{ev}", file=sys.stderr)

    if ds.eval_mode and acc_n > 0:
        summary = f"[eval] mean {args.eval_metric} score: {acc_sum / acc_n:.4f}  (n={acc_n})"
        print(summary, file=sys.stderr)

    lib.llama_model_free(model)

    out_str = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"results : {args.out}", flush=True)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
