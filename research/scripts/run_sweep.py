#!/usr/bin/env python3
"""run_sweep.py — KV cache quantization perplexity sweep.

Usage:
    python3 run_sweep.py models/Qwen3-8B-Q8_0.gguf research/data/wikitext2_test.txt \
        --n-ctx 128 --n-chunks 20 --n-threads 8 --out results.json
"""

import argparse, json, math, os, re, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import llama_bindings as llama
import parse_state
import quant as quant_mod
import strategies

try:
    import gpu_quant
    HAS_GPU_QUANT = gpu_quant.HAS_CUPY
except ImportError:
    HAS_GPU_QUANT = False


def make_kv_hook(lib, k_names, v_names, n_pos_per_embd=1,
                 use_gpu=False, n_layer=0, ctx_ptr=None,
                 default_group_size=128):
    """Create a KV hook for the given K/V quant name lists.

    k_names / v_names: list[str] of length n_layer.
    Returns None if all layers are fp16 (no-op).

    Per-layer group sizes are computed from each layer's quant type:
    - per-token quants → group_size=1 (fire every token)
    - per-channel quants → group_size=default_group_size (accumulate first)
    When layers on the same side have mixed group sizes (e.g. per-channel on
    layers 0-15 and per-token on layers 16-31), a stateful closure tracks each
    layer's pending count independently so per-channel layers still batch
    default_group_size tokens before quantizing.
    """
    if all(n == "fp16" for n in k_names) and all(n == "fp16" for n in v_names):
        return None

    k_layer_gs = [1 if n in quant_mod.PER_TOKEN_QUANTS else default_group_size
                  for n in k_names]
    v_layer_gs = [1 if n in quant_mod.PER_TOKEN_QUANTS else default_group_size
                  for n in v_names]
    k_uniform  = len(set(k_layer_gs)) == 1
    v_uniform  = len(set(v_layer_gs)) == 1

    if k_uniform and v_uniform:
        # Simple path: no per-layer tracking needed; hook fires at the right cadence
        if use_gpu and HAS_GPU_QUANT:
            return gpu_quant.make_kv_hook_gpu(lib, ctx_ptr, k_names, v_names, n_layer)
        k_fns = [quant_mod.get_quant_fn(n) for n in k_names]
        v_fns = [quant_mod.get_quant_fn(n) for n in v_names]
        def hook_simple(ctx, n_new_k=None, n_new_v=None, _kf=k_fns, _vf=v_fns):
            parse_state.apply_kv_hook(lib, ctx, llama.ContextPtr,
                                      k_fn=_kf, v_fn=_vf,
                                      seq_id=0, n_pos_per_embd=n_pos_per_embd,
                                      n_new_k=n_new_k, n_new_v=n_new_v)
        return hook_simple

    # Mixed granularities: stateful per-layer pending counters.
    # The hook fires every token (global group_size=1 from get_kv_group_sizes);
    # each layer only quantizes when its own threshold is met.
    k_pending = [0] * n_layer
    v_pending = [0] * n_layer

    def _per_layer_new(n_new, pending, layer_gs):
        """Accumulate n_new into pending; return per-layer list.
        None = quantize all (prefill), 0 = skip, int = quantize last N cells."""
        result = []
        for i in range(n_layer):
            if n_new is None:           # prefill: reset and quantize all
                pending[i] = 0
                result.append(None)
            else:
                pending[i] += n_new
                if pending[i] >= layer_gs[i]:
                    result.append(pending[i])
                    pending[i] = 0
                else:
                    result.append(0)    # not enough accumulated yet
        return result

    if use_gpu and HAS_GPU_QUANT:
        def hook_gpu_mixed(ctx, n_new_k=None, n_new_v=None):
            k_per = _per_layer_new(n_new_k, k_pending, k_layer_gs)
            v_per = _per_layer_new(n_new_v, v_pending, v_layer_gs)
            if n_new_k is None or any(k_per) or any(v_per):
                gpu_quant.apply_kv_hook_gpu(
                    lib, ctx, n_layer, k_names, v_names,
                    n_new_k=None if n_new_k is None else k_per,
                    n_new_v=None if n_new_v is None else v_per)
        return hook_gpu_mixed

    k_fns = [quant_mod.get_quant_fn(n) for n in k_names]
    v_fns = [quant_mod.get_quant_fn(n) for n in v_names]
    def hook_cpu_mixed(ctx, n_new_k=None, n_new_v=None):
        k_per = _per_layer_new(n_new_k, k_pending, k_layer_gs)
        v_per = _per_layer_new(n_new_v, v_pending, v_layer_gs)
        if n_new_k is None or any(k_per) or any(v_per):
            parse_state.apply_kv_hook(lib, ctx, llama.ContextPtr,
                                      k_fn=k_fns, v_fn=v_fns,
                                      seq_id=0, n_pos_per_embd=n_pos_per_embd,
                                      n_new_k=None if n_new_k is None else k_per,
                                      n_new_v=None if n_new_v is None else v_per)
    return hook_cpu_mixed


def get_kv_group_sizes(k_names, v_names, default_group_size):
    """Return (k_group_size, v_group_size) for the given K/V quant name lists.
    Per-token quants use group_size=1 (quantize immediately each token).
    Per-channel quants use default_group_size (need multiple tokens for scale).
    If any layer on a side uses per-token quant, group_size=1 for that side.
    """
    any_k_per_tok = any(n in quant_mod.PER_TOKEN_QUANTS for n in k_names)
    any_v_per_tok = any(n in quant_mod.PER_TOKEN_QUANTS for n in v_names)
    k_group_size = 1 if any_k_per_tok else default_group_size
    v_group_size = 1 if any_v_per_tok else default_group_size
    return k_group_size, v_group_size


def main():
    parser = argparse.ArgumentParser(description="KV cache quantization perplexity sweep")
    parser.add_argument("model")
    parser.add_argument("corpus_file",
                        help=".txt for flat mode, .jsonl for structured mode")
    parser.add_argument("--corpus-mode",    default="flat",
                        choices=["flat", "structured"],
                        help="flat: sliding-window over .txt (default). "
                             "structured: per-example (prompt,completion) from .jsonl")
    parser.add_argument("--n-ctx",          type=int, default=128,
                        help="Context window size. In structured mode: cap on "
                             "prompt+completion tokens per example (excess is truncated).")
    parser.add_argument("--n-chunks",       type=int, default=20,
                        help="Flat mode: number of non-overlapping chunks. "
                             "Structured mode: max examples to evaluate (0=all).")
    parser.add_argument("--n-threads",      type=int, default=8)
    parser.add_argument("--n-gpu-layers",   type=int, default=0)
    parser.add_argument("--flash-attn",     action="store_true",
                        help="Enable Flash Attention (required for GPU-side KV quantization).")
    parser.add_argument("--n-pos-per-embd", type=int, default=1,
                        help="Set to 4 for M-RoPE models (e.g. Qwen-VL)")
    parser.add_argument("--n-prompt",       type=int, default=0,
                        help="Flat mode only: leading tokens to treat as prompt. "
                             "In structured mode the prompt length comes from the JSONL.")
    parser.add_argument("--prefill-tokens", type=int, default=0,
                        help="Fixed prefill size for multi-window mode. "
                             "Use with --score-windows. Sets n_ctx = prefill + max(windows).")
    parser.add_argument("--score-windows",  type=int, nargs="+", default=None,
                        help="Multi-window scoring: decode max(windows) tokens once, "
                             "compute PPL at each cutoff (e.g. 512 1024 2048). "
                             "Requires --prefill-tokens. Replaces --n-prompt/--n-ctx.")
    parser.add_argument("--prefill-batch",  action="store_true",
                        help="Flat mode: batch-prefill the prompt. "
                             "Structured mode always uses batch-prefill.")
    parser.add_argument("--quantize-prompt-only", action="store_true",
                        help="Strategy D: quantize prompt KV only, decode stays FP16.")
    parser.add_argument("--quant-group-size",   type=int, default=128,
                        help="Tokens per quantization group during decode (default 128). "
                             "Scales are shared across the group, matching real KV quant systems. "
                             "Larger = more compression error but fewer scales stored.")
    parser.add_argument("--show-text",      action="store_true",
                        help="Print detokenized prompt/completion and fp16 top-1 predictions.")
    parser.add_argument("--show-text-chunk", type=int, default=None,
                        help="Which chunk/example index to show token predictions for. "
                             "Default: show all. Pass a non-negative int to show only that index.")
    parser.add_argument("--eval-accuracy",  action="store_true",
                        help="Structured mode: run greedy generation and compare extracted "
                             "answer against gold. Reports accuracy alongside PPL/KL.")
    parser.add_argument("--max-gen-tokens", type=int, default=512,
                        help="Max new tokens to generate per example for --eval-accuracy (default 512).")
    parser.add_argument("--answer-regex",   default=r"(?:####|[Tt]he answer is)\s*\$?\s*([\d,]+)",
                        help="Regex with one capture group to extract the answer from generated "
                             "text (default matches both GSM8K '#### 42' and 'The answer is 42' formats). "
                             "Ignored when --eval-metric f1.")
    parser.add_argument("--eval-metric",   default="exact", choices=["exact", "f1"],
                        help="How to compare generated answer to gold. "
                             "'exact': regex extraction + string match (default, good for GSM8K). "
                             "'f1': token-overlap F1 against all gold answers in jsonl 'answers' field "
                             "(better for LongBench QA where multiple valid answers exist).")
    parser.add_argument("--stop-strings",  nargs="*", default=["\n\nQuestion:", "\nassistant", "\n\n\n"],
                        help="Stop generation when any of these strings appear in the output. "
                             "Default includes '\\nassistant' to prevent chat role token leakage.")
    parser.add_argument("--out",            default="results.json")
    parser.add_argument("--quants",    nargs="+",
                        default=["fp16","bf16","fp8_e4m3","fp8_e5m2","int8","int8_ch","int4","int4_ch","nf4","int2"])
    parser.add_argument("--quant-k",   default=None,
                        help="K-cache quant spec added as one extra sweep entry alongside --quants. "
                             "Supports layer ranges: 'int8_ch@0-15/int4_ch@16-31'. "
                             "Combined with --quant-v (defaults to fp16 if omitted).")
    parser.add_argument("--quant-v",   default=None,
                        help="V-cache quant spec added as one extra sweep entry alongside --quants. "
                             "Supports layer ranges: 'int8_tok@0-15/int4_tok@16-31'. "
                             "Combined with --quant-k (defaults to fp16 if omitted).")
    args = parser.parse_args()

    # Merge --quant-k / --quant-v into the quants list as one extra entry
    if args.quant_k is not None or args.quant_v is not None:
        k_spec = args.quant_k or "fp16"
        v_spec = args.quant_v or "fp16"
        extra  = f"{k_spec}:{v_spec}"
        if extra not in args.quants:
            args.quants = list(args.quants) + [extra]

    # ── Tee stdout+stderr to a .log file alongside the .json output ──────────
    log_path = (args.out[:-5] if args.out.endswith(".json") else args.out) + ".log"
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, stream, f): self._s, self._f = stream, f
        def write(self, d):  self._s.write(d);  self._f.write(d)
        def flush(self):     self._s.flush();   self._f.flush()
        def fileno(self):    return self._s.fileno()

    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    print(f"command : {' '.join(sys.argv)}", flush=True)
    print(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"log     : {log_path}", flush=True)

    import atexit
    def _close_log():
        print(f"\nfinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()
    atexit.register(_close_log)

    lib = llama.load_lib()
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)
    n_layer = lib.llama_model_n_layer(model)

    def show_chunk_text(prompt_tokens, decode_tokens, label=""):
        """Print full detokenized prompt and decode target."""
        prompt_text = llama.detokenize(lib, vocab, prompt_tokens)
        decode_text = llama.detokenize(lib, vocab, decode_tokens)
        print(f"  [text{' ' + label if label else ''}]", flush=True)
        print(f"    prompt ({len(prompt_tokens)} tokens):", flush=True)
        print(prompt_text, flush=True)
        print(f"    decode target ({len(decode_tokens)} tokens):", flush=True)
        print(decode_text, flush=True)

    # decode_tokens: the full decode window token ids (chunk[n_prefill:] or ct)
    # actual[t] = decode_tokens[t+1]  (model reads decode_tokens[t], predicts decode_tokens[t+1])
    # Per-chunk/example token prediction data.
    # _text_per_chunk[idx] = {"actual_ids": [...], "top1s": {q: [...]},
    #                          "log_probs": {q: [...]}, "diags": {q: {...}}}
    _text_per_chunk = {}

    def _show_this(idx, total):
        """Return True if this chunk/example index should have its data collected."""
        if args.show_text_chunk is not None:
            target = args.show_text_chunk % total if args.show_text_chunk < 0 else args.show_text_chunk
            return idx == target
        return True  # default: collect all

    def show_predictions():
        """Print a table of actual vs predicted tokens for all collected quants.

        Each quant has five sub-columns (all computed without needing ground truth
        except lp):
          token : '-' when prediction matches actual; predicted piece when wrong
          lp    : log-prob of the correct token (needs ground truth; near 0 = confident)
          H     : output entropy in nats (low = sharp/confident, high = confused)
          p     : probability of top-1 prediction (model's own confidence)
          ss    : self-surprisal — log-prob of the previous step's top-1 under the
                  current distribution; drops when model's own continuations become
                  incoherent (NaN for step 0)
        """
        if not _text_per_chunk:
            return

        def _f(v, w, d):
            return f"{'—':>{w}}" if v != v else f"{v:>{w}.{d}f}"

        def _avg(lst):
            return sum(lst) / len(lst) if lst else float("nan")

        tok_w = 12
        lp_w  =  6
        h_w   =  5
        pm_w  =  5
        ss_w  =  6
        act_w = 12
        grp_w = tok_w + 1 + lp_w + 1 + h_w + 1 + pm_w + 1 + ss_w

        for chunk_idx in sorted(_text_per_chunk.keys()):
            cdata       = _text_per_chunk[chunk_idx]
            actual_ids  = cdata["actual_ids"]
            top1s_map   = cdata["top1s"]
            lp_map      = cdata["log_probs"]
            diag_map    = cdata["diags"]
            quant_names = list(top1s_map.keys())

            h1 = f"  {'step':>4}  {'actual':{act_w}}"
            h2 = f"  {'':>4}  {'':>{act_w}}"
            for q in quant_names:
                h1 += f"  {q:{grp_w}}"
                h2 += f"  {'token':{tok_w}} {'lp':>{lp_w}} {'H':>{h_w}} {'p':>{pm_w}} {'ss':>{ss_w}}"
            sep = "  " + "-" * (len(h1) - 2)

            print(f"\n  [token predictions — chunk/ex {chunk_idx}, {len(actual_ids)} steps]", flush=True)
            print(h1, flush=True)
            print(h2, flush=True)
            print(sep, flush=True)

            buckets = {q: {"match": {"lp":[],"H":[],"p_max":[],"ss":[]},
                            "miss":  {"lp":[],"H":[],"p_max":[],"ss":[]}}
                       for q in quant_names}

            for t, actual_id in enumerate(actual_ids):
                actual_piece = llama.token_to_piece(lib, vocab, actual_id).replace("\n", "↵")
                row = f"  {t:>4}  {repr(actual_piece):{act_w}}"
                for q in quant_names:
                    top1s = top1s_map.get(q, [])
                    lps   = lp_map.get(q, [])
                    diags = diag_map.get(q, {})
                    if t < len(top1s):
                        lp    = lps[t]   if t < len(lps)                        else float("nan")
                        H     = diags["H"][t]         if "H"         in diags and t < len(diags["H"])         else float("nan")
                        p_max = diags["p_max"][t]     if "p_max"     in diags and t < len(diags["p_max"])     else float("nan")
                        ss    = diags["self_surp"][t] if "self_surp" in diags and t < len(diags["self_surp"]) else float("nan")
                        matched  = top1s[t] == actual_id
                        tok_cell = "-" if matched else repr(
                            llama.token_to_piece(lib, vocab, top1s[t]).replace("\n", "↵"))
                        row += (f"  {tok_cell:{tok_w}} {_f(lp,lp_w,1)}"
                                f" {_f(H,h_w,2)} {_f(p_max,pm_w,3)} {_f(ss,ss_w,1)}")
                        bkt = buckets[q]["match" if matched else "miss"]
                        for key, val in (("lp", lp), ("H", H), ("p_max", p_max), ("ss", ss)):
                            if val == val:
                                bkt[key].append(val)
                    else:
                        row += f"  {'—':{tok_w}} {'':>{lp_w}} {'':>{h_w}} {'':>{pm_w}} {'':>{ss_w}}"
                print(row, flush=True)

            q_w   = max(len(q) for q in quant_names)
            n_max = max((len(buckets[q][b]["lp"])
                         for q in quant_names for b in ("match", "miss")), default=0)
            n_w   = len(str(max(n_max, 1)))
            lbl_w = 9 + 3 + n_w
            sh = (f"  {'quant':{q_w}}  {'bucket':{lbl_w}}"
                  f"  {'lp':>{lp_w}} {'H':>{h_w}} {'p':>{pm_w}} {'ss':>{ss_w}}")
            print(f"\n  [summary — averages by match/miss]", flush=True)
            print(sh, flush=True)
            print("  " + "-" * (len(sh) - 2), flush=True)
            for q in quant_names:
                for bucket_name, word in (("match", "matched"), ("miss", "unmatched")):
                    bkt = buckets[q][bucket_name]
                    n   = len(bkt["lp"])
                    tag = f"{word:{9}} N={n:{n_w}}"
                    print(f"  {q:{q_w}}  {tag:{lbl_w}}"
                          f"  {_f(_avg(bkt['lp']),  lp_w, 2)}"
                          f" {_f(_avg(bkt['H']),     h_w,  2)}"
                          f" {_f(_avg(bkt['p_max']), pm_w, 3)}"
                          f" {_f(_avg(bkt['ss']),    ss_w, 2)}", flush=True)

    def save_results(results):
        """Write results JSON, appending incrementally after each quant."""
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (saved → {args.out})", flush=True)

    def _f1_score(prediction, gold_list):
        """Token-overlap F1 between prediction and best-matching gold answer (0.0–1.0).
        Standard LongBench evaluation metric.
        """
        import collections
        def _toks(s):
            s = s.lower()
            s = re.sub(r'[^\w\s]', ' ', s)
            return s.split()
        pred_toks = _toks(prediction)
        if not pred_toks:
            return 0.0
        best = 0.0
        for gold in gold_list:
            gold_toks = _toks(gold)
            if not gold_toks:
                continue
            common    = collections.Counter(pred_toks) & collections.Counter(gold_toks)
            n_same    = sum(common.values())
            if n_same == 0:
                continue
            precision = n_same / len(pred_toks)
            recall    = n_same / len(gold_toks)
            f1        = 2 * precision * recall / (precision + recall)
            best      = max(best, f1)
        return best

    def eval_accuracy_pass(examples, gold_answers, kv_hook, k_group_size, v_group_size):
        """Run greedy generation on each example, score against gold, return (score_sum, n_total, per_ex).

        For --eval-metric exact: n_correct / n_total (accuracy).
        For --eval-metric f1:    mean F1 (0–1) over all examples.
        per_ex: list of (gold_display, pred_display, score) tuples.
        """
        use_f1 = (args.eval_metric == "f1")
        ans_re  = re.compile(args.answer_regex) if not use_f1 else None
        score_sum = 0.0
        n_total   = 0
        per_ex    = []
        for ei, (pt, ct, label) in enumerate(examples):
            gold = gold_answers[ei] if gold_answers else None
            max_ctx_avail = args.n_ctx - len(pt) - 1
            max_gen = min(args.max_gen_tokens, max_ctx_avail)
            if max_gen <= 0:
                per_ex.append((gold, None, 0.0))
                continue
            gen_ids  = strategies.run_generate(
                lib, ctx, vocab, pt, n_vocab,
                kv_hook=kv_hook,
                max_new_tokens=max_gen,
                stop_strings=args.stop_strings or None,
                k_group_size=k_group_size,
                v_group_size=v_group_size)
            gen_text = llama.detokenize(lib, vocab, gen_ids, remove_special=False).strip()

            if use_f1:
                gold_list   = gold if isinstance(gold, list) else ([gold] if gold else [])
                score       = _f1_score(gen_text, gold_list) if gold_list else 0.0
                pred_disp   = gen_text[:80].replace("\n", "↵")
                gold_disp   = (gold_list[0] if gold_list else "")[:80]
                mark        = f"{score:.2f}"
                n_total    += 1
                score_sum  += score
            else:
                m           = ans_re.search(gen_text)
                pred        = m.group(1).replace(",", "") if m else None
                gold_disp   = str(gold) if gold is not None else ""
                pred_disp   = str(pred) if pred is not None else "None"
                score       = 1.0 if (pred is not None and pred == gold) else 0.0
                mark        = "✓" if score == 1.0 else "✗"
                if gold is not None:
                    n_total    += 1
                    score_sum  += score

            per_ex.append((gold_disp, pred_disp, score))
            print(f"  acc ex {ei+1}/{len(examples)}: {label} | "
                  f"gold={gold_disp}  pred={pred_disp}  {mark}", flush=True)
            if args.show_text and _show_this(ei, len(examples)):
                print(f"    gen: {gen_text[:300]!r}", flush=True)
        mean_score = score_sum / n_total if n_total > 0 else 0.0
        return mean_score, n_total, per_ex

    use_gpu = args.n_gpu_layers > 0 and HAS_GPU_QUANT and args.flash_attn
    if args.n_gpu_layers > 0 and not args.flash_attn:
        print("NOTE: --n-gpu-layers set but --flash-attn not set. "
              "Using CPU parse_state for KV quantization (slow for per-token quants). "
              "Add --flash-attn to enable GPU-side quantization.", flush=True)
    if use_gpu:
        print(f"GPU-side KV quantization enabled (CuPy, n_layer={n_layer})", flush=True)

    print(f"Model loaded, n_vocab={n_vocab}, n_layer={n_layer}", flush=True)

    # ── Structured mode ───────────────────────────────────────────────────────
    if args.corpus_mode == "structured":
        if not args.corpus_file.endswith(".jsonl"):
            print("WARNING: --corpus-mode structured expects a .jsonl file", flush=True)

        records = []
        with open(args.corpus_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        max_examples = args.n_chunks if args.n_chunks > 0 else len(records)
        records = records[:max_examples]
        print(f"Structured mode: {len(records)} examples from {args.corpus_file}", flush=True)

        # Tokenize all examples; truncate to n_ctx
        examples         = []
        raw_answers_list = []   # parallel list: rec["answers"] or [] per example
        skipped = 0
        for rec in records:
            pt = llama.tokenize(lib, vocab, rec["prompt"])
            ct = llama.tokenize(lib, vocab, rec["completion"])
            total = len(pt) + len(ct)
            if total > args.n_ctx:
                # Truncate prompt to make room for completion
                max_pt = args.n_ctx - len(ct) - 1
                if max_pt < 64:
                    skipped += 1
                    continue   # completion alone already fills context
                pt = pt[-max_pt:]   # keep the END of the prompt (most recent context)
            # Build a short label from whatever identifier fields exist
            if "repo" in rec and "file" in rec:
                label = f"{rec['repo']}/{rec['file']}"
            elif "dataset" in rec:
                label = f"{rec['dataset']}#{rec.get('id', len(examples))}"
            else:
                label = str(rec.get("id", i))
            examples.append((pt, ct, label))
            raw_answers_list.append(rec.get("answers", []))
        print(f"  {len(examples)} examples fit in n_ctx={args.n_ctx} "
              f"({skipped} skipped — completion too long)", flush=True)

        if not examples:
            raise RuntimeError("No examples fit in n_ctx. Increase --n-ctx.")

        # Extract gold answers for accuracy evaluation
        gold_answers = []
        if args.eval_accuracy:
            if args.eval_metric == "f1":
                # Gold = all answers from jsonl 'answers' field; fall back to completion text
                for (pt, ct, _), raw_ans in zip(examples, raw_answers_list):
                    if raw_ans:
                        gold_answers.append(raw_ans)  # list of strings
                    else:
                        gold_answers.append([llama.detokenize(lib, vocab, ct, remove_special=False)])
                print(f"  F1 mode: {len(gold_answers)} examples with gold answers", flush=True)
            else:
                ans_re = re.compile(args.answer_regex)
                for pt, ct, _ in examples:
                    text = llama.detokenize(lib, vocab, ct, remove_special=False)
                    m    = ans_re.search(text)
                    gold_answers.append(m.group(1).replace(",", "") if m else None)
                n_gold = sum(a is not None for a in gold_answers)
                print(f"  Gold answers found: {n_gold}/{len(examples)}", flush=True)

        # n_ctx: cover teacher-forced PPL pass and, if needed, generation pass
        actual_n_ctx = max(len(pt) + len(ct) for pt, ct, _ in examples) + 1
        if args.eval_accuracy:
            gen_n_ctx    = max(len(pt) for pt, _, _ in examples) + args.max_gen_tokens + 1
            actual_n_ctx = max(actual_n_ctx, gen_n_ctx)
        actual_n_ctx = min(actual_n_ctx, args.n_ctx)
        max_prompt   = max(len(pt) for pt, _, _ in examples)

        cparams = lib.llama_context_default_params()
        cparams.n_ctx           = actual_n_ctx
        cparams.n_batch         = max(max_prompt, 512)
        cparams.n_ubatch        = min(max(max_prompt, 512), 512)
        cparams.n_threads       = args.n_threads
        cparams.n_threads_batch = args.n_threads
        cparams.no_perf         = True
        if args.flash_attn:
            cparams.flash_attn_type = 1
        ctx = lib.llama_init_from_model(model, cparams)

        # ── fp16 baseline: collect full log distributions for KL reference ──
        print("\n[fp16 baseline]", flush=True)
        t0 = time.time()
        base_lps_per_ex   = []   # list of log_probs per example
        base_dists_per_ex = []   # list of [n_completion, n_vocab] float16
        for ei, (pt, ct, label) in enumerate(examples):
            if args.show_text and _show_this(ei, len(examples)):
                show_chunk_text(pt, ct, label=label)
            r = strategies.run_structured(lib, ctx, pt, ct, n_vocab,
                                          kv_hook=None,
                                          quantize_prompt_only=False,
                                          return_log_dists=True,
                                          return_top1=(args.show_text and _show_this(ei, len(examples))),
                                          return_diagnostics=(args.show_text and _show_this(ei, len(examples))))
            base_lps_per_ex.append(r.log_probs)
            base_dists_per_ex.append(r.log_dists)
            if args.show_text and _show_this(ei, len(examples)) and r.top1s:
                c = _text_per_chunk.setdefault(ei, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                c["actual_ids"]    = ct[1:len(r.top1s) + 1]
                c["top1s"]["fp16"]     = r.top1s
                c["log_probs"]["fp16"] = r.log_probs
                c["diags"]["fp16"]     = r.diags
            ppl_so_far = math.exp(-np.mean([-lp for ex in base_lps_per_ex for lp in ex]))
            print(f"  ex {ei+1}/{len(examples)}: {label} | ppl={ppl_so_far:.4f}", flush=True)
        fp16_ppl = math.exp(-np.mean([-lp for ex in base_lps_per_ex for lp in ex]))
        elapsed = time.time() - t0
        print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  ({elapsed:.1f}s)", flush=True)

        results = {}
        if "fp16" in args.quants:
            n_tok = sum(len(lps) for lps in base_lps_per_ex)
            entry = {"ppl": fp16_ppl, "mean_kl": 0.0, "n_tokens": n_tok}
            if args.eval_accuracy:
                mean_score, nt, _ = eval_accuracy_pass(examples, gold_answers,
                                                       kv_hook=None,
                                                       k_group_size=args.quant_group_size,
                                                       v_group_size=args.quant_group_size)
                metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                entry[metric_label] = mean_score
                entry["n_total"]    = nt
                print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  {metric_label}={mean_score:.4f}  (n={nt})  ({elapsed:.1f}s)", flush=True)
            results["fp16"] = entry
            save_results(results)

        for quant_name in args.quants:
            if quant_name == "fp16":
                continue
            t0 = time.time()
            print(f"\n[{quant_name}]", flush=True)

            k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
            k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
            kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                                   use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                                   default_group_size=args.quant_group_size)

            all_lp = []
            all_kl = []
            for ei, (pt, ct, label) in enumerate(examples):
                r = strategies.run_structured(lib, ctx, pt, ct, n_vocab,
                                              kv_hook=kv_hook,
                                              quantize_prompt_only=args.quantize_prompt_only,
                                              k_group_size=k_group_size,
                                              v_group_size=v_group_size,
                                              base_log_dists=base_dists_per_ex[ei],
                                              return_top1=(args.show_text and _show_this(ei, len(examples))),
                                              return_diagnostics=(args.show_text and _show_this(ei, len(examples))))
                all_lp.extend(r.log_probs)
                all_kl.extend(r.kl_divs)
                if args.show_text and _show_this(ei, len(examples)) and r.top1s:
                    c = _text_per_chunk.setdefault(ei, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                    c["top1s"][quant_name]     = r.top1s
                    c["log_probs"][quant_name] = r.log_probs
                    c["diags"][quant_name]     = r.diags
                ppl_so_far = math.exp(-sum(all_lp) / len(all_lp))
                kl_so_far  = sum(all_kl) / len(all_kl)
                print(f"  ex {ei+1}/{len(examples)}: {label} | "
                      f"ppl={ppl_so_far:.4f}  kl={kl_so_far:.4f}", flush=True)

            ppl     = math.exp(-sum(all_lp) / len(all_lp))
            mean_kl = sum(all_kl) / len(all_kl)
            elapsed = time.time() - t0
            entry = {"ppl": ppl, "mean_kl": mean_kl, "n_tokens": len(all_lp)}
            if args.eval_accuracy:
                mean_score, nt, _ = eval_accuracy_pass(examples, gold_answers,
                                                       kv_hook=kv_hook,
                                                       k_group_size=k_group_size,
                                                       v_group_size=v_group_size)
                metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                entry[metric_label] = mean_score
                entry["n_total"]    = nt
                print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  {metric_label}={mean_score:.4f}  (n={nt})  ({elapsed:.1f}s)", flush=True)
            else:
                print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  ({elapsed:.1f}s)", flush=True)
            results[quant_name] = entry
            save_results(results)

        if args.show_text:
            show_predictions()
        lib.llama_free(ctx)
        lib.llama_model_free(model)
        lib.llama_backend_free()
        return

    # ── Flat mode (default) ───────────────────────────────────────────────────
    text = open(args.corpus_file, encoding="utf-8").read()
    all_tokens = llama.tokenize(lib, vocab, text)
    print(f"Total tokens: {len(all_tokens)}", flush=True)

    # ── Multi-window mode: fixed prefill, decode once, PPL at multiple cutoffs ─
    if args.score_windows:
        if not args.prefill_tokens:
            raise ValueError("--score-windows requires --prefill-tokens")
        score_windows = sorted(args.score_windows)
        max_window    = score_windows[-1]
        n_prefill     = args.prefill_tokens
        chunk_size    = n_prefill + max_window

        n_available = (len(all_tokens) - chunk_size) // chunk_size + 1
        n_want = min(args.n_chunks, n_available)
        stride = max(chunk_size, len(all_tokens) // n_want) if n_want > 1 else chunk_size
        chunks = []
        for c in range(n_want):
            start = c * stride
            end   = start + chunk_size
            if end > len(all_tokens):
                break
            chunks.append(all_tokens[start:end])

        print(f"Multi-window mode: {len(chunks)} chunks (strided across corpus), "
              f"prefill={n_prefill}, decode={max_window}, windows={score_windows}", flush=True)

        n_batch = max(n_prefill, 512)
        cparams = lib.llama_context_default_params()
        cparams.n_ctx           = chunk_size
        cparams.n_batch         = n_batch
        cparams.n_ubatch        = min(n_batch, 512)
        cparams.n_threads       = args.n_threads
        cparams.n_threads_batch = args.n_threads
        cparams.no_perf         = True
        if args.flash_attn:
            cparams.flash_attn_type = 1
        ctx = lib.llama_init_from_model(model, cparams)

        # ── fp16 baseline: collect full log distributions for KL reference ──
        print("\n[fp16 baseline]", flush=True)
        t0 = time.time()
        base_lps_per_chunk   = []
        base_dists_per_chunk = []   # [n_chunks][max_window, n_vocab] float16
        for ci, chunk in enumerate(chunks):
            if args.show_text and _show_this(ci, len(chunks)):
                show_chunk_text(chunk[:n_prefill], chunk[n_prefill:],
                                label=f"chunk {ci+1}/{len(chunks)}")
            r = strategies.run_chunk_batch_prefill(
                lib, ctx, chunk, n_vocab, None,
                n_prompt=n_prefill,
                return_log_dists=True,
                return_top1=(args.show_text and _show_this(ci, len(chunks))),
                return_diagnostics=(args.show_text and _show_this(ci, len(chunks))))
            base_lps_per_chunk.append(r.log_probs)
            base_dists_per_chunk.append(r.log_dists)
            if args.show_text and _show_this(ci, len(chunks)) and r.top1s:
                c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                c["actual_ids"]        = [chunk[n_prefill + t + 1] for t in range(len(r.top1s))]
                c["top1s"]["fp16"]     = r.top1s
                c["log_probs"]["fp16"] = r.log_probs
                c["diags"]["fp16"]     = r.diags
            ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
            print(f"  chunk {ci+1}/{len(chunks)}: ppl@{max_window}={ppl:.4f}", flush=True)
        elapsed = time.time() - t0
        fp16_entry = {}
        for w in score_windows:
            flat = [lp for lps in base_lps_per_chunk for lp in lps[:w]]
            fp16_entry[f"ppl_{w}"] = math.exp(-sum(flat) / len(flat))
        fp16_entry["mean_kl"] = 0.0
        ppl_str = "  ".join(f"ppl@{w}={fp16_entry[f'ppl_{w}']:.4f}" for w in score_windows)
        print(f"  => {ppl_str}  kl=0.000  ({elapsed:.1f}s)", flush=True)

        results = {}
        if "fp16" in args.quants:
            results["fp16"] = fp16_entry
            save_results(results)

        for quant_name in args.quants:
            if quant_name == "fp16":
                continue
            t0 = time.time()
            print(f"\n[{quant_name}]", flush=True)

            k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
            k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
            kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                                   use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                                   default_group_size=args.quant_group_size)

            all_lps = []
            all_kls = []
            for ci, chunk in enumerate(chunks):
                r = strategies.run_chunk_batch_prefill(
                    lib, ctx, chunk, n_vocab, kv_hook,
                    n_prompt=n_prefill,
                    _quantize_prompt_only=args.quantize_prompt_only,
                    k_group_size=k_group_size,
                    v_group_size=v_group_size,
                    base_log_dists=base_dists_per_chunk[ci],
                    return_top1=(args.show_text and _show_this(ci, len(chunks))),
                    return_diagnostics=(args.show_text and _show_this(ci, len(chunks))))
                all_lps.append(r.log_probs)
                all_kls.extend(r.kl_divs)
                if args.show_text and _show_this(ci, len(chunks)) and r.top1s:
                    c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                    c["top1s"][quant_name]     = r.top1s
                    c["log_probs"][quant_name] = r.log_probs
                    c["diags"][quant_name]     = r.diags
                ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
                kl  = sum(r.kl_divs) / len(r.kl_divs)
                print(f"  chunk {ci+1}/{len(chunks)}: ppl@{max_window}={ppl:.4f}  kl={kl:.4f}", flush=True)

            entry = {}
            for w in score_windows:
                flat = [lp for lps in all_lps for lp in lps[:w]]
                entry[f"ppl_{w}"] = math.exp(-sum(flat) / len(flat))
            entry["mean_kl"] = sum(all_kls) / len(all_kls)
            elapsed = time.time() - t0
            ppl_str = "  ".join(f"ppl@{w}={entry[f'ppl_{w}']:.4f}" for w in score_windows)
            print(f"  => {ppl_str}  kl={entry['mean_kl']:.4f}  ({elapsed:.1f}s)", flush=True)
            results[quant_name] = entry
            save_results(results)

        if args.show_text:
            show_predictions()
        print(f"\nDone. Results in {args.out}")
        lib.llama_free(ctx)
        lib.llama_model_free(model)
        lib.llama_backend_free()
        return

    # ── Standard flat mode ────────────────────────────────────────────────────
    n_available = (len(all_tokens) - args.n_ctx) // args.n_ctx + 1
    n_want = min(args.n_chunks, n_available)
    stride = max(args.n_ctx, len(all_tokens) // n_want) if n_want > 1 else args.n_ctx
    chunks = []
    for c in range(n_want):
        start = c * stride
        end   = start + args.n_ctx
        if end > len(all_tokens):
            break
        chunks.append(all_tokens[start:end])

    n_measured = args.n_ctx - args.n_prompt - 1
    assert n_measured > 0, f"--n-prompt ({args.n_prompt}) must be < n-ctx-1 ({args.n_ctx - 1})"
    if args.prefill_batch and args.n_prompt < 1:
        raise ValueError("--prefill-batch requires --n-prompt >= 1")
    if args.quantize_prompt_only and not args.prefill_batch:
        raise ValueError("--quantize-prompt-only requires --prefill-batch")
    if args.quantize_prompt_only:
        mode = "batch-prefill+prompt-only-quant"
    elif args.prefill_batch:
        mode = "batch-prefill"
    else:
        mode = "token-by-token"
    print(f"Running {len(chunks)} chunks × {args.n_ctx} tokens each "
          f"(prompt={args.n_prompt} [{mode}], measured={n_measured})", flush=True)

    n_batch = max(args.n_prompt, 512) if args.prefill_batch else 512
    cparams = lib.llama_context_default_params()
    cparams.n_ctx           = args.n_ctx
    cparams.n_batch         = n_batch
    cparams.n_ubatch        = min(n_batch, 512)
    cparams.n_threads       = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.no_perf         = True
    if args.flash_attn:
        cparams.flash_attn_type = 1
    ctx = lib.llama_init_from_model(model, cparams)

    # ── fp16 baseline: collect full log distributions for KL reference ──────
    def _run_one_flat(chunk, kv_hook, k_gs, v_gs, base_dists, ret_dists,
                      ret_top1=False, ret_diags=False):
        if args.prefill_batch:
            return strategies.run_chunk_batch_prefill(
                lib, ctx, chunk, n_vocab, kv_hook,
                n_prompt=args.n_prompt,
                _quantize_prompt_only=args.quantize_prompt_only,
                k_group_size=k_gs, v_group_size=v_gs,
                base_log_dists=base_dists, return_log_dists=ret_dists,
                return_top1=ret_top1, return_diagnostics=ret_diags)
        else:
            return strategies.run_chunk_token_by_token(
                lib, ctx, chunk, n_vocab, kv_hook,
                n_prompt=args.n_prompt,
                k_group_size=k_gs, v_group_size=v_gs,
                base_log_dists=base_dists, return_log_dists=ret_dists,
                return_top1=ret_top1, return_diagnostics=ret_diags)

    print("\n[fp16 baseline]", flush=True)
    t0 = time.time()
    base_lps_per_chunk   = []
    base_dists_per_chunk = []
    for ci, chunk in enumerate(chunks):
        if args.show_text and _show_this(ci, len(chunks)):
            show_chunk_text(chunk[:args.n_prompt], chunk[args.n_prompt:])
        r = _run_one_flat(chunk, None, args.quant_group_size, args.quant_group_size,
                          base_dists=None, ret_dists=True,
                          ret_top1=(args.show_text and _show_this(ci, len(chunks))),
                          ret_diags=(args.show_text and _show_this(ci, len(chunks))))
        base_lps_per_chunk.append(r.log_probs)
        base_dists_per_chunk.append(r.log_dists)
        if args.show_text and _show_this(ci, len(chunks)) and r.top1s:
            c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
            c["actual_ids"]        = [chunk[args.n_prompt + t + 1] for t in range(len(r.top1s))]
            c["top1s"]["fp16"]     = r.top1s
            c["log_probs"]["fp16"] = r.log_probs
            c["diags"]["fp16"]     = r.diags
        ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
        print(f"  chunk {ci+1}/{len(chunks)}: ppl={ppl:.4f}", flush=True)
    all_base_lps = [lp for lps in base_lps_per_chunk for lp in lps]
    fp16_ppl = math.exp(-sum(all_base_lps) / len(all_base_lps))
    elapsed  = time.time() - t0
    print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  ({elapsed:.1f}s)", flush=True)

    results = {}
    if "fp16" in args.quants:
        results["fp16"] = {"ppl": fp16_ppl, "mean_kl": 0.0, "n_tokens": len(all_base_lps)}
        save_results(results)

    for quant_name in args.quants:
        if quant_name == "fp16":
            continue
        t0 = time.time()
        print(f"\n[{quant_name}]", flush=True)

        k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
        k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
        kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                               use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                               default_group_size=args.quant_group_size)

        all_lp = []
        all_kl = []
        for ci, chunk in enumerate(chunks):
            r = _run_one_flat(chunk, kv_hook, k_group_size, v_group_size,
                              base_dists=base_dists_per_chunk[ci], ret_dists=False,
                              ret_top1=(args.show_text and _show_this(ci, len(chunks))),
                              ret_diags=(args.show_text and _show_this(ci, len(chunks))))
            all_lp.extend(r.log_probs)
            all_kl.extend(r.kl_divs)
            if args.show_text and _show_this(ci, len(chunks)) and r.top1s:
                c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                c["top1s"][quant_name]     = r.top1s
                c["log_probs"][quant_name] = r.log_probs
                c["diags"][quant_name]     = r.diags
            ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
            kl  = sum(r.kl_divs) / len(r.kl_divs)
            print(f"  chunk {ci+1}/{len(chunks)}: ppl={ppl:.4f}  kl={kl:.4f}", flush=True)

        ppl     = math.exp(-sum(all_lp) / len(all_lp))
        mean_kl = sum(all_kl) / len(all_kl)
        elapsed = time.time() - t0
        print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  ({elapsed:.1f}s)", flush=True)
        results[quant_name] = {"ppl": ppl, "mean_kl": mean_kl, "n_tokens": len(all_lp)}
        save_results(results)

    if args.show_text:
        show_predictions()
    print(f"\nDone. Results in {args.out}")

    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()


if __name__ == "__main__":
    main()
