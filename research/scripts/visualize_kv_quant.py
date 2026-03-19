#!/usr/bin/env python3
"""
visualize_kv_quant.py — Text visualization of KV cache quantization effects.

Extracts a real K/V slice from a model forward pass and prints, for each quant:
  - fp16 original values
  - quantized values with recovered integers (e.g. +0.1234(  27))
  - per-token scale appended as extra column  (tok quants)
  - per-channel scale as extra row            (ch quants)
  - error matrix |quant − fp16| with RMS / max

Output is plain text written to --out.  Best viewed with:  less -S kv_viz.txt

Usage:
  # single quant
  python3 research/scripts/visualize_kv_quant.py model.gguf \\
      --quants int8_ch \\
      --text-file research/data/wikitext2_test.txt \\
      --n-threads 8 --out kv_viz.txt

  # compare multiple quants; show 16×16 cells
  python3 research/scripts/visualize_kv_quant.py model.gguf \\
      --quants int8_ch int4_ch int2_ch int4_ch:int4_tok \\
      --layer 16 --head 0 --n-show 16 --out kv_viz.txt

  # --quant (singular) still works as a shorthand for one quant

Note: Q capture (for attention analysis) requires --n-gpu-layers 0 (default).
The eval callback reads tensor data via llama_tensor_data() which returns a CPU pointer.
"""

import argparse
import ctypes
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import llama_bindings as llama
import parse_state
import quant as quant_mod


# ── Q capture via eval callback ───────────────────────────────────────────────

def make_q_capture_cb(lib, target_layer, head_dim):
    """
    Return (eval_cb, q_store) where:
      eval_cb  — EVAL_CB_TYPE callback; assign to cparams.cb_eval before context creation
      q_store  — dict; after llama_decode, q_store[target_layer] is
                 float16 ndarray [n_tok, n_head, head_dim]  (post-RoPE Q)

    Captures the last "Qcur-{target_layer}" tensor seen whose ne[0]==head_dim
    (distinguishes 3-D post-reshape/post-RoPE tensor from the 2-D matmul output).
    Requires n_gpu_layers=0 (CPU tensors only; data pointer via llama_tensor_data).
    """
    q_store = {}
    target_name = f"Qcur-{target_layer}".encode()

    def _cb(tensor_ptr, ask, _user_data):
        name = lib.ggml_get_name(tensor_ptr)
        if name != target_name:
            return True
        if ask:
            return True   # request data sync (no-op for CPU backend)
        ne0 = lib.llama_tensor_ne(tensor_ptr, 0)
        if int(ne0) != head_dim:
            return True   # skip 2-D matmul output (ne0 = n_head * head_dim)
        ne1 = lib.llama_tensor_ne(tensor_ptr, 1)
        ne2 = lib.llama_tensor_ne(tensor_ptr, 2)
        n_el    = lib.ggml_nelements(tensor_ptr)
        n_bytes = lib.ggml_nbytes(tensor_ptr)
        data_ptr = lib.llama_tensor_data(tensor_ptr)
        if data_ptr and int(n_el) > 0:
            bytes_per = int(n_bytes) // int(n_el)
            raw = ctypes.string_at(data_ptr, int(n_bytes))
            if bytes_per == 4:
                arr = np.frombuffer(raw, dtype=np.float32).copy().astype(np.float16)
            elif bytes_per == 2:
                arr = np.frombuffer(raw, dtype=np.float16).copy()
            else:
                return True  # unexpected type; skip
            # GGML layout [ne0=head_dim, ne1=n_head, ne2=n_tok] → NumPy [n_tok, n_head, head_dim]
            q_store[target_layer] = arr.reshape(int(ne2), int(ne1), int(ne0))
        return True

    cb = llama.EVAL_CB_TYPE(_cb)
    return cb, q_store


# ── Attention analysis ─────────────────────────────────────────────────────────

def _attn_analysis_lines(Q_head, K_fp16, V_fp16, attn_entries, n_show, sep, label):
    """
    Q_head     : float32 [n_show, head_dim]  — real Q vectors (post-RoPE)
    K_fp16     : float32 [n_show, head_dim]  — fp16 K slice (full context for this head)
    V_fp16     : float32 [n_show, head_dim_v]
    attn_entries : list of (spec, K_q [n_show, head_dim], V_q [n_show, head_dim_v])

    Outputs per quant:
      - K/V error L2 norm per token and per dim
      - causal attention KL(fp16 || quant) per query token
      - output error decomposed into K-part and V-part
    """
    lines   = []
    nt      = len(Q_head)
    head_dim = Q_head.shape[1]
    sqrt_d  = float(head_dim) ** 0.5
    nd      = min(n_show, head_dim)
    eps     = 1e-10
    _FW     = 8    # field width for norms

    def _softmax(x):
        x = x - x.max()
        e = np.exp(x.astype(np.float64))
        return (e / e.sum()).astype(np.float32)

    for spec, K_q, V_q in attn_entries:
        K_q32 = K_q.astype(np.float32)
        V_q32 = V_q.astype(np.float32)

        K_err = K_q32 - K_fp16
        V_err = V_q32 - V_fp16
        k_per_tok = np.linalg.norm(K_err, axis=1)[:nt]
        k_per_dim = np.linalg.norm(K_err, axis=0)[:nd]
        v_per_tok = np.linalg.norm(V_err, axis=1)[:nt]
        v_per_dim = np.linalg.norm(V_err, axis=0)[:nd]

        lines.append('')
        lines.append(sep)
        lines.append(f'  {spec}  error norms  ({label})')
        lines.append(sep)
        lines.append('  K per-token L2: ' + '  '.join(f'{v:.5f}' for v in k_per_tok))
        lines.append('  K per-dim   L2: ' + '  '.join(f'{v:.5f}' for v in k_per_dim))
        lines.append('  V per-token L2: ' + '  '.join(f'{v:.5f}' for v in v_per_tok))
        lines.append('  V per-dim   L2: ' + '  '.join(f'{v:.5f}' for v in v_per_dim))

        lines.append('')
        lines.append(f'  {spec}  causal attention analysis  ({label})')
        lines.append(f'    {"tok":>4}  {"attn_KL":>10}  {"out_err":>10}  {"K_part":>10}  {"V_part":>10}')

        for t in range(nt):
            q_t      = Q_head[t]
            k_fp16_t = K_fp16[:t + 1]
            k_q_t    = K_q32[:t + 1]
            v_fp16_t = V_fp16[:t + 1]
            v_q_t    = V_q32[:t + 1]

            a_fp16 = _softmax((q_t @ k_fp16_t.T) / sqrt_d)
            a_q    = _softmax((q_t @ k_q_t.T)    / sqrt_d)

            kl = float(np.sum(a_fp16 * np.log((a_fp16 + eps) / (a_q + eps))))

            err_K = (a_q - a_fp16) @ v_fp16_t   # attn-weight shift × fp16 V
            err_V =  a_q           @ (v_q_t - v_fp16_t)   # quant attn × V error

            lines.append(
                f'    tok {t:>3}:  '
                f'{kl:>10.6f}  '
                f'{float(np.linalg.norm(err_K + err_V)):>10.6f}  '
                f'{float(np.linalg.norm(err_K)):>10.6f}  '
                f'{float(np.linalg.norm(err_V)):>10.6f}'
            )

    return lines


# ── Scale computation ─────────────────────────────────────────────────────────

def _compute_scales(arr: np.ndarray, quant_name: str):
    """
    Return (scale_row, scale_col):
      scale_row : per-token  scales [n_tokens] or None  (for _tok quants)
      scale_col : per-channel scales [head_dim] or None  (for _ch quants)
    """
    m = re.match(r"int(\d+)(?:_(ch|tok))?(?:_g\d+)?$", quant_name)
    if m is None:
        return None, None
    bits   = int(m.group(1))
    suffix = m.group(2)
    n_lev  = (1 << (bits - 1)) - 1
    f32    = arr.astype(np.float32)
    if suffix == "ch":
        return None, np.abs(f32).max(axis=0) / n_lev     # [head_dim]
    elif suffix == "tok":
        return np.abs(f32).max(axis=1) / n_lev, None     # [n_tokens]
    return None, None


def _side_lines(side_label, fp16_sl, quant_entries, n_verbose):
    """
    Build all output lines for one KV side (K or V).

    quant_entries : list of (quant_name, quant_sl, scale_row, scale_col)
    """
    lines  = []
    n_tok, n_dim = fp16_sl.shape
    fp32   = fp16_sl.astype(np.float32)

    sep  = '─' * 74
    _CEL = 7      # float cell width (+7.3f)
    _SEP = '  '
    nt   = min(n_verbose, n_tok)
    nd   = min(n_verbose, n_dim)

    lines.append('')
    lines.append(sep)
    lines.append(f'  {side_label}  tokens 0–{nt-1}, dims 0–{nd-1}')
    lines.append(sep)

    # ── fp16 baseline ─────────────────────────────────────────────────────────
    lines.append('  fp16:')
    for i in range(nt):
        vals = _SEP.join(f'{float(fp16_sl[i, j]):+{_CEL}.3f}' for j in range(nd))
        lines.append(f'    tok {i}: {vals}')

    # ── one block per quant ────────────────────────────────────────────────────
    for quant_name, quant_sl, scale_row, scale_col in quant_entries:
        error = quant_sl.astype(np.float32) - fp32
        rms_e = float(np.sqrt((error ** 2).mean()))
        max_e = float(np.abs(error).max())

        # Parse bits for integer recovery
        _m     = re.match(r'int(\d+)', quant_name)
        _bits  = int(_m.group(1)) if _m else None
        _n_lev = ((1 << (_bits - 1)) - 1) if _bits else None
        _ipad  = '      ' if _n_lev is not None else ''   # 6 spaces = width of (int:4d)

        def _cell(val, i, j):
            f_str = f'{float(val):+{_CEL}.3f}'
            if _n_lev is None:
                return f_str
            sc = (float(scale_col[j]) if scale_col is not None else
                  float(scale_row[i]) if scale_row is not None else None)
            if sc is None or sc == 0.0:
                return f_str
            iv = int(round(float(val) / sc))
            iv = max(-_n_lev, min(_n_lev, iv))
            return f'{f_str}({iv:4d})'

        lines.append(f'  {quant_name}:  RMS={rms_e:.4f}  max={max_e:.4f}')
        for i in range(nt):
            vals = _SEP.join(_cell(quant_sl[i, j], i, j) for j in range(nd))
            suffix = (f'  | {float(scale_row[i]):{_CEL}.4f}'
                      if scale_row is not None else '')
            lines.append(f'    tok {i}: {vals}{suffix}')
        if scale_col is not None:
            sc_row = _SEP.join(f'{float(scale_col[j]):{_CEL}.4f}{_ipad}' for j in range(nd))
            lines.append(f'    scale  : {sc_row}')
        lines.append(f'  {quant_name} error:')
        for i in range(nt):
            vals = _SEP.join(f'{float(error[i, j]):+{_CEL}.3f}{_ipad}' for j in range(nd))
            lines.append(f'    tok {i}: {vals}')

    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Text visualisation of KV cache quantization")
    parser.add_argument("model")
    parser.add_argument("--quants", nargs="+", default=None,
                        help="One or more quant specs to compare: "
                             "int8_ch int4_ch int2_ch int4_ch:int4_tok ...")
    parser.add_argument("--quant",  default=None,
                        help="Single quant spec (shorthand for --quants Q)")
    parser.add_argument("--text",         default=None,
                        help="Input text (inline)")
    parser.add_argument("--text-file",    default=None,
                        help="Path to text file (first --n-tokens tokens used)")
    parser.add_argument("--n-tokens",     type=int, default=128,
                        help="Tokens to prefill (default 128, max 128 shown)")
    parser.add_argument("--token-offset", type=int, default=0,
                        help="Skip this many tokens into the corpus before sampling (default 0)")
    parser.add_argument("--layer",        type=int, default=-1,
                        help="Layer to visualize (default: n_layer // 2)")
    parser.add_argument("--head",         type=int, default=0,
                        help="KV head index (default 0)")
    parser.add_argument("--head-dim",     type=int, default=128,
                        help="Head dimension (default 128)")
    parser.add_argument("--n-threads",    type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-pos-per-embd", type=int, default=1)
    parser.add_argument("--n-show",       type=int, default=8,
                        help="Rows × cols to print per block (default 8)")
    parser.add_argument("--out",          default="kv_viz.txt")
    args = parser.parse_args()

    # Resolve quant list
    quant_list = list(args.quants) if args.quants else []
    if args.quant and args.quant not in quant_list:
        quant_list.append(args.quant)
    if not quant_list:
        parser.error("Provide at least one quant via --quants or --quant")

    # ── Text ──────────────────────────────────────────────────────────────────
    if args.text_file:
        # Read only as much text as needed to obtain token_offset + n_tokens tokens.
        # Rough estimate: 6 bytes per token; double it for safety.
        char_limit = (args.token_offset + args.n_tokens) * 12 + 4096
        with open(args.text_file, encoding="utf-8") as f:
            raw_text = f.read(char_limit)
    elif args.text:
        raw_text = args.text
    else:
        raw_text = (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
            "company designed and built the tower from 1887 to 1889 as the entrance "
            "arch to the 1889 World's Fair. The tower is 330 metres tall and was the "
            "tallest man-made structure in the world for 41 years. It is now the most "
            "visited monument in the world. The tower has three levels for visitors. "
            "The top level's upper platform is the highest observation deck in the EU. "
        ) * 4

    # ── Load model ────────────────────────────────────────────────────────────
    lib = llama.load_lib()
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model   = lib.llama_model_load_from_file(args.model.encode(), mparams)
    vocab   = lib.llama_model_get_vocab(model)
    n_layer = lib.llama_model_n_layer(model)

    layer_idx = args.layer if args.layer >= 0 else n_layer // 2
    print(f"model  : {args.model}", flush=True)
    print(f"n_layer: {n_layer}  →  visualizing layer {layer_idx}", flush=True)
    print(f"quants : {quant_list}", flush=True)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    all_tokens = llama.tokenize(lib, vocab, raw_text)
    if len(all_tokens) < args.n_tokens:
        print(f"WARNING: only {len(all_tokens)} tokens available "
              f"(need {args.n_tokens}). Use --text-file with longer text.")
    tokens = all_tokens[args.token_offset : args.token_offset + args.n_tokens]
    n_tok  = len(tokens)
    print(f"tokens : {n_tok}", flush=True)

    # ── Eval callback for Q capture ───────────────────────────────────────────
    q_cb, q_store = make_q_capture_cb(lib, layer_idx, args.head_dim)

    # ── Prefill ───────────────────────────────────────────────────────────────
    cparams = lib.llama_context_default_params()
    cparams.n_ctx           = n_tok + 1
    cparams.n_batch         = n_tok
    cparams.n_ubatch        = min(n_tok, 512)
    cparams.n_threads       = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.no_perf         = True
    cparams.cb_eval         = ctypes.cast(q_cb, ctypes.c_void_p)
    ctx = lib.llama_init_from_model(model, cparams)

    batch = lib.llama_batch_init(n_tok, 0, 1)
    for i, tok in enumerate(tokens):
        batch.token[i]     = tok
        batch.pos[i]       = i
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]    = 0
    batch.n_tokens          = n_tok
    batch.logits[n_tok - 1] = 1
    ret = lib.llama_decode(ctx, batch)
    assert ret == 0, f"llama_decode failed: {ret}"
    lib.llama_batch_free(batch)
    print("prefill done", flush=True)

    # ── Extract captured Q ────────────────────────────────────────────────────
    if layer_idx in q_store:
        q_raw   = q_store[layer_idx]               # [n_tok, n_head, head_dim]
        Q_head  = q_raw[:, args.head, :].astype(np.float32)  # [n_tok, head_dim]
        print(f"Q captured: {q_raw.shape}  → head {args.head}: {Q_head.shape}", flush=True)
    else:
        Q_head  = None
        print("WARNING: Q not captured — attention analysis unavailable", flush=True)

    # ── Extract KV state ──────────────────────────────────────────────────────
    seq_id    = 0
    sz        = lib.llama_state_seq_get_size(ctx, seq_id)
    buf       = ctypes.create_string_buffer(sz)
    n_written = lib.llama_state_seq_get_data(ctx, buf, sz, seq_id)
    assert n_written > 0, "llama_state_seq_get_data returned 0"
    state = parse_state.parse_kv_state(bytes(buf[:n_written]), args.n_pos_per_embd)
    print(f"KV state: {state.n_layer} layers, {state.cell_count} cells", flush=True)

    if layer_idx >= state.n_layer:
        raise ValueError(
            f"--layer {layer_idx} out of range (model has {state.n_layer} layers)")

    k_full = state.k[layer_idx]
    v_full = state.v[layer_idx]

    n_show  = n_tok   # use all prefilled tokens for the analysis
    hd      = args.head_dim
    h_start = args.head * hd
    h_end   = h_start + hd

    if h_end > k_full.shape[1]:
        raise ValueError(
            f"head {args.head} with head_dim {hd} → dims {h_start}:{h_end} "
            f"exceeds k_dim={k_full.shape[1]}. "
            f"Try --head 0 or --head-dim {k_full.shape[1]}.")

    k_slice = k_full[:n_show, h_start:h_end].copy()
    v_h_end = min(h_end, v_full.shape[1])
    v_slice = v_full[:n_show, h_start:v_h_end].copy()
    print(f"K slice: {k_slice.shape}  V slice: {v_slice.shape}", flush=True)

    # ── Apply all quants ──────────────────────────────────────────────────────
    # k_entries / v_entries : list of (quant_name, quant_sl, scale_row, scale_col)
    k_entries = []
    v_entries = []
    for spec in quant_list:
        k_name, v_name = (spec.split(":", 1) if ":" in spec else (spec, spec))
        k_fn    = quant_mod.get_quant_fn(k_name)
        v_fn    = quant_mod.get_quant_fn(v_name)
        k_q     = k_fn(k_slice)
        v_q     = v_fn(v_slice)
        k_sr, k_sc = _compute_scales(k_slice, k_name)
        v_sr, v_sc = _compute_scales(v_slice, v_name)
        k_entries.append((spec, k_q, k_sr, k_sc))
        v_entries.append((spec, v_q, v_sr, v_sc))

    # ── Build output ──────────────────────────────────────────────────────────
    out = []
    out.append('=' * 74)
    out.append('  KV Cache Quantization Visualization')
    out.append('=' * 74)
    out.append(f'  model   : {os.path.basename(args.model)}')
    out.append(f'  layer   : {layer_idx} / {n_layer}')
    out.append(f'  head    : {args.head}  (dims {h_start}–{h_end-1})')
    out.append(f'  tokens  : {n_show} × {hd} dims')
    out.append(f'  quants  : {", ".join(quant_list)}')
    out.append(f'  K range : [{float(k_slice.min()):.4f}, {float(k_slice.max()):.4f}]')
    out.append(f'  V range : [{float(v_slice.min()):.4f}, {float(v_slice.max()):.4f}]')
    out.append(f'  showing  : {args.n_show} tokens × {args.n_show} dims per block')
    out.append(f'  format   : float(+8.4f) + integer(int:4d) where applicable')
    out.append(f'  per-token scale: appended as | sc=... on each row')
    out.append(f'  per-chan  scale: extra "scale" row below each quant block')

    out += _side_lines(
        side_label    = f'K  layer={layer_idx}  head={args.head}',
        fp16_sl       = k_slice,
        quant_entries = k_entries,
        n_verbose     = args.n_show,
    )

    out += _side_lines(
        side_label    = f'V  layer={layer_idx}  head={args.head}',
        fp16_sl       = v_slice,
        quant_entries = v_entries,
        n_verbose     = args.n_show,
    )

    # ── Attention analysis (requires captured Q) ──────────────────────────────
    if Q_head is not None:
        n_attn = min(len(Q_head), n_show)
        attn_entries = [
            (spec, k_q[:n_attn].astype(np.float32), v_q[:n_attn].astype(np.float32))
            for (spec, k_q, _, _), (_, v_q, _, _) in zip(k_entries, v_entries)
        ]
        out += _attn_analysis_lines(
            Q_head    = Q_head[:n_attn],
            K_fp16    = k_slice[:n_attn].astype(np.float32),
            V_fp16    = v_slice[:n_attn].astype(np.float32),
            attn_entries = attn_entries,
            n_show    = n_attn,
            sep       = '─' * 74,
            label     = f'layer={layer_idx}  head={args.head}',
        )

    out.append('')
    out.append('=' * 74)

    text = '\n'.join(out) + '\n'
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nWrote {len(out)} lines → {args.out}")
    print(f"View with:  less -S {args.out}")

    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()


if __name__ == "__main__":
    main()
