"""
parse_state.py — parse and modify K/V cache via llama_state_seq_get/set_data.

The state blob format for llama_state_seq_get_data(ctx, buf, size, seq_id=0):

  uint32  n_stream          (always 1 for seq-level state)
  for each stream s (0..n_stream-1):
    uint32  cell_count      (number of used KV cells for this sequence)
    if cell_count > 0:
      [meta section]  — cell_count * (4+4+n_seq_id*4) bytes
        for each cell:
          int32   pos
          uint32  n_seq_id
          [if n_pos_per_embd > 1: llama_kv_cell_ext (skipped for standard models)]
          int32   seq_id  * n_seq_id
      [data section]
        uint32  v_trans
        uint32  n_layer
        for each layer (K tensors):
          int32   k_type_i     (ggml type: 1=F16)
          uint64  k_size_row   (bytes per cell)
          uint8[] raw K data   (k_size_row * cell_count bytes, all ranges concat'd)
        for each layer (V tensors, if !v_trans):
          int32   v_type_i
          uint64  v_size_row
          uint8[] raw V data   (v_size_row * cell_count bytes)
        for each layer (V tensors, if v_trans):
          int32   v_type_i
          uint32  v_size_el
          uint32  n_embd_v_gqa
          for j in 0..n_embd_v_gqa:
            uint8[] raw V col data  (v_size_el * cell_count bytes)

GGML types: 0=F32, 1=F16, 2=Q4_0, ..., 30=BF16, 39=MXFP4 (Hopper GPU default for FA)

Note: only F16/F32/BF16 KV types can be modified via state blob.  If the model
stores KV in another type (e.g. MXFP4 on GH200 with flash-attn), apply_kv_hook
prints a one-time warning and becomes a no-op for the rest of the run.
"""

import ctypes
import struct
import time
import numpy as np

# Set to True after the first unsupported KV type is encountered so that all
# subsequent apply_kv_hook calls are instant no-ops (avoids repeated expensive
# state get/set and repeated warnings).
_kv_parse_disabled: bool = False

# GGML type to numpy dtype
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

GGML_DTYPE_MAP = {
    GGML_TYPE_F32: np.float32,
    GGML_TYPE_F16: np.float16,
    # BF16 has no direct numpy type; treat as uint16
    GGML_TYPE_BF16: np.uint16,
}

GGML_ELEM_SIZE = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
}


def _to_ggml_bytes(arr: np.ndarray, ggml_type: int) -> bytes:
    """Convert a float16 numpy array back to the original GGML storage format."""
    if ggml_type == GGML_TYPE_F16:
        return arr.astype(np.float16).tobytes()
    elif ggml_type == GGML_TYPE_F32:
        return arr.astype(np.float32).tobytes()
    elif ggml_type == GGML_TYPE_BF16:
        f32 = arr.astype(np.float32)
        u32 = f32.view(np.uint32)
        # Round to nearest bf16: add 0x8000 then truncate lower 16 bits
        u16 = ((u32 + 0x8000) >> 16).astype(np.uint16)
        return u16.tobytes()
    else:
        raise NotImplementedError(f"Cannot write back GGML type {ggml_type}")


def _read_elems(buf, offset, data_sz, ggml_type, n_elems):
    """Read n_elems from buf at offset using the correct GGML dtype, return float16 1-D array."""
    if ggml_type == GGML_TYPE_F16:
        return np.frombuffer(buf, dtype=np.float16, count=n_elems, offset=offset).copy()
    elif ggml_type == GGML_TYPE_F32:
        return np.frombuffer(buf, dtype=np.float32, count=n_elems, offset=offset).copy().astype(np.float16)
    elif ggml_type == GGML_TYPE_BF16:
        # BF16 has no native numpy type: pad lower 16 bits with zeros and view as float32
        raw_u16 = np.frombuffer(buf, dtype=np.uint16, count=n_elems, offset=offset).copy()
        f32 = np.zeros(n_elems, dtype=np.uint32)
        f32[:] = raw_u16.astype(np.uint32) << 16
        return f32.view(np.float32).astype(np.float16)
    else:
        raise NotImplementedError(
            f"KV cache GGML type {ggml_type} is not supported (only F16=1, F32=0, BF16=30). "
            f"Quantized KV types (Q4_0, etc.) cannot be modified via state blob."
        )


def _read_kv_block(buf, offset, data_sz, ggml_type, cell_count):
    """Read a 2-D [cell_count, dim] float16 array from a KV data block."""
    elem_sz = GGML_ELEM_SIZE.get(ggml_type, 2)
    n_elems = data_sz // elem_sz
    flat = _read_elems(buf, offset, data_sz, ggml_type, n_elems)
    dim = n_elems // cell_count
    return flat.reshape(cell_count, dim)


class KVState:
    """Parsed KV state: holds the raw blob and numpy views into K/V per layer."""

    def __init__(self, blob: bytearray, cell_count: int, n_layer: int,
                 v_trans: bool, k_arrays, v_arrays, k_offsets, v_offsets,
                 k_types=None, v_types=None):
        self.blob       = blob          # mutable bytearray — edit in-place
        self.cell_count = cell_count
        self.n_layer    = n_layer
        self.v_trans    = v_trans
        self.k          = k_arrays      # list[np.ndarray] shape [cell_count, k_dim], float16
        self.v          = v_arrays      # list[np.ndarray] shape [cell_count, v_dim], float16
        # offsets into blob for each layer's raw bytes (for debugging)
        self.k_offsets  = k_offsets
        self.v_offsets  = v_offsets
        # original GGML types (needed to write back correctly)
        self.k_types    = k_types or [GGML_TYPE_F16] * n_layer
        self.v_types    = v_types or [GGML_TYPE_F16] * n_layer


def parse_kv_state(blob: bytes, n_pos_per_embd: int = 1) -> KVState:
    """Parse the raw state blob returned by llama_state_seq_get_data."""
    buf = bytearray(blob)
    pos = 0

    def read_u32():
        nonlocal pos
        val = struct.unpack_from('<I', buf, pos)[0]
        pos += 4
        return val

    def read_i32():
        nonlocal pos
        val = struct.unpack_from('<i', buf, pos)[0]
        pos += 4
        return val

    def read_u64():
        nonlocal pos
        val = struct.unpack_from('<Q', buf, pos)[0]
        pos += 8
        return val

    # Header
    n_stream = read_u32()
    assert n_stream == 1, f"Expected n_stream=1, got {n_stream}"

    cell_count = read_u32()
    if cell_count == 0:
        raise ValueError("No KV cells found — did you decode any tokens first?")

    # Meta section: for each cell, read pos + n_seq_ids + seq_ids
    for _ in range(cell_count):
        _pos_val = read_i32()       # cell position
        n_seq_id = read_u32()       # number of seqs occupying this cell
        if n_pos_per_embd > 1:
            pos += 4 * 4            # llama_kv_cell_ext is 4 * int32 (skip)
        for _ in range(n_seq_id):
            read_i32()              # seq_id value

    # Data section
    v_trans = bool(read_u32())
    n_layer  = read_u32()

    k_arrays = []
    v_arrays = []
    k_offsets = []
    v_offsets = []
    k_types = []
    v_types = []

    # Keys: one block per layer
    for _ in range(n_layer):
        k_type   = read_i32()
        k_types.append(k_type)
        k_row_sz = read_u64()
        data_sz  = k_row_sz * cell_count

        k_offsets.append((pos, data_sz))
        k_arrays.append(_read_kv_block(buf, pos, data_sz, k_type, cell_count))
        pos += data_sz

    # Values
    for _ in range(n_layer):
        if not v_trans:
            v_type   = read_i32()
            v_types.append(v_type)
            v_row_sz = read_u64()
            data_sz  = v_row_sz * cell_count

            v_offsets.append((pos, data_sz))
            v_arrays.append(_read_kv_block(buf, pos, data_sz, v_type, cell_count))
            pos += data_sz
        else:
            v_type      = read_i32()
            v_types.append(v_type)
            v_size_el   = read_u32()
            n_embd_v    = read_u32()
            data_sz_col = v_size_el * cell_count

            v_offsets.append((pos, data_sz_col * n_embd_v))
            col_data = []
            for _ in range(n_embd_v):
                raw = _read_elems(buf, pos, data_sz_col, v_type, cell_count)
                col_data.append(raw)
                pos += data_sz_col
            # Shape [cell_count, n_embd_v] — transposed back for consistency
            v_arrays.append(np.stack(col_data, axis=1))

    return KVState(buf, cell_count, n_layer, v_trans, k_arrays, v_arrays,
                   k_offsets, v_offsets, k_types, v_types)


def pack_kv_state(state: KVState, n_pos_per_embd: int = 1) -> bytearray:
    """Write modified numpy arrays back into the state blob and return it."""
    buf = bytearray(state.blob)
    pos = 0

    def skip_u32():
        nonlocal pos; pos += 4

    def skip_i32():
        nonlocal pos; pos += 4

    def skip_u64():
        nonlocal pos; pos += 8

    def read_u32():
        nonlocal pos
        val = struct.unpack_from('<I', buf, pos)[0]; pos += 4; return val

    def read_i32():
        nonlocal pos
        val = struct.unpack_from('<i', buf, pos)[0]; pos += 4; return val

    def read_u64():
        nonlocal pos
        val = struct.unpack_from('<Q', buf, pos)[0]; pos += 8; return val

    # Skip header
    read_u32()  # n_stream
    cell_count = read_u32()

    # Skip meta
    for _ in range(cell_count):
        read_i32()  # pos
        n_seq_id = read_u32()
        if n_pos_per_embd > 1:
            pos += 4 * 4
        for _ in range(n_seq_id):
            read_i32()

    # Data header
    v_trans_val = read_u32()
    v_trans = bool(v_trans_val)
    n_layer = read_u32()

    # Write keys back
    for il in range(n_layer):
        read_i32()    # k_type
        k_row_sz = read_u64()
        data_sz = k_row_sz * cell_count

        arr = _to_ggml_bytes(state.k[il], state.k_types[il])
        buf[pos:pos + data_sz] = arr
        pos += data_sz

    # Write values back
    for il in range(n_layer):
        if not v_trans:
            read_i32()     # v_type
            v_row_sz = read_u64()
            data_sz = v_row_sz * cell_count

            arr = _to_ggml_bytes(state.v[il], state.v_types[il])
            buf[pos:pos + data_sz] = arr
            pos += data_sz
        else:
            read_i32()           # v_type
            v_size_el = read_u32()
            n_embd_v  = read_u32()
            data_sz_col = v_size_el * cell_count

            # v[il] is [cell_count, n_embd_v]; write column by column
            for j in range(n_embd_v):
                col = _to_ggml_bytes(state.v[il][:, j], state.v_types[il])
                buf[pos:pos + data_sz_col] = col
                pos += data_sz_col

    return buf


def apply_kv_hook(lib, ctx, CP, k_fn=None, v_fn=None, seq_id: int = 0,
                  n_pos_per_embd: int = 1, n_new_k=None, n_new_v=None,
                  start_k=None, start_v=None, profile=None):
    """
    Get KV state for seq_id, apply k_fn/v_fn to each layer's K/V arrays,
    write modified state back.

    k_fn: callable, list[callable] (one per layer), or None (skip K entirely).
    v_fn: callable, list[callable] (one per layer), or None (skip V entirely).

    n_new_k: None  = quantize all K cells (initial prefill).
             int   = quantize N K cells (position determined by start_k, see below).
             list  = per-layer: list[int] where 0 means skip that layer.
    n_new_v: same semantics as n_new_k, applied to V.

    start_k: None  = quantize the LAST n_new_k cells (default, "just-added" tokens).
             int   = quantize cells [start_k : start_k + n_new_k] (absolute offset).
             Used for recent-window mode where tokens that fell out of the window
             are at a specific position, not the tail.
    start_v: same as start_k, applied to V.

    profile: optional kv_profile.KvProfile — if set, accumulates per-phase CPU times.

    Returns: KVState (with modified arrays)
    """
    global _kv_parse_disabled
    if k_fn is None and v_fn is None or _kv_parse_disabled:
        return None

    if profile is not None:
        profile.n_hook_calls += 1

    t0 = time.perf_counter()
    sz = lib.llama_state_seq_get_size(ctx, seq_id)
    buf = ctypes.create_string_buffer(sz)
    n_written = lib.llama_state_seq_get_data(ctx, buf, sz, seq_id)
    assert n_written > 0, "llama_state_seq_get_data returned 0"
    if profile is not None:
        profile.kv_get_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        state = parse_kv_state(bytes(buf[:n_written]), n_pos_per_embd)
    except NotImplementedError as e:
        _kv_parse_disabled = True
        import re
        m = re.search(r'type (\d+)', str(e))
        kv_type = int(m.group(1)) if m else '?'
        print(f"\n[WARN] KV quantization disabled for this model: {e}"
              f"\n       (GGML type {kv_type} in state blob; likely MXFP4 on Hopper GPU)"
              f"\n       KV cache will remain unmodified (no int2/int3/int4 applied).",
              flush=True)
        return None
    if profile is not None:
        profile.kv_parse_s += time.perf_counter() - t0

    # Normalize k_fn / v_fn: accept single callable or list[callable] (one per layer)
    if callable(k_fn):
        k_fn = [k_fn] * state.n_layer
    if callable(v_fn):
        v_fn = [v_fn] * state.n_layer

    def _q(fn, arr):
        if profile is None:
            return fn(arr)
        tq = time.perf_counter()
        out = fn(arr)
        profile.kv_quant_s += time.perf_counter() - tq
        return out

    def _apply_fn(arr, fn, n_new, start):
        if fn is None or n_new == 0:
            return arr
        if n_new is None:                          # quantize all (prefill)
            return _q(fn, arr)
        if start is not None:                      # absolute offset (recent-window mode)
            end = min(start + n_new, arr.shape[0])
            start = min(start, arr.shape[0])
            if end <= start:
                return arr
            return np.concatenate([arr[:start], _q(fn, arr[start:end]), arr[end:]], axis=0)
        if n_new < state.cell_count:               # last-N mode (default)
            s = state.cell_count - n_new
            return np.concatenate([arr[:s], _q(fn, arr[s:])], axis=0)
        return _q(fn, arr)                         # n_new >= cell_count: quantize all

    if k_fn is not None:
        state.k = [
            _apply_fn(arr,
                      k_fn[il] if il < len(k_fn) else None,
                      n_new_k[il] if isinstance(n_new_k, list) else n_new_k,
                      start_k[il] if isinstance(start_k, list) else start_k)
            for il, arr in enumerate(state.k)
        ]

    if v_fn is not None:
        state.v = [
            _apply_fn(arr,
                      v_fn[il] if il < len(v_fn) else None,
                      n_new_v[il] if isinstance(n_new_v, list) else n_new_v,
                      start_v[il] if isinstance(start_v, list) else start_v)
            for il, arr in enumerate(state.v)
        ]

    t0 = time.perf_counter()
    modified_blob = pack_kv_state(state, n_pos_per_embd)
    if profile is not None:
        profile.kv_pack_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    c_buf = ctypes.c_char_p(bytes(modified_blob))
    n_read = lib.llama_state_seq_set_data(ctx, c_buf, len(modified_blob), seq_id)
    assert n_read > 0, "llama_state_seq_set_data returned 0"
    if profile is not None:
        profile.kv_set_s += time.perf_counter() - t0

    return state
