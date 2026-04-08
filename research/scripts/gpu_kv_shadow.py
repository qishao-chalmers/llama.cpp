"""
gpu_kv_shadow.py — device-resident KV checkpoints for adaptive-gen (no PCIe blob).

save / restore copy the live K/V tensors via cudaMemcpy D→D using llama_get_kv_layer_info.
Used when --adaptive-gen-gpu-shadow and GPU KV + CuPy are available; otherwise fall back to
save_kv_state / restore_kv_state (CPU bytes, PCIe for GPU backend).
"""

from __future__ import annotations

import ctypes

import llama_bindings as llama

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # type: ignore


def _ptr(x) -> int:
    if x is None:
        return 0
    return int(ctypes.cast(x, ctypes.c_void_p).value)


def _u8_view(ptr, nbytes: int):
    """uint8 CuPy array view of `nbytes` bytes at device pointer `ptr` (UnownedMemory)."""
    mem = cp.cuda.UnownedMemory(_ptr(ptr), nbytes, owner=None)
    return cp.ndarray(
        (nbytes,), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))


def _copyto_d2d(dst_u8, src_u8) -> None:
    """Device-to-device copy; uses cp.copyto (portable; some CuPy builds lack memcpyDtoD)."""
    cp.copyto(dst_u8, src_u8)


class GpuKvShadowCheckpoint:
    """One KV snapshot: per-layer raw byte copies of K and V (device memory)."""

    __slots__ = ("_layers", "_n_cells_saved")

    def __init__(self) -> None:
        self._layers: list[tuple[object, object, int, int]] | None = None
        # Each entry: (k_uint8_cp, v_uint8_cp, nbytes_k, nbytes_v)
        self._n_cells_saved: int = 0   # logical KV length when save() ran (layer 0)

    def save(self, lib, ctx: llama.ContextPtr, n_layer: int) -> None:
        """Copy live KV tensors to GPU shadow buffers (DtoD)."""
        if not HAS_CUPY:
            raise RuntimeError("GpuKvShadowCheckpoint requires CuPy")
        infos = (llama.LlamaKVLayerInfo * n_layer)()
        n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
        if n_filled <= 0:
            raise RuntimeError(f"llama_get_kv_layer_info returned {n_filled}")

        layers: list[tuple[object, object, int, int]] = []
        for i in range(n_filled):
            info = infos[i]
            if info.v_trans:
                raise RuntimeError(
                    "gpu_kv_shadow requires Flash Attention (v_trans=0)")
            n_cells = int(info.n_cells)
            k_stride = int(info.k_stride)
            v_stride = int(info.v_stride)
            nbytes_k = n_cells * k_stride
            nbytes_v = n_cells * v_stride

            k_dst = cp.empty(nbytes_k, dtype=cp.uint8)
            v_dst = cp.empty(nbytes_v, dtype=cp.uint8)
            _copyto_d2d(k_dst, _u8_view(info.k_data, nbytes_k))
            _copyto_d2d(v_dst, _u8_view(info.v_data, nbytes_v))
            layers.append((k_dst, v_dst, nbytes_k, nbytes_v))

        cp.cuda.Device().synchronize()
        self._layers = layers
        self._n_cells_saved = int(infos[0].n_cells)

    def restore(self, lib, ctx: llama.ContextPtr, n_layer: int) -> None:
        """Copy shadow buffers back to live KV tensors (DtoD).

        Live cache may hold *more* logical positions than this snapshot (e.g. restore
        pre-prefix checkpoint while KV still has bootstrap). We memcpy the snapshot into
        the leading bytes of each layer buffer, then trim sequence positions
        [n_cells_saved, inf) so metadata matches the blob restore path.
        """
        if self._layers is None:
            raise RuntimeError("GpuKvShadowCheckpoint.restore: empty checkpoint")
        if not HAS_CUPY:
            raise RuntimeError("GpuKvShadowCheckpoint requires CuPy")

        infos = (llama.LlamaKVLayerInfo * n_layer)()
        n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
        if n_filled != len(self._layers):
            raise RuntimeError(
                f"layer count mismatch: checkpoint {len(self._layers)} vs ctx {n_filled}")

        n_cells_saved = self._n_cells_saved
        n_cells_live0 = int(infos[0].n_cells)

        for i in range(n_filled):
            info = infos[i]
            k_sh, v_sh, nbytes_k, nbytes_v = self._layers[i]
            n_cells = int(info.n_cells)
            k_stride = int(info.k_stride)
            v_stride = int(info.v_stride)
            if n_cells * k_stride < nbytes_k or n_cells * v_stride < nbytes_v:
                raise RuntimeError(
                    f"KV buffer at layer {i} too small for checkpoint: live bytes "
                    f"{n_cells * k_stride}/{n_cells * v_stride} need {nbytes_k}/{nbytes_v}")

            _copyto_d2d(_u8_view(info.k_data, nbytes_k), k_sh)
            _copyto_d2d(_u8_view(info.v_data, nbytes_v), v_sh)

        # Match llama_state_seq_set_data: logical length must equal snapshot.
        if n_cells_live0 > n_cells_saved:
            mem = lib.llama_get_memory(ctx)
            lib.llama_memory_seq_rm(mem, 0, n_cells_saved, -1)

        cp.cuda.Device().synchronize()
