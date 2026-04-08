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


class GpuKvShadowCheckpoint:
    """One KV snapshot: per-layer raw byte copies of K and V (device memory)."""

    __slots__ = ("_layers",)

    def __init__(self) -> None:
        self._layers: list[tuple[object, object, int, int]] | None = None
        # Each entry: (k_uint8_cp, v_uint8_cp, nbytes_k, nbytes_v)

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
            cp.cuda.runtime.memcpyDtoD(k_dst.data.ptr, _ptr(info.k_data), nbytes_k)
            cp.cuda.runtime.memcpyDtoD(v_dst.data.ptr, _ptr(info.v_data), nbytes_v)
            layers.append((k_dst, v_dst, nbytes_k, nbytes_v))

        cp.cuda.Device().synchronize()
        self._layers = layers

    def restore(self, lib, ctx: llama.ContextPtr, n_layer: int) -> None:
        """Copy shadow buffers back to live KV tensors (DtoD)."""
        if self._layers is None:
            raise RuntimeError("GpuKvShadowCheckpoint.restore: empty checkpoint")
        if not HAS_CUPY:
            raise RuntimeError("GpuKvShadowCheckpoint requires CuPy")

        infos = (llama.LlamaKVLayerInfo * n_layer)()
        n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
        if n_filled != len(self._layers):
            raise RuntimeError(
                f"layer count mismatch: checkpoint {len(self._layers)} vs ctx {n_filled}")

        for i in range(n_filled):
            info = infos[i]
            k_dst, v_dst, nbytes_k, nbytes_v = self._layers[i]
            n_cells = int(info.n_cells)
            k_stride = int(info.k_stride)
            v_stride = int(info.v_stride)
            if n_cells * k_stride != nbytes_k or n_cells * v_stride != nbytes_v:
                raise RuntimeError(
                    f"KV shape mismatch at layer {i}: live "
                    f"{n_cells * k_stride}/{n_cells * v_stride} vs shadow {nbytes_k}/{nbytes_v}")

            cp.cuda.runtime.memcpyDtoD(_ptr(info.k_data), k_dst.data.ptr, nbytes_k)
            cp.cuda.runtime.memcpyDtoD(_ptr(info.v_data), v_dst.data.ptr, nbytes_v)

        cp.cuda.Device().synchronize()
