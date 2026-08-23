"""
kv_profile.py — accumulate timings for KV-quant research runs.

CPU path (parse_state.apply_kv_hook):
  decode   — llama_decode (forward pass; includes GPU matmuls when layers offloaded)
  kv_get   — llama_state_seq_get_data (serialize KV blob to host)
  kv_parse — parse_kv_state (numpy views)
  kv_quant — numpy quantization on K/V slices
  kv_pack  — pack_kv_state
  kv_set   — llama_state_seq_set_data (restore blob)

GPU KV path (gpu_quant.apply_kv_hook_gpu, CuPy in-place on device pointers):
  gpu_kv_s — wall time for the whole hook including cp.cuda.Device().synchronize()
             at the end (kernels complete before return). No PCIe KV blob copy.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass


@dataclass
class KvProfile:
    decode_s: float = 0.0
    kv_get_s: float = 0.0
    kv_parse_s: float = 0.0
    kv_quant_s: float = 0.0
    kv_pack_s: float = 0.0
    kv_set_s: float = 0.0
    gpu_kv_s: float = 0.0
    n_decode_calls: int = 0
    n_hook_calls: int = 0
    n_gpu_kv_hook_calls: int = 0

    def reset(self) -> None:
        self.decode_s = 0.0
        self.kv_get_s = 0.0
        self.kv_parse_s = 0.0
        self.kv_quant_s = 0.0
        self.kv_pack_s = 0.0
        self.kv_set_s = 0.0
        self.gpu_kv_s = 0.0
        self.n_decode_calls = 0
        self.n_hook_calls = 0
        self.n_gpu_kv_hook_calls = 0

    def total_kv_s(self) -> float:
        return (
            self.kv_get_s + self.kv_parse_s + self.kv_quant_s
            + self.kv_pack_s + self.kv_set_s
        )

    def total_kv_all_s(self) -> float:
        return self.total_kv_s() + self.gpu_kv_s

    def total_s(self) -> float:
        return self.decode_s + self.total_kv_all_s()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_kv_s"] = self.total_kv_s()
        d["total_kv_all_s"] = self.total_kv_all_s()
        d["total_s"] = self.total_s()
        d["decode_frac"] = (
            self.decode_s / self.total_s() if self.total_s() > 0 else 0.0
        )
        d["kv_quant_frac_of_kv"] = (
            self.kv_quant_s / self.total_kv_s() if self.total_kv_s() > 0 else 0.0
        )
        return d

    def summary_lines(self) -> list[str]:
        t = self.total_s()
        lines = [
            f"  profile: decode={self.decode_s:.3f}s ({100*self.decode_s/t:.1f}% of total)"
            if t > 0 else "  profile: decode=0s",
            f"           kv_cpu={self.total_kv_s():.3f}s  "
            f"(get={self.kv_get_s:.3f} parse={self.kv_parse_s:.3f} "
            f"quant={self.kv_quant_s:.3f} pack={self.kv_pack_s:.3f} set={self.kv_set_s:.3f})  "
            f"n_cpu_hook={self.n_hook_calls}",
            f"           kv_gpu={self.gpu_kv_s:.3f}s  n_gpu_kv_hook={self.n_gpu_kv_hook_calls}",
        ]
        return lines


def wrap_llama_decode(lib, profile: KvProfile):
    """Monkey-patch lib.llama_decode to accumulate decode time. Returns original."""
    orig = lib.llama_decode

    def wrapped(ctx, batch):
        t0 = time.perf_counter()
        try:
            return orig(ctx, batch)
        finally:
            profile.decode_s += time.perf_counter() - t0
            profile.n_decode_calls += 1

    lib.llama_decode = wrapped
    return orig


def save_json(path: str, profile: KvProfile, extra: dict | None = None) -> None:
    d = profile.to_dict()
    if extra:
        d["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
