#!/usr/bin/env python3
"""convert_gguf_to_hf.py — Convert a GGUF model to HuggingFace safetensors for vLLM.

Dequantises all weight tensors and writes safetensors shards + config.json so the
output directory loads directly in vLLM, transformers, etc.

Supports input tensor types: Q8_0, Q4_K, Q6_K, Q5_K, Q4_0, Q3_K, Q2_K, F32, F16, BF16.
Architectures: llama, qwen2, qwen3, mistral, gemma2, gemma3 (and most llama-like variants).
config.json is built from GGUF metadata — no HF model download required.
Tokenizer: copied from --hf-model if provided; otherwise supply --tokenizer to vLLM.
Gemma/Gemma2/Gemma3: RMSNorm weights are shifted by -1.0 (llama.cpp stores weight+1).

Usage:
    # Minimal — build config from GGUF, point vLLM at tokenizer separately
    python3 research/scripts/convert_gguf_to_hf.py \\
        --input  models/Qwen3-8B-Q8_0-attn4-ffn3-ess6.gguf \\
        --output /scratch/Qwen3-8B-attn4-ffn3-ess6-hf/

    # With tokenizer (recommended)
    python3 research/scripts/convert_gguf_to_hf.py \\
        --input  models/Qwen3-8B-Q8_0-attn4-ffn3-ess6.gguf \\
        --output /scratch/Qwen3-8B-attn4-ffn3-ess6-hf/ \\
        --hf-model /scratch/models/Qwen3-8B/

    # Run with vLLM:
    vllm serve /scratch/Qwen3-8B-attn4-ffn3-ess6-hf/ --dtype bfloat16
    # or if no tokenizer was copied:
    vllm serve /scratch/Qwen3-8B-attn4-ffn3-ess6-hf/ --tokenizer Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))

try:
    import gguf
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    from gguf.quants import dequantize as _gguf_dequantize
except ImportError:
    sys.exit("gguf-py not found — run from the llama.cpp root or add gguf-py/ to PYTHONPATH")

try:
    from safetensors.numpy import save_file as _st_save_numpy
    _HAS_ST_NUMPY = True
except ImportError:
    _HAS_ST_NUMPY = False

try:
    import torch
    from safetensors.torch import save_file as _st_save_torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if not _HAS_ST_NUMPY and not _HAS_TORCH:
    sys.exit("pip install safetensors  (and optionally torch for bf16 support)")


# ── GGML type IDs ─────────────────────────────────────────────────────────────
_F32  = 0
_F16  = 1
_BF16 = 30


# ── Architecture → HuggingFace config mapping ─────────────────────────────────

_ARCH_CONFIG: dict[str, dict] = {
    "llama": {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
    },
    "mistral": {
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"],
    },
    "qwen2": {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
    },
    "qwen3": {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    },
    "qwen2moe": {
        "model_type": "qwen2_moe",
        "architectures": ["Qwen2MoeForCausalLM"],
    },
    "gemma2": {
        "model_type": "gemma2",
        "architectures": ["Gemma2ForCausalLM"],
    },
    "gemma3": {
        "model_type": "gemma3_text",
        "architectures": ["Gemma3ForCausalLM"],
    },
}

# Architectures where llama.cpp stores RMSNorm as (HF_weight + 1.0)
_NORM_SHIFT_ARCHES = frozenset({"gemma", "gemma2", "gemma3"})

# GGUF metadata key suffix → HF config key
_GGUF_TO_HF_CONFIG: dict[str, str] = {
    "block_count":                       "num_hidden_layers",
    "embedding_length":                  "hidden_size",
    "feed_forward_length":               "intermediate_size",
    "attention.head_count":              "num_attention_heads",
    "attention.head_count_kv":           "num_key_value_heads",
    "context_length":                    "max_position_embeddings",
    "rope.freq_base":                    "rope_theta",
    "attention.layer_norm_rms_epsilon":  "rms_norm_eps",
    "attention.key_length":              "head_dim",
    "attention.sliding_window":          "sliding_window",
    "rope.dimension_count":              "rope_scaling_dim",  # informational only
    "expert_count":                      "num_experts",
    "expert_used_count":                 "num_experts_per_tok",
    "final_logit_softcapping":           "final_logit_softcapping",
    "attn_logit_softcapping":            "attn_logit_softcapping",
}

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.json",
]


# ── Tensor name mapping: GGUF → HuggingFace ──────────────────────────────────

_GLOBAL_NAME_MAP: dict[str, str] = {
    "token_embd.weight":  "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight":      "lm_head.weight",
}

_BLOCK_ROLE_MAP: dict[str, str] = {
    # Attention projections
    "attn_q":           "self_attn.q_proj",
    "attn_k":           "self_attn.k_proj",
    "attn_v":           "self_attn.v_proj",
    "attn_output":      "self_attn.o_proj",
    # Qwen2/3 QK norms
    "attn_q_norm":      "self_attn.q_norm",
    "attn_k_norm":      "self_attn.k_norm",
    # FFN
    "ffn_gate":         "mlp.gate_proj",
    "ffn_up":           "mlp.up_proj",
    "ffn_down":         "mlp.down_proj",
    # MoE FFN (qwen2_moe / mixtral style)
    "ffn_gate_inp":     "mlp.gate",
    "ffn_gate_exps":    "mlp.experts.gate_proj",
    "ffn_up_exps":      "mlp.experts.up_proj",
    "ffn_down_exps":    "mlp.experts.down_proj",
    "ffn_gate_shexp":   "mlp.shared_expert.gate_proj",
    "ffn_up_shexp":     "mlp.shared_expert.up_proj",
    "ffn_down_shexp":   "mlp.shared_expert.down_proj",
    # Layer norms
    "attn_norm":        "input_layernorm",
    "ffn_norm":         "post_attention_layernorm",
    # Gemma2/3 extra norms (ffn_norm remapped for gemma* in gguf_to_hf_name)
    "post_attention_norm": "post_attention_layernorm",
    "post_ffw_norm":       "post_feedforward_layernorm",
    # MLA (DeepSeek)
    "attn_q_a":         "self_attn.q_a_proj",
    "attn_q_b":         "self_attn.q_b_proj",
    "attn_kv_a_mqa":    "self_attn.kv_a_proj_with_mqa",
    "attn_kv_b":        "self_attn.kv_b_proj",
    "attn_q_a_norm":    "self_attn.q_a_layernorm",
    "attn_kv_a_norm":   "self_attn.kv_a_layernorm",
}

# Gemma2/3: ffn_norm is pre-FFN, not post-attention
_BLOCK_ROLE_MAP_GEMMA: dict[str, str] = {
    **_BLOCK_ROLE_MAP,
    "ffn_norm": "pre_feedforward_layernorm",
}

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.*?)\.weight$")


def detect_arch(reader: gguf.GGUFReader) -> str:
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            return str(kv.contents()).lower()
    return "llama"


def gguf_to_hf_name(name: str, arch: str = "llama") -> str | None:
    """Map GGUF tensor name → HF name. Returns None for unknown / non-weight tensors."""
    if name in _GLOBAL_NAME_MAP:
        return _GLOBAL_NAME_MAP[name]
    role_map = _BLOCK_ROLE_MAP_GEMMA if arch in _NORM_SHIFT_ARCHES else _BLOCK_ROLE_MAP
    m = _BLOCK_RE.match(name)
    if m:
        layer, role = m.group(1), m.group(2)
        hf_role = role_map.get(role)
        if hf_role:
            return f"model.layers.{layer}.{hf_role}.weight"
    return None


def apply_gemma_norm_shift(hf_name: str, arr: np.ndarray, arch: str) -> np.ndarray:
    """Undo llama.cpp Gemma RMSNorm +1.0 storage convention."""
    if arch not in _NORM_SHIFT_ARCHES:
        return arr
    # covers *.layernorm.weight, *.q_norm.weight, *.k_norm.weight, model.norm.weight
    if hf_name.endswith("norm.weight"):
        return arr - 1.0
    return arr


# ── Dequantisation ────────────────────────────────────────────────────────────

def dequant_to_float32(t) -> np.ndarray:
    """
    Dequantise a GGUFReader tensor to float32 in HF weight shape.

    GGUFReader stores tensor data with numpy shape = reversed(ggml_shape), which
    is already the correct HF layout (out_features, in_features) — no transpose.
    """
    ttype = int(t.tensor_type)
    ggml_shape = list(t.shape)           # [ne0, ne1, ...]  (GGML dim order)
    hf_shape   = tuple(reversed(ggml_shape))  # (ne_{n-1}, ..., ne0) = HF shape

    if ttype == _F32:
        return np.asarray(t.data).view(np.float32).reshape(hf_shape)

    if ttype == _F16:
        return np.asarray(t.data).view(np.float16).astype(np.float32).reshape(hf_shape)

    if ttype == _BF16:
        raw = np.asarray(t.data).view(np.uint16)
        f32 = (raw.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(hf_shape)

    # All other types: use gguf-py's general dequantize
    try:
        qtype = GGMLQuantizationType(ttype)
        block_size, block_bytes = GGML_QUANT_SIZES[qtype]
        n_elem   = int(np.prod(ggml_shape))
        n_blocks = n_elem // block_size
        raw      = np.asarray(t.data, dtype=np.uint8).reshape(n_blocks, block_bytes)
        return _gguf_dequantize(raw, qtype).reshape(hf_shape).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Cannot dequantise tensor {t.name!r} (type {ttype}): {e}") from e


def to_target_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Convert float32 array to target dtype for safetensors."""
    if dtype == "fp32":
        return arr.astype(np.float32)
    if dtype == "fp16":
        return arr.astype(np.float16)
    if dtype == "bf16":
        if not _HAS_TORCH:
            raise RuntimeError("bf16 requires torch: pip install torch")
        return arr  # handled in save_sharded via torch conversion
    raise ValueError(f"Unknown dtype {dtype!r}")


# ── config.json builder ───────────────────────────────────────────────────────

def build_config(reader: gguf.GGUFReader, vocab_size: int) -> dict:
    """Build a minimal HF config.json from GGUF metadata."""
    arch = detect_arch(reader)

    arch_defaults = _ARCH_CONFIG.get(arch, _ARCH_CONFIG["llama"])
    config: dict = {
        "model_type":    arch_defaults["model_type"],
        "architectures": arch_defaults["architectures"],
        "vocab_size":    vocab_size,
        "torch_dtype":   "bfloat16",
        "transformers_version": "4.50.0",
    }

    # Extract numeric fields from GGUF KV pairs (key format: "{arch}.{suffix}")
    rope_scale_type = None
    rope_scale_factor = None
    for kv in reader.fields.values():
        name = kv.name
        # Strip arch prefix: "qwen3.block_count" → "block_count"
        for prefix in (arch + ".", "llama.", "general."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name == "rope.scaling.type":
            rope_scale_type = str(kv.contents())
            continue
        if name == "rope.scaling.factor":
            rope_scale_factor = float(kv.contents())
            continue
        hf_key = _GGUF_TO_HF_CONFIG.get(name)
        if hf_key:
            val = kv.contents()
            if isinstance(val, (int, float)):
                config[hf_key] = int(val) if isinstance(val, float) and val == int(val) else val

    # Qwen3 always has head_dim set via attention.key_length; compute if missing
    if "head_dim" not in config and "hidden_size" in config and "num_attention_heads" in config:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    if rope_scale_type is not None and rope_scale_factor is not None:
        # HF gemma3 uses rope_type; older configs used "type"
        config["rope_scaling"] = {
            "factor": rope_scale_factor,
            "rope_type": rope_scale_type if rope_scale_type != "linear" else "linear",
        }

    # Tokenizer ids from GGUF when present
    for gguf_key, hf_key in (
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
        ("tokenizer.ggml.padding_token_id", "pad_token_id"),
    ):
        for kv in reader.fields.values():
            if kv.name == gguf_key:
                config[hf_key] = int(kv.contents())
                break

    # Tied embeddings if no output.weight tensor
    has_output = any(t.name == "output.weight" for t in reader.tensors)
    config["tie_word_embeddings"] = not has_output

    if arch in ("gemma2", "gemma3"):
        config["hidden_activation"] = "gelu_pytorch_tanh"
        config["attention_bias"] = False
        if "query_pre_attn_scalar" not in config and "hidden_size" in config and "num_attention_heads" in config:
            # 27B uses 168 (= hidden/heads); matches official HF configs
            config["query_pre_attn_scalar"] = config["hidden_size"] // config["num_attention_heads"]
        if arch == "gemma3":
            config.setdefault("rope_local_base_freq", 10000.0)
            n_layers = int(config.get("num_hidden_layers", 0))
            # Official Gemma3: every 6th layer is full attention
            pattern = 6
            config["sliding_window_pattern"] = pattern
            config["layer_types"] = [
                "full_attention" if ((i + 1) % pattern == 0) else "sliding_attention"
                for i in range(n_layers)
            ]

    # Drop informational-only keys
    config.pop("rope_scaling_dim", None)
    return config


# ── Safetensors shard writer ──────────────────────────────────────────────────

def _nbytes(arr: np.ndarray, dtype: str) -> int:
    elem_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    return arr.size * elem_bytes


def _prepare_shard_for_save(shard: dict[str, np.ndarray], dtype: str):
    if dtype == "bf16" and _HAS_TORCH:
        return {k: torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)
                for k, v in shard.items()}, "torch"
    np_shard = {
        k: v.astype(np.float16 if dtype == "bf16" else
                     np.float32 if dtype == "fp32" else np.float16)
        for k, v in shard.items()
    }
    return np_shard, "numpy"


class ShardWriter:
    """Write safetensors shards incrementally to keep peak memory bounded."""

    def __init__(self, out_dir: Path, dtype: str, max_shard_bytes: int):
        self.out_dir = out_dir
        self.dtype = dtype
        self.max_shard_bytes = max_shard_bytes
        self.current: dict[str, np.ndarray] = {}
        self.current_size = 0
        self.shards_meta: list[tuple[str, list[str]]] = []  # (fname, names)
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self._shard_idx = 0

    def add(self, name: str, arr: np.ndarray) -> None:
        sz = _nbytes(arr, self.dtype)
        if self.current and self.current_size + sz > self.max_shard_bytes:
            self._flush()
        self.current[name] = arr
        self.current_size += sz
        self.total_size += sz

    def _flush(self) -> None:
        if not self.current:
            return
        self._shard_idx += 1
        # Filename finalized in finish() once shard count known; use temp names first
        fname = f"_shard_{self._shard_idx:05d}.safetensors"
        fpath = self.out_dir / fname
        print(f"  writing {fname}  ({len(self.current)} tensors, "
              f"{self.current_size / 1e9:.2f} GB)", flush=True)
        prepared, kind = _prepare_shard_for_save(self.current, self.dtype)
        if kind == "torch":
            _st_save_torch(prepared, str(fpath))
        else:
            _st_save_numpy(prepared, str(fpath))
        names = list(self.current.keys())
        self.shards_meta.append((fname, names))
        self.current = {}
        self.current_size = 0

    def finish(self) -> None:
        self._flush()
        n_shards = len(self.shards_meta)
        # Rename temp shards to final HF names and build weight_map
        for i, (tmp_name, names) in enumerate(self.shards_meta, 1):
            final = ("model.safetensors"
                     if n_shards == 1
                     else f"model-{i:05d}-of-{n_shards:05d}.safetensors")
            src = self.out_dir / tmp_name
            dst = self.out_dir / final
            if src != dst:
                src.rename(dst)
            for name in names:
                self.weight_map[name] = final
            print(f"  finalized {final}", flush=True)

        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )


def save_sharded(
    tensors: dict[str, np.ndarray],
    out_dir: Path,
    dtype: str,
    max_shard_bytes: int,
) -> None:
    """Split tensors into ≤max_shard_bytes shards and write safetensors + index."""
    writer = ShardWriter(out_dir, dtype, max_shard_bytes)
    for name in sorted(tensors):
        writer.add(name, tensors[name])
    writer.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  "-i", required=True, type=Path,
                    help="Input GGUF file (any quant type)")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output directory for safetensors + config")
    ap.add_argument("--hf-model", type=Path, default=None,
                    help="HF model directory to copy config.json + tokenizer from (optional)")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp16",
                    help="Output tensor dtype (default fp16; bf16 requires torch)")
    ap.add_argument("--max-shard-size", type=float, default=5.0,
                    help="Max shard size in GB (default 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print tensor mapping only, no files written")
    args = ap.parse_args()

    if args.dtype == "bf16" and not _HAS_TORCH:
        print("[warn] bf16 requires torch — falling back to fp16", file=sys.stderr)
        args.dtype = "fp16"

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"Input not found: {src}")

    out_dir = args.output.expanduser().resolve()
    max_shard_bytes = int(args.max_shard_size * 1024 ** 3)

    print(f"input   : {src}  ({src.stat().st_size / 1e9:.2f} GB)")
    print(f"output  : {out_dir}")
    print(f"dtype   : {args.dtype}   max_shard={args.max_shard_size:.0f} GB")
    print()

    reader = gguf.GGUFReader(str(src), "r")
    arch = detect_arch(reader)
    print(f"arch    : {arch}")

    # Determine vocab size from token_embd shape
    vocab_size = 0
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            vocab_size = int(t.shape[1])  # ne1 = vocab_size (ne0 = hidden_dim)
            break

    if args.dry_run:
        print("Tensor mapping (GGUF name → HF name):")
        n_mapped = n_skip = 0
        for t in reader.tensors:
            hf_name = gguf_to_hf_name(t.name, arch)
            ttype   = int(t.tensor_type)
            try:
                tname = GGMLQuantizationType(ttype).name
            except ValueError:
                tname = f"type{ttype}"
            if hf_name:
                print(f"  {t.name:<50} → {hf_name}  [{tname}]")
                n_mapped += 1
            else:
                print(f"  {t.name:<50}   (skipped — no HF mapping)  [{tname}]")
                n_skip += 1
        print(f"\nMapped: {n_mapped}   Skipped: {n_skip}   vocab_size: {vocab_size}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dequantise + stream-write shards (bounded memory) ─────────────────────
    print("Dequantising tensors (streaming shards) ...")
    writer = ShardWriter(out_dir, args.dtype, max_shard_bytes)
    t0 = time.time()
    n_skip = 0
    n_done = 0

    for idx, t in enumerate(reader.tensors):
        hf_name = gguf_to_hf_name(t.name, arch)
        if hf_name is None:
            n_skip += 1
            continue

        ttype = int(t.tensor_type)
        try:
            tname = GGMLQuantizationType(ttype).name
        except ValueError:
            tname = f"type{ttype}"

        try:
            f32 = dequant_to_float32(t)
        except RuntimeError as e:
            print(f"  [warn] skipping {t.name}: {e}", file=sys.stderr)
            n_skip += 1
            continue

        f32 = apply_gemma_norm_shift(hf_name, f32, arch)
        # Store as target dtype immediately to cut memory (bf16 kept as f32 until save)
        if args.dtype == "fp16":
            arr = f32.astype(np.float16)
        elif args.dtype == "fp32":
            arr = f32
        else:
            arr = f32  # bf16 converted at write time
        del f32

        writer.add(hf_name, arr)
        n_done += 1
        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{len(reader.tensors)}] {t.name:<50}  {tname:<8}"
              f"  → {hf_name}  ({elapsed:.1f}s)", flush=True)

    print(f"\nDequantised {n_done} tensors, skipped {n_skip}.")
    print(f"Writing remaining shard(s) ...")
    writer.finish()
    print(f"Output size (index total_size): {writer.total_size / 1e9:.2f} GB")

    # ── config.json ───────────────────────────────────────────────────────────
    config_copied = False
    if args.hf_model:
        hf_src = args.hf_model.expanduser().resolve()
        cfg_src = hf_src / "config.json"
        if cfg_src.is_file():
            shutil.copy2(cfg_src, out_dir / "config.json")
            print(f"\nCopied config.json from {hf_src}")
            config_copied = True
        else:
            print(f"[warn] config.json not found in {hf_src}, building from GGUF", file=sys.stderr)

    if not config_copied:
        config = build_config(reader, vocab_size)
        if args.dtype == "bf16":
            config["torch_dtype"] = "bfloat16"
        elif args.dtype == "fp16":
            config["torch_dtype"] = "float16"
        else:
            config["torch_dtype"] = "float32"
        cfg_path = out_dir / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"\nWrote config.json (built from GGUF metadata, arch={config['model_type']})")

    # ── Tokenizer files ──────────────────────────────────────────────────────
    if args.hf_model:
        hf_src = args.hf_model.expanduser().resolve()
        copied = []
        for fname in _TOKENIZER_FILES:
            src_f = hf_src / fname
            if src_f.is_file():
                shutil.copy2(src_f, out_dir / fname)
                copied.append(fname)
        if copied:
            print(f"Copied tokenizer files: {', '.join(copied)}")
        else:
            print("[warn] No tokenizer files found in --hf-model directory", file=sys.stderr)
    else:
        print("\n[note] No tokenizer copied. Run vLLM with:")
        print(f"  vllm serve {out_dir} --tokenizer <original-model-id-or-path>")

    # ── Done ─────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    out_size = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1e9
    print(f"\ndone  → {out_dir}  ({out_size:.2f} GB total)  [{total_elapsed:.0f}s]")
    print(f"\nRun with vLLM:")
    print(f"  vllm serve {out_dir} --dtype {'bfloat16' if args.dtype == 'bf16' else args.dtype.replace('fp', 'float')}")


if __name__ == "__main__":
    main()
