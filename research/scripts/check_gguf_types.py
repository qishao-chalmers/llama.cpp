#!/usr/bin/env python3
"""Check a GGUF file for tensor types that would fail the CUDA binbcast assert."""
import sys
sys.path.insert(0, "gguf-py")
import gguf

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-8B-Q8_0-Q4_K_M_recovered.gguf"
r = gguf.GGUFReader(path)

type_names = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16", 32: "BF16"}

problems = []
counts = {}
for t in r.tensors:
    ttype = int(t.tensor_type)
    counts[ttype] = counts.get(ttype, 0) + 1
    if ttype not in (0, 1, 8):
        problems.append((t.name, ttype, type_names.get(ttype, f"type{ttype}")))

print(f"File: {path}")
print(f"Total tensors: {len(r.tensors)}")
print("\nType counts:")
for ttype, n in sorted(counts.items()):
    print(f"  {type_names.get(ttype, f'type{ttype}'):8s} (id={ttype}): {n}")

if problems:
    print(f"\nPROBLEM tensors (not F32/F16/Q8_0): {len(problems)}")
    for name, ttype, tname in problems:
        print(f"  {name}  ({tname}, id={ttype})")
else:
    print("\nAll tensors are F32, F16, or Q8_0 — no type problems found.")
