#!/usr/bin/env python3
"""Decode Volta+/Hopper SASS control codes from cuobjdump --dump-sass output.

Usage: python3 decode_sass_control.py k3_sass.txt > k3_sass_decoded.txt

Each instruction in cuobjdump's dump prints two hex comment lines:
  /*0010*/   OPCODE ... ;   /* 0x<low64> */
                            /* 0x<high64> */
The control word lives in the high64 comment. Standard layout (bit offset
within high64, i.e. absolute bit-64 of the 128-bit instruction):
  bits 41-44 (4b): stall count
  bit  45        : yield (1 = do NOT yield, i.e. this warp can keep going)
  bits 46-48 (3b): write barrier index (7 = none set)
  bits 49-51 (3b): read barrier index  (7 = none set)
  bits 52-57 (6b): wait barrier mask (bit i set = wait on barrier i)

Robust against any single line's opcode-text formatting (predicated
instructions, branch targets, etc.): we only ever sync off the
"/*ADDR*/"-prefixed line itself, never off matching the full instruction
text, so a single odd-looking line can never desync the 2-line pairing for
the rest of the file.
"""
import re
import sys

addr_re = re.compile(r'^\s*/\*([0-9a-fA-F]+)\*/\s*(.*)$')
last_hex_re = re.compile(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/\s*$')
any_hex_re = re.compile(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/')

def decode_control(hi64):
    stall = (hi64 >> 41) & 0xF
    yield_bit = (hi64 >> 45) & 0x1
    wrtbar = (hi64 >> 46) & 0x7
    rdbar = (hi64 >> 49) & 0x7
    waitmask = (hi64 >> 52) & 0x3F
    return stall, yield_bit, wrtbar, rdbar, waitmask

def main(path):
    with open(path) as f:
        lines = f.readlines()
    i = 0
    n = 0
    skipped = 0
    nonzero_stall = 0
    while i < len(lines):
        m = addr_re.match(lines[i])
        if not m:
            i += 1
            continue
        addr = m.group(1)
        rest = m.group(2)
        # instruction text is everything up to the trailing /* 0x... */ comment
        hex_m = last_hex_re.search(rest)
        instr_text = rest[:hex_m.start()].rstrip() if hex_m else rest.strip()
        instr_text = instr_text.rstrip(';').strip()

        # the control word is on the NEXT line's hex comment, unconditionally
        if i + 1 < len(lines):
            hi_m = any_hex_re.search(lines[i + 1])
        else:
            hi_m = None

        if hi_m and instr_text:
            hi64 = int(hi_m.group(1), 16)
            stall, yld, wrtbar, rdbar, waitmask = decode_control(hi64)
            n += 1
            if stall > 0:
                nonzero_stall += 1
            print(f"{addr} {instr_text:<55} stall={stall} yield={yld} "
                  f"wrtbar={wrtbar if wrtbar!=7 else '-'} rdbar={rdbar if rdbar!=7 else '-'} "
                  f"waitmask={waitmask:06b}")
        else:
            skipped += 1
        i += 2  # always advance by exactly 2 -- never let content-parsing failures desync us

    print(f"\n# total decoded={n} skipped={skipped} nonzero_stall={nonzero_stall} "
          f"({100.0*nonzero_stall/max(n,1):.1f}%)", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1])

