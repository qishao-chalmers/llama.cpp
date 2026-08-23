#!/usr/bin/env python3
"""
fetch_aime.py — Download AIME and format as JSONL for run_sweep structured mode.

AIME (American Invitational Mathematics Examination) problems with integer answers 0–999.
Uses the di-zhang-fdu/AIME_1983_2024 dataset from HuggingFace (933 problems, 1983–2024).

Each output record:
  {"prompt": "<few-shot examples + problem>",
   "completion": "<answer line>",
   "answer": "<str>",
   "dataset": "aime", "id": <int>, "year": <int>, "problem_number": <int>}

The completion is just the final boxed-answer line so PPL measures how likely the model
is to produce the correct answer.  Accuracy evaluation generates the full chain-of-thought
and extracts via the boxed-answer regex.

Usage:
  # All problems 1983-2024
  python3 research/scripts/fetch_aime.py --out research/data/aime.jsonl

  # Recent years only (less contamination risk)
  python3 research/scripts/fetch_aime.py --out research/data/aime_2024.jsonl --year-min 2024

  # Zero-shot (no few-shot examples prepended)
  python3 research/scripts/fetch_aime.py --out research/data/aime.jsonl --n-shot 0

Evaluate with run_sweep.py:
  python3 research/scripts/run_sweep.py model.gguf research/data/aime_2024.jsonl \\
      --corpus-mode structured --eval-accuracy --skip-ppl \\
      --answer-regex 'boxed{(\\d+)}' \\
      --stop-strings $'\\n\\n---' $'\\n\\nQuestion:' $'\\nassistant' $'\\n\\n\\n' \\
      --n-ctx 8192 --max-gen-tokens 4096 \\
      --quants fp16 int8_ch int4_ch int3_ch int2_ch
"""

import argparse
import json
import os

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install datasets")


# 4 few-shot examples from pre-2022 AIME (well-known, high-confidence answers).
# Solutions end with \\boxed{N} so the answer-regex fires on few-shot completions too.
FEW_SHOT = [
    {
        "problem": (
            "Let $x$, $y$, and $z$ all exceed $1$ and let $w$ be a positive number "
            "such that $\\log_x w = 24$, $\\log_y w = 40$, and $\\log_{xyz} w = 12$. "
            "Find $\\log_z w$."
        ),
        "solution": (
            "From $\\log_x w = 24$ we get $\\log_w x = \\frac{1}{24}$. "
            "Similarly $\\log_w y = \\frac{1}{40}$ and $\\log_w (xyz) = \\frac{1}{12}$. "
            "So $\\log_w z = \\frac{1}{12} - \\frac{1}{24} - \\frac{1}{40} "
            "= \\frac{10 - 5 - 3}{120} = \\frac{2}{120} = \\frac{1}{60}$. "
            "Therefore $\\log_z w = \\boxed{60}$."
        ),
        "answer": "60",
    },
    {
        "problem": (
            "What is the largest positive integer $n$ for which $n^3 + 100$ is "
            "divisible by $n + 10$?"
        ),
        "solution": (
            "Write $n^3 + 100 = (n+10)(n^2 - 10n + 100) - 900$. "
            "So $(n+10) \\mid (n^3+100)$ iff $(n+10) \\mid 900$. "
            "The largest value of $n+10$ that divides 900 is 900 itself, "
            "giving $n = 890$. "
            "The answer is $\\boxed{890}$."
        ),
        "answer": "890",
    },
    {
        "problem": (
            "The sum of the lengths of the twelve edges of a rectangular box is $140$, "
            "and the distance from one corner of the box to the farthest corner is $21$. "
            "What is the total surface area of the box?"
        ),
        "solution": (
            "Let the dimensions be $a, b, c$. "
            "Then $4(a+b+c) = 140$, so $a+b+c = 35$. "
            "The space diagonal gives $a^2+b^2+c^2 = 21^2 = 441$. "
            "Surface area $= 2(ab+bc+ca) = (a+b+c)^2 - (a^2+b^2+c^2) = 1225 - 441 = 784$. "
            "The answer is $\\boxed{784}$."
        ),
        "answer": "784",
    },
    {
        "problem": (
            "In the sequence $2001$, $2002$, $2003$, $\\ldots$, each term after the third "
            "is found by subtracting the previous term from the sum of the two terms that "
            "precede it. What is the $2001$st term in this sequence?"
        ),
        "solution": (
            "Let the first three terms be $a, b, c$. "
            "The recurrence is $t_n = t_{n-2} + t_{n-3} - t_{n-1}$. "
            "Computing: $t_1=2001, t_2=2002, t_3=2003, "
            "t_4=2001+2002-2003=2000, t_5=2002+2003-2000=2005, "
            "t_6=2003+2000-2005=1998, t_7=2000+2005-1998=2007, "
            "t_8=2005+1998-2007=1996, t_9=1998+2007-1996=2009, "
            "t_{10}=2007+1996-2009=1994, t_{11}=1996+2009-1994=2011, "
            "t_{12}=2009+1994-2011=1992$. "
            "The sequence has period $12$ starting from $t_1$. "
            "$2001 = 12 \\cdot 166 + 9$, so $t_{2001} = t_9 = 2009$. "
            "The answer is $\\boxed{2009}$."
        ),
        "answer": "2009",
    },
]


def build_few_shot_prompt(examples, n_shot):
    parts = []
    for ex in examples[:n_shot]:
        parts.append(
            f"Problem: {ex['problem']}\n\nSolution: {ex['solution']}"
        )
    return "\n\n---\n\n".join(parts) + ("\n\n---\n\n" if parts else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",          default="research/data/aime.jsonl")
    parser.add_argument("--dataset",      default="di-zhang-fdu/AIME_1983_2024",
                        help="HuggingFace dataset name (default: di-zhang-fdu/AIME_1983_2024)")
    parser.add_argument("--year-min",     type=int, default=0,
                        help="Only include problems from this year onward (0 = all)")
    parser.add_argument("--year-max",     type=int, default=9999,
                        help="Only include problems up to this year (default: all)")
    parser.add_argument("--n-shot",       type=int, default=0,
                        help="Number of few-shot examples prepended to each prompt (default 0). "
                             "AIME reasoning models need no format demonstration; use 0 for "
                             "results comparable to published benchmarks.")
    parser.add_argument("--n-examples",   type=int, default=0,
                        help="Max examples to export after filtering (0 = all)")
    args = parser.parse_args()

    print(f"Downloading {args.dataset}...", flush=True)
    ds = load_dataset(args.dataset, split="train")

    records = list(ds)

    # Filter by year
    if args.year_min > 0 or args.year_max < 9999:
        before = len(records)
        records = [r for r in records
                   if args.year_min <= int(r.get("Year", r.get("year", 0))) <= args.year_max]
        print(f"  Year filter [{args.year_min}, {args.year_max}]: "
              f"{before} → {len(records)} problems")

    if args.n_examples > 0:
        records = records[:args.n_examples]

    few_shot_prompt = build_few_shot_prompt(FEW_SHOT, args.n_shot)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prompt_lens = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            # di-zhang-fdu/AIME_1983_2024 uses "Question"; fall back to "Problem" variants.
            problem = (rec.get("Question") or rec.get("Problem") or rec.get("problem") or "").strip()
            answer  = str(rec.get("Answer") or rec.get("answer") or "").strip()
            year    = int(rec.get("Year",   rec.get("year",   0)))
            pnum    = int(rec.get("Problem Number", rec.get("problem_number", 0)))

            prompt     = few_shot_prompt + f"Problem: {problem}\n\nSolution:"
            # Completion: the canonical boxed answer line (short — for PPL scoring).
            # Accuracy eval generates full chain-of-thought and extracts via regex.
            completion = f" The answer is $\\boxed{{{answer}}}$."

            prompt_lens.append(len(prompt.split()))

            out = {
                "prompt":          prompt,
                "completion":      completion,
                "answer":          answer,
                "dataset":         "aime",
                "id":              i,
                "year":            year,
                "problem_number":  pnum,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(records)
    avg_p = sum(prompt_lens) // n if n else 0
    print(f"Wrote {n} examples to {args.out}")
    print(f"  Few-shot shots : {args.n_shot}")
    print(f"  Avg prompt len : ~{avg_p} words (~{avg_p * 4 // 3} tokens)")
    print(f"  Completion     : short boxed-answer line (for PPL); "
          f"generate up to 2048 tokens for accuracy")
    print()
    print("Evaluate with:")
    print(f"  python3 research/scripts/run_sweep.py model.gguf {args.out} \\")
    print(f"      --corpus-mode structured --eval-accuracy --skip-ppl \\")
    print(f"      --answer-regex 'boxed{{(\\d+)}}' \\")
    print(f"      --stop-strings $'\\n\\n---' $'\\n\\nQuestion:' $'\\nassistant' $'\\n\\n\\n' \\")
    print(f"      --n-ctx 8192 --max-gen-tokens 4096 \\")
    print(f"      --quants fp16 int8_ch int4_ch int3_ch int2_ch")


if __name__ == "__main__":
    main()
