#!/usr/bin/env python3
"""
fetch_gsm8k.py — Download GSM8K and format as JSONL for run_sweep structured mode.

Each output record:
  {"prompt": "<few-shot examples + question>", "completion": "<chain-of-thought + answer>",
   "dataset": "gsm8k", "id": <int>}

The prompt contains 8 few-shot examples (standard GSM8K eval setup) so the model
sees worked examples before being asked to solve the test question. This mimics
real few-shot inference and gives the KV cache a meaningful long context to compress.

Usage:
  python3 research/scripts/fetch_gsm8k.py --out research/data/gsm8k_test.jsonl
  python3 research/scripts/fetch_gsm8k.py --out research/data/gsm8k_test.jsonl --n-shot 4 --n-examples 200
"""

import argparse
import json
import os
import random

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install datasets")


# Standard 8-shot examples from the GSM8K paper / EleutherAI eval harness
FEW_SHOT_8 = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. Christmas is coming and he receives two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.",
    },
]


def build_few_shot_prompt(examples, n_shot):
    """Build a few-shot prompt string from the first n_shot examples."""
    lines = []
    for ex in examples[:n_shot]:
        lines.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(lines) + "\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default="research/data/gsm8k_test.jsonl")
    parser.add_argument("--split",      default="test", choices=["test", "train"])
    parser.add_argument("--n-shot",     type=int, default=8,
                        help="Number of few-shot examples prepended to each prompt (default 8)")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Max examples to export (0 = all, test set has 1319)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Downloading GSM8K {args.split} split...", flush=True)
    ds = load_dataset("gsm8k", "main", split=args.split)

    few_shot_prompt = build_few_shot_prompt(FEW_SHOT_8, args.n_shot)

    records = list(ds)
    if args.n_examples > 0:
        records = records[:args.n_examples]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            question = rec["question"].strip()
            answer   = rec["answer"].strip()

            # Prompt = few-shot context + test question header
            # Completion = chain-of-thought + final answer
            prompt     = few_shot_prompt + f"Question: {question}\nAnswer:"
            completion = " " + answer   # space before CoT (common LM convention)

            out = {
                "prompt":     prompt,
                "completion": completion,
                "dataset":    "gsm8k",
                "id":         i,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} examples to {args.out}")
    print(f"  Few-shot: {args.n_shot} examples in each prompt")
    print(f"  Typical prompt length: ~{len(few_shot_prompt.split()) * 4 // 3} tokens "
          f"(few-shot) + question")
    print(f"  Typical completion length: ~200-400 tokens (chain-of-thought + answer)")


if __name__ == "__main__":
    main()
