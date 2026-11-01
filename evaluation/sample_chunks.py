import json
import random
import sys

SEED = 42          # fixed seed → same sample every run; recorded in design notes
N = 24

chunks = [json.loads(line) for line in open(".index-hedge-fund/chunks.jsonl", encoding="utf-8")]
random.seed(SEED)
picks = random.sample(range(len(chunks)), N)

show_text = "--text" in sys.argv
for i in picks:
    c = chunks[i]
    first = c["text"].strip().splitlines()[0] if c["text"].strip() else ""
    print(f"{i:3}  {c['path']}:{c['start_line']}-{c['end_line']}  {c['kind']:9} {c['name'] or '-':18} {first[:60]}")
    if show_text:
        print("-" * 70)
        print(c["text"][:700])
        print("=" * 70)
