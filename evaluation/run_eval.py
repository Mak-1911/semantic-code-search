import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer.embedder import MODEL_NAME, Embedder
from search.retrieve import CANDIDATES, RERANKER_NAME, Reranker
from storage.baseline import load_index

parser = argparse.ArgumentParser(description="run the eval")
parser.add_argument("--index-dir", default=".index")
parser.add_argument("--queries", default="evaluation/queries.jsonl")
parser.add_argument("--rerank", action="store_true",
                    help="re-rank the top 20 with a cross-encoder")
parser.add_argument("--rescue", action="store_true",
                    help="rescue-mode CE: swap-in only, dense top-4 sacred")
parser.add_argument("--why", action="store_true",
                    help="diagnose non-rank-1 queries: dense score margins + "
                         "lexical overlap vs the dense top-5 (always stage-1, "
                         "independent of --rerank/--rescue)")
args = parser.parse_args()
if args.rerank and args.rescue:
    parser.error("--rerank and --rescue are different experiments")

queries = []
with open(args.queries, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            queries.append(json.loads(line))

chunks, vectors = load_index(Path(args.index_dir), MODEL_NAME)
embedder = Embedder()
reranker = Reranker() if (args.rerank or args.rescue) else None
if args.rerank:
    print(f"reranking top {CANDIDATES} with {RERANKER_NAME}")
elif args.rescue:
    print("rescue mode: single-slot swap-in, dense top-4 sacred")
print(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.0f} MB")

latencies = []
results = []
details = []
attribution = []
# why-mode: (row, final_rank, dense_rank, ans_idx, ans_score, dense_top5_idx, scores)
whys = []
swap_log = []

def grade(chunk, row) -> bool:
    # accepted_alternatives: (path, symbol) pairs that are equally correct
    # answers to a human. Decided once, on validity grounds, never added
    # to rescue a failing score (2026-11-08).
    def matches(path, symbol):
        return chunk.path == path and (symbol == "" or chunk.name == symbol)

    if matches(row["expected_path"], row["expected_symbol"]):
        return True
    return any(
        matches(a["path"], a["symbol"])
        for a in row.get("accepted_alternatives", [])
    )

for row in queries:
    t0 = time.perf_counter()
    q = embedder.embed_query(row["query"])
    scores = vectors @ q
    dense_order = np.argsort(scores)[::-1]
    # answer's rank in the full dense order: stage-1 quality, CE-independent
    dense_rank = next(
        (pos + 1 for pos, i in enumerate(dense_order) if grade(chunks[i], row)), None
    )
    ans_i = next((i for i in dense_order if grade(chunks[i], row)), None)
    ans_score = float(scores[ans_i]) if ans_i is not None else None
    if args.rescue:
        ranked, swaps = reranker.rescue(
            row["query"],
            [(chunks[i], float(scores[i])) for i in dense_order[:CANDIDATES]],
        )
        top_chunks = [c for c, _ in ranked[:5]]
        top_scores = [s for _, s in ranked[:5]]
        swap_log.extend((row["query"], row["difficulty"], *s) for s in swaps)
    elif reranker:
        stage1 = np.argsort(scores)[::-1][:CANDIDATES]
        ranked = reranker.rerank(
            row["query"], [(chunks[i], float(scores[i])) for i in stage1]
        )
        top_chunks = [c for c, _ in ranked[:5]]
        top_scores = [s for _, s in ranked[:5]]
    else:
        top5 = np.argsort(scores)[::-1][:5]
        top_chunks = [chunks[i] for i in top5]
        top_scores = [float(scores[i]) for i in top5]
    elapsed = time.perf_counter() - t0
    latencies.append(elapsed)

    rank = next((i + 1 for i, c in enumerate(top_chunks) if grade(c, row)), None)
    details.append((row, rank, top_chunks, top_scores))
    results.append((row["query"], row["difficulty"], rank))
    attribution.append((row, dense_rank, rank))
    whys.append((row, rank, dense_rank, ans_i, ans_score, dense_order[:5], scores))


def metrics(ranks):
    n = len(ranks)
    return(
        sum(r is not None for r in ranks) / n,
        sum(r == 1 for r in ranks) / n,
        sum((1 / r if r else 0) for r in ranks) / n,
    )

mode = "full CE" if args.rerank else ("rescue" if args.rescue else "dense only")
print(f"\nstage attribution (pool={CANDIDATES}, {mode}):")
buckets = {}
for row, dr, fr in attribution:
    if dr is None or dr > CANDIDATES:
        b = "beyond pool"
    elif fr is None:
        b = "outside_top5"
    elif fr > dr:
        b = "demoted"
    elif fr < dr:
        b = "promoted"
    else:
        b = "same"
    buckets[b] = buckets.get(b, 0) + 1
    print(f"  [{row['difficulty']:6}] dense={str(dr):>4}  final={str(fr):>4}  "f"{b:12}  {row['query']}")
print("  " + "  ".join(f"{k}={v}" for k, v in sorted(buckets.items())))

far = [dr for _, dr, _ in attribution if dr and dr > CANDIDATES]
if far:
    print(f"    beyond_pool dense ranks: {sorted(far)}")

if swap_log:
    print(f"\nrescue swaps ({len(swap_log)}):")
    for q, diff, ev, ins, ce_best, ds in swap_log:
        print(f"  [{diff:6}] {q}")
        print(f"      evicted  {ev[0]}  {ev[2] or ev[1]}")
        print(f"      inserted {ins[0]}  {ins[2] or ins[1]}   ce={ce_best:.3f}  dense_cos={ds:.3f}")
else:
    print("\nrescue swaps: none")


r5, h1, m = metrics([r for _, _, r in results])
print(f"\noverall    recall@5={r5:.2f}  hit@1={h1:.2f}  mrr={m:.2f}   (n={len(results)})")
for diff in ("easy", "medium", "trap"):
    sub = [r for _, d, r in results if d == diff]
    if sub:
        d5, d1, dm = metrics(sub)
        print(f"{diff:9} recall@5={d5:.2f}  hit@1={d1:.2f}  mrr={dm:.2f}   (n={len(sub)})")


p50, p95 = np.percentile(latencies, [50, 95])
print(f"\nquery latency  p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms   (model load excluded)")

print("\nmissed (answer not in top 5):")
for q, d, r in results:
    if r is None:
        print(f"  [{d}] {q}")

if args.why:
    # question/connector words carry no signal — drop them so the overlap
    # column only shows content words
    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "we", "do",
        "does", "did", "how", "what", "where", "why", "which", "when",
        "who", "to", "of", "in", "for", "on", "with", "and", "or", "not",
        "it", "its", "that", "this", "these", "those", "from", "by", "at",
        "as", "into", "about", "s", "t", "don",
    }

    def tokens(s):
        # lowercase alphanumeric words; snake_case identifiers split apart
        return set(re.findall(r"[a-z0-9]+", s.lower())) - STOP

    print("\nwhy: every query that didn't rank 1, vs the dense top-5"
          "  (delta = how much it beat the answer by)")
    for row, rank, dr, ai, asc, top5, sc in whys:
        if rank == 1:
            continue
        qtok = tokens(row["query"])
        print(f"\n[{row['difficulty']}] {row['query']}")
        if ai is None:
            print("  answer is not in the index at all — stale index or dead expectation")
            continue
        ans = chunks[ai]
        hit = qtok & tokens(f"{ans.path} {ans.name} {ans.text}")
        print(f"  answer  {asc:.3f} dense={dr}  {ans.path}  {ans.name or ans.kind}"
              f"   overlap({len(hit)}): {' '.join(sorted(hit)) or '-'}")
        for pos, i in enumerate(top5, 1):
            c = chunks[i]
            h = qtok & tokens(f"{c.path} {c.name} {c.text}")
            d = float(sc[i]) - asc
            print(f"  #{pos}    {float(sc[i]):.3f} {d:+.3f}  {c.path}  {c.name or c.kind}"
                  f"   overlap({len(h)}): {' '.join(sorted(h)) or '-'}")

print("\nper-query details (rank != 1):")
for row, rank, top_chunks, top_scores in details:
    if rank == 1:
        continue
    print(f"\n[rank {rank}] [{row['difficulty']}] {row['query']}")
    for pos, (c, s) in enumerate(zip(top_chunks, top_scores), start=1):
        mark = "  <-- answer" if grade (c, row) else ""
        print(f"    {pos}. {s:.3f}  {c.path}    {c.name or c.kind}{mark}")

total = 0
for f in sorted(Path(args.index_dir).iterdir()):
    print(f"  {f.name:15} {f.stat().st_size/1024:8.1f} KB")
    total += f.stat().st_size
print(f"index storage: {total/1024:.1f} KB")

