import argparse
import json
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
args = parser.parse_args()

queries = []
with open(args.queries, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            queries.append(json.loads(line))

chunks, vectors = load_index(Path(args.index_dir), MODEL_NAME)
embedder = Embedder()
reranker = Reranker() if args.rerank else None
if reranker:
    print(f"reranking top {CANDIDATES} with {RERANKER_NAME}")
print(f"RSS: {psutil.Process().memory_info().rss / 1024**2:.0f} MB")

latencies = []
results = []
details = []

def grade(chunk, row) -> bool:
    if chunk.path != row["expected_path"]:
        return False
    return row["expected_symbol"] == "" or chunk.name == row["expected_symbol"]

for row in queries:
    t0 = time.perf_counter()
    q = embedder.embed_query(row["query"])
    scores = vectors @ q
    if reranker:
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


def metrics(ranks):
    n = len(ranks)
    return(
        sum(r is not None for r in ranks) / n,
        sum(r == 1 for r in ranks) / n,
        sum((1 / r if r else 0) for r in ranks) / n,
    )

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

