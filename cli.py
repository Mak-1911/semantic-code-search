import argparse
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from indexer.chunker import ChunkCode, print_stats
from indexer.embedder import Embedder, MODEL_NAME, chunk_to_text
from indexer.repo_reader import read_repository
from storage.baseline import load_index, save_index
from search.retrieve import CANDIDATES, Reranker


TOP_K = 5

def track_peak_rss(peak, stop):
    proc = psutil.Process()
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.1)


def cmd_index(args):
    root = Path(args.repo).resolve()
    files = read_repository(root)
    print(f"{len(files)} eligible files")

    t0 = time.perf_counter()
    chunks = ChunkCode(str(root)).chunk_repo(files)
    print(f"chunking: {time.perf_counter() - t0:.2f}s")
    print_stats(chunks)

    peak, stop = {"rss": 0}, threading.Event()
    thread = threading.Thread(target=track_peak_rss, args=(peak, stop))
    thread.start()

    t0 = time.perf_counter()
    vectors = Embedder().embed_texts([chunk_to_text(c) for c in chunks])
    print(f"embedding: {time.perf_counter() - t0:.1f}s")

    stop.set()
    thread.join()
    print(f"peak RSS {peak['rss']/1024**2:.0f} MB")

    save_index(chunks, vectors, Path(args.index_dir), MODEL_NAME)
    print(f"index saved to {Path(args.index_dir)}/")


def cmd_search(args):
    chunks, vectors = load_index(Path(args.index_dir), MODEL_NAME)
    q = Embedder().embed_query(args.query)
    scores = vectors @ q

    n = CANDIDATES if args.rerank else TOP_K
    stage1 = np.argsort(scores)[::-1][:n]
    ranked = [(chunks[i], float(scores[i])) for i in stage1]

    if args.rerank:
        ranked = Reranker().rerank(args.query, ranked)

    for c, s in ranked[:TOP_K]:
        first = c.text.strip().splitlines()[0] if c.text.strip() else ""
        print(f"{c.path}:{c.start_line}-{c.end_line}  score={s:.3f}  {c.name or c.kind}")
        print(f"    {first}")

def main():
    parser = argparse.ArgumentParser(description="semantic code search")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a repository")
    p_index.add_argument("repo", nargs="?", default=".")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.set_defaults(func=cmd_search)

    p_index.add_argument("--index-dir", default=".index")
    p_search.add_argument("--index-dir", default=".index")
    p_search.add_argument("--rerank", action="store_true",
                          help="re-rank the top 20 with a cross-encoder")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()