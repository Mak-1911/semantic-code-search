"""Isolate the embedding segfault: small batches, progress prints, flush everything"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from indexer.chunker import ChunkCode
from indexer.embedder import Embedder, chunk_to_text
from indexer.repo_reader import read_repository

root = Path(__file__).resolve().parent.parent
chunks = ChunkCode(str(root)).chunk_repo(read_repository(root))
texts = [chunk_to_text(c) for c in chunks]
print("chunks:", len(chunks), flush=True)
for i, c in enumerate(chunks):
    print(f"    {i}: {len(c.text)} chars {c.path}  {c.name}", flush=True)

torch.set_num_threads(2)

e = Embedder()
BATCH = 4
for start in range(0, len(texts), BATCH):
    print(f"--- embedding {start}..{start + BATCH}", flush=True)
    v = e.embed_texts(texts[start:start + BATCH], )
    print("    ok", v.shape, flush=True)
print("ALL OK", flush=True)