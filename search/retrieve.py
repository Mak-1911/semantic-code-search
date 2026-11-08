import os
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer.chunker import Chunk
from indexer.embedder import chunk_to_text

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # stage-1 pool fed to the reranker
ALPHA_CE = 0.7   # weight of the cross-encoder's order vs the bi-encoder's
CODE_KINDS = {"function", "class", "docstring"}  # prose fallbacks never enter the pool
RESCUE_SLOT = 5  # only this position of the dense top-5 is contestable

class Reranker:
    """Stage 2: read (query, chunk) pairs jointly, re-order the stage-1 pool.

    Unlike the bi-encoder, this model sees query and chunk in one pass, so
    attention can compare query tokens against chunk tokens directly.

    First attempt (2026-11-01) regressed both eval sets — this ms-marco model
    prefers prose passages over code, so docs fallback chunks colonized the
    top-5 (recall@5 0.88 -> 0.62 own repo, 0.92 -> 0.62 foreign). Two guards:
    only code-ish chunks enter the pool (prose keeps its stage-1 order behind
    them), and the cross-encoder's order is *blended* with the bi-encoder's
    instead of replacing it, so a confused ranking can't fully invert.
    """

    def __init__(self):
        # the CE is tiny (~22M params); the Embedder's 2-thread cap would make
        # it several times slower than it needs to be
        torch.set_num_threads(min(4, os.cpu_count() or 2))
        self.model = CrossEncoder(RERANKER_NAME, max_length=512)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        """candidates: (chunk, bi-encoder cosine) pairs in stage-1 order.

        Returns all candidates in final order: reranked code chunks first,
        prose fallbacks after in stage-1 order. Scores are raw CE logits —
        order is the only thing that matters.
        """
        pool = [(c, s) for c, s in candidates if c.kind in CODE_KINDS]
        prose = [(c, s) for c, s in candidates if c.kind not in CODE_KINDS]
        if not pool:
            return prose

        # raw query: the "Instruct:" prefix is a Qwen-embedding convention,
        # not something the cross-encoder was trained on
        pairs = [(query, chunk_to_text(c)) for c, _ in pool]
        ce = self.model.predict(pairs, batch_size=8, convert_to_numpy=True)

        # normalized rank per scorer: 1.0 for best, 0.0 for worst — scale-free
        # fusion between a cosine and a raw logit
        def ranks(x):
            x = np.asarray(x)
            return (len(x) - 1 - np.argsort(np.argsort(-x))) / max(len(x) - 1, 1)

        blended = ALPHA_CE * ranks(ce) + (1 - ALPHA_CE) * ranks([s for _, s in pool])
        order = np.argsort(-blended)
        ranked = [(pool[i][0], float(ce[i])) for i in order]
        return ranked + prose

    def rescue(self, query: str, candidates: list[tuple[Chunk, float]]):
        """Rescue-mode CE: never reorders, never demotes.
        Slots 1-4 of the dense order are untouchable. A candidate from the 
        pool tail swaps into the last slot only if the CE logit beats every code
        chunk in the dense top-5. Prose holds no logit, so it never blocks a rescue
        Returns (top5, swaps) — swaps: (query-side context filled in by the caller),
        one entry per swap.
        """
        pool = [(c, s) for c, s in candidates if c.kind in CODE_KINDS]
        if not pool:
            return candidates[:RESCUE_SLOT], []

        pairs = [(query, chunk_to_text(c)) for c, _ in pool]
        ce = self.model.predict(pairs, batch_size=8, convert_to_numpy=True)

        def key(c):
            return (c.path, c.start_line, c.name)

        top5 = candidates[:RESCUE_SLOT]
        inside = {key(c) for c, _ in top5 if c.kind in CODE_KINDS}
        inside_max = max(
            (float(ce[i]) for i, (c, _) in enumerate(pool) if key(c) in inside),
            default=float("-inf"),
        )

        swaps = []
        best = None
        for i, (c, s) in enumerate(pool):
            if key(c) in inside:
                continue
            if float(ce[i]) > inside_max and (best is None or ce[i] > best[0]):
                best = (float(ce[i]), i, s)
        if best is not None:
            ce_best, i, dense_score = best
            evicted = top5[-1]
            swaps.append((key(evicted[0]), key(pool[i][0]), ce_best, dense_score))
            top5 = top5[:-1] + [(pool[i][0], dense_score)]
        return top5, swaps
            
if __name__ == "__main__":
    from indexer.chunker import ChunkCode

    root = Path(".").resolve()
    chunks = ChunkCode(root).chunk_file(str(root / "cli.py"))
    r = Reranker()
    for c, s in r.rerank("where are search queries embedded differently?",
                         [(c, 0.0) for c in chunks[:20]]):
        print(f"{s:8.4f}  {c.path}:{c.start_line}-{c.end_line}  {c.name or c.kind}")
