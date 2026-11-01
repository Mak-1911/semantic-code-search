# semantic-code-search

**Ask a question in plain English. Get the code that answers it — file, line numbers, and all.**

![Python](https://img.shields.io/badge/python-3.12+-blue)
![CPU only](https://img.shields.io/badge/hardware-CPU--only-informational)
![tree-sitter](https://img.shields.io/badge/parsing-tree--sitter-orange)
![eval](https://img.shields.io/badge/recall@5-0.88–0.92-success)

A local-first semantic code search engine built from scratch: tree-sitter
structural chunking → transformer embeddings → exact NumPy search → an
experimental cross-encoder re-rank stage. Everything runs on a laptop CPU.
No API calls, no vector database.

```text
$ python cli.py search "where do we skip directories when walking the repository?"
indexer/repo_reader.py:16-57  score=0.619  read_repository
    def read_repository(root: Path) -> list[Path]:
indexer/repo_reader.py:1-4  score=0.612  repo_reader
    """
indexer/repo_reader.py:8-14  score=0.583  under_nested_repo
    def under_nested_repo(path: Path, root: Path) -> bool:
```

## Why

`grep` finds strings. IDE go-to-definition finds symbols you already know.
Neither answers *"where is the code that decides if a stock is too risky?"*
That query lives in the gap between natural language and source code — this
project closes it, entirely offline.

## Architecture

```
                 ┌──────────────┐   ┌───────────────┐   ┌────────────────┐
 repo ──────────►│ repo_reader  │──►│    chunker    │──►│    embedder    │
                 │ (walk+filter)│   │ (tree-sitter) │   │ (Qwen3-0.6B)   │
                 └──────────────┘   └───────────────┘   └───────┬────────┘
                                                                │ save_index
                                                                ▼
                 ┌──────────────┐   ┌───────────────┐   ┌────────────────┐
 query ─────────►│   embedder   │──►│  cosine top-5 │──►│   reranker*    │
                 │ (instructed) │   │   (1 matmul)  │   │  (optional)    │
                 └──────────────┘   └───────────────┘   └────────────────┘
```

| module | role |
|---|---|
| `indexer/repo_reader.py` | walks the repo; extension + junk-dir filters; skips nested git repos |
| `indexer/chunker.py` | tree-sitter walk → functions, classes, docstrings, fallback windows |
| `indexer/embedder.py` | Qwen3-Embedding-0.6B (1024-dim), instruction-prefixed queries, unit-normalized rows |
| `storage/baseline.py` | `vectors.npy` + `chunks.jsonl` + `meta.json`; row-alignment asserted on save, verified on load |
| `search/retrieve.py` | *experimental* cross-encoder re-rank stage (`--rerank`), see [the reranker experiment](#the-reranker-experiment-a-measured-negative-result) |
| `evaluation/run_eval.py` | graded eval: recall@5 / hit@1 / MRR by difficulty, per-query miss details, latency |

Design invariants: chunk *i* ↔ jsonl line *i* ↔ vector row *i*; unit-norm rows
so cosine is one matmul; `meta.json` records model + dim and **fails loudly**
on mismatch; nothing silently dropped from the index.

## Results

Two eval sets, hand-written queries with one expected `(path, symbol)` each,
graded on top-5:

- **own repo** — n=42 queries (easy / medium / trap), the codebase itself;
- **foreign repo** — n=24 queries sampled from a real codebase this pipeline
  had never seen (sampled with a fixed seed so the set is reproducible);
  removes the self-authorship bias of writing queries against your own code.

| metric    | own repo (n=42) | foreign repo (n=24) |
|-----------|-----------------|---------------------|
| recall@5  | 0.88            | 0.92                |
| hit@1     | 0.57            | 0.62                |
| MRR       | 0.69            | 0.75                |

Coverage is strong; ranking is the weak spot — and the gap *between* the two
numbers is what drove every design decision below.

Resources: indexing ≈ 3 min for 37 chunks on CPU (embedding dominates);
query p50 ≈ 0.5 s (≈99% model inference — the matmul is microseconds); peak
RSS ≈ 3 GB; index storage 150.7 KB, i.e. 4 KB/chunk → ~4 GB per million
chunks, which is the argument for a compact binary index later.

## The interesting parts

**Structural chunking, not sliding windows.** A 4-branch tree-sitter walk
emits decorated definitions (decorator span kept, inner def's name), nested
defs (methods found by recursion), and module docstrings (natural-language
searchable). Unparsable files still produce 50-line fallback windows —
nothing silently disappears. Every feature is pinned by a hand-computed
oracle file; any regression fails the chunker instantly.

**Asymmetric retrieval.** Queries get Qwen3's instruction prefix
(*"Given a natural language query, retrieve code snippets…"*); indexed chunks
get none. The model was trained for exactly this asymmetry.

**A real segfault hunt.** Indexing a foreign repo crashed natively — no
traceback, uncatchable by `try/except`. Per-unit `print(..., flush=True)`
instrumentation established the signature: ~1-in-3 processes died, always in
the same parse window; every survivor ran 570+ parses clean ("dies early or
never" — a per-process heap-layout coin flip). One-variable bisection
(input → cyclic GC → parser reuse → Python version → binding version) landed
on py-tree-sitter 0.26.0's Node/Tree lifetime bug; 0.25.2 runs 5 processes ×
570 parses clean. Corroborated by upstream issues
[#435](https://github.com/tree-sitter/py-tree-sitter/issues/435) and
[#308](https://github.com/tree-sitter/py-tree-sitter/issues/308). Full case
file and debugging playbook in [`docs/design-notes.md`](docs/design-notes.md).

**Measure → diagnose → fix, in that order.** hit@1 lagged recall@5 by 30+
points, so before touching the ranking path, all 28 rank≠1 eval queries were
classified by hand. The assumed culprit — parent/child chunks splitting the
vote — turned out *not* to be the main cause. The leaders were **concept
collision** (same idea living in two files), a **vocabulary hub** (a short
field-name-dense chunk magnetically winning queries it can't answer), and
**prose beats code** (doc chunks out-competing the target function).

**An eval that fights back.** Difficulty rubric with deliberately planted
split-vote pairs and trap queries; a blind-sampled foreign validation set;
and a recorded drift lesson — when the codebase grew 29 → 32 chunks,
unchanged queries produced new distractors and hit@1 fell. An eval set is
valid for exactly one version of the code.

## The reranker experiment: a measured negative result

The miss analysis said the failures were *ordering* failures — the answer was
almost always already in the candidate set. The textbook fix for that is a
cross-encoder re-ranker: instead of comparing two independently-compressed
vectors, feed (query, chunk) through a transformer *together* and let
attention judge the pair. Textbook also says it works. It didn't:

| stage                       | recall@5 (own/foreign) | hit@1 (own/foreign) | p50 latency |
|-----------------------------|------------------------|---------------------|-------------|
| baseline (bi-encoder only)  | 0.88 / 0.92            | 0.57 / 0.62         | 0.5 s       |
| + reranker, first attempt   | 0.62 / 0.62            | 0.43 / 0.46         | 3.2 s       |
| + prose filter + rank fusion| 0.81 / 0.75            | 0.48 / 0.42         | 1.5 s       |

Three things were learned, each the hard way:

1. **A general-text reranker reads prose better than code.** The ms-marco
   model judged a doc file that *describes* directory-skipping more relevant
   than the function that *does* it — docs chunks colonized the top-5 and
   recall collapsed from 0.88 to 0.62. Diagnosis from per-query logs, fixed
   structurally: only code-ish chunks enter the rerank pool; prose keeps its
   retrieval order behind them.
2. **When nothing in the pool truly answers, the cross-encoder orders noise.**
   On twin-function queries the top-5 logits sat within 0.2 of each other —
   plausible-sounding near-misses beat the real answer. Blending the CE's
   order with the bi-encoder's (rank fusion, α=0.7) capped the damage without
   recovering the loss: still below baseline on every headline metric.
3. **Latency compounds honestly.** 20 pairs × 512-token sequences on a CPU
   tripled p50 even after moving off a 2-thread cap.

The stage stays in the repo behind `--rerank` — off by default — because a
negative result you can reproduce is worth more than one you deleted.
Recorded next steps, none attempted yet: a code-trained reranker
(e.g. jina-reranker-v2-base-code), a fusion-weight sweep (deliberately not
done here — tuning α against n=42 until it passes *is* overfitting the eval),
or fine-tuning the reranker on code pairs.

## Quickstart

```bash
uv sync                                   # Python 3.12+

python cli.py index                       # index the current directory → .index/
python cli.py index path/to/other/repo --index-dir .index-other

python cli.py search "your question here"
python cli.py search "your question" --rerank    # experimental two-stage retrieval

python evaluation/run_eval.py             # graded eval on evaluation/queries.jsonl
python evaluation/run_eval.py --rerank --index-dir .index-other \
       --queries evaluation/queries_other.jsonl
```

## Limitations, honestly

- Eval sets are hand-written and small (n=42, n=24) — numbers are directional.
- Full structural parsing for Python; 8 other extensions indexed as fallback
  windows only.
- 0.6B fp32 model = ~3 GB RAM.
- The re-rank stage is a documented negative result (above) — don't turn it on
  expecting better ranking yet.
- No incremental indexing yet — any code change means a full re-index.

## Design notes

The full decision log, chunker mechanics, segfault case file, eval-set
history, and per-component reasoning live in
[`docs/design-notes.md`](docs/design-notes.md).
