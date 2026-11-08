# Design Notes — semantic-code-search

Living notes on what was built, why each choice was made, and how it was verified.
Update as components land. Started 2026-10-24.

## Scope of the baseline

Python-only baseline: `chunker → embedder → storage → CLI`, exact NumPy cosine
search. No C++ engine, no summarization model — both come after retrieval quality
is proven on real queries.

Goal for this phase: index one repository and have a natural-language query
return 5 code snippets with file paths and line numbers.

**Achieved 2026-10-25.** First real query: *"where do we skip directories when
walking the repository?"* → top-2 hits were the `repo_reader.py` module
docstring (0.612) and `read_repository` (0.601); tail was sensible
(`chunk_repo`, `ChunkCode`, `__init__`). Docstring chunking (decision above)
won the #1 spot in its first live test. Index run: 11 files, 29 chunks
(22 function, 3 class, 2 fallback, 2 docstring), embedding 89s on CPU.

## Environment

- uv project, `.venv/` (Python 3.12.10 — downgraded from 3.14 during the
  segfault hunt and kept; `requires-python` relaxed to `>=3.12`; uv 0.9.9)
- torch 2.14.1+cpu (deliberate: see model decision), sentence-transformers 6.2.0
- tree-sitter 0.25.2 (downgraded from 0.26.0 — see "Case file" below),
  tree-sitter-python 0.25.0
- GPU available (GTX 1650 Ti, 4 GB) but unused for now

Note: tree-sitter's Python API changed across versions; current form is
`Language(tree_sitter_python.language())` — no language-name argument.

## Decision log

| Decision | Choice | Reasoning |
|---|---|---|
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` (1024-dim) | Native transformers/sentence-transformers support. First pick was `jinaai/jina-embeddings-v2-base-code`, but its `trust_remote_code` modeling file imports `find_pruneable_heads_and_indices`, removed in transformers 5.x — failed to load on day one. Swapped to Qwen3: no remote code, no rot; stronger model, ~3-4x slower on CPU. Lesson: native support beats remote code for a baseline |
| Torch flavor | CPU | Indexing speed is a baseline *metric*, not today's problem; CUDA torch = 2.5 GB download + index config. Switch only if the measured indexing time demands it |
| Chunk granularity | functions, classes, module docstrings | Queries target symbols, not files |
| Decorated defs | Emit **wrapper's** span + **inner** def's name/kind | Decorator lines (`@app.route("/login")`) carry search-relevant info that plain `function_definition` spans lose |
| Nested defs | Recurse everywhere, emit all defs (methods, nested) | Duplicate storage is cheap; recall is the goal. Revisit only if baseline numbers demand it |
| Chunk `text` field | Raw source span only | Chunker = faithful extraction; embedder composes the model-facing text (`path + kind + name + text`). Presentation policy stays out of the index format |
| Module docstrings | Top-level string statements become chunks | Docstrings are written in the natural language users search with |
| Unknown/unparsable files | ~50-line windows, `kind="fallback"` | Nothing silently disappears from the index |

## Architecture (as of first working baseline)

```
indexer/repo_reader.py    walks the repo → list[Path]        (extension + junk-dir filter)
indexer/chunker.py        files → list[Chunk]                (tree-sitter, 4-branch walk, fallback)
indexer/embedder.py       text → float32 vectors             (Qwen3-Embedding-0.6B, fp32, threads=2, batch=4)
storage/baseline.py       list[Chunk] + matrix ↔ disk        (save_index / load_index)
cli.py                    the two verbs: index, search       (glue only)
```

Index: `read_repository` → `chunk_repo` (tree-sitter walk; unparsable /
non-Python → 50-line fallback windows) → `chunk_to_text` composes
`path + kind + name + raw span` → `embed_texts` (rows unit-normalized) →
`save_index`.

Search: `load_index` (meta guard) → `embed_query` (instruction prefix —
asymmetric by design, chunks get none) → `scores = V @ q` (cosine, because
rows are pre-normalized) → top-5.

Invariants the system depends on:

1. Row alignment: chunk *i* ↔ `chunks.jsonl` line *i* ↔ vector row *i* —
   asserted on save, verified on load.
2. Unit-norm rows → cosine = one matmul.
3. `meta.json` records model_name + dim; a mismatch raises instead of
   silently returning garbage.
4. Nothing silently dropped (fallback chunks).
5. One model for index and query, enforced by (3).

Deliberately absent until measurements justify them: ANN / quantization,
C++ engine (replaces this storage+search *implementation*, not the
architecture — the narrow save/load interface is the seam), non-Python
parsing, summarization model.

## Chunker — how it works

Pipeline: `read_repository()` walks eligible files → per `.py` file:
`chunk_file` reads bytes → tree-sitter parse → `_walk` recursion → each emit
creates `Chunk(path, start_line, end_line, kind, name, text)`.

`_walk` branches:

1. `decorated_definition` — find inner def, emit wrapper span + inner
   name/kind, recurse into the **inner's children, never the wrapper's**.
   This single rule prevents double emission (the wrapper's children contain
   the def; the def's own children don't).
2. `function_definition` / `class_definition` — emit, recurse into children
   (this is how methods and nested functions are found).
3. `expression_statement` with `parent == "module"` and a `string` child —
   module docstring, `name` = file stem. The parent guard prevents string
   statements inside function bodies from becoming fake docstrings.
4. everything else — recurse deeper (also finds defs inside top-level
   `if` / `try` blocks).

Mechanical details: lines are `start_point.row + 1` (0-indexed rows);
`node.text` provides the source slice; non-`.py` or unparsable files go through
`_fallback_chunks` (50-line windows, `name=""`); `read_file` uses
`errors="replace"` so one bad file can't kill an index run.

## Verification method

1. **Smoke test first** — print the raw tree of a real file before writing any
   logic; learn the tree's shape instead of guessing.
2. **DEMO oracle** — synthetic snippet containing all hard cases (decorator,
   nested def, class + method) with hand-computed expected output. Every
   feature must keep DEMO's 4 chunks identical → instant regression check.
3. **Incremental steps, each independently runnable** — skeleton → top-level
   defs → decorators → recursion → docstring → fallback.
4. **Stats as acceptance** — current repo: 16 chunks (13 function, 2 class,
   1 docstring), 0 fallback, text length min/median/max = 57/289/3979 chars.

### What the stats told us

- Max chunk ≈ 3979 chars (~1000 tokens) — the `ChunkCode` class chunk
  containing its methods, as designed. Fits the model's 8k window; would have
  been silently truncated by MiniLM.
- Median 289 chars — small, focused chunks.
- 0 fallback chunks — no dependency/cachedir leakage through the walker.

## Baseline results (Step 3, final — 2026-10-27)

Eval set: 42 hand-verified queries (19 easy / 15 medium / 8 trap), one expected
(path, symbol) per query, graded on top-5. `evaluation/run_eval.py` reuses the
CLI's own code path (load_index + embed_query + one matmul). Eval set covers
every meaningful chunk in the current codebase (stub/throwaway files excluded
by design).

| metric    | overall | easy | medium | trap |
|-----------|---------|------|--------|------|
| recall@5  | 0.90    | 1.00 | 0.87   | 0.75 |
| hit@1     | 0.55    | 0.68 | 0.53   | 0.25 |
| MRR       | 0.70    | 0.81 | 0.67   | 0.46 |

Random-guess floor on this repo (32 chunks): 0.16 recall@5. Recall degrades
smoothly with query difficulty — the eval discriminates rather than flatters.
Persistent misses (4): two trap queries (semantic ceiling), two medium
(bug-report phrasing of "ignored dirs"; "where is embedding timed" —
distracted by sibling timing code).

Resource / speed table:

| metric | value | reading |
|---|---|---|
| indexing time | 0.03 s chunking + ~70–90 s embedding (32 chunks, fp32 CPU, batch 4) | embedding owns indexing; machine variance between runs |
| query latency | p50 525 ms, p95 536–749 ms (model load excluded) | ~99% is CPU model inference; matmul is microseconds. **A C++ search engine cannot improve query latency at this scale — its case is memory and storage.** |
| peak RSS | 3027 MB (index) / 2792 MB (query) | model floor: 0.6B params × 4 B fp32 ≈ 2.4 GB + torch runtime |
| storage | 150.7 KB (`vectors.npy` 128.1 KB = 32 × 4 KB exactly) | 4 KB/chunk → 1M chunks ≈ 4 GB. This projection justifies the compact C++ index (Step 4). |

Key reading: coverage is strong (0.89 recall@5), ranking is the weak spot
(hit@1 0.53) — overlapping class/method chunks (decision: emit both) split the
vote. Trap misses are model-semantics limited, not pipeline bugs.

**Eval-set drift lesson:** between two eval runs the repo grew 29 → 32 chunks;
queries were unchanged. recall@5 held but hit@1 fell (0.58 → 0.53; trap 0.29 →
0.14) because new chunks (e.g. `track_peak_rss`) became distractors. An eval
set is valid for one version of the codebase — when code grows, ground truth
must grow with it.

## Open items

- Fallback chunking path exercised on real input (2 windows from
  `docs/design-notes.md`) — works.
- Two independent native segfault cases — see "Case file: native segfaults".
- Empty-file hole — **fixed 2026-11-01** (found by the first foreign repo: a
  file that parses cleanly but yields no definitions — an empty `__init__.py`
  is the everyday real-world case — silently produced zero chunks, violating
  invariant 4). Two-level guard at the end of `chunk_source`: walk yielded
  nothing → `_fallback_chunks`; even the fallback empty (0-byte file) → one
  synthetic chunk, `kind="fallback"`, `name` = file stem, `text=""` — the
  stem is real signal for the embedder. DEMO oracle unchanged.
- Old `scs/` venv disappeared during the first `uv add` run (unexplained;
  `.venv/` is now the standard layout anyway).

## Foreign-repo validation (2026-10-28)

First foreign codebase: ai-hedge-fund (54 chunks from 25 files; own repo and
foreign indexed separately via `--index-dir`). Eval set: 24 queries sampled
uniformly from the index (seed 42, `evaluation/sample_chunks.py`), one
hand-written query per chunk — the query author did not author the code,
removing the self-reference bias of the own-repo set.

| metric    | foreign (n=24) | own repo (n=42, same day) |
|-----------|----------------|---------------------------|
| recall@5  | 0.92           | 0.88                      |
| hit@1     | 0.62           | 0.57                      |
| MRR       | 0.75           | 0.69                      |

Foreign by difficulty: easy 1.00 / medium 0.92 / trap 0.80 recall@5; trap
hit@1 0.40 (own repo's was 0.25). **Verdict: retrieval holds up on code the
pipeline has never seen — the baseline was not flattered by self-authorship.**

Two misses: *"how do all the pieces fit together?"* (vague query vs the HLD
doc — legitimate trap loss) and *"the daily loop asking every agent for its
signal"* (genuine concept collision: `backtester.run` vs `engine.run_backtest`
are the same concept in two files). The deliberately planted split-vote pairs
(RiskManager class/method, `aggregate_decisions` vs `decision_node`) did not
sink hit@1.

Honest caveats: the two eval sets were not written under equal adversarial
pressure (own-repo queries evolved over weeks; the foreign set was written in
one pass), ai-hedge-fund is small and unusually docstring-rich, and n=24 vs
n=42 — the cross-set delta is directional, not exact. Separately noted: own
repo misses shifted this week because prose chunks (design-notes, upstream
draft) now out-compete code for doc-ish queries.

## Two-stage retrieval experiment (2026-11-01)

hit@1 diagnosis first: all 28 rank≠1 queries across both eval sets classified
by hand. Cause table (own/foreign): concept collision ~5/~4 (same idea in two
files, e.g. `backtester.run` vs `engine.run_backtest`), vocabulary hub ~7/0
(the `Chunk` dataclass's 7-line field-name text is a keyword magnet winning
chunker queries it can't answer), prose-beats-code ~4/0 (design-notes fallback
chunks out-competing the target function), overlap-split ~2/~2 (RiskManager
class vs its own `process`, Δ=0.009), model-semantics traps ~3/~1. The assumed
culprit — parent/child overlap-split — is NOT the main cause. Conclusion:
failures are *ordering* failures with the answer already in the candidate set
(recall@5 0.88–0.92), the textbook profile for a cross-encoder re-ranker.

Implementation: bi-encoder top-20 → `cross-encoder/ms-marco-MiniLM-L-6-v2`
(~22M params, ms-marco trained) re-orders → top-5. `search/retrieve.py`,
`--rerank` on `search` and `run_eval`; pairs fed as
`(raw query, path+kind+name+text)`, truncation at 512.

| stage | recall@5 own/foreign | hit@1 own/foreign | MRR own/foreign | p50 |
|---|---|---|---|---|
| baseline | 0.88 / 0.92 | 0.57 / 0.62 | 0.69 / 0.75 | 525 ms |
| naive rerank | 0.62 / 0.62 | 0.43 / 0.46 | 0.48 / 0.50 | 3175 ms |
| + prose filter + rank fusion (α=0.7) + 4 threads | 0.81 / 0.75 | 0.48 / 0.42 | 0.60 / 0.52 | 1543 ms |

Lessons, in the order they were paid for:

1. **A general-text reranker reads prose better than code.** ms-marco judged a
   doc file that *describes* directory-skipping more relevant than the
   function that *does* it: docs fallback chunks colonized the top-5 (logits
   +0.58 for prose vs −9 for code) and recall collapsed 0.88 → 0.62. Guard:
   only function/class/docstring chunks enter the rerank pool; prose keeps
   its stage-1 order behind them. That guard alone recovered most of the loss.
2. **When no candidate truly answers, the CE orders noise.** Twin-function
   queries produced five logits within 0.2 of each other; plausible
   near-misses beat the real answer. Medium/trap suffered most (foreign trap
   recall 0.80 → 0.40, trap hit@1 0.40 → 0.00). Rank fusion with the
   bi-encoder (α=0.7) capped the damage but did not recover the loss.
3. **Latency compounds honestly.** 20 × 512-token pairs ≈ 2.6 s of CE at the
   Embedder's 2-thread cap; 4 of 12 cores → ~1 s. The Embedder's 2-thread
   segfault insurance (Case 1) does not transfer to a 22M-param model.

Decision gate was "hit@1 up AND recall@5 not down": failed twice, so the
stage ships **off by default** as a reproducible negative result. Unexplored,
recorded deliberately: code-trained reranker (e.g. jina-reranker-v2-base-code),
a fusion-weight sweep (skipped on purpose — tuning α against n=42 until it
passes is overfitting the eval), fine-tuning a reranker on code pairs.

## Drift update + empty-file fix (2026-11-01)

Empty-file hole closed (see Open items): two-level guard at the end of
`chunk_source`. Re-indexing to verify exposed the drift rule firing on our
own repo: since the last index the codebase grew 37 → 47 chunks (README.md,
docs/upstream-comment-draft.md, search/retrieve.py, the reranker-experiment
section below — fallback windows went 2 → 13). The eval set grew with it,
42 → 45 queries (Reranker class, rerank method, one bare-name trap).

| metric | 37 chunks (2026-10-28) | 47 chunks (2026-11-01) |
|---|---|---|
| recall@5 | 0.88 | 0.82 |
| hit@1 | 0.57 | 0.56 |
| MRR | 0.69 | 0.66 |

Coverage, not ranking, absorbed the hit — and the two new full misses are
the prose-beats-code cause eating its own cooking: *"where is the
instruction prefix added?"* and *"why is the model loaded in float32?"*
now lose to README/design-notes windows that mention the same words more
prominently than the code does. Prose windows are 13 of 47 chunks and are
the standing distractor class — the strongest argument yet for the round-2
candidates (hybrid lexical+dense retrieval, a code-trained reranker).
Lesson restated: published numbers must describe the exact code and index
they were measured on.

## Re-baseline (2026-11-02)

Fresh measurement on the frozen exam after the 2026-11-01 drift: 45
queries against the 47-chunk index, bi-encoder only. Verbatim eval output:

```
overall    recall@5=0.82  hit@1=0.56  mrr=0.66   (n=45)
easy       recall@5=0.95  hit@1=0.74  mrr=0.81   (n=19)
medium     recall@5=0.76  hit@1=0.53  mrr=0.62   (n=17)
trap       recall@5=0.67  hit@1=0.22  mrr=0.41   (n=9)
query latency  p50=373ms  p95=425ms   (model load excluded)
```

These are the new reference: 0.82 / 0.56 / 0.66 replaces the
0.88 / 0.57 / 0.69 (p50 525 ms) quoted as the reranker-experiment
baseline row — that row was measured on the pre-drift 37-chunk exam and
is not comparable to this one.

Latency honesty: the same machine measured p50 608 ms / p95 739 ms on the
reranker-era run and 373 / 425 here. Same code path, same eval — the gap
is machine-state noise, recorded once and not chased.

Going-forward rule: experiments compare against 0.82 / 0.56 / 0.66 at
this frozen exam, measured once — no tuning iterations against the set.
If the codebase grows again, re-baseline rather than chase the drift.
evaluation/results/2026-11-02-own.txt.

## Second reranker: bge-reranker-base (2026-11-04)

Round-2 candidate was `jinaai/jina-reranker-v2-base-code` (code-trained). It is
**gone from HuggingFace** — 401 on the repo, absent from the jinaai org listing
(models?author=jinaai&search=reranker). The multilingual sibling still exists but
its remote `embedding.py` imports `create_position_ids_from_input_ids` as a
module-level function; transformers 5.x keeps it only as a static method of
`XLMRobertaEmbeddings`, so load dies in import. Patching would have meant
maintaining a fork of remote code — the day-one lesson ("native support beats
remote code") says no. `einops` was installed for that attempt and kept.

Pulled instead: `BAAI/bge-reranker-base` (278M, XLM-RoBERTa, Apache-2.0,
native load, no `trust_remote_code`). Same guards, same rank fusion (α=0.7),
one measurement against the frozen exam:

| stage | recall@5 | hit@1 | MRR | p50 | RSS |
|---|---|---|---|---|---|
| frozen baseline (bi-encoder only) | 0.82 | 0.56 | 0.66 | 373 ms | ~2.8 GB |
| bge-reranker-base + guards | 0.82 | 0.49 | 0.61 | 5685 ms | 4136 MB |

Per difficulty: easy 0.95/0.68/0.79, medium 0.71/0.35/0.47, trap 0.78/0.33/0.50.
Decision gate ("hit@1 up AND recall@5 not down") **failed a third time.**

Diagnosis — same failure mode as ms-marco lesson #2, now confirmed across two
model families: **flat logits on weak pools.** The venv-exclusion query scored
all 20 candidates ≤ 0.012 — the CE cannot discriminate, and scrambles answers
stage-1 had placed correctly (medium hit@1 0.53 → 0.35). Logit magnitudes are
not even monotone with answer quality (one query: non-answer 0.759, answer
0.154). The one real win: **traps improved** (recall@5 0.67 → 0.78, hit@1
0.22 → 0.33) — when candidates are genuinely comparable, joint query-chunk
reading adds signal the bi-encoder lacks.

Conclusion: the reranker is not the fix for this failure profile. The measured
stage-1 problem stands (prose-beats-code, vocabulary hub). Round-2 candidates
narrow to: markdown-aware prose chunking (header sections instead of 50-line
fallback windows) and hybrid lexical+dense retrieval (BM25 + RRF, no tunable
weight). Results: `evaluation/results/2026-11-04-own-rerank-bge.txt`.

**Reverted the same day.** The bge trade — recall flat, hit@1 down, p50
373 → 5685 ms — was declined: worse on every axis. `--rerank` runs ms-marco
again and reproduces the documented 2026-11-01 experiment; bge survives only
in this section and the results file. `einops` (pulled for the dead jina
attempt) was dropped again.

## Hybrid lexical+dense retrieval: BM25 + RRF (2026-11-06)

Round-2 candidate: fuse exact-token matching with the dense ranking.
`search/lexicon.py` (since deleted): tokenizer (lowercase; split at `_` and
camelCase — "read_repository" → read, repository), Okapi BM25 (k1=1.5, b=0.75,
Lucene-style IDF), Reciprocal Rank Fusion (k=60, dense ranks every chunk, BM25
only chunks with overlap > 0, ties broken by corpus index). All three pinned
by hand-computed oracles.

Diagnostics before the eval (BM25-only top-5, two probe queries) — both
predictions half-confirmed, half-inverted:

- **Vocabulary hub: broken by IDF, as predicted.** The Chunk dataclass is
  absent from the parse-tree query's top-5. The one falsifiable test passed.
- **Prose: wins BM25 too.** The skip-dirs query's top-3 were README /
  design-notes / upstream windows; the answer (`read_repository`) fell out of
  top-5 entirely. Doc-phrased queries meet prose where their exact words live —
  IDF only demotes corpus-common words, and "skip directories" is
  prose-specific, not corpus-common.

Eval (`--hybrid`, one run): **recall@5 0.62, hit@1 0.20, MRR 0.35** — gate
failed catastrophically. Honest labeling per the standing rule: measured on
the 49-chunk index (lexicon.py itself had been indexed — the drift rule fired
mid-experiment, caught via the storage line: vectors 196.1 KB = 49 chunks vs
188.1 KB = 47 that morning), so not directly comparable to the 0.82/0.56/0.66
47-chunk reference. But the collapse is 20+ points, far beyond what +2 chunks
can cause — the verdict stands on the code, not the excuse.

Two mechanics the forecast missed, recorded because they generalize:

1. **k=60 RRF is a near-uniform blend** (rank 1 → 1/61, rank 10 → 1/70 — a 7%
   spread), so fusion ≈ rank averaging, and averaging with a much-worse
   ranker lands below the better one. "Graceful degradation" only holds for
   chunks the weak ranker *ignores* (score 0 → no bonus); chunks it actively
   misranks still collect top-list bonuses — and BM25's top was
   systematically prose.
2. **The hub fix inverted.** The Chunk dataclass has zero lexical overlap
   with "chunk" queries — `chunker` tokenizes to [chunker], stemless, never
   [chunk] — so BM25 ranked the answer near-bottom while every chunker
   path/name chunk scored high. Dense's #1 answer collected one bonus; a
   prose window at BM25 #2 + dense #8 collected two.

Lesson: **unweighted fusion is a bet that both rankers are comparably good** —
the fusion partner must not share the target failure mode (here, both lose to
prose on doc-phrased queries). Weighted fusion and a smaller k are the escape
hatches; both are tuning against the frozen exam, refused per the standing
rule.

Reverted same day: `lexicon.py` deleted, `run_eval.py` restored to HEAD.
Evidence: `evaluation/results/2026-11-06-own-hybrid.txt`. Standing consequence:
prose windows are now the confirmed bottleneck under the bi-encoder, under a
cross-encoder, and under fusion — markdown-aware chunking is the only
remaining round-2 candidate with evidence behind it. `.index/` still holds the
49-chunk state including the deleted lexicon.py — re-index before the next
experiment.

## Case file: native segfaults (Windows, 2026-10-27)

Native crashes print no traceback and `try/except` cannot catch them; only
instrumentation plus repetition finds them.

**Case 1 — torch, embedding (2026-10-25, open, mitigated).** Single large
batch (28 sequences padded to ~1000 tokens) at default thread count crashed.
Mitigated with `torch.set_num_threads(2)` + `batch_size=4` — changed
together, culprit never isolated; both kept as cheap insurance.

**Case 2 — tree-sitter, chunking (2026-10-27, closed).** First foreign repo
(ai-hedge-fund) crashed `cli.py index` during chunking on tree-sitter 0.26.0.
Signature established by bisect: ~1-in-3 processes died, always during
roughly the 13th–14th file parse; every process that survived that window
ran 570+ parses clean ("dies early or never" — a per-process heap-layout /
ASLR coin flip). Ruled out: file content (three boring victim files),
cyclic GC (`gc.disable()` still crashed), parser reuse (fresh Parser per
file still crashed), Python 3.14 (still crashed on 3.12). Fix: binding
downgrade 0.26.0 → 0.25.2 → 5/5 processes × 570 parses clean. Honesty note:
5 clean processes is suggestive, not proof (≈8% chance the old death rate
was unchanged); "0.25 fixed it" is the working verdict — a recurring
segfault reopens the case.

**Upstream corroboration.** py-tree-sitter has recorded history in exactly
this bug class: [#435](https://github.com/tree-sitter/py-tree-sitter/issues/435)
(open) — segfault in `node_get_text` during GC, reporter survives by
disabling gc; [#308](https://github.com/tree-sitter/py-tree-sitter/issues/308)
(closed) — segfault in `node_get_text` via a null from `PyList_GetItem`;
[#216](https://github.com/tree-sitter/py-tree-sitter/pull/216) (unmerged PR)
— missing bounds check on child access. Our `_walk` calls `node.text`
constantly — the same C function both crashes point at. Retrospective
correction: `gc.disable()` only stops the *cyclic* collector, not
reference-counted frees, so that experiment ruled out less than we assumed.
Working root cause: Node/Tree lifetime bug in the binding — text accessed
on memory already freed; fatal or harmless is a per-process layout coin
flip.

**Playbook this taught.** (1) Instrument with per-unit `print(...,
flush=True)` — flush is mandatory, a segfault discards buffered output.
(2) One clean run after a crash proves nothing (n=1 vs n=1); "the print
fixed it" is cargo cult — repetition is the arbiter. (3) When suspect
content and suspect position are confounded, remove the suspect and see
whether the crash follows the chair or the content. (4) Bisect one variable
at a time: input → code tweak → GC → Python version → dependency version.
(5) Timebox it: upstream-shaped bugs (Rust/pyo3 binding) get a mitigation
(subprocess per file), not unbounded hunting.

## Status / next steps

- [x] `indexer/repo_reader.py` — file walker (fixed `__pycache__`, venv leak)
- [x] `indexer/chunker.py` — Steps 1–5 complete, oracle-verified
- [x] `indexer/embedder.py` — smoke check passed: shape `(2, 1024)`,
      unrelated-pair similarity 0.226. Loads in fp32 on CPU (bf16 default is
      slow on CPU). `embed_query` prepends the Qwen3 instruction prefix
      (asymmetric retrieval); `embed_texts` stays prefix-free
- [x] `storage/baseline.py` — `vectors.npy` + `chunks.jsonl` + `meta.json`,
      row-aligned, model/dim mismatch fails loudly
- [x] `cli.py` — `index` / `search` subcommands; index writes `.index/`
- [x] Baseline metrics — final (2026-10-27): recall@5 0.90 / hit@1 0.55 /
      MRR 0.70, p50 511 ms / p95 737 ms, peak RSS 2790 MB, 150.7 KB storage
- [x] Foreign-repo validation (2026-10-28): recall@5 0.92 / hit@1 0.62 /
      MRR 0.75 on 24 sampled queries — see "Foreign-repo validation"
- [x] `--index-dir` flag on cli.py + run_eval.py so own-repo and foreign
      indexes coexist
- [x] Chunker cleanups: remove crash-bisect print in `chunk_repo`; fix
      fallback window slice (`lines[i:1 + W]` → `lines[i:i + W]` — window 2
      currently collapses to 1 line, windows 3+ vanish); `print_stats`
      guard for the empty case
- [x] Hit@1 diagnosis: all 28 rank≠1 queries across both eval sets classified
      (concept collision, vocabulary hub, prose-beats-code, overlap-split,
      model-semantics) — parent/child overlap-split turned out NOT to be the
      main cause
- [x] Two-stage retrieval — built, measured, **regressed both eval sets**
      (see "Two-stage retrieval experiment"); ships off by default behind
      `--rerank` as a documented negative result
- [x] Second reranker (bge-reranker-base, after the planned jina code reranker
      turned out to be dead on HF) — gate failed a third time; swapped back to
      ms-marco the same day (see "Second reranker: bge-reranker-base")
- [x] Hybrid BM25 + RRF — built, measured, **gate failed catastrophically**
      (recall@5 0.62, hit@1 0.20; see "Hybrid lexical+dense retrieval");
      reverted same day, lexicon.py deleted. Prose windows confirmed as the
      bottleneck under every ranking scheme tried on top of them
