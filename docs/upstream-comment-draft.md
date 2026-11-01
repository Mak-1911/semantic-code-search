# Upstream contribution draft — py-tree-sitter segfault (0.26.0)

Written 2026-10-27, after closing the case (see design-notes, "Case file:
native segfaults"). When you're ready to contribute:

1. Check the **latest** py-tree-sitter version on PyPI — if a 0.26.x point
   release landed after 0.26.0, reproduce on it first. Fixed upstream? Then
   this becomes a "confirmed fixed in X" comment instead of a bug report.
2. Re-read [#435](https://github.com/tree-sitter/py-tree-sitter/issues/435)
   for new comments — if still open and unfixed, post the draft below as a
   comment there (it's the closest open issue and help-wanted). If it was
   closed without a fix, open a new issue linking it.
3. Don't post the leaked-API-key detail or anything about ai-hedge-fund
   beyond "a small Python repo" — irrelevant to the bug.

---

## Draft comment (paste as-is)

**Environment:** Windows 11, CPython 3.12.10 (also observed on 3.14.0),
py-tree-sitter 0.26.0, grammar tree-sitter-python 0.25.0. Single-threaded
script, no queries, no tree editing — just `parser.parse()` + recursive walk
reading `node.type`, `node.children`, `node.parent`, `node.text`,
`child_by_field_name`.

**Symptom:** `Segmentation fault` with no traceback while walking parsed
trees. Across ~20 process runs: ~1-in-3 processes died, always during
roughly the 13th–14th parse+walk of the process. Every process that
survived that early window then ran 500+ more parses without crashing.
Crash position was content-independent — three different ordinary `.py`
files were the "victim" across crashes, all syntactically boring.

**What we ruled out:**
- **Cyclic GC:** crash still occurs with `gc.disable()`. (Caveat: refcount
  frees are still active with gc disabled, so this only rules out
  cycle-collection *timing*, not refcount-driven frees — possibly relevant
  to #435, where disabling gc "fixed" it.)
- **Parser reuse:** still crashes with a fresh `Parser(language)` per parse.
- **File content:** skipping/permuted the suspect files — the crash followed
  the position, not the content.

**Fix we found:** downgrading py-tree-sitter 0.26.0 → 0.25.2 eliminated the
crash: 5/5 fresh processes × 570 parses each ran clean, versus ~35%
process-death rate on 0.26.0.

Happy to provide the reproducer script or test patches.
