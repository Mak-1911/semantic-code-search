
from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython
from dataclasses import dataclass


@dataclass
class Chunk:
    path: str   # relative, posix-style
    start_line: int
    end_line: int
    kind: str       # "function" | "class" | "docstring" | "fallback"
    name: str
    text: str

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

FALLBACK_WINDOW = 50

DEFINITION_TYPES = ("function_definition", "class_definition")

class ChunkCode:
    def __init__(self, repo_root):
        self.repo_root = repo_root

    def read_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            content = file.read()
            return content

    def chunk_file(self, file_path: str) -> list[Chunk]:
        source = self.read_file(file_path).encode("utf-8")
        rel_path = Path(file_path).relative_to(self.repo_root).as_posix()
        return self.chunk_source(source, rel_path)

    def chunk_source(self, source: bytes, rel_path: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        tree = Parser(PY_LANGUAGE).parse(source)
        self._walk(tree.root_node, rel_path, chunks)
        # invariant 4: nothing silently dropped — walk yielded nothing ->
        # fallback windows; a truly empty file still gets one named chunk
        # (the stem is real signal for the embedder)
        if not chunks:
            chunks.extend(self._fallback_chunks(
                source.decode("utf-8", errors="replace"), rel_path))
        if not chunks:
            chunks.append(Chunk(
                path=rel_path, start_line=1, end_line=1,
                kind="fallback", name=Path(rel_path).stem, text="",
            ))
        return chunks

    def chunk_repo(self, files: list[Path]) -> list[Chunk]:
            chunks: list[Chunk] = []
            for path in files:
                if path.suffix == ".py":
                    try:
                        chunks.extend(self.chunk_file(str(path)))
                        continue
                    except Exception:
                        pass
                rel_path = Path(path).relative_to(self.repo_root).as_posix()
                chunks.extend(self._fallback_chunks(self.read_file(str(path)), rel_path))
            return chunks

    def _fallback_chunks(self, content: str, rel_path: str) -> list[Chunk]:
        lines = content.splitlines()
        out = []
        for i in range(0, len(lines), FALLBACK_WINDOW):
            window = lines[i:i + FALLBACK_WINDOW]
            text = "\n".join(window)
            if text.strip():
                out.append(Chunk(
                    path=rel_path,
                    start_line=i + 1,
                    end_line=i + len(window),
                    kind = "fallback",
                    name = "",
                    text = text,
                ))
        return out

    def _node_name(self, node: Node) -> str:
        name_node = node.child_by_field_name("name")
        return (
            name_node.text.decode("utf-8")
            if name_node is not None and name_node.text is not None
            else ""
        )

    def _emit(self, node: Node, kind: str, name: str, rel_path: str, chunks: list[Chunk]) -> None:
        chunks.append(Chunk(
            path=rel_path,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            kind=kind,
            name=name,
            text=node.text.decode("utf-8") if node.text is not None else "",
        ))

    def _walk(self, node: Node, rel_path: str, chunks: list[Chunk]) -> None:
        if node.type == "decorated_definition":
            # Decision A: wrappers span, inner definition's name/kind.
            inner = next(c for c in node.children if c.type in DEFINITION_TYPES)
            kind = "function" if inner.type == "function_definition" else "class"
            self._emit(node, kind, self._node_name(inner), rel_path, chunks)

            # Recurse into the inner definition's childer — never the wrapper's, or the
            # inner def gets found (and emitted) a second time
            for child in inner.children:
                self._walk(child, rel_path, chunks)
        elif node.type in DEFINITION_TYPES:
            kind = "function" if node.type == "function_definition" else "class"
            self._emit(node, kind, self._node_name(node), rel_path, chunks)
            for child in node.children:
                self._walk(child, rel_path, chunks)
        elif node.type == "expression_statement" and node.parent.type == "module":
            # top-of-file docstring
            if any(c.type == "string" for c in node.children):
                self._emit(node, "docstring", Path(rel_path).stem, rel_path, chunks)
            for child in node.children:
                self._walk(child, rel_path, chunks)
        else:
            for child in node.children:
                self._walk(child, rel_path, chunks)


def print_stats(chunks: list[Chunk]) -> None:
    import statistics
    from collections import Counter
    if not chunks:
        print("no chunks")
        return
    print("total chunks:", len(chunks))
    print("by kind:", dict(Counter(c.kind for c in chunks)))
    lens = [len(c.text) for c in chunks]
    print(f"test chars min/median/max:{min(lens)}/{statistics.median(lens)}/{max(lens)}")


DEMO = b'''
import functools

@functools.lru_cache
def cached_load(path):
    def helper(x):
        return x
    return helper

class RepoReader:
    ignored = {".git"}

    def read(self, root):
        return []
'''

if __name__ == "__main__":
    from repo_reader import read_repository
    root = Path(__file__).resolve().parent.parent
    cc = ChunkCode(str(root))

    print_stats(cc.chunk_repo(read_repository(root)))

    print("--- repo_reader.py ---")
    for c in cc.chunk_file(str(root / "indexer" / "repo_reader.py")):
        print(c.path, c.start_line, c.end_line, c.kind, c.name)

    print("--- DEMO oracle ---")
    for c in cc.chunk_source(DEMO, "demo.py"):
        print(c.path, c.start_line, c.end_line, c.kind, c.name)