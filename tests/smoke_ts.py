from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser

PY_LANG = Language(tree_sitter_python.language())

parser = Parser(PY_LANG)

# ---- Part A ----
source = Path("indexer/repo_reader.py").read_bytes()    # tree-sitter wants bytes, not str
tree = parser.parse(source)

for node in tree.root_node.children:
    name_node = node.child_by_field_name("name")        # only DIRECT children = top-level
    name_text = name_node.text if name_node is not None else None
    name = name_text.decode("utf-8") if name_text is not None else ""
    start = node.start_point.row + 1
    end = node.end_point.row + 1
    print(f"{node.type:25} {name:20} [{start}-{end}]")


# ----- Part B: What does the tree actually look like? ----
DEMO = b'''
import functools

@functools.lru_cache
def cached_load(path):
    def helper(x):
        return x
    return helper

class RepoReader:
    ignored = {'.git'}

    def read(self, root):
        return []
'''

def dump(node, depth=0):
    name_node = node.child_by_field_name("name")
    name = f" name={name_node.text.decode()}" if name_node else ""
    span = f"[{node.start_point.row + 1}-{node.end_point.row + 1}]"
    print("    " * depth + f"{node.type}{name} {span}")
    for child in node.children:
        dump(child, depth + 1)

demo_tree = parser.parse(DEMO)
dump(demo_tree.root_node)