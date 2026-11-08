"""
Search & Retrieve: Read a local repository (v0)
Walk the files, ignore binaries, dependencies, and generated files.
"""
from pathlib import Path


def under_nested_repo(path: Path, root: Path) -> bool:
    p = path.parent
    while p != root:
        if (p / ".git").exists():
            return True
        p = p.parent
    return False

def read_repository(root: Path) -> list[Path]:
    """return eligible source files under repository root
    Excludes hidden folders (like .git), build artifacts, and caches, 
    while gathering files matching common programming extensions.
    """

    ignored_dirs = {
        '.git', 
        '.github', 
        '__pycache__', 
        'node_modules', 
        'venv', 
        '.venv', 
        'build',
        'dist'
    }

    # No prose extensions (.md, .txt, .rst): in the embedding space docs
    # out-compete the code they describe — 3 of 8 eval misses had a README/
    # design-notes window sitting above the real function (2026-11-08).
    # Documentation is a separate retrieval problem: own collection, own index.
    eligible_extensions = {
        '.js',
        '.ts',
        '.py',
        '.go',
        '.cpp',
        '.c',
        '.h',
        '.java',
        '.rs'
    }

    eligible_files = []

    # using rglob to recursively traverse files
    for path in root.rglob('*'):
            # Combined all criteria into a single conditional statement
            if (path.is_file() and 
                path.suffix in eligible_extensions and 
                not any(parent.name in ignored_dirs for parent in path.parents) and
                not under_nested_repo(path, root)):
                eligible_files.append(path)

    return eligible_files