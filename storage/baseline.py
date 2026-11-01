import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from indexer.chunker import Chunk


def save_index(chunks: list[Chunk], vectors: np.ndarray, out_dir: Path, model_name: str) -> None:
    if len(chunks) != vectors.shape[0]:
        raise ValueError(f"{len(chunks)} chunks vs {vectors.shape[0]} vectors — row mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors.astype(np.float32))
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        meta = {
            "model_name": model_name,
            "dim": int(vectors.shape[1]),
            "count": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(out_dir: Path, model_name: str) -> tuple[list[Chunk], np.ndarray]:
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["model_name"] != model_name:
        raise ValueError(
            f"index was built with '{meta['model_name']}' but you're querying with "
            f"'{model_name}' — delete the index and re-run `index`"
        )
    vectors = np.load(out_dir / "vectors.npy")
    with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    if len(chunks) != vectors.shape[0] or vectors.shape[1] != meta["dim"]:
        raise ValueError("index files are inconsistent — re-index")
    return chunks, vectors