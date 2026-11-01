import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

QUERY_INSTRUCTION = "Given a natural language query, retrieve code snippets that answer it."

class Embedder:
    def __init__(self):
        torch.set_num_threads(2)
        self.model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=4, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        prompted = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        return self.model.encode(prompted, normalize_embeddings=True, convert_to_numpy=True)


def chunk_to_text(chunk) -> str:
    return f"{chunk.path}\n{chunk.kind} {chunk.name}\n{chunk.text}".strip()


if __name__ == "__main__":
    e = Embedder()
    v = e.embed_texts(["validate authentication token", "numpy array reshape"])
    print("shape:", v.shape)
    print("similarity:", float(v[0] @ v[1])) # want well below 1.0