"""
embeddings.py — Convert text into vectors using a local model.

We use sentence-transformers with 'all-MiniLM-L6-v2':
  - 384-dimensional output (small but effective)
  - Runs fully locally — no API key, no cost, no rate limits
  - Loaded once at module level so it's not reloaded on every call
"""

from sentence_transformers import SentenceTransformer

# Load the model once when this module is first imported.
# The first run downloads ~90MB from HuggingFace and caches it locally.
# Subsequent runs load from cache instantly.
_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """
    Convert a string into a 384-dimensional embedding vector.

    normalize_embeddings=True ensures each vector has unit length (norm=1).
    This makes cosine similarity equivalent to a simple dot product,
    which is slightly faster and produces scores cleanly in [-1, 1].
    """
    embedding = _model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
