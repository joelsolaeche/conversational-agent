"""
embeddings.py — Convert text into vectors using OpenAI's embedding API.

We use text-embedding-3-small:
  - 1536-dimensional output
  - Runs remotely via API (no local model, no PyTorch, no RAM overhead)
  - Very cheap: $0.02 per 1M tokens
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> list[float]:
    """
    Convert a string into a 1536-dimensional embedding vector via OpenAI API.
    """
    response = _client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
