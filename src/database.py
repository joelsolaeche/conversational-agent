"""
database.py — SQLite persistence layer for long-term agent memory.

Two storage mechanisms work together:
  - `memories` table: stores content + a binary blob of the embedding vector
  - `memories_fts` (FTS5 virtual table): full-text index for BM25 keyword search

A trigger keeps them in sync automatically whenever a new memory is inserted.
"""

import sqlite3
import struct
import math
import secrets
from datetime import datetime
from typing import Optional


import os

DB_PATH = os.environ.get("DB_PATH", "memories.db")

# Ensure the parent directory exists (needed when using a mounted disk like /data)
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity measures the angle between two vectors, not their length.
    A score of 1.0 means identical direction (semantically the same).
    A score of 0.0 means completely unrelated.

    Formula: dot(a, b) / (||a|| * ||b||)
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _serialize_embedding(embedding: list[float]) -> bytes:
    """
    SQLite has no native float-array type, so we pack the vector into bytes.
    struct.pack with format "Nf" packs N 4-byte floats into a binary blob.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack binary blob back into a Python list of floats."""
    count = len(blob) // 4  # each float is 4 bytes
    return list(struct.unpack(f"{count}f", blob))


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create tables, the FTS5 virtual table, and the sync trigger — only if
    they don't already exist (safe to call on every startup).

    Why FTS5?
      SQLite's FTS5 extension gives us BM25 full-text search for free.
      BM25 (Best Match 25) is the same algorithm powering Elasticsearch and
      Lucene. It ranks documents by term frequency, rewarding rare words more
      than common ones.

    Why a trigger?
      We want both tables to stay in sync without manual inserts. The trigger
      fires automatically after every INSERT on `memories`.
    """
    conn = _get_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                category     TEXT    NOT NULL,
                embedding    BLOB    NOT NULL,
                created_at   TEXT    NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT
            );

            -- FTS5 virtual table: indexes content + category for BM25 search.
            -- content_rowid links each FTS row back to the memories table.
            -- porter tokenizer applies stemming: "running" matches "run".
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                content,
                category,
                content='memories',
                content_rowid='id',
                tokenize='porter ascii'
            );

            -- Trigger: after every INSERT into memories, mirror the row into
            -- the FTS index so keyword search stays current.
            CREATE TRIGGER IF NOT EXISTS memories_ai
            AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;

            -- API keys table: maps a secret token to a user_id.
            -- The token is a random 32-byte hex string (64 chars).
            -- We store it as plain text here — in production you'd hash it.
            CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
        """)
    conn.close()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_memory(
    user_id: str,
    content: str,
    category: str,
    embedding: list[float],
) -> dict:
    """
    Persist a memory and return its stored representation.
    The trigger automatically inserts into memories_fts.
    """
    created_at = datetime.utcnow().isoformat()
    blob = _serialize_embedding(embedding)

    conn = _get_connection()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO memories (user_id, content, category, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, content, category, blob, created_at),
        )
        memory_id = cursor.lastrowid
    conn.close()

    return {
        "id": memory_id,
        "content": content,
        "category": category,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_by_vector(
    user_id: str,
    query_embedding: list[float],
    limit: int = 10,
) -> list[dict]:
    """
    Retrieve the top-N memories by cosine similarity.

    Why do we load all rows and compute similarity in Python?
      SQLite has no native vector-math support (unlike pgvector for Postgres).
      For a personal agent with thousands of memories this is fast enough.
      For millions of rows you'd move to a dedicated vector DB.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, content, category, embedding, created_at FROM memories WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()

    scored = []
    for row in rows:
        stored_emb = _deserialize_embedding(row["embedding"])
        score = _cosine_similarity(query_embedding, stored_emb)
        scored.append({
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "created_at": row["created_at"],
            "vector_score": score,
        })

    scored.sort(key=lambda x: x["vector_score"], reverse=True)
    return scored[:limit]


def search_by_bm25(
    user_id: str,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Full-text keyword search using FTS5's built-in BM25 ranking.

    Why negate `rank`?
      FTS5 returns rank as a negative number (more negative = better match).
      We negate it so higher is better, consistent with vector_score.
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT m.id, m.content, m.category, m.created_at,
               (-memories_fts.rank) AS bm25_score
        FROM memories_fts
        JOIN memories m ON memories_fts.rowid = m.id
        WHERE memories_fts MATCH ?
          AND m.user_id = ?
        ORDER BY bm25_score DESC
        LIMIT ?
        """,
        (query, user_id, limit),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def hybrid_search(
    user_id: str,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """
    Combine vector + BM25 results into a single ranked list.

    Why hybrid?
      - Vector search is great at semantic similarity ("I love dogs" ≈ "I like puppies")
        but can miss exact keywords (product codes, names, jargon).
      - BM25 is great at exact keywords but misses paraphrases.
      - Together they cover each other's weaknesses.

    Algorithm:
      1. Get candidates from both searches.
      2. Min-max normalize each score set to [0, 1] independently.
      3. For memories appearing in only one result set, their missing score = 0.
      4. Final score = vector_weight * norm_vector + bm25_weight * norm_bm25.
    """
    vector_results = search_by_vector(user_id, query_embedding, limit=limit * 3)
    bm25_results = search_by_bm25(user_id, query, limit=limit * 3)

    # Index by memory id
    combined: dict[int, dict] = {}

    for r in vector_results:
        combined[r["id"]] = {**r, "bm25_score": 0.0}

    for r in bm25_results:
        if r["id"] in combined:
            combined[r["id"]]["bm25_score"] = r["bm25_score"]
        else:
            combined[r["id"]] = {**r, "vector_score": 0.0}

    memories = list(combined.values())

    def _normalize(key: str) -> None:
        """Min-max normalization in-place."""
        scores = [m[key] for m in memories]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi != lo else 1.0
        for m in memories:
            m[f"norm_{key}"] = (m[key] - lo) / rng

    _normalize("vector_score")
    _normalize("bm25_score")

    for m in memories:
        m["hybrid_score"] = (
            vector_weight * m["norm_vector_score"]
            + bm25_weight * m["norm_bm25_score"]
        )

    memories.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return memories[:limit]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def create_api_key(user_id: str) -> str:
    """
    Generate a new API key for a user_id and persist it.

    Why secrets.token_hex?
      It uses the OS's cryptographically secure random number generator.
      A 32-byte (64 hex char) token has 256 bits of entropy —
      practically impossible to brute-force.

    Returns the raw key string (shown to the user once at registration).
    """
    key = secrets.token_hex(32)
    created_at = datetime.utcnow().isoformat()

    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO api_keys (key, user_id, created_at) VALUES (?, ?, ?)",
            (key, user_id, created_at),
        )
    conn.close()
    return key


def validate_api_key(key: str) -> Optional[str]:
    """
    Look up an API key and return the associated user_id, or None if invalid.

    This is called on every request — SQLite primary key lookup is O(log n),
    fast enough for this scale.
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT user_id FROM api_keys WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["user_id"] if row else None
