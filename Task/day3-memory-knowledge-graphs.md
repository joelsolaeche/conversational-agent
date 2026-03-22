# Day 3: Memory Systems & Knowledge Graphs (8h)

## Learning Objectives

By the end of this day, participants will be able to:

- Understand and implement different memory architectures for AI agents
- Build a Clawbot-inspired memory system with hybrid search
- Work with Neo4j for knowledge graph construction
- Combine LLMs with graph databases for enhanced reasoning
- Build and deploy a conversational agent with persistent memory

---

## Table of Contents

1. [Memory Architectures for AI](#1-memory-architectures-for-ai-15h)
2. [The Clawbot Pattern](#2-the-clawbot-pattern-15h)
3. [Building a Memory System](#3-building-a-memory-system-1h)
4. [Neo4j & Knowledge Graphs](#4-neo4j--knowledge-graphs-15h)
5. [GraphRAG: Combining Graphs + LLMs](#5-graphrag-combining-graphs--llms-1h)
6. [Lab 03: Conversational Agent with Memory](#6-lab-03-conversational-agent-with-persistent-memory-15h)

---

## 1. Memory Architectures for AI (1.5h)

### 1.1 Why Memory Matters

LLMs are stateless by default. Every API call starts fresh with zero knowledge of prior interactions. This creates three fundamental problems:

**Context window limitations** -- Even the largest context windows (200K tokens for Claude) have a hard ceiling. For long-running agents or multi-session workflows, you cannot fit everything into a single prompt.

**Session persistence** -- When a conversation ends, all context is lost. Users expect continuity. A support agent that forgets every prior interaction is unusable.

**Types of memory** that agents need:

| Memory Type | Description | Example |
|-------------|-------------|---------|
| **Episodic** | Specific past events and interactions | "The user asked about billing on March 3rd" |
| **Semantic** | General knowledge and facts | "This user prefers Python over TypeScript" |
| **Procedural** | How to perform tasks | "To deploy, run `make deploy-prod`" |

A production agent typically needs all three. The challenge is implementing them efficiently.

### 1.2 Memory Design Patterns

#### Conversation Buffer

The simplest approach: store the entire conversation history and pass it back each time.

<details>
<summary>Python: Conversation Buffer</summary>

```python
class ConversationBuffer:
    def __init__(self):
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        return self.messages.copy()

# Usage
buffer = ConversationBuffer()
buffer.add_message("user", "What is RAG?")
buffer.add_message("assistant", "RAG stands for Retrieval-Augmented Generation...")
buffer.add_message("user", "How does it differ from fine-tuning?")

# Pass full history to the LLM
response = model.generate_content(
    "\n".join(f"{m['role']}: {m['content']}" for m in buffer.get_context())
)
```

</details>

<details>
<summary>TypeScript: Conversation Buffer</summary>

```typescript
interface Message {
  role: "user" | "assistant";
  content: string;
}

class ConversationBuffer {
  private messages: Message[] = [];

  addMessage(role: Message["role"], content: string): void {
    this.messages.push({ role, content });
  }

  getContext(): Message[] {
    return [...this.messages];
  }
}

// Usage
const buffer = new ConversationBuffer();
buffer.addMessage("user", "What is RAG?");
buffer.addMessage("assistant", "RAG stands for Retrieval-Augmented Generation...");

const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
const history = buffer.getContext().map(m => `${m.role}: ${m.content}`).join("\n");
const response = await model.generateContent(history);
```

</details>

**Limitation**: Token costs grow linearly. After 50+ exchanges, you blow through the context window.

#### Sliding Window

Keep only the last N messages. Simple and bounded.

<details>
<summary>Python: Sliding Window</summary>

```python
class SlidingWindowMemory:
    def __init__(self, window_size: int = 20):
        self.messages: list[dict] = []
        self.window_size = window_size

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        return self.messages[-self.window_size:]
```

</details>

**Limitation**: Old context is permanently lost. The agent forgets everything beyond the window.

#### Summary Memory

Periodically compress older messages into a summary, keeping recent messages intact.

<details>
<summary>Python: Summary Memory</summary>

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

class SummaryMemory:
    def __init__(self, recent_limit: int = 10):
        self.summary: str = ""
        self.recent_messages: list[dict] = []
        self.recent_limit = recent_limit

    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        if len(self.recent_messages) > self.recent_limit * 2:
            self._compress()

    def _compress(self):
        """Summarize older messages and keep only recent ones."""
        old_messages = self.recent_messages[:-self.recent_limit]
        self.recent_messages = self.recent_messages[-self.recent_limit:]

        old_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )

        response = model.generate_content(
            f"Summarize this conversation concisely, preserving key facts and decisions:\n\n{old_text}\n\nPrevious summary:\n{self.summary}"
        )
        self.summary = response.text

    def get_context(self) -> list[dict]:
        system_context = f"Conversation summary so far:\n{self.summary}" if self.summary else ""
        messages = self.recent_messages.copy()
        if system_context:
            messages.insert(0, {"role": "user", "content": system_context})
            messages.insert(1, {"role": "assistant", "content": "Understood, I have that context."})
        return messages
```

</details>

**Limitation**: Lossy. Summaries inevitably lose detail.

#### Vector Memory

Store every interaction as an embedding. Retrieve semantically relevant past interactions on demand.

<details>
<summary>Python: Vector Memory (Conceptual)</summary>

```python
import numpy as np
from typing import Optional

class VectorMemory:
    def __init__(self, embedding_fn):
        self.entries: list[dict] = []  # {text, embedding, metadata}
        self.embed = embedding_fn

    def store(self, text: str, metadata: Optional[dict] = None):
        embedding = self.embed(text)
        self.entries.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        })

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self.embed(query)
        scored = []
        for entry in self.entries:
            score = cosine_similarity(query_emb, entry["embedding"])
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
```

</details>

**Limitation**: No temporal ordering. Retrieval depends entirely on embedding quality.

#### Hybrid Memory (The Clawbot Approach)

Combine multiple memory types for robustness. This is what production systems use, and what we will build in this module.

### 1.3 Storage Options

| Storage | Pros | Cons | Best For |
|---------|------|------|----------|
| **Markdown files** | Human-readable, version-controllable, zero dependencies | No built-in search | Small agents, developer tools |
| **SQLite + embeddings** | Single file, no server, fast local access | Not distributed | Single-user agents, prototypes |
| **Redis** | Sub-millisecond reads, TTL support | Volatile by default, no vector search (without RedisSearch) | Session caches |
| **PostgreSQL + pgvector** | Full SQL, ACID, vector search, production-ready | Requires a server | Production systems |
| **Dedicated vector DBs** | Purpose-built for embeddings, managed scaling | Another service to manage, cost | Large-scale RAG pipelines |

---

## 2. The Clawbot Pattern (1.5h)

Clawbot is an open-source AI agent whose memory architecture is a practical reference for building persistent, searchable agent memory. This section dissects how it works.

### 2.1 Two-Layer Architecture

Clawbot separates memory into two layers:

**Layer 1: Daily Logs** (`memory/YYYY-MM-DD.md`)
- Append-only files, one per day
- Raw observations, decisions, conversation snippets
- Automatically created and written during each session
- Never edited after creation (immutable log)

**Layer 2: Long-term Memory** (`MEMORY.md`)
- Curated, structured knowledge base
- Updated deliberately (by the agent or user)
- Contains durable facts: preferences, project context, key decisions
- Think of it as the agent's "profile" of the user/project

**Context vs. Memory -- the critical distinction:**

| | Context | Memory |
|---|---------|--------|
| Lifetime | Single API call | Persistent across sessions |
| Size limit | Token window | Unbounded |
| Cost | Tokens billed per call | No API cost to store |
| Searchability | Linear scan | Indexed (vector + FTS) |
| Mutability | Read-only within a call | Append/update |

The key insight: **move knowledge out of expensive, ephemeral context and into cheap, persistent memory**.

### 2.2 Retrieval Mechanism

Clawbot exposes two tools to the LLM:

**`memory_search`** -- Semantic querying
- Takes a natural language query
- Runs hybrid search (embeddings + full-text search)
- Returns the top-N most relevant memory chunks

**`memory_get`** -- Targeted file reading
- Takes a file path within the memory directory
- Returns the full contents of that file
- Used when the agent knows exactly what it needs

**Hybrid scoring formula:**

```
finalScore = (0.7 * vectorScore) + (0.3 * textScore)
```

Where:
- `vectorScore` = cosine similarity between query embedding and chunk embedding
- `textScore` = BM25/FTS5 relevance score, normalized to [0, 1]

The 70/30 weighting favors semantic similarity but preserves exact-match signals. This is tunable per use case.

### 2.3 Indexing Pipeline

When a memory file changes, the indexing pipeline runs:

```
File change detected (Chokidar, 1.5s debounce)
    |
    v
Read file contents
    |
    v
Chunk text (~400 tokens, 80 token overlap)
    |
    v
Generate embeddings (384 dimensions)
    |
    v
Store in SQLite:
    - chunks table (id, file_path, chunk_index, text, metadata)
    - chunks_vec (vector embeddings via sqlite-vec)
    - chunks_fts (FTS5 full-text search index)
```

**File watching**: Chokidar monitors the `memory/` directory. A 1.5-second debounce prevents re-indexing during rapid writes.

**Chunking**: ~400 tokens per chunk with 80 tokens of overlap ensures that information at chunk boundaries is not lost.

**Embedding**: 384-dimensional vectors (sentence-transformers `all-MiniLM-L6-v2`, free and runs locally). Each chunk gets its own embedding.

**SQLite schema:**

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT,  -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- sqlite-vec virtual table for vector similarity
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[384]
);

-- FTS5 for full-text search
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);
```

### 2.4 Compaction Strategy

Over time, daily logs accumulate. Compaction keeps memory efficient:

**Pre-compaction flush**: Before summarizing a session, the agent writes durable facts to `MEMORY.md`. This ensures nothing important is lost when daily logs are compressed.

**Session lifecycle boundaries**: Compaction happens at session end, not mid-conversation. This prevents the agent from losing context it might need moments later.

**Cache-TTL pruning**: Old daily logs beyond a configurable threshold (e.g., 30 days) can be archived or deleted. The important facts have already been extracted to `MEMORY.md`.

### 2.5 Implementation

<details>
<summary>Python: Complete Clawbot-Style Memory System</summary>

```python
import sqlite3
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# --- Configuration ---
MEMORY_DIR = Path("./memory")
DB_PATH = Path("./memory.db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, local model via sentence-transformers
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
CHUNK_SIZE = 400  # tokens (approx)
CHUNK_OVERLAP = 80  # tokens (approx)
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_TEXT_WEIGHT = 0.3


# --- Embedding Generation ---
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using sentence-transformers (free, runs locally)."""
    embeddings = _embedding_model.encode(texts)
    return embeddings.tolist()


def get_query_embedding(query: str) -> list[float]:
    """Generate a single embedding for a search query."""
    embedding = _embedding_model.encode([query])
    return embedding[0].tolist()


# --- Chunking ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Uses word boundaries as a proxy for token boundaries.
    For production, use a proper tokenizer (tiktoken, etc.).
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# --- Database Setup ---
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with FTS5 and vector support."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)

    # Load sqlite-vec extension (must be installed separately)
    # See: https://github.com/asg017/sqlite-vec
    try:
        conn.load_extension("vec0")
    except Exception:
        print("Warning: sqlite-vec extension not found. Vector search disabled.")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT,
            content_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # FTS5 full-text search index
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        )
    """)

    # Vector table (requires sqlite-vec)
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except Exception:
        pass

    # Triggers to keep FTS in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END
    """)

    conn.commit()
    return conn


# --- Indexing ---
def index_file(conn: sqlite3.Connection, file_path: Path, embed_fn) -> int:
    """Index a memory file: chunk, embed, and store.

    Returns the number of chunks indexed.
    """
    text = file_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    # Check if already indexed with same content
    existing = conn.execute(
        "SELECT content_hash FROM chunks WHERE file_path = ? LIMIT 1",
        (str(file_path),)
    ).fetchone()

    if existing and existing[0] == content_hash:
        return 0  # No changes

    # Remove old chunks for this file
    old_ids = [
        row[0] for row in
        conn.execute("SELECT id FROM chunks WHERE file_path = ?", (str(file_path),)).fetchall()
    ]
    if old_ids:
        conn.execute(f"DELETE FROM chunks WHERE file_path = ?", (str(file_path),))
        for old_id in old_ids:
            try:
                conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (old_id,))
            except Exception:
                pass

    # Chunk and embed
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embed_fn(chunks)

    # Store chunks
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = json.dumps({
            "file_path": str(file_path),
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

        cursor = conn.execute(
            "INSERT INTO chunks (file_path, chunk_index, text, metadata, content_hash) VALUES (?, ?, ?, ?, ?)",
            (str(file_path), i, chunk, metadata, content_hash),
        )
        chunk_id = cursor.lastrowid

        # Store embedding
        try:
            conn.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(embedding)),
            )
        except Exception:
            pass

    conn.commit()
    return len(chunks)


# --- Hybrid Search ---
def memory_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    text_weight: float = HYBRID_TEXT_WEIGHT,
) -> list[dict]:
    """Hybrid search combining vector similarity and full-text search."""

    query_embedding = get_query_embedding(query)

    # Vector search
    vector_results = {}
    try:
        rows = conn.execute(
            """
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (json.dumps(query_embedding), top_k * 3),
        ).fetchall()

        if rows:
            max_dist = max(r[1] for r in rows) or 1.0
            for rowid, distance in rows:
                # Convert distance to similarity score [0, 1]
                vector_results[rowid] = 1.0 - (distance / max_dist)
    except Exception:
        pass

    # Full-text search
    text_results = {}
    try:
        rows = conn.execute(
            """
            SELECT rowid, rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k * 3),
        ).fetchall()

        if rows:
            max_rank = max(abs(r[1]) for r in rows) or 1.0
            for rowid, rank in rows:
                text_results[rowid] = abs(rank) / max_rank
    except Exception:
        pass

    # Combine scores
    all_ids = set(vector_results.keys()) | set(text_results.keys())
    scored = []
    for chunk_id in all_ids:
        v_score = vector_results.get(chunk_id, 0.0)
        t_score = text_results.get(chunk_id, 0.0)
        final_score = (vector_weight * v_score) + (text_weight * t_score)
        scored.append((chunk_id, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in scored[:top_k]]

    # Fetch chunk details
    results = []
    for chunk_id in top_ids:
        row = conn.execute(
            "SELECT text, file_path, metadata FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            results.append({
                "text": row[0],
                "file_path": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "score": next(s for cid, s in scored if cid == chunk_id),
            })

    return results


# --- Memory File Management ---
def write_daily_log(entry: str, memory_dir: Path = MEMORY_DIR):
    """Append an entry to today's daily log."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = memory_dir / f"{today}.md"

    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp}\n\n{entry}\n")


def update_long_term_memory(fact: str, memory_dir: Path = MEMORY_DIR):
    """Add a durable fact to MEMORY.md."""
    memory_path = memory_dir / "MEMORY.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    if not memory_path.exists():
        memory_path.write_text("# Long-Term Memory\n\n", encoding="utf-8")

    with open(memory_path, "a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")


def memory_get(file_path: str, memory_dir: Path = MEMORY_DIR) -> str:
    """Read a specific memory file."""
    full_path = memory_dir / file_path
    if not full_path.exists():
        return f"File not found: {file_path}"
    return full_path.read_text(encoding="utf-8")


# --- Memory Compaction ---
def compact_daily_logs(
    conn: sqlite3.Connection,
    memory_dir: Path = MEMORY_DIR,
    days_to_keep: int = 7,
):
    """Compact old daily logs by extracting key facts to MEMORY.md."""
    log_files = sorted(memory_dir.glob("????-??-??.md"))
    cutoff = datetime.now().strftime("%Y-%m-%d")

    for log_file in log_files:
        date_str = log_file.stem
        if date_str >= cutoff:
            continue

        # Check age
        from datetime import timedelta
        log_date = datetime.strptime(date_str, "%Y-%m-%d")
        if (datetime.now() - log_date).days <= days_to_keep:
            continue

        content = log_file.read_text(encoding="utf-8")

        # Extract durable facts using Gemini
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gemini = genai.GenerativeModel("gemini-2.0-flash")

        response = gemini.generate_content(
            f"Extract the key durable facts from this daily log that should be "
            f"remembered long-term. Return each fact as a bullet point. "
            f"Ignore transient details.\n\n{content}"
        )

        facts = response.text
        if facts.strip():
            update_long_term_memory(
                f"[Extracted from {date_str}]\n{facts}",
                memory_dir=memory_dir,
            )

        # Archive or delete the old log
        archive_dir = memory_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        log_file.rename(archive_dir / log_file.name)

    # Re-index MEMORY.md after updates
    memory_md = memory_dir / "MEMORY.md"
    if memory_md.exists():
        index_file(conn, memory_md, get_embeddings)
```

</details>

<details>
<summary>TypeScript: Complete Clawbot-Style Memory System</summary>

```typescript
import Database from "better-sqlite3";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

// --- Configuration ---
const MEMORY_DIR = "./memory";
const DB_PATH = "./memory.db";
const EMBEDDING_DIM = 384;  // all-minilm produces 384-dimensional embeddings
const OLLAMA_BASE_URL = "http://localhost:11434";
const CHUNK_SIZE = 400;
const CHUNK_OVERLAP = 80;
const HYBRID_VECTOR_WEIGHT = 0.7;
const HYBRID_TEXT_WEIGHT = 0.3;

// --- Types ---
interface Chunk {
  id: number;
  filePath: string;
  chunkIndex: number;
  text: string;
  metadata: Record<string, unknown>;
  contentHash: string;
}

interface SearchResult {
  text: string;
  filePath: string;
  metadata: Record<string, unknown>;
  score: number;
}

// --- Embedding Generation (using Ollama - free, runs locally) ---
async function getEmbeddings(texts: string[]): Promise<number[][]> {
  // Using Ollama with all-minilm model (free, local)
  // Install: ollama pull all-minilm
  const embeddings: number[][] = [];
  for (const text of texts) {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "all-minilm",
        prompt: text,
      }),
    });

    const data = (await response.json()) as { embedding: number[] };
    embeddings.push(data.embedding);
  }
  return embeddings;
}

async function getQueryEmbedding(query: string): Promise<number[]> {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "all-minilm",
      prompt: query,
    }),
  });

  const data = (await response.json()) as { embedding: number[] };
  return data.embedding;
}

// --- Chunking ---
function chunkText(
  text: string,
  chunkSize = CHUNK_SIZE,
  overlap = CHUNK_OVERLAP
): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    const end = start + chunkSize;
    chunks.push(words.slice(start, end).join(" "));
    start += chunkSize - overlap;
  }

  return chunks;
}

// --- Database Setup ---
function initDb(dbPath = DB_PATH): Database.Database {
  const db = new Database(dbPath);

  // Enable WAL mode for better concurrent access
  db.pragma("journal_mode = WAL");

  db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_path TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      text TEXT NOT NULL,
      metadata TEXT,
      content_hash TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      text,
      content='chunks',
      content_rowid='id'
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, text)
        VALUES('delete', old.id, old.text);
    END;
  `);

  return db;
}

// --- Indexing ---
async function indexFile(
  db: Database.Database,
  filePath: string
): Promise<number> {
  const text = fs.readFileSync(filePath, "utf-8");
  const contentHash = crypto.createHash("sha256").update(text).digest("hex");

  // Check if already indexed with same content
  const existing = db
    .prepare("SELECT content_hash FROM chunks WHERE file_path = ? LIMIT 1")
    .get(filePath) as { content_hash: string } | undefined;

  if (existing?.content_hash === contentHash) {
    return 0; // No changes
  }

  // Remove old chunks
  db.prepare("DELETE FROM chunks WHERE file_path = ?").run(filePath);

  // Chunk and embed
  const chunks = chunkText(text);
  if (chunks.length === 0) return 0;

  const embeddings = await getEmbeddings(chunks);

  const insertChunk = db.prepare(`
    INSERT INTO chunks (file_path, chunk_index, text, metadata, content_hash)
    VALUES (?, ?, ?, ?, ?)
  `);

  const insertMany = db.transaction(() => {
    for (let i = 0; i < chunks.length; i++) {
      const metadata = JSON.stringify({
        file_path: filePath,
        chunk_index: i,
        total_chunks: chunks.length,
      });
      insertChunk.run(filePath, i, chunks[i], metadata, contentHash);
    }
  });

  insertMany();
  return chunks.length;
}

// --- Hybrid Search ---
async function memorySearch(
  db: Database.Database,
  query: string,
  topK = 5,
  vectorWeight = HYBRID_VECTOR_WEIGHT,
  textWeight = HYBRID_TEXT_WEIGHT
): Promise<SearchResult[]> {
  // Full-text search
  const textResults = new Map<number, number>();
  try {
    const ftsRows = db
      .prepare(
        `SELECT rowid, rank FROM chunks_fts
         WHERE chunks_fts MATCH ?
         ORDER BY rank LIMIT ?`
      )
      .all(query, topK * 3) as Array<{ rowid: number; rank: number }>;

    if (ftsRows.length > 0) {
      const maxRank = Math.max(...ftsRows.map((r) => Math.abs(r.rank))) || 1;
      for (const row of ftsRows) {
        textResults.set(row.rowid, Math.abs(row.rank) / maxRank);
      }
    }
  } catch {
    // FTS query may fail on certain query strings; fall back to vector only
  }

  // Vector search (in-memory cosine similarity if sqlite-vec is unavailable)
  const vectorResults = new Map<number, number>();
  try {
    const queryEmbedding = await getQueryEmbedding(query);
    // Note: For production with sqlite-vec, use the vec0 virtual table query.
    // This fallback computes cosine similarity in JS.
    const allChunks = db
      .prepare("SELECT id, text FROM chunks")
      .all() as Array<{ id: number; text: string }>;

    const chunkTexts = allChunks.map((c) => c.text);
    const chunkEmbeddings = await getEmbeddings(chunkTexts);

    const scored: Array<{ id: number; score: number }> = [];
    for (let i = 0; i < allChunks.length; i++) {
      const score = cosineSimilarity(queryEmbedding, chunkEmbeddings[i]);
      scored.push({ id: allChunks[i].id, score });
    }
    scored.sort((a, b) => b.score - a.score);

    const maxScore = scored[0]?.score || 1;
    for (const item of scored.slice(0, topK * 3)) {
      vectorResults.set(item.id, item.score / maxScore);
    }
  } catch {
    // Vector search failed; rely on text results only
  }

  // Combine scores
  const allIds = new Set([...vectorResults.keys(), ...textResults.keys()]);
  const combined: Array<{ id: number; score: number }> = [];

  for (const id of allIds) {
    const vScore = vectorResults.get(id) ?? 0;
    const tScore = textResults.get(id) ?? 0;
    combined.push({ id, score: vectorWeight * vScore + textWeight * tScore });
  }

  combined.sort((a, b) => b.score - a.score);
  const topIds = combined.slice(0, topK);

  // Fetch details
  const getChunk = db.prepare(
    "SELECT text, file_path, metadata FROM chunks WHERE id = ?"
  );

  return topIds.map(({ id, score }) => {
    const row = getChunk.get(id) as {
      text: string;
      file_path: string;
      metadata: string;
    };
    return {
      text: row.text,
      filePath: row.file_path,
      metadata: row.metadata ? JSON.parse(row.metadata) : {},
      score,
    };
  });
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// --- Memory File Management ---
function writeDailyLog(entry: string, memoryDir = MEMORY_DIR): void {
  fs.mkdirSync(memoryDir, { recursive: true });
  const today = new Date().toISOString().split("T")[0];
  const logPath = path.join(memoryDir, `${today}.md`);
  const timestamp = new Date().toTimeString().split(" ")[0];

  fs.appendFileSync(logPath, `\n## ${timestamp}\n\n${entry}\n`, "utf-8");
}

function updateLongTermMemory(
  fact: string,
  memoryDir = MEMORY_DIR
): void {
  fs.mkdirSync(memoryDir, { recursive: true });
  const memoryPath = path.join(memoryDir, "MEMORY.md");

  if (!fs.existsSync(memoryPath)) {
    fs.writeFileSync(memoryPath, "# Long-Term Memory\n\n", "utf-8");
  }
  fs.appendFileSync(memoryPath, `- ${fact}\n`, "utf-8");
}

function memoryGet(filePath: string, memoryDir = MEMORY_DIR): string {
  const fullPath = path.join(memoryDir, filePath);
  if (!fs.existsSync(fullPath)) {
    return `File not found: ${filePath}`;
  }
  return fs.readFileSync(fullPath, "utf-8");
}
```

</details>

---

## 3. Building a Memory System (1h)

### 3.1 Practical Implementation

This section is a step-by-step walkthrough. Participants build a working memory system from scratch.

**Step 1: Set up SQLite with FTS5**

<details>
<summary>Python: SQLite + FTS5 Setup</summary>

```python
import sqlite3

def setup_memory_db(db_path: str = "agent_memory.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,        -- 'conversation', 'observation', 'fact'
            content TEXT NOT NULL,
            embedding BLOB,              -- serialized float array
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            metadata TEXT                -- JSON
        )
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            content='memories',
            content_rowid='id'
        )
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END
    """)

    conn.commit()
    return conn
```

</details>

<details>
<summary>TypeScript: SQLite + FTS5 Setup</summary>

```typescript
import Database from "better-sqlite3";

function setupMemoryDb(dbPath = "agent_memory.db"): Database.Database {
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");

  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source TEXT NOT NULL,
      content TEXT NOT NULL,
      embedding BLOB,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      session_id TEXT,
      metadata TEXT
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      content,
      content='memories',
      content_rowid='id'
    );

    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
    END;
  `);

  return db;
}
```

</details>

**Step 2: Create an embedding pipeline**

<details>
<summary>Python: Embedding Pipeline</summary>

```python
import struct
from typing import Optional

from sentence_transformers import SentenceTransformer

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_and_store(
    conn: sqlite3.Connection,
    content: str,
    source: str = "conversation",
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Embed content and store in the memory database."""
    embedding = _embed_model.encode([content])[0].tolist()

    # Serialize embedding to bytes for BLOB storage
    embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

    conn.execute(
        """
        INSERT INTO memories (source, content, embedding, session_id, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            source,
            content,
            embedding_bytes,
            session_id,
            json.dumps(metadata) if metadata else None,
        ),
    )
    conn.commit()


def search_memories(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid search over memories."""
    query_embedding = _embed_model.encode([query])[0].tolist()

    # Full-text search scores
    fts_rows = conn.execute(
        "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
        (query, top_k * 3),
    ).fetchall()

    fts_scores = {}
    if fts_rows:
        max_rank = max(abs(r[1]) for r in fts_rows)
        for rowid, rank in fts_rows:
            fts_scores[rowid] = abs(rank) / max_rank if max_rank else 0

    # Vector search (brute force over all rows -- fine for < 100K memories)
    all_rows = conn.execute(
        "SELECT id, content, embedding FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()

    vector_scores = {}
    for row_id, content, emb_bytes in all_rows:
        emb = list(struct.unpack(f"{len(emb_bytes)//4}f", emb_bytes))
        score = cosine_similarity(query_embedding, emb)
        vector_scores[row_id] = score

    # Normalize vector scores
    if vector_scores:
        max_v = max(vector_scores.values())
        vector_scores = {k: v / max_v for k, v in vector_scores.items()}

    # Combine
    all_ids = set(fts_scores.keys()) | set(vector_scores.keys())
    combined = []
    for mid in all_ids:
        v = vector_scores.get(mid, 0.0)
        t = fts_scores.get(mid, 0.0)
        combined.append((mid, 0.7 * v + 0.3 * t))

    combined.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in combined[:top_k]:
        row = conn.execute(
            "SELECT content, source, metadata, timestamp FROM memories WHERE id = ?",
            (mid,),
        ).fetchone()
        if row:
            results.append({
                "content": row[0],
                "source": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "timestamp": row[3],
                "score": score,
            })

    return results
```

</details>

**Step 3: Integrate with Gemini function calling**

### 3.2 Integration with Gemini

The memory system is exposed to Gemini as tools. The LLM decides when to read and write memory.

<details>
<summary>Python: Gemini Integration with Memory Tools</summary>

```python
import google.generativeai as genai
import json
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define memory tools for Gemini function calling
memory_search_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="memory_search",
            description=(
                "Search the memory system for relevant past context. "
                "Use this when you need to recall previous conversations, "
                "user preferences, or past decisions."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Natural language search query"),
                    "top_k": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of results to return (default: 5)"),
                },
                required=["query"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="memory_save",
            description=(
                "Save an important fact or observation to long-term memory. "
                "Use this for user preferences, key decisions, or information "
                "that should persist across sessions."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "content": genai.protos.Schema(type=genai.protos.Type.STRING, description="The fact or observation to remember"),
                    "source": genai.protos.Schema(type=genai.protos.Type.STRING, description="Category: fact, preference, decision, or observation"),
                },
                required=["content", "source"],
            ),
        ),
    ]
)


def handle_tool_call(tool_name: str, tool_input: dict, conn: sqlite3.Connection) -> str:
    """Route tool calls to the memory system."""
    if tool_name == "memory_search":
        results = search_memories(conn, tool_input["query"], tool_input.get("top_k", 5))
        return json.dumps(results, default=str)
    elif tool_name == "memory_save":
        embed_and_store(conn, tool_input["content"], source=tool_input["source"])
        return json.dumps({"status": "saved"})
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def chat_with_memory(user_message: str, conn: sqlite3.Connection, chat_history: list):
    """Run a conversation turn with memory-augmented Gemini."""

    # Bootstrap: search memory for relevant context
    context_results = search_memories(conn, user_message, top_k=3)
    system_prompt = "You are a helpful assistant with access to a persistent memory system."
    if context_results:
        memory_context = "\n".join(
            f"- [{r['source']}] {r['content']}" for r in context_results
        )
        system_prompt += f"\n\nRelevant memories:\n{memory_context}"

    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        tools=[memory_search_tool],
        system_instruction=system_prompt,
    )
    chat = model.start_chat(history=chat_history)

    response = chat.send_message(user_message)

    # Handle function call loop
    while response.candidates[0].content.parts[0].function_call.name:
        fc = response.candidates[0].content.parts[0].function_call
        result = handle_tool_call(fc.name, dict(fc.args), conn)

        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": json.loads(result)},
                    )
                )]
            )
        )

    # Save the interaction to daily log
    assistant_text = response.text
    write_daily_log(f"User: {user_message}\nAssistant: {assistant_text[:500]}")

    return assistant_text
```

</details>

<details>
<summary>TypeScript: Gemini Integration with Memory Tools</summary>

```typescript
import { GoogleGenerativeAI, FunctionCallingMode } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);

const memoryTools = {
  functionDeclarations: [
    {
      name: "memory_search",
      description:
        "Search the memory system for relevant past context. " +
        "Use this when you need to recall previous conversations, " +
        "user preferences, or past decisions.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: { type: "STRING", description: "Natural language search query" },
          top_k: { type: "INTEGER", description: "Number of results (default: 5)" },
        },
        required: ["query"],
      },
    },
    {
      name: "memory_save",
      description: "Save an important fact or observation to long-term memory.",
      parameters: {
        type: "OBJECT",
        properties: {
          content: { type: "STRING", description: "The fact or observation to remember" },
          source: { type: "STRING", description: "Category: fact, preference, decision, or observation" },
        },
        required: ["content", "source"],
      },
    },
  ],
};

async function chatWithMemory(
  userMessage: string,
  db: Database.Database,
  chatHistory: Array<{ role: string; parts: Array<{ text: string }> }>
): Promise<string> {
  // Bootstrap context from memory
  const contextResults = await memorySearch(db, userMessage, 3);
  let systemInstruction =
    "You are a helpful assistant with access to a persistent memory system.";
  if (contextResults.length > 0) {
    const memoryContext = contextResults
      .map((r) => `- [${r.metadata.source ?? "memory"}] ${r.text}`)
      .join("\n");
    systemInstruction += `\n\nRelevant memories:\n${memoryContext}`;
  }

  const model = genAI.getGenerativeModel({
    model: "gemini-2.0-flash",
    tools: [memoryTools],
    systemInstruction,
  });

  const chat = model.startChat({ history: chatHistory });
  let response = await chat.sendMessage(userMessage);

  // Function call loop
  let fc = response.response.functionCalls();
  while (fc && fc.length > 0) {
    const functionResponses = [];

    for (const call of fc) {
      const args = call.args as Record<string, unknown>;
      let result: string;

      if (call.name === "memory_search") {
        const results = await memorySearch(
          db,
          args.query as string,
          (args.top_k as number) ?? 5
        );
        result = JSON.stringify(results);
      } else if (call.name === "memory_save") {
        db.prepare(
          "INSERT INTO memories (source, content) VALUES (?, ?)"
        ).run(args.source, args.content);
        result = JSON.stringify({ status: "saved" });
      } else {
        result = JSON.stringify({ error: `Unknown tool: ${call.name}` });
      }

      functionResponses.push({
        functionResponse: { name: call.name, response: JSON.parse(result) },
      });
    }

    response = await chat.sendMessage(functionResponses);
    fc = response.response.functionCalls();
  }

  const assistantText = response.response.text();
  writeDailyLog(`User: ${userMessage}\nAssistant: ${assistantText.slice(0, 500)}`);

  return assistantText;
}
```

</details>

---

## 4. Neo4j & Knowledge Graphs (1.5h)

### 4.1 Why Knowledge Graphs?

Vector search excels at finding semantically similar content, but it has blind spots:

- **Relationships**: "Which modules depend on module X?" requires traversal, not similarity.
- **Multi-hop reasoning**: "Who manages the person who wrote the auth service?" requires following a chain.
- **Structured queries**: "List all services that use PostgreSQL and are owned by the platform team" is a filter, not a fuzzy match.

Knowledge graphs fill these gaps. They store entities (nodes) and explicit relationships (edges), enabling precise structural queries.

### 4.2 Neo4j Fundamentals

**Core concepts:**
- **Nodes**: Entities with labels and properties. `(:Person {name: "Alice", role: "Engineer"})`
- **Relationships**: Directed, typed connections. `-[:MANAGES]->`, `-[:DEPENDS_ON]->`
- **Properties**: Key-value pairs on nodes and relationships

**Cypher query language basics:**

```cypher
// Create nodes
CREATE (p:Person {name: "Alice", role: "Engineer"})
CREATE (s:Service {name: "auth-service", language: "Python"})

// Create relationships
MATCH (p:Person {name: "Alice"}), (s:Service {name: "auth-service"})
CREATE (p)-[:MAINTAINS]->(s)

// Query: Who maintains auth-service?
MATCH (p:Person)-[:MAINTAINS]->(s:Service {name: "auth-service"})
RETURN p.name

// Multi-hop: What services depend on services maintained by Alice?
MATCH (p:Person {name: "Alice"})-[:MAINTAINS]->(s:Service)<-[:DEPENDS_ON]-(dep:Service)
RETURN dep.name

// Pattern matching with filters
MATCH (s:Service)
WHERE s.language = "Python" AND s.created > date("2024-01-01")
RETURN s.name, s.language
```

**Neo4j Aura setup (free tier -- no credit card required):**
1. Go to https://neo4j.com/cloud/aura-free/
2. Create a free-tier instance (includes 200K nodes, 400K relationships)
3. Note the connection URI, username, and password
4. Install the driver: `pip install neo4j` or `npm install neo4j-driver`

### 4.3 Building Knowledge Graphs from Text

The pipeline: **Raw Text -> Entity Extraction (Gemini) -> Relationship Extraction (Gemini) -> Graph Construction (Neo4j)**

<details>
<summary>Python: Knowledge Graph Construction Pipeline</summary>

```python
import json
import os
from neo4j import GraphDatabase
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Neo4j Connection ---
NEO4J_URI = "neo4j+s://xxxxx.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# --- Entity Extraction with Gemini ---
def extract_entities_and_relations(text: str) -> dict:
    """Use Gemini to extract entities and relationships from text."""
    response = model.generate_content(
        f"""Extract entities and relationships from the following text.
Return a JSON object with:
- "entities": array of {{"name": string, "type": string, "properties": object}}
- "relationships": array of {{"source": string, "target": string, "type": string, "properties": object}}

Entity types: Person, Organization, Service, Technology, Concept
Relationship types: WORKS_AT, MANAGES, MAINTAINS, DEPENDS_ON, USES, RELATES_TO

Text:
{text}

Return ONLY valid JSON, no markdown formatting."""
    )

    return json.loads(response.text)


# --- Graph Construction ---
def create_entity(tx, name: str, entity_type: str, properties: dict):
    """Create a node in Neo4j."""
    props = {k: v for k, v in properties.items() if v is not None}
    props["name"] = name

    # Dynamic label from entity_type
    query = f"MERGE (n:{entity_type} {{name: $name}}) SET n += $props"
    tx.run(query, name=name, props=props)


def create_relationship(tx, source: str, target: str, rel_type: str, properties: dict):
    """Create a relationship between two nodes."""
    props = {k: v for k, v in properties.items() if v is not None}

    query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
    """
    tx.run(query, source=source, target=target, props=props)


def build_knowledge_graph(text: str):
    """Full pipeline: text -> entities/relations -> Neo4j."""
    extracted = extract_entities_and_relations(text)

    with driver.session() as session:
        # Create entities
        for entity in extracted.get("entities", []):
            session.execute_write(
                create_entity,
                entity["name"],
                entity["type"],
                entity.get("properties", {}),
            )

        # Create relationships
        for rel in extracted.get("relationships", []):
            session.execute_write(
                create_relationship,
                rel["source"],
                rel["target"],
                rel["type"],
                rel.get("properties", {}),
            )

    return extracted


# --- Querying ---
def query_graph(cypher: str, params: dict = None) -> list[dict]:
    """Execute a Cypher query and return results."""
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]


# --- Example Usage ---
text = """
The authentication service is maintained by Alice Chen from the Platform team.
It depends on the user-database service and the Redis cache.
Bob Martinez manages Alice and oversees all platform infrastructure.
The auth service is written in Python and uses FastAPI and JWT tokens.
"""

# Build the graph
extracted = build_knowledge_graph(text)
print(f"Created {len(extracted['entities'])} entities and {len(extracted['relationships'])} relationships")

# Query: What does the auth service depend on?
results = query_graph("""
    MATCH (s:Service {name: "authentication service"})-[:DEPENDS_ON]->(dep)
    RETURN dep.name AS dependency, labels(dep)[0] AS type
""")
for r in results:
    print(f"  Depends on: {r['dependency']} ({r['type']})")
```

</details>

<details>
<summary>TypeScript: Knowledge Graph Construction Pipeline</summary>

```typescript
import neo4j, { Driver, Session } from "neo4j-driver";
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// --- Neo4j Connection ---
const driver: Driver = neo4j.driver(
  "neo4j+s://xxxxx.databases.neo4j.io",
  neo4j.auth.basic("neo4j", "your-password")
);

// --- Types ---
interface Entity {
  name: string;
  type: string;
  properties: Record<string, unknown>;
}

interface Relationship {
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

interface ExtractionResult {
  entities: Entity[];
  relationships: Relationship[];
}

// --- Entity Extraction with Gemini ---
async function extractEntitiesAndRelations(
  text: string
): Promise<ExtractionResult> {
  const response = await model.generateContent(
    `Extract entities and relationships from the following text.
Return a JSON object with:
- "entities": array of {"name": string, "type": string, "properties": object}
- "relationships": array of {"source": string, "target": string, "type": string, "properties": object}

Entity types: Person, Organization, Service, Technology, Concept
Relationship types: WORKS_AT, MANAGES, MAINTAINS, DEPENDS_ON, USES, RELATES_TO

Text:
${text}

Return ONLY valid JSON, no markdown formatting.`
  );

  const content = response.response.text();
  return JSON.parse(content) as ExtractionResult;
}

// --- Graph Construction ---
async function createEntity(
  session: Session,
  entity: Entity
): Promise<void> {
  const query = `MERGE (n:${entity.type} {name: $name}) SET n += $props`;
  await session.run(query, {
    name: entity.name,
    props: { ...entity.properties, name: entity.name },
  });
}

async function createRelationship(
  session: Session,
  rel: Relationship
): Promise<void> {
  const query = `
    MATCH (a {name: $source}), (b {name: $target})
    MERGE (a)-[r:${rel.type}]->(b)
    SET r += $props
  `;
  await session.run(query, {
    source: rel.source,
    target: rel.target,
    props: rel.properties,
  });
}

async function buildKnowledgeGraph(
  text: string
): Promise<ExtractionResult> {
  const extracted = await extractEntitiesAndRelations(text);
  const session = driver.session();

  try {
    for (const entity of extracted.entities) {
      await createEntity(session, entity);
    }
    for (const rel of extracted.relationships) {
      await createRelationship(session, rel);
    }
  } finally {
    await session.close();
  }

  return extracted;
}

// --- Querying ---
async function queryGraph(
  cypher: string,
  params: Record<string, unknown> = {}
): Promise<Record<string, unknown>[]> {
  const session = driver.session();
  try {
    const result = await session.run(cypher, params);
    return result.records.map((record) => record.toObject());
  } finally {
    await session.close();
  }
}

// --- Example Usage ---
async function main() {
  const text = `
    The authentication service is maintained by Alice Chen from the Platform team.
    It depends on the user-database service and the Redis cache.
    Bob Martinez manages Alice and oversees all platform infrastructure.
    The auth service is written in Python and uses FastAPI and JWT tokens.
  `;

  const extracted = await buildKnowledgeGraph(text);
  console.log(
    `Created ${extracted.entities.length} entities and ${extracted.relationships.length} relationships`
  );

  const deps = await queryGraph(`
    MATCH (s:Service {name: "authentication service"})-[:DEPENDS_ON]->(dep)
    RETURN dep.name AS dependency, labels(dep)[0] AS type
  `);

  for (const r of deps) {
    console.log(`  Depends on: ${r.dependency} (${r.type})`);
  }

  await driver.close();
}

main();
```

</details>

---

## 5. GraphRAG: Combining Graphs + LLMs (1h)

### 5.1 The GraphRAG Pattern

Standard RAG retrieves text chunks by semantic similarity. GraphRAG adds a structural layer:

1. **Vector search** finds semantically relevant chunks
2. **Graph traversal** expands context along known relationships
3. **Combined context** gives the LLM both semantic matches and structural knowledge

This is particularly powerful for questions like:
- "What are the downstream effects of changing this service?" (dependency traversal)
- "Who should I talk to about the payments system?" (ownership + team relationships)
- "What technologies are used by services that our service depends on?" (multi-hop)

### 5.2 Implementation

<details>
<summary>Python: GraphRAG Pipeline</summary>

```python
import google.generativeai as genai
import json
import os
from neo4j import GraphDatabase

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Assumes Neo4j driver is already configured (see Section 4)


def graphrag_query(
    user_query: str,
    conn,  # SQLite connection for vector search
    neo4j_driver,  # Neo4j driver for graph queries
    top_k: int = 5,
) -> str:
    """Full GraphRAG pipeline: query -> entities -> graph -> context -> LLM."""

    # Step 1: Extract entities from the query using Gemini
    entity_response = gemini.generate_content(
        f"Extract the key entities (names, services, concepts) from this query. "
        f"Return a JSON array of strings.\n\nQuery: {user_query}"
    )
    entities = json.loads(entity_response.text)

    # Step 2: Vector search for semantic context
    vector_results = search_memories(conn, user_query, top_k=top_k)
    semantic_context = "\n".join(
        f"- {r['content']}" for r in vector_results
    )

    # Step 3: Graph traversal for structural context
    graph_context_parts = []
    with neo4j_driver.session() as session:
        for entity in entities:
            # Find the entity and its immediate neighborhood
            result = session.run("""
                MATCH (n {name: $name})-[r]-(connected)
                RETURN n.name AS source,
                       type(r) AS relationship,
                       connected.name AS target,
                       labels(connected)[0] AS target_type
                LIMIT 20
            """, name=entity)

            for record in result:
                graph_context_parts.append(
                    f"{record['source']} --[{record['relationship']}]--> "
                    f"{record['target']} ({record['target_type']})"
                )

    graph_context = "\n".join(graph_context_parts) if graph_context_parts else "No graph relationships found."

    # Step 4: Generate answer with combined context
    response = gemini.generate_content(
        f"""Answer the following question using the provided context.

Question: {user_query}

Semantic context (from document search):
{semantic_context}

Structural context (from knowledge graph):
{graph_context}

Provide a comprehensive answer that leverages both the semantic and structural information."""
    )

    return response.text


# --- Cypher Query Generation with LLMs ---
def natural_language_to_cypher(question: str) -> str:
    """Use Gemini to generate a Cypher query from a natural language question."""
    schema_description = """
    Node types and properties:
    - Person: name, role, team
    - Service: name, language, status
    - Technology: name, category
    - Organization: name, type

    Relationship types:
    - MANAGES (Person -> Person)
    - MAINTAINS (Person -> Service)
    - DEPENDS_ON (Service -> Service)
    - USES (Service -> Technology)
    - WORKS_AT (Person -> Organization)
    """

    response = gemini.generate_content(
        f"""Given this Neo4j graph schema:
{schema_description}

Generate a Cypher query to answer: {question}

Return ONLY the Cypher query, no explanation."""
    )

    return response.text.strip()
```

</details>

<details>
<summary>TypeScript: GraphRAG Pipeline</summary>

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";
import neo4j, { Driver } from "neo4j-driver";
import Database from "better-sqlite3";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function graphragQuery(
  userQuery: string,
  db: Database.Database,  // SQLite for vector search
  neo4jDriver: Driver,    // Neo4j for graph queries
  topK = 5
): Promise<string> {
  // Step 1: Extract entities from the query
  const entityResponse = await model.generateContent(
    `Extract the key entities (names, services, concepts) from this query. Return a JSON array of strings.\n\nQuery: ${userQuery}`
  );

  const entityText = entityResponse.response.text();
  const entities = JSON.parse(entityText) as string[];

  // Step 2: Vector search for semantic context
  const vectorResults = await memorySearch(db, userQuery, topK);
  const semanticContext = vectorResults
    .map((r) => `- ${r.text}`)
    .join("\n");

  // Step 3: Graph traversal for structural context
  const graphParts: string[] = [];
  const session = neo4jDriver.session();

  try {
    for (const entity of entities) {
      const result = await session.run(
        `MATCH (n {name: $name})-[r]-(connected)
         RETURN n.name AS source,
                type(r) AS relationship,
                connected.name AS target,
                labels(connected)[0] AS targetType
         LIMIT 20`,
        { name: entity }
      );

      for (const record of result.records) {
        graphParts.push(
          `${record.get("source")} --[${record.get("relationship")}]--> ` +
            `${record.get("target")} (${record.get("targetType")})`
        );
      }
    }
  } finally {
    await session.close();
  }

  const graphContext =
    graphParts.length > 0
      ? graphParts.join("\n")
      : "No graph relationships found.";

  // Step 4: Generate answer with combined context
  const response = await model.generateContent(
    `Answer the following question using the provided context.

Question: ${userQuery}

Semantic context (from document search):
${semanticContext}

Structural context (from knowledge graph):
${graphContext}

Provide a comprehensive answer that leverages both the semantic and structural information.`
  );

  return response.response.text();
}


async function naturalLanguageToCypher(question: string): Promise<string> {
  const schemaDescription = `
    Node types and properties:
    - Person: name, role, team
    - Service: name, language, status
    - Technology: name, category
    - Organization: name, type

    Relationship types:
    - MANAGES (Person -> Person)
    - MAINTAINS (Person -> Service)
    - DEPENDS_ON (Service -> Service)
    - USES (Service -> Technology)
    - WORKS_AT (Person -> Organization)
  `;

  const response = await model.generateContent(
    `Given this Neo4j graph schema:\n${schemaDescription}\n\nGenerate a Cypher query to answer: ${question}\n\nReturn ONLY the Cypher query, no explanation.`
  );

  return response.response.text().trim();
}
```

</details>

### 5.3 Use Cases

**Codebase understanding**: Build a graph from import/require statements. Query: "What breaks if I change the User model?" Traverse all modules that import User, then their dependents.

**Documentation navigation**: Nodes are doc pages, edges are cross-references and topic relationships. Query: "How do I set up authentication?" traverses from the auth page to related setup guides, config references, and troubleshooting pages.

**Organizational knowledge**: People, teams, services, and ownership. Query: "Who should review a change to the payment service?" traverses maintainers, their managers, and the oncall rotation.

---

## 6. Lab 03: Conversational Agent with Persistent Memory (1.5h)

See `labs/lab03-memory-system.md` for the full lab instructions and starter code.

**Lab objective**: Build a conversational agent that remembers context across sessions using the Clawbot pattern.

**Deliverables**:
1. SQLite-backed memory system with hybrid search
2. Gemini integration with `memory_search` and `memory_save` tools
3. Daily log writing and long-term memory extraction
4. A multi-turn conversation demonstrating cross-session recall

---

## Exercise: Memory System Design

Design a memory architecture for a **customer support bot** that handles returns, billing questions, and product troubleshooting.

Consider:
- What goes in short-term memory (session buffer)?
- What goes in long-term memory (persistent store)?
- What relationships would benefit from a knowledge graph?
- How would you handle memory for thousands of concurrent users?

---

## Extra Exercises

1. **Multi-agent memory isolation**: Extend the memory system so that multiple agents can share a database but only access their own memories. Implement namespace isolation with shared cross-namespace search for "public" memories.

2. **Codebase knowledge graph**: Write a script that parses a project's import/dependency tree (Python or TypeScript) and builds a Neo4j knowledge graph. Query it to find circular dependencies, most-depended-on modules, and isolated components.

3. **Memory pruning**: Implement a pruning strategy that considers both access frequency and recency. Memories accessed often should persist; memories never retrieved should decay. Use an exponential decay formula with access-count boosting.
