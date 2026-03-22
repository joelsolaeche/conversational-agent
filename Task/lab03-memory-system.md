# Lab 03: Conversational Agent with Persistent Memory

## Objective

Build and deploy a conversational agent with a persistent memory system inspired by the Clawbot architecture. The agent uses hybrid search (vector similarity + BM25 keyword matching) to store and retrieve memories across sessions.


---

## Learning Goals

- Implement persistent memory with file-based storage backed by SQLite
- Build hybrid search combining vector embeddings and full-text search (BM25)
- Use Gemini function calling for memory operations (`memory_search`, `memory_save`)
- Understand memory layering: session logs vs. long-term indexed memory
- Deploy the agent with a persistent data store

---

## What You Will Build

A conversational agent that:

1. Remembers information across sessions (user preferences, facts, decisions)
2. Maintains two memory layers: session-scoped conversation logs and long-term indexed memories
3. Uses hybrid search to retrieve the most relevant memories given a query
4. Exposes `memory_search` and `memory_save` as tools that Claude calls autonomously
5. Deploys with persistent SQLite storage

---

## Architecture

```
User ──> POST /chat ──> Agent Loop ──> Gemini API (with memory tools)
                                           │
                          ┌────────────────┼────────────────┐
                          v                                 v
                   [memory_search]                   [memory_save]
                          │                                 │
                          v                                 v
                  ┌───────────────┐                ┌───────────────┐
                  │ Hybrid Search │                │  Write to DB  │
                  │               │                │  + Index      │
                  │ vector (0.7)  │                └───────────────┘
                  │ + BM25 (0.3)  │
                  └───────────────┘
                          │
                          v
                   SQLite + FTS5
                   (vectors + full-text)
```

---

## Prerequisites

- Google AI API key (`GOOGLE_API_KEY`) -- free tier available at [aistudio.google.com](https://aistudio.google.com)
- Python 3.11+ or Node.js 20+
- `sqlite-vec` extension (Python) or a lightweight vector library (TypeScript)
- `sentence-transformers` (Python) or Ollama running locally (TypeScript) for embeddings
- Claude Code for development

---

## Step 1: Set Up the Database Layer

**Task:** Create a SQLite-backed persistence layer for memories with both vector storage and full-text search capabilities.

### What to build

You need a database module that provides:

- A `memories` table storing: id (autoincrement primary key), user_id, content, category, embedding (as binary/blob in Python or JSON string in TypeScript), created_at timestamp, access_count, and last_accessed timestamp.
- An FTS5 virtual table (`memories_fts`) indexing the `content` and `category` columns, linked to the main table via `content_rowid`. Use the `porter` tokenizer.
- A trigger that automatically inserts into the FTS table whenever a new row is added to `memories`.

### Functions to implement

1. **`init_db`** -- Creates the tables, virtual table, and trigger if they do not already exist.
2. **`save_memory(user_id, content, category, embedding)`** -- Inserts a memory row and returns a dict/object with id, content, category, and created_at.
3. **`search_by_vector(user_id, query_embedding, limit)`** -- Retrieves all memories for the user, computes cosine similarity between the stored embeddings and the query embedding, sorts by similarity descending, and returns the top results. Each result should include a `vector_score` field.
4. **`search_by_bm25(user_id, query, limit)`** -- Queries the FTS5 table using `MATCH` and joins back to the main table. Use FTS5's built-in `rank` column for BM25 scoring (note: FTS5 rank values are negative, so negate them). Each result should include a `bm25_score` field.
5. **`hybrid_search(user_id, query, query_embedding, limit, vector_weight, bm25_weight)`** -- Calls both search functions, normalizes each score set to [0, 1] using min-max normalization, merges the candidate sets by ID, and computes a final score as `vector_weight * normalized_vector + bm25_weight * normalized_bm25`. Default weights: 0.7 vector, 0.3 BM25.

### Hints

- **Python:** Use `sqlite3` from the standard library. Serialize float vectors to bytes using `struct.pack` with format `f"{length}f"`. Deserialize with `struct.unpack`.
- **TypeScript:** Use the `better-sqlite3` library. Store embeddings as JSON strings (`JSON.stringify` / `JSON.parse`).
- For cosine similarity: `dot(a, b) / (norm(a) * norm(b))`. Implement it yourself -- no external library needed.
- For min-max normalization: `(score - min) / (max - min)`. Handle the edge case where max equals min by treating the range as 1.0.

### How to verify

- After calling `init_db`, open the database file and confirm both tables and the trigger exist (use `sqlite3` CLI or a DB browser).
- Save a memory and verify a row appears in both `memories` and `memories_fts`.
- Save 3-4 memories with different content. Run `search_by_bm25` with a keyword that matches only one of them. Confirm only the matching memory is returned.
- Run `hybrid_search` and confirm the results have all three score fields (`hybrid_score`, `vector_score`, `bm25_score`) and are sorted by `hybrid_score` descending.

---

## Step 2: Build the Embedding Pipeline

**Task:** Create a module that converts text into embedding vectors for use in vector similarity search.

### What to build

A function `get_embedding(text)` that takes a string and returns a list/array of floats representing the semantic embedding.

### Approach by language

- **Python:** Use the `sentence-transformers` library with the `all-MiniLM-L6-v2` model. This produces 384-dimensional embeddings. Load the model once at module level to avoid reloading on every call. Use `normalize_embeddings=True` when encoding.
- **TypeScript:** Use Ollama's local HTTP API at `http://localhost:11434/api/embeddings` with the `all-minilm` model. Send a POST request with `{ model: "all-minilm", prompt: text }` and extract the `embedding` field from the response.

### Install commands

Python:
```bash
uv add sentence-transformers
```

TypeScript (requires Ollama installed and running):
```bash
ollama pull all-minilm
```

### How to verify

- Call `get_embedding("hello world")` and confirm the result is a list/array of floats with a consistent length (384 for all-MiniLM-L6-v2).
- Call it with two semantically similar sentences (e.g., "I like dogs" and "I love puppies") and two dissimilar ones (e.g., "I like dogs" and "The stock market crashed"). Compute cosine similarity between each pair. The similar pair should score noticeably higher.

---

## Step 3: Define Memory Tools for Gemini

**Task:** Define two function declarations (`memory_search` and `memory_save`) that Gemini can call autonomously, and implement a dispatcher function that executes them.

### What to build

1. **Tool declarations** conforming to the Gemini function-calling schema:
   - `memory_search` -- Parameters: `query` (string, required), `limit` (integer, optional). Description should tell the model when to use it: when the user references past conversations, asks about preferences, or when prior context would help.
   - `memory_save` -- Parameters: `content` (string, required), `category` (string, required; one of: preference, fact, decision, context, general). Description should tell the model to use it when the user shares important information worth remembering.

2. **An executor function** `execute_memory_tool(name, input_data, user_id)` that:
   - For `memory_search`: generates an embedding for the query, calls `hybrid_search`, and formats the results as a readable string. Each result should show its score, category, content, and save date.
   - For `memory_save`: generates an embedding for the content, calls `save_memory`, and returns a confirmation string.
   - Returns an error message for unknown tool names.

### Hints

- **Python:** Use `genai.protos.FunctionDeclaration` and `genai.protos.Schema` to define tools. Wrap them in a `genai.protos.Tool` object.
- **TypeScript:** Use `FunctionDeclarationSchemaType` from `@google/generative-ai`. Structure as an array of objects with a `functionDeclarations` array inside.
- The formatted search results should be human-readable. A format like `[Score: 0.85] (preference) User prefers Python (saved: 2025-01-15T...)` works well.

### How to verify

- Verify that the tool declarations are accepted by the Gemini SDK without schema errors (you will test this fully in Step 4).
- Call `execute_memory_tool("memory_save", {"content": "Test memory", "category": "general"}, "test_user")` and confirm it returns a success message and the memory appears in the database.
- Call `execute_memory_tool("memory_search", {"query": "test"}, "test_user")` and confirm it returns formatted results containing the memory you just saved.

---

## Step 4: Build the Conversation Agent

**Task:** Implement the agent loop that connects user messages to Gemini, handles tool calls, and manages chat sessions.

### What to build

1. **Session management:** Maintain a dictionary/map of active chat sessions keyed by `session_id`. Each session holds a Gemini `ChatSession` object.
2. **A `chat(user_id, user_message, session_id)` function** that:
   - Gets or creates a chat session for the given session_id.
   - Sends the user message to Gemini.
   - Enters a loop (max 10 iterations) that checks whether the response contains function calls.
   - If no function calls are present, returns the text response.
   - If function calls are present, executes each one via `execute_memory_tool`, packages the results as function responses, and sends them back to Gemini.
   - Returns a fallback message if the loop is exhausted.
3. **A system prompt** instructing the model to:
   - Search memory at the start of conversations for user context.
   - Save important information (preferences, facts, decisions) when shared.
   - Search memory when the user references past interactions.
   - Avoid saving trivial or redundant information.
   - Be natural about memory usage -- do not narrate every operation.

### Hints

- **Python:** Use `genai.GenerativeModel` with `tools=[MEMORY_TOOLS]` and `system_instruction`. Create a chat with `model.start_chat()`. Send messages with `chat.send_message()`. Check `response.candidates[0].content.parts` for `function_call` attributes. Return function results as `genai.protos.Part(function_response=...)`.
- **TypeScript:** Use `genai.getGenerativeModel(...)` with `tools` and `systemInstruction`. Start chat with `model.startChat()`. Use `response.response.functionCalls()` to detect tool use. Return results as `{ functionResponse: { name, response: { result } } }`.
- The loop is necessary because a single user message can trigger multiple rounds of tool use (e.g., the model searches memory, then decides to save something based on what it found).

### How to verify

- Send a message like "Hi, I prefer Python and work at Acme Corp." The agent should respond conversationally AND save the relevant facts to memory (check the database).
- In the same session, ask "What language do I prefer?" The agent should answer from the current conversation context.
- Start a NEW session (different session_id, same user_id) and ask "What do you know about me?" The agent should search memory and recall the facts from the first conversation.

---

## Step 5: Build the API

**Task:** Wrap the agent in an HTTP API with the following endpoints.

### Endpoints

| Method | Path           | Description                                       |
|--------|----------------|---------------------------------------------------|
| POST   | `/chat`        | Send a message. Body: `{ user_id, message, session_id? }`. Returns `{ response, session_id }`. |
| POST   | `/chat/reset`  | Clear a session. Body/param: `session_id`. Returns `{ status }`. |
| GET    | `/health`      | Health check. Returns `{ status: "ok" }`.         |

### Guidance

- **Python:** Use FastAPI with Pydantic models for request/response validation. Call `init_db()` on startup. Run with Uvicorn.
- **TypeScript:** Use Hono with `@hono/node-server`. Call `initDb()` at module load. Validate that `user_id` and `message` are present in the request body.
- If `session_id` is not provided in the `/chat` request, generate one (e.g., from user_id + timestamp or request id).
- Wrap the chat call in a try/catch and return a 500 with the error message on failure.

### Install commands

Python:
```bash
uv add fastapi uvicorn google-generativeai sentence-transformers aiosqlite
uv run uvicorn main:app --reload --port 8000
```

TypeScript:
```bash
npm install hono @hono/node-server @google/generative-ai better-sqlite3
npm install -D @types/better-sqlite3 tsx typescript
```

### How to verify

- Start the server and confirm `/health` returns `{"status": "ok"}`.
- Send a POST to `/chat` without `user_id` or `message` and confirm you get a 400 or 500 error.
- Run the full conversation test from Step 6 below.

---

## Step 6: Test

### Conversation 1 -- Establishing memories

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "Hi! I prefer Python over TypeScript, and I work at Acme Corp as a senior engineer.", "session_id": "s1"}'

# Second message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "I am working on migrating our monolith to microservices. We use PostgreSQL and Redis.", "session_id": "s1"}'
```

### Conversation 2 -- Testing recall (new session)

```bash
# New session -- agent should search memory for context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "Can you recommend a good framework for my project?", "session_id": "s2"}'
```

The agent should search memory, find that the user prefers Python and is doing a microservices migration, and tailor its recommendation accordingly.

### How to verify

- Conversation 1 responses should acknowledge the shared information naturally.
- After Conversation 1, check the database: there should be multiple memories saved for user1 covering preferences (Python), facts (Acme Corp, senior engineer), and project context (monolith migration, PostgreSQL, Redis).
- Conversation 2 response should reference Python and/or the microservices migration without the user restating those facts. This confirms cross-session recall via hybrid search.

---

## Deliverables

- [ ] SQLite database with FTS5 full-text search index
- [ ] Embedding pipeline (sentence-transformers / Ollama -- free, local)
- [ ] Hybrid search combining vector similarity (0.7 weight) and BM25 (0.3 weight)
- [ ] Memory tools (`memory_search`, `memory_save`) integrated with Gemini API
- [ ] Conversational agent that autonomously manages memory
- [ ] API with session management
- [ ] Deployed to Railway or Render

---

## Extension Challenges

1. **Add Neo4j for entity-relationship memory** -- Store entities (people, projects, technologies) and their relationships in a graph database. Use Cypher queries to traverse connections.

2. **Implement memory compaction** -- When a user has more than N memories, use Claude to summarize and consolidate older memories into higher-level summaries. This keeps the memory store manageable.

3. **Add multi-user isolation** -- Ensure that memory search is strictly scoped to the current user. Add API key authentication so each user only accesses their own memories.

4. **Build memory analytics dashboard** -- Add an endpoint that returns statistics: total memories per user, category distribution, most-accessed memories, and search hit rates.

---

## References

- [Gemini Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling)
- [SQLite FTS5 Documentation](https://www.sqlite.org/fts5.html)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Embeddings](https://ollama.com/blog/embedding-models)
- [Hybrid Search Explained](https://www.pinecone.io/learn/hybrid-search-intro/)
