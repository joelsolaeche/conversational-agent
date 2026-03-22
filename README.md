# Conversational Agent with Persistent Memory

A conversational agent that remembers information across sessions using hybrid search (vector similarity + BM25 keyword matching). Built with FastAPI, OpenAI, and SQLite.

## Architecture

```
User ──> POST /chat ──> Agent Loop ──> OpenAI API (with memory tools)
                                           │
                          ┌────────────────┼────────────────┐
                          v                                 v
                   [memory_search]                   [memory_save]
                          │                                 │
                          v                                 v
                  ┌───────────────┐                ┌───────────────┐
                  │ Hybrid Search │                │  Write to DB  │
                  │ vector (0.7)  │                │  + FTS Index  │
                  │ + BM25 (0.3)  │                └───────────────┘
                  └───────────────┘
                          │
                          v
                   SQLite + FTS5
```

## Features

- **Persistent memory** across sessions via SQLite
- **Hybrid search** combining vector embeddings (0.7) and BM25 keyword search (0.3)
- **Two memory layers**: session-scoped conversation history + long-term indexed memories
- **API key authentication** with strict per-user memory isolation
- **OpenAI function calling** — the agent autonomously decides when to search or save memories

## Project Structure

```
src/
  database.py      — SQLite schema, vector storage, FTS5, hybrid search, auth
  embeddings.py    — Text → vector via OpenAI text-embedding-3-small
  memory_tools.py  — memory_search / memory_save tool declarations + executor
  agent.py         — Agentic conversation loop with session management
  main.py          — FastAPI HTTP endpoints
```

## Setup

**Requirements:** Python 3.11+, OpenAI API key

```bash
uv init --no-workspace
uv add fastapi uvicorn openai python-dotenv
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
DB_PATH=memories.db
```

Run locally:
```bash
uv run uvicorn src.main:app --reload --port 8000
```

## API

### Register
```bash
POST /register
{"user_id": "joel"}

# Response
{"user_id": "joel", "api_key": "<64-char hex key>"}
```

> Save your key — it's shown only once.

### Chat
```bash
POST /chat
Authorization: Bearer <your-api-key>
{"message": "I prefer Python and work at Acme Corp.", "session_id": "s1"}

# Response
{"response": "...", "session_id": "s1"}
```

### Reset Session
```bash
POST /chat/reset
Authorization: Bearer <your-api-key>
{"session_id": "s1"}
```

### Health Check
```bash
GET /health
# {"status": "ok"}
```

## How Memory Works

1. At the start of each session the agent searches long-term memory for user context
2. When the user shares important information the agent saves it automatically
3. On the next session (different `session_id`, same user) the agent recalls past facts via hybrid search
4. Memory is strictly scoped per user — API keys enforce isolation

## Deployment

Deployed on Render. Push to `main` triggers an automatic redeploy.

> **Note:** The free tier uses an ephemeral filesystem — memories reset on redeploy. Add a persistent disk and set `DB_PATH=/data/memories.db` for production use.
