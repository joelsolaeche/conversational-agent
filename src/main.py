"""
main.py — FastAPI HTTP API wrapping the conversational agent.

Three endpoints:
  POST /chat        — send a message, get a response
  POST /chat/reset  — clear a session's conversation history
  GET  /health      — liveness check
"""

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import chat, reset_session

app = FastAPI(title="Conversational Agent with Memory")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
# Pydantic models validate incoming JSON automatically.
# If a required field is missing, FastAPI returns a 422 before your code runs.

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None  # optional — generated if not provided


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Send a message to the agent and receive a response.

    If no session_id is provided, one is generated from user_id + random uuid.
    This means each call without a session_id starts a fresh conversation,
    while calls with the same session_id continue the same thread.
    """
    session_id = req.session_id or f"{req.user_id}-{uuid.uuid4().hex[:8]}"

    try:
        response = chat(
            user_id=req.user_id,
            user_message=req.message,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=response, session_id=session_id)


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest):
    """
    Clear a session's in-memory conversation history.
    Long-term memories in SQLite are NOT affected.
    """
    reset_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
def health():
    return {"status": "ok"}
