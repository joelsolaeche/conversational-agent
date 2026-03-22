"""
main.py — FastAPI HTTP API with authentication.

Endpoints:
  POST /register    — create a user and receive an API key
  POST /chat        — send a message (requires Bearer token)
  POST /chat/reset  — clear a session (requires Bearer token)
  GET  /health      — liveness check (public)
"""

import uuid
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.agent import chat, reset_session
from src.auth import require_api_key
from src.database import create_api_key

app = FastAPI(title="Conversational Agent with Memory")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    user_id: str


class RegisterResponse(BaseModel):
    user_id: str
    api_key: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # user_id now comes from the API key, not the body


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    """
    Create a new user and return their API key.

    The key is shown only once — if lost, the user must register again.
    In production you'd add email verification, password hashing, etc.
    """
    try:
        key = create_api_key(req.user_id)
    except Exception as e:
        # UNIQUE constraint on user_id will fail if already registered
        raise HTTPException(status_code=400, detail=f"Could not register user: {e}")

    return RegisterResponse(user_id=req.user_id, api_key=key)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, user_id: str = Depends(require_api_key)):
    """
    Send a message to the agent.

    user_id is injected by the require_api_key dependency — the caller
    cannot spoof it. This enforces strict memory isolation between users.
    """
    session_id = req.session_id or f"{user_id}-{uuid.uuid4().hex[:8]}"

    try:
        response = chat(
            user_id=user_id,
            user_message=req.message,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=response, session_id=session_id)


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest, user_id: str = Depends(require_api_key)):
    reset_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
def health():
    return {"status": "ok"}
