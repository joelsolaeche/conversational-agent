"""
auth.py — FastAPI dependency for API key authentication.

How FastAPI dependencies work:
  Any function annotated with Depends() is called automatically before the
  route handler. If it raises HTTPException, the request is rejected.
  If it returns a value, that value is injected into the route handler.

Usage in a route:
  @app.post("/chat")
  def chat_endpoint(req: ChatRequest, user_id: str = Depends(require_api_key)):
      ...  # user_id is guaranteed valid here
"""

from fastapi import Header, HTTPException
from src.database import validate_api_key


def require_api_key(authorization: str = Header(...)) -> str:
    """
    Extract and validate a Bearer token from the Authorization header.

    Expected header format:
      Authorization: Bearer <your-api-key>

    Returns the user_id associated with the key.
    Raises 401 if the header is missing, malformed, or the key is invalid.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header. Use: Authorization: Bearer <key>",
        )

    key = authorization.removeprefix("Bearer ").strip()
    user_id = validate_api_key(key)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or unknown API key.")

    return user_id
