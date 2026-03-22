"""
agent.py — Conversation loop using OpenAI with memory tools.

The agentic loop:
  1. Send messages to the model.
  2. If it responds with tool_calls, execute them and add results to history.
  3. Repeat until the model responds with plain text.
  4. Return that text to the user.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from src.database import init_db
from src.memory_tools import MEMORY_TOOLS, execute_memory_tool

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful personal assistant with long-term memory.

Memory behavior:
- At the start of each conversation, search memory for context about this user.
- When the user shares preferences, facts about themselves, decisions, or project
  context, save those to memory automatically.
- When the user references past conversations or asks what you know about them,
  search memory before answering.
- Do not save trivial or redundant information.
- Do not announce every memory operation — use memory naturally and seamlessly.
- If memory search returns results, use that context to personalize your response.
"""

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# OpenAI uses a simple list of message dicts as history.
# We store one list per session_id.

_sessions: dict[str, list[dict]] = {}


def _get_session_history(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Chat function
# ---------------------------------------------------------------------------

def chat(user_id: str, user_message: str, session_id: str) -> str:
    """
    Send a message and return the agent's response.

    The agentic loop handles chains of tool calls before the final reply.
    """
    history = _get_session_history(session_id)
    history.append({"role": "user", "content": user_message})

    for _ in range(10):
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=history,
            tools=MEMORY_TOOLS,
            tool_choice="auto",  # let the model decide when to use tools
        )

        message = response.choices[0].message

        # Add the assistant's response to history (may include tool_calls)
        history.append(message.model_dump(exclude_unset=True))

        # No tool calls → done, return the text
        if not message.tool_calls:
            return message.content

        # Execute each tool call and add results to history
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_memory_tool(
                name=tc.function.name,
                input_data=args,
                user_id=user_id,
            )
            # OpenAI requires tool results as role="tool" messages
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I'm sorry, I couldn't complete that request (max iterations reached)."


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

init_db()
