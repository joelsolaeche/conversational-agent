"""
memory_tools.py — OpenAI function-calling tool definitions + executor.

OpenAI's function-calling flow is the same concept as Gemini's:
  1. We declare tools as JSON schemas.
  2. The model decides when to call them and returns a tool_call object.
  3. We execute the real function and send the result back.
  4. The model uses the result to compose its final text response.
"""

from src.database import hybrid_search, save_memory
from src.embeddings import get_embedding


# ---------------------------------------------------------------------------
# Tool declarations (OpenAI JSON schema format)
# ---------------------------------------------------------------------------

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search the user's long-term memory for relevant context. "
                "Use this when the user references past conversations, asks about "
                "their preferences or previous decisions, or when knowing prior "
                "context would help give a better answer. Always search at the "
                "start of a new session to load user context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant memories.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": (
                "Save important information to the user's long-term memory. "
                "Use this when the user shares preferences, facts about themselves, "
                "decisions they've made, or project context worth remembering. "
                "Do not save trivial, redundant, or transient information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember, written as a clear, self-contained fact.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["preference", "fact", "decision", "context", "general"],
                        "description": "Category for this memory.",
                    },
                },
                "required": ["content", "category"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_memory_tool(name: str, input_data: dict, user_id: str) -> str:
    """
    Dispatch an OpenAI tool call to the real implementation.
    Returns a plain string — the model reads this as the tool's output.
    """
    if name == "memory_search":
        query = input_data["query"]
        limit = input_data.get("limit", 5)

        query_embedding = get_embedding(query)
        results = hybrid_search(
            user_id=user_id,
            query=query,
            query_embedding=query_embedding,
            limit=limit,
        )

        if not results:
            return "No relevant memories found."

        lines = ["Relevant memories found:"]
        for r in results:
            lines.append(
                f"[Score: {r['hybrid_score']:.2f}] ({r['category']}) "
                f"{r['content']} (saved: {r['created_at']})"
            )
        return "\n".join(lines)

    elif name == "memory_save":
        content = input_data["content"]
        category = input_data["category"]

        embedding = get_embedding(content)
        saved = save_memory(
            user_id=user_id,
            content=content,
            category=category,
            embedding=embedding,
        )
        return f"Memory saved (id={saved['id']}): [{category}] {content}"

    else:
        return f"Unknown tool: {name}"
