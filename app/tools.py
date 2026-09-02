"""Tools exposed to the agent, plus their execution dispatch.

Two tools are registered:

* ``search_policies`` - retrieves relevant policy documents (the real work).
* ``submit_answer``   - the model's way to end the loop with a structured answer;
  its schema mirrors :class:`app.models.PolicyAnswer`.
"""

from __future__ import annotations

from typing import Any

from app.policies import DEFAULT_SEARCH_LIMIT, PolicyStore

SEARCH_POLICIES = "search_policies"
SUBMIT_ANSWER = "submit_answer"


class ToolError(RuntimeError):
    """Raised for an unknown tool or a tool wiring problem."""


SEARCH_POLICIES_TOOL: dict[str, Any] = {
    "toolSpec": {
        "name": SEARCH_POLICIES,
        "description": (
            "Search the company HR policy documents for passages relevant to a query. "
            "Returns a list of matching policies with their title, source filename, and "
            "full text. Call this before answering any policy question."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or a short phrase describing the information needed.",
                    }
                },
                "required": ["query"],
            }
        },
    }
}

SUBMIT_ANSWER_TOOL: dict[str, Any] = {
    "toolSpec": {
        "name": SUBMIT_ANSWER,
        "description": (
            "Provide the final answer to the employee. Call this exactly once, when you "
            "have enough policy evidence (or have determined the policies do not cover "
            "the question). This ends the task."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Concise answer for a non-technical employee. If the policies do "
                            "not contain enough information, clearly say the assistant cannot "
                            "determine the answer."
                        ),
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Titles of the policies actually used. Empty if none were used.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence between 0 and 1.",
                    },
                },
                "required": ["answer", "sources", "confidence"],
            }
        },
    }
}

TOOL_SPECS: list[dict[str, Any]] = [SEARCH_POLICIES_TOOL, SUBMIT_ANSWER_TOOL]


def execute_search_policies(
    tool_input: dict[str, Any],
    store: PolicyStore,
) -> dict[str, Any]:
    """Run ``search_policies`` and return a JSON-serialisable tool result payload."""

    query = tool_input.get("query", "")
    if not isinstance(query, str):
        return {"error": "query must be a string"}

    limit_raw = tool_input.get("limit", DEFAULT_SEARCH_LIMIT)
    limit = limit_raw if isinstance(limit_raw, int) and limit_raw > 0 else DEFAULT_SEARCH_LIMIT

    hits = store.search(query, limit)
    return {
        "query": query,
        "count": len(hits),
        "results": [hit.to_tool_result() for hit in hits],
    }


def execute_tool(
    name: str,
    tool_input: dict[str, Any],
    store: PolicyStore,
) -> tuple[dict[str, Any], bool]:
    """Dispatch a tool call.

    Returns ``(payload, is_error)``. Unexpected exceptions inside a known tool are
    caught and reported as an error payload rather than crashing the agent loop.
    An unknown tool name raises :class:`ToolError` (a wiring bug, not a runtime one).
    """

    if name != SEARCH_POLICIES:
        raise ToolError(f"unknown tool: {name!r}")

    try:
        return execute_search_policies(tool_input, store), False
    except Exception as exc:  # noqa: BLE001 - deliberately broad; surfaced to the model
        return {"error": f"{type(exc).__name__}: {exc}"}, True
