"""Tests for tool specs and dispatch."""

from __future__ import annotations

import pytest

from app.policies import PolicyStore
from app.tools import (
    SEARCH_POLICIES,
    SUBMIT_ANSWER,
    TOOL_SPECS,
    ToolError,
    execute_tool,
)


def test_tool_specs_expose_both_tools() -> None:
    names = {spec["toolSpec"]["name"] for spec in TOOL_SPECS}
    assert names == {SEARCH_POLICIES, SUBMIT_ANSWER}


def test_execute_search_policies_returns_hits(policy_store: PolicyStore) -> None:
    payload, is_error = execute_tool(
        SEARCH_POLICIES, {"query": "remote work another state"}, policy_store
    )

    assert is_error is False
    assert payload["count"] >= 1
    assert payload["results"][0]["title"] == "Remote Work Policy"
    assert set(payload["results"][0]) == {"title", "source", "content"}


def test_execute_search_policies_empty_query(policy_store: PolicyStore) -> None:
    payload, is_error = execute_tool(SEARCH_POLICIES, {"query": "   "}, policy_store)
    assert is_error is False
    assert payload["count"] == 0
    assert payload["results"] == []


def test_unknown_tool_raises(policy_store: PolicyStore) -> None:
    with pytest.raises(ToolError):
        execute_tool("delete_everything", {}, policy_store)


def test_store_failure_is_reported_not_raised() -> None:
    class BoomStore:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("index unavailable")

    payload, is_error = execute_tool(SEARCH_POLICIES, {"query": "x"}, BoomStore())  # type: ignore[arg-type]

    assert is_error is True
    assert "index unavailable" in payload["error"]
