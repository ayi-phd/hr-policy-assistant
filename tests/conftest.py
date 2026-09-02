"""Shared test fixtures. The LLM/Bedrock boundary is the only thing mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent import PolicyAgent
from app.api import app, get_agent
from app.llm import LLMResponse, ToolCall
from app.policies import PolicyStore

# --------------------------------------------------------------------------- #
# Fake LLM
# --------------------------------------------------------------------------- #


class FakeLLM:
    """Scripted :class:`app.llm.LLMClient`.

    Hand it a list of :class:`LLMResponse` objects; each ``converse`` call pops
    the next one. Every call's arguments are recorded on ``.calls``.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def converse(self, system, messages, tools) -> LLMResponse:  # noqa: ANN001
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)


def search_call(query: str, *, call_id: str = "search-1") -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=[ToolCall(id=call_id, name="search_policies", input={"query": query})],
    )


def submit_call(
    answer: str,
    sources: list[str],
    confidence: float,
    *,
    call_id: str = "submit-1",
    extra: dict[str, Any] | None = None,
) -> LLMResponse:
    payload: dict[str, Any] = {"answer": answer, "sources": sources, "confidence": confidence}
    if extra is not None:
        payload = {**payload, **extra}
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=[ToolCall(id=call_id, name="submit_answer", input=payload)],
    )


def raw_submit_call(payload: dict[str, Any], *, call_id: str = "submit-1") -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=[ToolCall(id=call_id, name="submit_answer", input=payload)],
    )


def text_reply(text: str) -> LLMResponse:
    return LLMResponse(stop_reason="end_turn", text=text)


# --------------------------------------------------------------------------- #
# Policy fixtures
# --------------------------------------------------------------------------- #

_SAMPLE_POLICIES = {
    "Remote Work Policy.md": (
        "# Remote Work Policy\n\n"
        "Employees may temporarily work from another U.S. state for up to 10 business "
        "days per calendar year with manager approval. Longer arrangements require "
        "People Operations approval.\n"
    ),
    "Paid Time Off Policy.md": (
        "# Paid Time Off Policy\n\n"
        "Full-time employees accrue between 15 and 25 vacation days per year based on "
        "tenure. Up to 5 unused days carry over into the next calendar year.\n"
    ),
    "Business Travel Policy.md": (
        "# Business Travel Policy\n\n"
        "Domestic travel requires manager approval before booking. Economy airfare is "
        "expected for flights under six hours. Alcohol is not reimbursable.\n"
    ),
}


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    for name, body in _SAMPLE_POLICIES.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    # A non-markdown file that must be ignored by the loader.
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    return tmp_path


@pytest.fixture
def policy_store(policy_dir: Path) -> PolicyStore:
    return PolicyStore.from_directory(policy_dir)


# --------------------------------------------------------------------------- #
# API fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_client(policy_store: PolicyStore):
    """Factory: given a list of scripted LLMResponses, yield a TestClient whose
    /ask route is backed by a PolicyAgent over the FakeLLM and the temp store."""

    created: list[TestClient] = []

    def _make(responses: list[LLMResponse], *, max_iterations: int = 5) -> TestClient:
        fake = FakeLLM(responses)
        agent = PolicyAgent(fake, policy_store, max_iterations=max_iterations)
        app.dependency_overrides[get_agent] = lambda: agent
        client = TestClient(app)
        client.fake_llm = fake  # type: ignore[attr-defined]
        created.append(client)
        return client

    yield _make

    app.dependency_overrides.clear()
    for client in created:
        client.close()
