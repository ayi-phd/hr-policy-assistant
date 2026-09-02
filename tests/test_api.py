"""Tests for the FastAPI endpoints. LLM is faked via the make_client fixture."""

from __future__ import annotations

from conftest import search_call, submit_call

from app.agent import AgentIterationLimitError


def test_health_returns_ok(make_client) -> None:
    client = make_client([])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_happy_path(make_client) -> None:
    client = make_client(
        [
            search_call("remote work another state"),
            submit_call(
                "You may work from another state for up to 10 business days per year.",
                sources=["Remote Work Policy"],
                confidence=0.92,
            ),
        ]
    )

    resp = client.post(
        "/ask",
        json={"question": "Can I work remotely from another state for three weeks?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"answer", "sources", "confidence"}
    assert body["sources"] == ["Remote Work Policy"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_ask_rejects_empty_question(make_client) -> None:
    client = make_client([])
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_rejects_whitespace_question(make_client) -> None:
    client = make_client([])
    resp = client.post("/ask", json={"question": "    "})
    assert resp.status_code == 422


def test_ask_rejects_missing_body(make_client) -> None:
    client = make_client([])
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_maps_agent_failure_to_502_without_leaking_internals(make_client, monkeypatch) -> None:
    from app.agent import PolicyAgent

    def boom(_self, _question):
        raise AgentIterationLimitError("agent did not finish within 5 iterations")

    monkeypatch.setattr(PolicyAgent, "run", boom)
    client = make_client([])

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 502
    detail = resp.json()["detail"].lower()
    assert "iteration" not in detail
    assert "agent" not in detail


def test_build_agent_with_stub_backend_answers_without_aws(monkeypatch) -> None:
    from app import api
    from app.config import get_settings

    monkeypatch.setenv("LLM_BACKEND", "stub")
    get_settings.cache_clear()
    try:
        agent = api.build_agent()
        answer = agent.run("Can I work remotely from another state for three weeks?")
    finally:
        get_settings.cache_clear()

    assert answer.answer.startswith("[STUB]")
    assert answer.sources == ["Remote Work Policy"]


def test_ask_unexpected_error_maps_to_500(make_client, monkeypatch) -> None:
    from app.agent import PolicyAgent

    def kaboom(_self, _question):
        raise ValueError("something unexpected")

    monkeypatch.setattr(PolicyAgent, "run", kaboom)
    client = make_client([])

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 500
    assert resp.json()["detail"]
    assert "unexpected" not in resp.json()["detail"].lower()
