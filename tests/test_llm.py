"""Tests for the LLM layer: response parsing and the offline StubLLM."""

from __future__ import annotations

from app.agent import PolicyAgent
from app.llm import LLMError, StubLLM, parse_converse_response
from app.policies import PolicyStore

# --------------------------------------------------------------------------- #
# parse_converse_response
# --------------------------------------------------------------------------- #


def test_parse_splits_text_and_tool_use() -> None:
    raw = {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "let me look"},
                    {"toolUse": {"toolUseId": "abc", "name": "search_policies", "input": {"query": "pto"}}},
                ],
            }
        },
    }

    parsed = parse_converse_response(raw)

    assert parsed.stop_reason == "tool_use"
    assert parsed.text == "let me look"
    assert parsed.has_tool_calls
    assert parsed.tool_calls[0].id == "abc"
    assert parsed.tool_calls[0].input == {"query": "pto"}


def test_parse_empty_response_raises() -> None:
    try:
        parse_converse_response({})
    except LLMError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected LLMError")


# --------------------------------------------------------------------------- #
# StubLLM
# --------------------------------------------------------------------------- #


def test_stub_first_turn_requests_search_with_question() -> None:
    messages = [{"role": "user", "content": [{"text": "How much PTO do I get?"}]}]

    resp = StubLLM().converse([], messages, [])

    assert resp.tool_calls[0].name == "search_policies"
    assert resp.tool_calls[0].input == {"query": "How much PTO do I get?"}


def test_stub_second_turn_submits_answer_from_top_hit() -> None:
    messages = [
        {"role": "user", "content": [{"text": "pto question"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "s", "name": "search_policies", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "s",
                        "status": "success",
                        "content": [
                            {
                                "json": {
                                    "query": "pto question",
                                    "count": 1,
                                    "results": [
                                        {
                                            "title": "Paid Time Off Policy",
                                            "source": "Paid Time Off Policy.md",
                                            "content": "Full-time employees accrue 15 to 25 days.",
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                }
            ],
        },
    ]

    resp = StubLLM().converse([], messages, [])
    call = resp.tool_calls[0]

    assert call.name == "submit_answer"
    assert call.input["answer"].startswith("[STUB] Based on 'Paid Time Off Policy'")
    assert call.input["sources"] == ["Paid Time Off Policy"]
    assert 0.0 <= call.input["confidence"] <= 1.0


def test_stub_second_turn_no_results_says_cannot_determine() -> None:
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"content": [{"json": {"query": "q", "count": 0, "results": []}}]}}
            ],
        },
    ]

    call = StubLLM().converse([], messages, []).tool_calls[0]

    assert call.input["sources"] == []
    assert "cannot determine" in call.input["answer"].lower()


def test_stub_backend_end_to_end_through_the_agent(policy_store: PolicyStore) -> None:
    agent = PolicyAgent(StubLLM(), policy_store, max_iterations=5)

    answer = agent.run("Can I work remotely from another state?")

    assert answer.answer.startswith("[STUB]")
    assert answer.sources == ["Remote Work Policy"]
    assert answer.confidence == 0.5
