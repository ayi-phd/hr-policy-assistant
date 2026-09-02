"""Tests for the explicit agent loop. The LLM is always a FakeLLM."""

from __future__ import annotations

import pytest
from conftest import (
    FakeLLM,
    raw_submit_call,
    search_call,
    submit_call,
    text_reply,
)

from app.agent import (
    AgentIterationLimitError,
    AgentOutputError,
    PolicyAgent,
)
from app.models import PolicyAnswer
from app.policies import PolicyStore


def _agent(responses, store: PolicyStore, *, max_iterations: int = 5) -> PolicyAgent:
    return PolicyAgent(FakeLLM(responses), store, max_iterations=max_iterations)


def test_search_then_submit_returns_validated_answer(policy_store: PolicyStore) -> None:
    agent = _agent(
        [
            search_call("remote work another state"),
            submit_call(
                "You may work from another state for up to 10 business days per year.",
                sources=["Remote Work Policy"],
                confidence=0.9,
            ),
        ],
        policy_store,
    )

    answer = agent.run("Can I work remotely from another state for three weeks?")

    assert isinstance(answer, PolicyAnswer)
    assert "10 business days" in answer.answer
    assert answer.sources == ["Remote Work Policy"]
    assert answer.confidence == pytest.approx(0.9)


def test_multiple_iterations_before_final_answer(policy_store: PolicyStore) -> None:
    fake = FakeLLM(
        [
            search_call("remote work", call_id="s1"),
            search_call("paid time off carryover", call_id="s2"),
            submit_call("Combined answer.", sources=["Remote Work Policy"], confidence=0.7),
        ]
    )
    agent = PolicyAgent(fake, policy_store, max_iterations=5)

    answer = agent.run("Two-part question")

    assert answer.answer == "Combined answer."
    assert len(fake.calls) == 3  # two searches + final turn


def test_no_relevant_policy_yields_low_confidence_cannot_determine(
    policy_store: PolicyStore,
) -> None:
    agent = _agent(
        [
            search_call("pet insurance reimbursement"),
            submit_call(
                "I cannot determine the answer from the available HR policies.",
                sources=[],
                confidence=0.1,
            ),
        ],
        policy_store,
    )

    answer = agent.run("Does the company reimburse pet insurance?")

    assert answer.sources == []
    assert answer.confidence <= 0.2
    assert "cannot determine" in answer.answer.lower()


def test_sources_are_filtered_to_actually_retrieved_policies(
    policy_store: PolicyStore,
) -> None:
    agent = _agent(
        [
            search_call("remote work another state"),
            submit_call(
                "Answer.",
                # 'Parental Leave Policy' was never retrieved -> must be dropped.
                sources=["Remote Work Policy", "Parental Leave Policy", "Remote Work Policy"],
                confidence=0.8,
            ),
        ],
        policy_store,
    )

    answer = agent.run("remote work question")

    assert answer.sources == ["Remote Work Policy"]


def test_malformed_submit_then_valid_retry_recovers(policy_store: PolicyStore) -> None:
    agent = _agent(
        [
            search_call("remote work"),
            raw_submit_call({"answer": "x", "sources": [], "confidence": 5}),  # out of range
            submit_call("Corrected answer.", sources=["Remote Work Policy"], confidence=0.6),
        ],
        policy_store,
    )

    answer = agent.run("q")

    assert answer.answer == "Corrected answer."
    assert answer.confidence == pytest.approx(0.6)


def test_persistently_malformed_submit_raises_output_error(policy_store: PolicyStore) -> None:
    agent = _agent(
        [
            search_call("remote work"),
            raw_submit_call({"answer": "x", "sources": [], "confidence": 9}),
            raw_submit_call({"answer": "", "sources": [], "confidence": 0.5}),
        ],
        policy_store,
        max_iterations=3,
    )

    with pytest.raises(AgentOutputError):
        agent.run("q")


def test_tool_failure_is_surfaced_to_model_without_crashing(
    policy_store: PolicyStore,
) -> None:
    class BoomStore:
        def search(self, *_a, **_k):
            raise RuntimeError("search backend down")

    agent = PolicyAgent(
        FakeLLM(
            [
                search_call("anything"),
                submit_call(
                    "I cannot determine the answer right now.", sources=[], confidence=0.1
                ),
            ]
        ),
        BoomStore(),  # type: ignore[arg-type]
        max_iterations=4,
    )

    answer = agent.run("q")
    assert answer.confidence <= 0.2


def test_iteration_limit_exceeded_raises(policy_store: PolicyStore) -> None:
    # Model keeps searching, never submits.
    agent = _agent(
        [search_call("loop", call_id=f"s{i}") for i in range(10)],
        policy_store,
        max_iterations=3,
    )

    with pytest.raises(AgentIterationLimitError):
        agent.run("q")


def test_parallel_search_calls_return_one_tool_results_message(
    policy_store: PolicyStore,
) -> None:
    from app.llm import LLMResponse, ToolCall

    parallel = LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(id="a", name="search_policies", input={"query": "remote work"}),
            ToolCall(id="b", name="search_policies", input={"query": "travel approval"}),
        ],
    )
    fake = FakeLLM(
        [
            parallel,
            submit_call("Done.", sources=["Remote Work Policy"], confidence=0.5),
        ]
    )
    agent = PolicyAgent(fake, policy_store, max_iterations=5)

    agent.run("q")

    # Second converse call sees: user question, assistant(2x toolUse), ONE user tool-results msg.
    second_messages = fake.calls[1]["messages"]
    tool_result_msgs = [
        m
        for m in second_messages
        if m["role"] == "user"
        and any("toolResult" in b for b in m.get("content", []))
    ]
    assert len(tool_result_msgs) == 1
    assert len(tool_result_msgs[0]["content"]) == 2
    assert {b["toolResult"]["toolUseId"] for b in tool_result_msgs[0]["content"]} == {"a", "b"}


def test_json_string_sources_are_coerced(policy_store: PolicyStore) -> None:
    agent = _agent(
        [
            search_call("remote work another state"),
            raw_submit_call(
                {
                    "answer": "ok",
                    "sources": '["Remote Work Policy"]',  # double-encoded by the model
                    "confidence": 0.7,
                }
            ),
        ],
        policy_store,
    )

    answer = agent.run("q")
    assert answer.sources == ["Remote Work Policy"]


def test_text_only_turn_is_nudged_toward_submit(policy_store: PolicyStore) -> None:
    fake = FakeLLM(
        [
            search_call("remote work"),
            text_reply("Here is what I found..."),  # no tool call
            submit_call("Final.", sources=["Remote Work Policy"], confidence=0.5),
        ]
    )
    agent = PolicyAgent(fake, policy_store, max_iterations=5)

    answer = agent.run("q")

    assert answer.answer == "Final."
    # The nudge should have been appended as a user message before the 3rd call.
    third_call_messages = fake.calls[2]["messages"]
    assert any(
        block.get("text", "").startswith("Please finish by calling")
        for msg in third_call_messages
        for block in msg.get("content", [])
    )
