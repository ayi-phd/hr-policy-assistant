"""PolicyAgent: an explicit LLM -> tool -> LLM loop.

No agent framework. The loop below is the whole orchestration:

    user question
        -> LLM (may call search_policies)
        -> execute tool, feed result back
        -> LLM ...
        -> LLM calls submit_answer
        -> validate with Pydantic -> PolicyAnswer

The loop is bounded by ``max_iterations``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.llm import LLMClient, LLMResponse, ToolCall
from app.models import PolicyAnswer
from app.policies import PolicyStore
from app.tools import SEARCH_POLICIES, SUBMIT_ANSWER, TOOL_SPECS, ToolError, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the company HR Policy Assistant. Answer the employee's question using ONLY "
    "the company HR policy documents returned by the search_policies tool.\n\n"
    "Rules:\n"
    "- Always call search_policies at least once before answering.\n"
    "- Base every statement on the retrieved policy text. Never invent policy details, "
    "numbers, or conditions.\n"
    "- If the retrieved policies do not contain enough information to answer, say that you "
    "cannot determine the answer from the available policies, and use a low confidence.\n"
    "- Keep the answer concise and understandable to a non-technical employee.\n"
    "- Finish by calling submit_answer exactly once. In 'sources', list only the titles of "
    "policies you actually relied on; use [] if none. 'confidence' is between 0 and 1.\n"
)


class AgentError(RuntimeError):
    """Base class for agent failures."""


class AgentIterationLimitError(AgentError):
    """The agent hit its iteration budget without producing a final answer."""


class AgentOutputError(AgentError):
    """The model called submit_answer but never produced a schema-valid payload."""


def _user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"text": text}]}


def _tool_result_content(tool_use_id: str, payload: dict[str, Any], *, is_error: bool) -> dict[str, Any]:
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [{"json": payload}],
            "status": "error" if is_error else "success",
        }
    }


def _tool_results_message(*blocks: dict[str, Any]) -> dict[str, Any]:
    """All tool results for one assistant turn must go back in a single user message."""

    return {"role": "user", "content": list(blocks)}


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Reconstruct the assistant turn to append to the running transcript."""

    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"text": response.text})
    for call in response.tool_calls:
        content.append(
            {"toolUse": {"toolUseId": call.id, "name": call.name, "input": call.input}}
        )
    if not content:
        content.append({"text": ""})
    return {"role": "assistant", "content": content}


class PolicyAgent:
    """Answers HR policy questions with a bounded tool-calling loop."""

    def __init__(
        self,
        llm: LLMClient,
        store: PolicyStore,
        max_iterations: int = 5,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._llm = llm
        self._store = store
        self._max_iterations = max_iterations

    def run(self, question: str) -> PolicyAnswer:
        system = [{"text": SYSTEM_PROMPT}]
        messages: list[dict[str, Any]] = [_user_text(question)]
        retrieved_titles: set[str] = set()
        last_validation_error: str | None = None

        for iteration in range(1, self._max_iterations + 1):
            response = self._llm.converse(system, messages, TOOL_SPECS)
            messages.append(_assistant_message(response))
            logger.debug(
                "agent iteration %d/%d: stop_reason=%s tool_calls=%s",
                iteration,
                self._max_iterations,
                response.stop_reason,
                [c.name for c in response.tool_calls],
            )

            if not response.tool_calls:
                # No tool call at all: nudge the model toward submit_answer.
                messages.append(
                    _user_text(
                        "Please finish by calling the submit_answer tool with your final answer."
                    )
                )
                continue

            # Reject unknown tools up front (a wiring bug, not a runtime one).
            unknown = _find_unknown_call(response.tool_calls)
            if unknown is not None:
                raise ToolError(f"model requested unknown tool: {unknown.name!r}")

            # Respond to *every* tool call from this turn in one user message, so the
            # Converse transcript stays well-formed even if the model mixes calls.
            result_blocks: list[dict[str, Any]] = []
            for call in response.tool_calls:
                if call.name == SUBMIT_ANSWER:
                    try:
                        answer = self._finalize(call, retrieved_titles)
                    except ValidationError as exc:
                        last_validation_error = _format_validation_error(exc)
                        result_blocks.append(
                            _tool_result_content(
                                call.id,
                                {
                                    "error": "invalid submit_answer payload",
                                    "details": last_validation_error,
                                    "instructions": (
                                        "Call submit_answer again with a valid payload: "
                                        "'answer' non-empty string, 'sources' array of "
                                        "strings, 'confidence' number between 0 and 1."
                                    ),
                                },
                                is_error=True,
                            )
                        )
                        continue
                    return answer

                payload, is_error = execute_tool(call.name, call.input, self._store)
                if not is_error:
                    for result in payload.get("results", []):
                        retrieved_titles.add(result["title"])
                result_blocks.append(
                    _tool_result_content(call.id, payload, is_error=is_error)
                )

            messages.append(_tool_results_message(*result_blocks))

        if last_validation_error is not None:
            raise AgentOutputError(
                f"submit_answer never validated after {self._max_iterations} iterations: "
                f"{last_validation_error}"
            )
        raise AgentIterationLimitError(
            f"agent did not finish within {self._max_iterations} iterations"
        )

    def _finalize(self, call: ToolCall, retrieved_titles: set[str]) -> PolicyAnswer:
        raw = dict(call.input)

        # Bedrock tool inputs are already parsed dicts, but a model may double-encode.
        if isinstance(raw.get("sources"), str):
            raw["sources"] = _coerce_sources(raw["sources"])

        answer = PolicyAnswer.model_validate(raw)

        # Deterministic post-processing: keep only sources we actually retrieved,
        # de-duplicated and order-stable.
        seen: set[str] = set()
        cleaned: list[str] = []
        for src in answer.sources:
            if retrieved_titles and src not in retrieved_titles:
                continue
            if src in seen:
                continue
            seen.add(src)
            cleaned.append(src)

        return answer.model_copy(update={"sources": cleaned})


def _find_unknown_call(calls: list[ToolCall]) -> ToolCall | None:
    known = {SEARCH_POLICIES, SUBMIT_ANSWER}
    for call in calls:
        if call.name not in known:
            return call
    return None


def _coerce_sources(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    return parsed


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
