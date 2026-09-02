"""Bedrock Converse API abstraction.

The rest of the app depends only on the :class:`LLMClient` protocol and the two
small dataclasses below, so the LLM boundary can be faked in tests without any
AWS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.config import Settings

# JSON-ish aliases for readability.
Message = dict[str, Any]
ToolSpec = dict[str, Any]
SystemBlock = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the underlying LLM/Bedrock call fails or returns nothing usable."""


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Normalised view of one Converse turn."""

    stop_reason: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract the agent needs from an LLM."""

    def converse(
        self,
        system: list[SystemBlock],
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse: ...


def parse_converse_response(response: dict[str, Any]) -> LLMResponse:
    """Turn a raw Bedrock Converse response dict into an :class:`LLMResponse`."""

    stop_reason = response.get("stopReason", "")
    message = (response.get("output") or {}).get("message") or {}
    content = message.get("content") or []

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if "text" in block:
            texts.append(block["text"])
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tool_use.get("toolUseId", ""),
                    name=tool_use.get("name", ""),
                    input=tool_use.get("input") or {},
                )
            )

    if not stop_reason and not texts and not tool_calls:
        raise LLMError("empty response from Bedrock Converse")

    return LLMResponse(
        stop_reason=stop_reason,
        text="\n".join(texts).strip(),
        tool_calls=tool_calls,
    )


class BedrockLLM:
    """:class:`LLMClient` backed by AWS Bedrock's Converse API via boto3."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client  # lazily created if not injected

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # imported lazily so tests never need boto3 configured

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.aws_region,
            )
        return self._client

    def converse(
        self,
        system: list[SystemBlock],
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            raw = self.client.converse(
                modelId=self._settings.bedrock_model_id,
                system=system,
                messages=messages,
                toolConfig={"tools": tools},
                inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
            )
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - needs AWS
            raise LLMError("Bedrock Converse call failed") from exc

        return parse_converse_response(raw)


class StubLLM:
    """Offline :class:`LLMClient` for local runs without AWS (``LLM_BACKEND=stub``).

    Deterministic two-step behaviour that drives the real agent loop and the real
    ``search_policies`` tool:

    1. First turn - request ``search_policies`` with the user's question as the query.
    2. Next turn - read the tool results already in the transcript and return a
       canned ``submit_answer`` derived from the top hit (or a "cannot determine"
       answer if nothing matched).
    """

    _SNIPPET_CHARS = 240

    def converse(
        self,
        system: list[SystemBlock],
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        search_payload = _latest_search_payload(messages)

        if search_payload is None:
            question = _first_user_text(messages) or "HR policy question"
            return LLMResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="stub-search", name="search_policies", input={"query": question})],
            )

        results = search_payload.get("results", [])
        if results:
            top = results[0]
            snippet = " ".join(top.get("content", "").split())[: self._SNIPPET_CHARS]
            answer = f"[STUB] Based on '{top.get('title', 'a policy')}': {snippet}"
            sources = [top.get("title", "")]
            confidence = 0.5
        else:
            answer = "[STUB] I cannot determine the answer from the available HR policies."
            sources = []
            confidence = 0.1

        return LLMResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(
                    id="stub-submit",
                    name="submit_answer",
                    input={"answer": answer, "sources": sources, "confidence": confidence},
                )
            ],
        )


def _first_user_text(messages: list[Message]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            if "text" in block:
                return block["text"]
    return None


def _latest_search_payload(messages: list[Message]) -> dict[str, Any] | None:
    """Return the most recent ``search_policies`` tool-result payload, if any."""

    for message in reversed(messages):
        for block in message.get("content", []):
            result = block.get("toolResult")
            if not result:
                continue
            for item in result.get("content", []):
                payload = item.get("json")
                if isinstance(payload, dict) and "results" in payload:
                    return payload
    return None
