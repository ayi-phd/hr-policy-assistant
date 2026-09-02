"""Pydantic models: API request/response contracts and policy domain objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Incoming HR policy question."""

    question: str = Field(min_length=1, description="Natural-language HR policy question.")

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace")
        return stripped


class PolicyAnswer(BaseModel):
    """Validated, application-facing answer.

    This is also the schema the LLM must fill via the ``submit_answer`` tool.
    """

    answer: str = Field(min_length=1, description="Concise answer for a non-technical employee.")
    sources: list[str] = Field(
        default_factory=list,
        description="Titles of policies actually used to support the answer.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the answer, 0..1.")


class HealthResponse(BaseModel):
    """Service health payload."""

    status: Literal["ok"] = "ok"


class Policy(BaseModel):
    """A single HR policy document loaded from disk."""

    title: str
    source: str = Field(description="Source filename, e.g. 'Remote Work Policy.md'.")
    content: str


class PolicySearchHit(BaseModel):
    """A policy returned by a search, with its relevance score.

    ``score`` is retained for observability; only ``title``/``source``/``content``
    are surfaced to the LLM as tool output.
    """

    title: str
    source: str
    content: str
    score: float

    def to_tool_result(self) -> dict[str, str]:
        """Shape handed back to the LLM as a tool result entry."""
        return {"title": self.title, "source": self.source, "content": self.content}
