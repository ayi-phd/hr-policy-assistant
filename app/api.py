"""FastAPI application: HTTP surface over the PolicyAgent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.agent import (
    AgentError,
    AgentIterationLimitError,
    AgentOutputError,
    PolicyAgent,
)
from app.config import Settings, get_settings
from app.llm import BedrockLLM, LLMClient, LLMError, StubLLM
from app.models import AskRequest, HealthResponse, PolicyAnswer
from app.policies import PolicyStore
from app.tools import ToolError

logger = logging.getLogger(__name__)

_UPSTREAM_ERROR = "The assistant is temporarily unable to answer. Please try again later."


def _build_llm(settings: Settings) -> LLMClient:
    if settings.llm_backend == "stub":
        logger.warning("LLM_BACKEND=stub: returning canned answers, Bedrock is not called")
        return StubLLM()
    return BedrockLLM(settings)


def build_agent() -> PolicyAgent:
    settings = get_settings()
    store = PolicyStore.from_directory(settings.policy_dir)
    llm = _build_llm(settings)
    return PolicyAgent(llm, store, max_iterations=settings.agent_max_iterations)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.agent = build_agent()
    logger.info("PolicyAgent ready")
    yield
    app.state.agent = None


app = FastAPI(title="HR Policy Assistant", version="0.1.0", lifespan=lifespan)


def get_agent() -> PolicyAgent:
    agent = getattr(app.state, "agent", None)
    if agent is None:  # pragma: no cover - only if used outside the app lifespan
        raise HTTPException(status_code=503, detail="Service not ready")
    return agent


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=PolicyAnswer)
async def ask(request: AskRequest, agent: PolicyAgent = Depends(get_agent)) -> PolicyAnswer:
    try:
        return await run_in_threadpool(agent.run, request.question)
    except (AgentIterationLimitError, AgentOutputError, LLMError, ToolError, AgentError) as exc:
        logger.warning("agent failed for question %r: %s", request.question, exc)
        raise HTTPException(status_code=502, detail=_UPSTREAM_ERROR) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort guard, never leak internals
        logger.exception("unexpected error handling /ask")
        raise HTTPException(status_code=500, detail=_UPSTREAM_ERROR) from exc
