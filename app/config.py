"""Environment-driven configuration.

Kept deliberately tiny: a frozen dataclass populated from ``os.getenv`` with
sensible defaults. No secrets live here - AWS credentials are resolved by the
standard AWS credential chain inside boto3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_POLICY_DIR = "document-base"
DEFAULT_AGENT_MAX_ITERATIONS = 5
DEFAULT_LLM_BACKEND = "bedrock"
LLM_BACKENDS = ("bedrock", "stub")

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    aws_region: str = DEFAULT_AWS_REGION
    bedrock_model_id: str = DEFAULT_BEDROCK_MODEL_ID
    policy_dir: Path = _REPO_ROOT / DEFAULT_POLICY_DIR
    agent_max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS
    llm_backend: str = DEFAULT_LLM_BACKEND


def _resolve_policy_dir(raw: str | None) -> Path:
    if not raw:
        return _REPO_ROOT / DEFAULT_POLICY_DIR
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path)


def _resolve_max_iterations(raw: str | None) -> int:
    if not raw:
        return DEFAULT_AGENT_MAX_ITERATIONS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"AGENT_MAX_ITERATIONS must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError("AGENT_MAX_ITERATIONS must be >= 1")
    return value


def _resolve_llm_backend(raw: str | None) -> str:
    if not raw:
        return DEFAULT_LLM_BACKEND
    value = raw.strip().lower()
    if value not in LLM_BACKENDS:
        raise ValueError(
            f"LLM_BACKEND must be one of {LLM_BACKENDS}, got {raw!r}"
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) settings from the current environment."""

    return Settings(
        aws_region=os.getenv("AWS_REGION") or DEFAULT_AWS_REGION,
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID,
        policy_dir=_resolve_policy_dir(os.getenv("POLICY_DIR")),
        agent_max_iterations=_resolve_max_iterations(os.getenv("AGENT_MAX_ITERATIONS")),
        llm_backend=_resolve_llm_backend(os.getenv("LLM_BACKEND")),
    )
