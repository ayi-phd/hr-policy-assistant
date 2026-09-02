"""Tests for environment-driven settings resolution."""

from __future__ import annotations

import pytest

from app.config import LLM_BACKENDS, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AWS_REGION", "BEDROCK_MODEL_ID", "POLICY_DIR", "AGENT_MAX_ITERATIONS", "LLM_BACKEND"):
        monkeypatch.delenv(var, raising=False)

    settings = get_settings()

    assert settings.aws_region == "us-east-1"
    assert settings.agent_max_iterations == 5
    assert settings.llm_backend == "bedrock"
    assert settings.policy_dir.name == "document-base"


def test_llm_backend_stub_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "STUB")  # case-insensitive
    assert get_settings().llm_backend == "stub"


def test_llm_backend_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    with pytest.raises(ValueError):
        get_settings()


def test_agent_max_iterations_must_be_positive_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "0")
    with pytest.raises(ValueError):
        get_settings()

    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "not-a-number")
    with pytest.raises(ValueError):
        get_settings()


def test_known_backends() -> None:
    assert LLM_BACKENDS == ("bedrock", "stub")
