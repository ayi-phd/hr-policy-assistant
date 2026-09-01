# CLAUDE.md

## Project

Build the **HR Policy Assistant**, a small production-oriented AI application demonstrating Python backend and agentic workflow development.

The detailed product requirements are in `REQUIREMENTS.md`. Treat that document as the source of truth for product scope.

## Technology

Use:

- Python 3.12+
- FastAPI
- Pydantic 2
- httpx
- pytest / pytest-asyncio
- boto3
- AWS Bedrock
- Anthropic LLM via Bedrock
- Local Markdown files for policy documents

Do not introduce additional frameworks unless there is a clear need.

## Architecture

Keep responsibilities separated:

```text
FastAPI API
    ↓
PolicyAgent
    ↓
LLM via Bedrock
    ↓
Tool calls
    ↓
Policy search/domain functions
    ↓
Policy documents
```

Primary modules:

- `api` — HTTP endpoints
- `agent` — custom agent orchestration loop
- `llm` — Bedrock/LLM abstraction
- `tools` — tools exposed to the agent
- `policies` — policy loading and search
- `models` — Pydantic models

## Agent Implementation

**Do NOT use an agent framework.**

Implement the agent explicitly in Python.

The agent must:

1. Send the user question and system instructions to the LLM.
2. Allow the LLM to request tools.
3. Execute requested tools.
4. Return tool results to the LLM.
5. Repeat until the LLM produces a final response.
6. Enforce a maximum number of iterations.
7. Validate the final response with Pydantic.

The primary tool is:

```text
search_policies(query)
```

Keep the agent loop explicit and easy to understand. Do not hide orchestration behind abstractions.

## LLM

Use AWS Bedrock's Converse API through `boto3`.

Keep the model ID configurable through:

```text
AWS_REGION
BEDROCK_MODEL_ID
```

Do not hard-code credentials.

Use structured output where supported. Never blindly trust free-form LLM JSON; validate all application-facing output with Pydantic.

## Testing

Use pytest.
Tests must not require AWS credentials or network access.

Mock the LLM/Bedrock boundary.

Test:

- policy search
- successful tool calling
- multiple agent iterations
- final structured output
- malformed LLM output
- tool failures
- agent iteration limits
- FastAPI endpoints
- invalid requests

## Engineering Principles

- Keep the implementation small and readable.
- Prefer explicit code over framework magic.
- Separate deterministic business logic from LLM reasoning.
- Never allow the LLM to invent policy information.
- Fail safely when policy evidence is insufficient.
- Keep external integrations behind interfaces that can be mocked.
- Add type hints throughout.
- Do not over-engineer the MVP.

## Definition of Done

The application must run locally with FastAPI, answer HR policy questions using the policy documents, demonstrate an explicit LLM → tool → LLM agent loop, return validated structured responses, and have a passing test suite without requiring AWS access.

## Git Flow

- `master` is the production branch.
- `develop` is the default working branch.
- All feature branches start from `develop`.

For each new task or feature:

1. Ask whether I want a new feature branch.
2. If yes, find the greatest `NNN` among existing `feat/NNN-*` branches and create `feat/NNN-brief-feature-name` from `develop`, incrementing `NNN` by 1.
3. Implement and test the task on that branch.
4. When complete, stage changes and draft a commit message beginning with the task number, e.g. `Feature 003 Implement Tree-sitter predicates`. **Ask for approval before committing.**
5. After commit approval, commit the changes. **Ask for approval before pushing.**
6. After push, prompt: **"Please create a PR `feat/NNN-...` → `develop` in GitHub, review it, and let me know when it's merged."**
7. After merge is explicitly confirmed:
   ```bash
   git switch develop
   git pull origin develop
   ```

**Never commit, push, or switch branches without explicit approval at that step.**