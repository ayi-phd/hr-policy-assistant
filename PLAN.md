# HR Policy Assistant — Implementation Plan

## Context

This is a greenfield project. The repo currently contains only specs (`REQUIREMENTS.md`,
`CLAUDE.md`), a drafted `README.md`, and six sample HR policy documents in `document-base/`
(as both `.md` and `.pdf`; only the `.md` files are used).

We need to build a small, production-oriented AI application: an HTTP API that answers
natural-language HR policy questions by running an **explicit LLM → tool → LLM agent loop**
against AWS Bedrock, grounding every answer in the local policy Markdown and returning a
Pydantic-validated structured response. The test suite must pass with no AWS access by
mocking the Bedrock boundary.

Source of truth: `REQUIREMENTS.md` (product scope) + `CLAUDE.md` (engineering rules).
Hard constraints from those docs:
- No agent framework — hand-write the loop.
- Bedrock **Converse API** via `boto3`; model ID + region from env; no hard-coded creds.
- Validate all LLM-facing output with Pydantic 2; never trust free-form JSON.
- Enforce a max iteration count.
- Keep modules separated: `api`, `agent`, `llm`, `tools`, `policies`, `models`.
- Tests: no AWS creds / no network; mock the LLM boundary.

## Target structure

```
app/
  __init__.py
  config.py       # env-driven settings (region, model id, policy dir, max iterations)
  models.py       # Pydantic models (requests, responses, domain objects)
  policies.py     # load Markdown policies + keyword search
  llm.py          # Bedrock Converse wrapper + LLMClient Protocol + LLMError
  tools.py        # tool specs (search_policies, submit_answer) + dispatch
  agent.py        # PolicyAgent: explicit tool-calling loop
  api.py          # FastAPI app: POST /ask, GET /health, lifespan wiring
tests/
  conftest.py     # FakeLLM, temp policy dir, TestClient fixtures
  test_policies.py
  test_tools.py
  test_agent.py
  test_api.py
.env.example
pyproject.toml
```

`document-base/` stays as the default policy directory; loader reads `*.md` only.

## Modules

### `app/config.py`
Plain dataclass `Settings` populated from `os.getenv` (no new deps):
`aws_region` (`AWS_REGION`, default `us-east-1`), `bedrock_model_id` (`BEDROCK_MODEL_ID`,
default a current Bedrock Sonnet inference-profile id), `policy_dir` (`POLICY_DIR`, default
`document-base`), `agent_max_iterations` (`AGENT_MAX_ITERATIONS`, default `5`).
`get_settings()` cached with `functools.lru_cache`.

### `app/models.py` (Pydantic 2)
- `AskRequest`: `question: str`; field + validator rejecting empty/whitespace (`min_length=1`
  after `.strip()`).
- `PolicyAnswer`: `answer: str`, `sources: list[str]`, `confidence: float = Field(ge=0, le=1)`.
  This is the application-facing contract and the schema the LLM must fill via `submit_answer`.
- `HealthResponse`: `status: Literal["ok"]`.
- `Policy`: `title: str`, `source: str` (filename), `content: str`.
- `PolicySearchHit`: `title`, `source`, `content`, `score: float` (score for observability;
  the tool result serialized to the LLM includes `title`/`source`/`content`).

### `app/policies.py`
- `load_policies(policy_dir: Path) -> list[Policy]`: glob `*.md`, `title` from the first
  `# ` heading (fallback to filename stem), `source` = filename, `content` = full text.
- `search_policies(policies, query, limit=3) -> list[PolicySearchHit]`: deterministic
  keyword relevance. Lowercase + tokenize on non-alphanumerics, drop a small stopword set,
  score each doc by summed term frequency with a title-match boost (×3); return hits with
  `score > 0` sorted desc, capped at `limit`. Empty/whitespace query → `[]`.
- `PolicyStore`: holds the loaded list; `.search(query, limit)` delegates to `search_policies`.
  Injected into tools/agent so it is trivial to fake.

### `app/llm.py`
- `ToolCall` dataclass: `id`, `name`, `input: dict`.
- `LLMResponse` dataclass: `stop_reason: str`, `text: str`, `tool_calls: list[ToolCall]`.
- `LLMClient` `Protocol`: `converse(system, messages, tools) -> LLMResponse`.
- `BedrockLLM(settings)`: lazily builds `boto3.client("bedrock-runtime", region_name=...)`,
  calls `.converse(modelId=..., system=[...], messages=..., toolConfig={"tools": tools})`,
  parses `output.message.content` blocks into `LLMResponse` (text blocks joined; `toolUse`
  blocks → `ToolCall`). Wraps `botocore` `ClientError` / `BotoCoreError` in `LLMError`.
  Credentials come from the default boto3 chain — none in code.

### `app/tools.py`
- `SEARCH_POLICIES_TOOL`: Bedrock `toolSpec` — name `search_policies`, input schema
  `{query: string (required)}`.
- `SUBMIT_ANSWER_TOOL`: Bedrock `toolSpec` — name `submit_answer`, input schema mirroring
  `PolicyAnswer` (`answer: string`, `sources: string[]`, `confidence: number`). This is the
  "structured output where supported" mechanism: the loop ends when the model calls it.
- `TOOL_SPECS = [SEARCH_POLICIES_TOOL, SUBMIT_ANSWER_TOOL]`.
- `execute_tool(name, tool_input, store) -> list[dict]`: dispatch `search_policies` →
  `store.search`; unknown name → `ToolError`; catch unexpected exceptions and return an
  error marker so the agent can surface it as an `is_error` tool result.

### `app/agent.py`
- `PolicyAgent(llm: LLMClient, store: PolicyStore, max_iterations: int)`.
- `run(question: str) -> PolicyAnswer`:
  1. `system` = grounding rules: answer only from retrieved policies; if evidence is
     insufficient, say you cannot determine it, set low `confidence`, leave `sources`
     to only those actually used; you MUST finish by calling `submit_answer`.
  2. `messages = [{"role": "user", "content": [{"text": question}]}]`.
  3. Loop up to `max_iterations`:
     - `resp = llm.converse(system, messages, TOOL_SPECS)`; append assistant message.
     - If `submit_answer` called → validate input with `PolicyAnswer`. On success:
       post-process (clamp `confidence` to [0,1], dedupe `sources`, drop any source not
       in the set of policies actually retrieved this run) and return.
       On `ValidationError` → append an `is_error` tool result asking for a corrected
       call and continue (counts against the iteration budget).
     - If `search_policies` called → `execute_tool`, append `toolResult` content, continue.
     - If the model stops with text and no tool call → append a nudge to call
       `submit_answer`, continue.
  4. Budget exhausted → raise `AgentIterationLimitError`.
  5. Repeated invalid `submit_answer` until budget end → `AgentOutputError`.
- Exceptions live here: `AgentError` base, `AgentIterationLimitError`, `AgentOutputError`.

### `app/api.py`
- `lifespan`: load policies once → `PolicyStore`; build `BedrockLLM`; construct
  `PolicyAgent`; stash on `app.state`.
- `get_agent()` dependency returns `app.state.agent` (overridable in tests).
- `POST /ask` (async) → `response_model=PolicyAnswer`. Runs the agent in a threadpool
  (`fastapi.concurrency.run_in_threadpool`) since boto3 is sync. Maps `AgentIterationLimitError`
  / `AgentOutputError` / `LLMError` / `ToolError` to `HTTPException` (502) with a generic
  message; never leaks exception text, tracebacks, or AWS detail. Empty question → 422 via
  Pydantic.
- `GET /health` → `HealthResponse(status="ok")`.

## Sample policies

The six existing `document-base/*.md` files are realistic and sufficient (Remote Work,
PTO, Business Travel, Expense Reimbursement, Parental Leave, Information Security &
Privacy). No new policy docs needed. The canonical example question
("work remotely from another state for three weeks") is answerable from
`Remote Work Policy.md` §3 (10 business-day limit) — good for a docstring/README example.

## Dependencies (`pyproject.toml`)

Managed with **uv**: `pyproject.toml` + committed `uv.lock`; tests run as `uv run pytest`.
Runtime: `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `boto3`, `httpx`.
Dev (`[dependency-groups]` / `--group dev`): `pytest`, `pytest-asyncio` (`asyncio_mode = "auto"`).
No agent framework, no pydantic-settings, no vector/embedding libs.

## Decisions (confirmed with user)

- Packaging: **uv + pyproject.toml**.
- Structured output: **dedicated `submit_answer` tool** ends the loop.
- Policy docs: **keep `document-base/`** as-is; update `README.md` to match.

## Addendum: offline stub backend

- `LLM_BACKEND` setting (`config.py`): `bedrock` (default) or `stub`.
- `StubLLM` (`app/llm.py`) implements `LLMClient` with no AWS: turn 1 requests
  `search_policies` with the question; turn 2 reads the tool results already in
  the transcript and returns a canned `submit_answer` from the top hit.
- `api.build_agent()` selects the backend via `_build_llm(settings)`.
- Run offline: `LLM_BACKEND=stub uv run uvicorn app.api:app --reload`.

## Tests (mock the LLM boundary only)

- `conftest.py`: `FakeLLM` implementing `LLMClient` with a scripted queue of `LLMResponse`
  objects popped per `converse` call; `policy_store` fixture from a `tmp_path` with 2–3 tiny
  Markdown files; `client` fixture = `TestClient` with `get_agent` overridden to use `FakeLLM`.
- `test_policies.py`: relevant doc returned; irrelevant doc excluded; empty query → `[]`;
  title parsed from `#` heading; ranking/limit respected.
- `test_tools.py`: `execute_tool("search_policies", …)` returns expected hits; unknown tool
  → `ToolError`; store raising → error marker (not a crash).
- `test_agent.py`: (a) search → submit → valid `PolicyAnswer`; (b) multi-iteration: two
  searches then submit; (c) no relevant policy → low-confidence "cannot determine" answer,
  empty `sources`; (d) malformed submit (`confidence` 5 / missing field) → recovers or
  raises `AgentOutputError`; (e) tool failure surfaced without crashing; (f) LLM always
  calls `search_policies` → `AgentIterationLimitError`.
- `test_api.py`: valid `/ask` → 200 + schema; empty/whitespace question → 422; agent raises
  → 502 with generic body (assert no internals leaked); `/health` → 200 `{"status":"ok"}`.

## Deliverables beyond code

- `.env.example` with `AWS_REGION`, `BEDROCK_MODEL_ID`, `POLICY_DIR`, `AGENT_MAX_ITERATIONS`.
- Update `README.md`: correct the "Project Structure" block to the real layout and the
  policy path (`document-base/`), add concrete setup / run / test / example-request steps.

## Verification

1. `pytest` (or `uv run pytest`) — full suite green, no AWS creds, no network.
2. `uvicorn app.api:app --reload`, then `curl localhost:8000/health` → `{"status":"ok"}`.
3. With real AWS creds + Bedrock model access exported:
   `curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"Can I work remotely from another state for three weeks?"}'`
   → JSON with a grounded `answer`, `sources: ["Remote Work Policy"]`, `confidence` in [0,1].
4. `curl -X POST .../ask -d '{"question":""}'` → 422.
5. Swagger UI at `/docs` exercises both endpoints.

## Build order

1. `pyproject.toml`, `.env.example`, package skeleton.
2. `models.py` → `config.py`.
3. `policies.py` + `test_policies.py`.
4. `llm.py` (Bedrock wrapper + Protocol).
5. `tools.py` + `test_tools.py`.
6. `agent.py` + `test_agent.py`.
7. `api.py` + `test_api.py`.
8. README update + final full-suite run.
