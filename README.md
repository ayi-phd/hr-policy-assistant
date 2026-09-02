# AI HR Policy Assistant

An AI-powered HR policy assistant that provides employees with grounded, policy-aware answers to natural-language questions.

The system combines an explicit agent orchestration layer with deterministic application logic, policy retrieval, structured LLM output, and a production-oriented Python backend. The initial implementation is intentionally lightweight while establishing an architecture that can evolve toward enterprise-scale AI workflow automation.

## Capabilities

- Natural-language HR policy questions
- Agent-driven policy retrieval and reasoning
- Tool calling for policy search
- Grounded responses with policy citations
- Structured, validated responses using Pydantic
- Explicit agent execution limits and failure handling
- Modular LLM integration through AWS Bedrock
- Automated testing with the LLM boundary isolated from application logic

## Architecture

```text id="8tcvf2"
                         User
                           |
                           v
                    FastAPI REST API
                           |
                           v
                     Policy Agent
                           |
                    +------+------+
                    |             |
                    v             v
              Policy Search     LLM
                  Tool        via Bedrock
                    |             |
                    v             |
             Policy Documents     |
                    |             |
                    +------+------+
                           |
                           v
                  Structured Response
                           |
                           v
                         User
```

### Agent Execution

The Policy Agent implements an explicit tool-calling loop rather than delegating orchestration to an agent framework.

```text id="qjlnav"
User Request
     |
     v
    LLM
     |
     | tool request
     v
search_policies()
     |
     | retrieved policy context
     v
    LLM
     |
     v
Validated Response
```

The agent runtime is responsible for:

- Maintaining conversation/tool-call state
- Executing approved tools
- Returning tool results to the LLM
- Detecting completion
- Enforcing execution limits
- Validating final structured output

This separation keeps orchestration logic explicit and allows additional tools, workflows, and agent capabilities to be introduced without coupling them to the API layer.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.12+ |
| API | FastAPI |
| Data modeling | Pydantic 2 |
| HTTP | httpx |
| AI runtime | AWS Bedrock |
| LLM | Anthropic LLM |
| AWS SDK | boto3 |
| Testing | pytest, pytest-asyncio |
| Policy storage | Markdown |
| Deployment target | AWS / Kubernetes |

## Project Structure

```text id="39napd"
.
├── app/
│   ├── config.py         # Environment-driven settings
│   ├── models.py         # Pydantic models (requests, responses, domain objects)
│   ├── policies.py       # Policy loading and keyword search
│   ├── llm.py            # Bedrock Converse abstraction + LLMClient protocol
│   ├── tools.py          # Tool specs (search_policies, submit_answer) + dispatch
│   ├── agent.py          # PolicyAgent: explicit LLM -> tool -> LLM loop
│   └── api.py            # FastAPI endpoints (POST /ask, GET /health)
│
├── document-base/                       # Policy source documents (only *.md is loaded)
│   ├── Remote Work Policy.md
│   ├── Paid Time Off Policy.md
│   ├── Business Travel Policy.md
│   ├── Expense Reimbursement Policy.md
│   ├── Parental Leave Policy.md
│   └── Information Security and Privacy Policy.md
│
├── tests/                # pytest suite; FakeLLM mocks the Bedrock boundary
│   ├── conftest.py
│   ├── test_policies.py
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_api.py
├── .env.example
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
├── REQUIREMENTS.md
└── PLAN.md
```

## Design Principles

### Explicit Agent Orchestration

The initial implementation uses a custom Python agent runtime rather than an agent framework.

This provides direct control over:

- Tool selection and execution
- Agent state
- Execution limits
- Error handling
- Structured output
- Observability boundaries

The architecture can later incorporate an agent framework if it provides meaningful capabilities without obscuring these controls.

### Deterministic Logic Around LLM Reasoning

The system deliberately separates deterministic operations from probabilistic reasoning.

**Application layer:**

- API validation
- Policy loading
- Policy retrieval
- Tool execution
- Output validation
- Execution limits
- Error handling

**LLM layer:**

- Semantic interpretation
- Policy reasoning
- Answer generation
- Tool selection where appropriate

This provides greater predictability, testability, and operational control while avoiding unnecessary LLM calls.

### Grounded Responses

Policy documents are treated as the authoritative source for HR answers.

The agent must not manufacture policy when sufficient evidence is unavailable. Responses identify the policy sources used to support the answer.

## Current Scope

The current implementation uses a small collection of Markdown policy documents and lightweight retrieval.

This intentionally avoids introducing infrastructure that is not yet required by the workload.

The architecture is designed to evolve toward:

```text id="saxr2r"
PDF / DOCX
    ↓
Document ingestion
    ↓
Parsing / chunking
    ↓
Embedding / hybrid indexing
    ↓
Policy retrieval
    ↓
Policy Agent
```

## API

### `POST /ask`

Request:

```json id="ga8lmc"
{
  "question": "Can I work remotely from another state for three weeks?"
}
```

Response:

```json id="bv7unb"
{
  "answer": "Employees may temporarily work from another U.S. state for up to 10 business days per calendar year. A three-week arrangement requires additional approval.",
  "sources": [
    "Remote Work Policy"
  ],
  "confidence": 0.92
}
```

### `GET /health`

Returns the service health status.

## Reliability & Safety

The application explicitly handles:

- Invalid requests
- Missing policy information
- LLM/API failures
- Tool failures
- Invalid structured output
- Agent execution limits

The LLM does not have unrestricted access to application capabilities. Tools are explicitly registered and executed by application code.

## Security Foundation

The current implementation establishes boundaries appropriate for an enterprise AI service:

- AWS credentials use the standard AWS credential chain.
- Credentials are never stored in source code.
- LLM-generated output is validated before entering the application response model.
- External capabilities are exposed through explicit tools.
- Agent execution is bounded.
- External integrations can be isolated behind mockable interfaces.

A production deployment would extend this foundation with:

- Enterprise authentication and SSO
- RBAC and policy-level authorization
- Tenant isolation
- Audit logging
- AWS Secrets Manager
- PII/PHI controls
- Data classification and retention
- Encryption and key management
- Security and compliance monitoring

## Testing Strategy

The test suite isolates the LLM/Bedrock boundary so application behavior can be tested deterministically without AWS credentials or network access.

Coverage includes:

- Policy retrieval
- Agent tool execution
- Multi-step agent interactions
- Structured output validation
- Tool failures
- LLM failures
- Execution limits
- API validation
- API error handling

Run:

```bash id="iu9xnl"
uv run pytest
```

The suite needs no AWS credentials and makes no network calls.

## Configuration

The project uses [uv](https://docs.astral.sh/uv/) for dependency management
(`pyproject.toml` + `uv.lock`). All settings are environment variables; copy
`.env.example` to `.env` and adjust:

| Variable | Default | Purpose |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region for the Bedrock runtime client. |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Converse API model / inference-profile ID your account can access. |
| `POLICY_DIR` | `document-base` | Directory of policy `*.md` files (relative to repo root or absolute). |
| `AGENT_MAX_ITERATIONS` | `5` | Maximum LLM ↔ tool iterations before the agent gives up. |
| `LLM_BACKEND` | `bedrock` | `bedrock` calls AWS Bedrock; `stub` returns canned answers with no AWS calls (local API/UI testing). |

AWS authentication uses the standard AWS credential chain (environment, shared
config/credentials file, SSO, instance role). Credentials are never read from
source or committed.

## Local Development

Install dependencies and start the API:

```bash id="642uoi"
uv sync
uv run uvicorn app.api:app --reload
```

Check health and ask a question:

```bash id="hf83kd"
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Can I work remotely from another state for three weeks?"}'
# {"answer": "...", "sources": ["Remote Work Policy"], "confidence": 0.9}
```

`/ask` requires valid AWS credentials with Bedrock access to the configured
model. The interactive API docs are available at `http://127.0.0.1:8000/docs`.

To run the full HTTP surface with no AWS at all, use the stub backend:

```bash id="stub01"
LLM_BACKEND=stub uv run uvicorn app.api:app --reload
```

It drives the real agent loop and real policy search, then returns a canned
`[STUB]` answer instead of calling Bedrock.

## Evolution Path

The architecture provides a foundation for extending the assistant into a broader enterprise AI automation platform.

Potential capabilities include:

### Enterprise Knowledge

- PDF/DOCX ingestion
- Semantic and hybrid retrieval
- PostgreSQL / pgvector
- Document versioning
- Access-controlled knowledge sources

### Agentic Workflows

- Additional specialized tools
- Multi-agent coordination
- Agent registries
- Workflow state and persistence
- Scheduled agent execution
- Human approval gates
- Feedback and evaluation loops

### Enterprise Integrations

- HRIS
- CRM
- ERP
- Project management systems
- Email and collaboration systems
- OAuth2-based delegated access
- MCP-based enterprise connectors

### Production Platform

- AWS EKS
- Helm
- ArgoCD
- CI/CD
- Distributed tracing
- LLM observability
- Token and cost monitoring
- Automated evaluation
- Reliability and performance monitoring

The initial Policy Agent is intentionally narrow, but its execution model is designed to support progressively more capable enterprise AI workflows.