# AI HR Policy Assistant

A production-oriented prototype demonstrating how to build an **LLM-powered business workflow agent** using Python, FastAPI, AWS Bedrock, and Pydantic.

The application allows employees to ask natural-language questions about company HR policies. A dedicated Policy Agent determines when policy information is needed, retrieves relevant documents through a tool, and uses an LLM to produce a grounded, structured response with policy citations.

> **Purpose:** Demonstrate practical AI application engineering and agentic workflow design, rather than build a full-scale HR product.

## What It Demonstrates

- **Agentic workflow:** Explicit LLM → tool → LLM execution loop implemented in Python
- **LLM integration:** Anthropic LLM accessed through AWS Bedrock
- **Production Python:** FastAPI, Pydantic, async programming, dependency separation
- **Tool calling:** Agent dynamically invokes policy-search capabilities
- **Grounded AI:** Responses are based on retrieved policy documents rather than model knowledge
- **Structured output:** Pydantic models validate LLM-generated application data
- **Testing:** pytest-based tests with the LLM/Bedrock boundary mocked
- **Engineering discipline:** Clear separation between API, agent, LLM, tools, and domain logic
- **Safety:** Explicit handling of insufficient evidence, tool failures, malformed output, and agent iteration limits

## Architecture

```text
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

### Agent Execution Loop

The agent is intentionally implemented without an agent framework so that the orchestration mechanics remain explicit and easy to understand.

```text
User Question
      |
      v
    LLM
      |
      | tool request
      v
search_policies()
      |
      | policy results
      v
    LLM
      |
      v
Final Structured Answer
```

The loop supports multiple tool iterations and enforces a maximum iteration limit to prevent runaway execution.

## Example

### Request

```http
POST /ask
Content-Type: application/json

{
  "question": "Can I work remotely from another state for three weeks?"
}
```

### Response

```json
{
  "answer": "Employees may temporarily work from another U.S. state for up to 10 business days per calendar year without changing their primary work location. A three-week arrangement requires additional approval from People Operations.",
  "sources": [
    "Remote Work Policy"
  ],
  "confidence": 0.92
}
```

If the available policies do not provide sufficient information, the assistant explicitly says so rather than inventing an answer.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.12+ |
| API | FastAPI |
| Data validation | Pydantic 2 |
| HTTP | httpx |
| AI runtime | AWS Bedrock |
| LLM | Anthropic LLM |
| AWS SDK | boto3 |
| Testing | pytest, pytest-asyncio |
| Documents | Markdown |
| Deployment target | AWS / Kubernetes-compatible architecture |

## Project Structure

```text
.
├── app/
│   ├── agent.py          # Policy Agent and orchestration loop
│   ├── api.py            # FastAPI endpoints
│   ├── llm.py            # Bedrock/LLM integration
│   ├── models.py         # Pydantic models
│   ├── policies.py       # Policy loading and search
│   └── tools.py          # Agent tools
│
├── policies/
│   ├── remote-work.md
│   ├── paid-time-off.md
│   ├── business-travel.md
│   ├── expense-reimbursement.md
│   ├── parental-leave.md
│   └── information-security-and-privacy.md
│
├── tests/
├── CLAUDE.md
├── REQUIREMENTS.md
├── .gitignore
└── pyproject.toml
```

## Design Decisions

### Why a custom agent loop?

The MVP deliberately does not use an agent framework.

The goal is to make the fundamental agent architecture explicit:

1. Provide the LLM with a goal and available tools.
2. Allow the LLM to select a tool.
3. Execute the tool in application code.
4. Return the tool result to the LLM.
5. Continue until a final answer is produced.

This keeps the implementation transparent and makes it possible to evaluate agent frameworks later based on a concrete understanding of what they abstract.

### Why Markdown instead of a vector database?

The prototype contains only a small set of policy documents. Keyword-based retrieval is sufficient for demonstrating the agent workflow while keeping the implementation focused.

A production implementation could evolve toward:

```text
PDF / DOCX
    ↓
Document ingestion
    ↓
Chunking
    ↓
Embedding generation
    ↓
Vector / hybrid search
    ↓
Policy Agent
```

without changing the core API or agent responsibilities.

### Why separate deterministic logic from LLM reasoning?

The system intentionally keeps deterministic operations such as:

- document loading
- policy retrieval
- validation
- API handling
- iteration limits

outside the LLM.

The LLM is used where semantic reasoning and natural-language generation provide value.

This reduces unnecessary model usage and makes the system easier to test, debug, and control.

## Security Considerations

Although this is an educational prototype, the design reflects patterns required for enterprise AI applications.

The system:

- Does not store AWS credentials in source code.
- Uses the standard AWS credential chain.
- Validates LLM output before returning it to the API consumer.
- Limits agent execution.
- Does not allow the LLM to directly modify company systems.
- Explicitly handles insufficient policy evidence.
- Separates external integrations behind application interfaces that can be mocked.

A production implementation would additionally require authentication, authorization/RBAC, audit logging, tenant isolation, secrets management, data classification, and appropriate handling of PII/PHI.

## Testing

The test suite is designed to run without AWS credentials or network access.

The Bedrock/LLM boundary is mocked so that tests can deterministically verify:

- Policy retrieval
- Agent tool calling
- Multiple agent iterations
- Structured output validation
- Tool failures
- Malformed LLM responses
- Agent iteration limits
- API validation and error handling

Run:

```bash
pytest
```

## Running Locally

Install dependencies and configure AWS access using the standard AWS credential mechanism.

Set:

```text
AWS_REGION=<aws-region>
BEDROCK_MODEL_ID=<bedrock-model-id>
```

Start the API:

```bash
uvicorn app.api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Interactive API documentation is available through FastAPI's generated documentation.

## Future Extensions

Potential next steps for evolving the prototype toward a production enterprise AI application include:

- PostgreSQL + pgvector or hybrid search
- PDF/DOCX document ingestion
- Authentication and RBAC
- Per-user and per-document authorization
- Audit trails
- Enterprise identity integration
- MCP-based enterprise system connectors
- Multi-agent workflow orchestration
- Human approval steps for sensitive actions
- LLM evaluation and regression testing
- Token and latency monitoring
- PII/PHI detection and data controls
- AWS EKS deployment

## Why This Project

This project explores a practical pattern for **AI-native enterprise software**:

> **Combine conventional software engineering with LLM reasoning where it provides measurable value, while keeping control, validation, security, and deterministic business logic in the application layer.**

The same architecture generalizes beyond HR policies to enterprise workflows such as document analysis, compliance assistance, customer operations, finance automation, and internal knowledge systems.