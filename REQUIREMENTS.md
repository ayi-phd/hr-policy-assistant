# HR Policy Assistant — Product Requirements & Implementation Intent

## 1. Product Intent

Build a small internal **AI HR Policy Assistant** that allows an employee to ask natural-language questions about company HR policies and receive a concise, grounded answer with citations to the relevant policy documents.

The primary purpose of this project is to demonstrate production-oriented Python AI application development, including:

- FastAPI backend development
- Pydantic data modeling and validation
- LLM integration
- Tool-calling / agent workflow
- Retrieval of relevant policy documents
- Structured LLM responses
- Automated testing
- Clean separation between API, agent, tools, and domain logic

This is a learning/MVP project. Favor simplicity, correctness, testability, and clear architecture over production-scale infrastructure.

---

## 2. Example User Flow

User asks:

> "Can I work remotely from another state for three weeks?"

The system:

1. Receives the question through an HTTP API.
2. Invokes an HR Policy Agent.
3. The agent searches available HR policies using a tool.
4. The agent uses the retrieved policies as authoritative context.
5. The LLM generates a concise answer.
6. The system returns the answer together with the policy sources used.
7. If the available policies do not contain sufficient information, the assistant must explicitly say so rather than hallucinating.

---

## 3. Scope

### In scope

- Natural-language HR policy questions
- Policy document ingestion from local Markdown files
- Basic policy retrieval/search
- LLM-powered policy reasoning
- Agent tool calling
- Structured response validation
- REST API
- Automated unit and API tests
- Basic logging and error handling

### Out of scope

- Authentication / SSO
- Multi-tenancy
- Production deployment
- Vector database
- RAG embeddings
- Complex frontend
- Real HRIS integrations
- Email
- Persistent conversation history
- Autonomous actions affecting employee records

The architecture should make these capabilities possible to add later without requiring a major rewrite.

---

## 4. Technology Stack

### Backend

- **Python 3.12+**
- **FastAPI**
- **Pydantic 2**
- **httpx**
- **pytest**
- pytest-asyncio where appropriate

### AI

- **AWS Bedrock**
- Anthropic LLM (use a current Sonnet model available through Bedrock)
- Bedrock Converse API
- LLM tool/function calling
- Structured JSON output

Use environment/configuration for the Bedrock model ID; do not hard-code credentials.

### Storage / Documents

- Local Markdown policy documents for the MVP
- In-memory representation at application startup
- No database required for V1

### Optional

- boto3 for AWS Bedrock
- uv or pip for dependency management

---

## 5. High-Level Architecture

```text
                    HTTP Client
                        |
                        v
                   FastAPI API
                        |
                        v
                  Policy Agent
                   /        \
                  /          \
                 v            v
        Search Policy Tool   LLM
                 |          (Bedrock)
                 v            |
           Policy Documents   |
                 |            |
                 +------------+
                        |
                        v
               Structured Answer
                        |
                        v
                  API Response
```

Keep the following responsibilities separate:

- `api` — HTTP/API concerns
- `agent` — agent orchestration and LLM interaction
- `tools` — tools exposed to the agent
- `policies` — policy retrieval/domain logic
- `models` — Pydantic models
- `llm` — Bedrock client abstraction

---

## 6. Policy Agent

Implement a dedicated **Policy Agent** responsible for answering HR questions.

The agent should:

1. Receive the user's question.
2. Determine what information it needs.
3. Use `search_policies` when policy information is required.
4. Analyze the retrieved policies.
5. Produce a grounded answer.
6. Cite the policy documents used.
7. Refuse to invent policy when evidence is insufficient.

The agent should have access to at least this tool:

### `search_policies`

Input:

```json
{
  "query": "remote work from another state"
}
```

Output:

```json
[
  {
    "title": "Remote Work Policy",
    "source": "remote_work.md",
    "content": "..."
  }
]
```

Implement the agent using an explicit tool-calling loop rather than hiding all orchestration inside a framework.

The loop should support:

```text
User Question
      |
      v
     LLM
      |
      +---- tool call ----> search_policies()
      |                          |
      |<------- tool result -----+
      |
      v
     LLM
      |
      v
Final structured response
```

Limit the number of agent/tool iterations to prevent runaway execution.

---

## 7. Structured Output

The final response must conform to a Pydantic model similar to:

```python
class PolicyAnswer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

Requirements:

- `answer` must be concise and understandable to a non-technical employee.
- `sources` must contain only policies actually used.
- `confidence` must be between 0 and 1.
- If the policies do not provide enough information, the answer must clearly state that the assistant cannot determine the answer.

Do not rely on unvalidated free-form JSON from the LLM.

---

## 8. Policy Documents

Create 4–6 realistic sample HR policies, for example:

- Remote Work Policy
- Paid Time Off Policy
- Business Travel Policy
- Expense Reimbursement Policy
- Parental Leave Policy
- Work From Another State Policy

Each document should contain enough detail to support realistic questions.

The application should load these documents at startup.

Implement basic keyword/relevance search for V1. Do not introduce embeddings or a vector database.

---

## 9. API

Expose:

### `POST /ask`

Request:

```json
{
  "question": "Can I work remotely from another state for three weeks?"
}
```

Response:

```json
{
  "answer": "Employees may work from another state for up to ...",
  "sources": [
    "Remote Work Policy"
  ],
  "confidence": 0.92
}
```

### `GET /health`

Return:

```json
{
  "status": "ok"
}
```

Use Pydantic request/response models.

Use async FastAPI endpoints where appropriate.

---

## 10. Error Handling

Handle at minimum:

- Empty/invalid questions
- No relevant policies found
- LLM/API failure
- Malformed LLM output
- Tool failure
- Agent iteration limit exceeded

Do not expose AWS credentials, internal exceptions, or sensitive implementation details through API responses.

---

## 11. Testing

Use **pytest**.

Tests should cover:

### Policy search

- Relevant policy is returned.
- Irrelevant policy is not returned.
- Empty search behaves correctly.

### Agent

Mock the LLM/Bedrock dependency.

Test:

- Successful tool call → final answer.
- No relevant policy → appropriate response.
- Malformed LLM output.
- Tool failure.
- Maximum iteration protection.

### API

Test:

- Valid `/ask` request.
- Invalid/empty question.
- Successful response.
- Internal failure handling.
- `/health`.

The test suite must run without AWS credentials by mocking the Bedrock integration.

---

## 12. Configuration

Use environment variables for configuration, including:

```text
AWS_REGION
BEDROCK_MODEL_ID
```

AWS credentials must use the standard AWS credential chain. Never store credentials in source code.

---

## 13. Implementation Priorities

Implement in this order:

1. Project structure and dependencies
2. Pydantic models
3. Policy document loading
4. Policy search
5. Bedrock/LLM abstraction
6. Policy Agent and tool-calling loop
7. FastAPI endpoints
8. Error handling
9. Automated tests
10. README with setup, architecture, and example API calls

Keep the implementation small and understandable. Avoid unnecessary frameworks or infrastructure.

---

## 14. Definition of Done

The project is complete when:

- `uvicorn` can start the FastAPI application locally.
- `/health` works.
- `/ask` accepts natural-language HR questions.
- The Policy Agent can call `search_policies`.
- The LLM produces a validated structured answer.
- Answers are grounded in the supplied policy documents.
- Sources are returned with every successful answer.
- The system explicitly handles insufficient information.
- Tests pass without requiring AWS credentials.
- README documents setup, architecture, configuration, and example requests.

The final implementation should demonstrate a clean **Python + FastAPI + Pydantic + httpx + pytest + AWS Bedrock + LLM + tool-calling agent** architecture suitable as a small production-oriented AI application prototype.