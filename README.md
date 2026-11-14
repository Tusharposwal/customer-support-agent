# Customer Support AI Agent

An AI customer-support agent that answers questions from a small knowledge base
(`data/kb.json`) using Cohere, cites the articles it used, and hands off to a
human when the articles don't contain the answer.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env   # then put your Cohere API key in .env
```

`.env` settings:

| Variable | Required | Purpose |
| --- | --- | --- |
| `COHERE_API_KEY` | Yes | Cohere API key (never commit `.env`). |
| `COHERE_CHAT_MODEL` | No | Chat model; defaults to `command-a-03-2025`. |

## Run the API locally

```powershell
uvicorn support_agent.api:app --reload
# or: python -m support_agent.api   (uses HOST / PORT env vars, default 127.0.0.1:8000)
```

Then open http://127.0.0.1:8000/docs for interactive API docs.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Returns `{"status": "ok"}`. Does not call Cohere. |
| POST | `/chat` | Answers a customer question. |

### Example

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"question": "How do I reset my password?"}'
```

```json
{
  "answer": "To reset your password, open the sign-in page, select Forgot password, and follow the instructions sent to your registered email address.",
  "sources": [
    {"id": "KB-001", "title": "Reset your password", "url": "https://support.example.com/password-reset"}
  ],
  "escalate": false,
  "reason": "answered"
}
```

When the agent can't answer from the knowledge base, `escalate` is `true`,
`sources` is empty, and `answer` is a fixed hand-off message.

Responses: `200` answer or hand-off · `422` invalid request (missing, empty or
over 1000-character `question`) · `500` unexpected server error (safe hand-off body).

## Command line

```powershell
python -m support_agent.agent "How do I reset my password?"
```

## Tests

```powershell
pytest tests/test_kb.py tests/test_answer.py tests/test_agent.py tests/test_api.py
```

These tests run offline. `tests/test_semantic_kb.py` calls the Cohere API.

## Docs

- `docs/LEARNING_NOTES.md`: how each part works, in plain language.
- `docs/PROJECT_STATUS.md`: goals and phases.
