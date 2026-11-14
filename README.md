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

## Run the application locally

From the project folder, with the virtual environment activated:

```powershell
uvicorn support_agent.api:app --reload
# or: python -m support_agent.api   (uses HOST / PORT env vars, default 127.0.0.1:8000)
```

| What | URL |
| --- | --- |
| Chat interface (customers) | http://127.0.0.1:8000/ |
| Interactive API docs | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

Stop the server with `Ctrl+C`.

### The chat interface

A single responsive page (plain HTML, CSS and JavaScript in
`src/support_agent/static/`, no framework) served by the same FastAPI app:

- Type a question and press **Enter** (Shift+Enter adds a new line) or **Send**.
- While the agent works, a loading indicator appears and sending is disabled.
- Answers show their **source articles** as links.
- When the agent can't answer from the docs, the reply is shown in an amber
  **"Handed off to a human agent"** card.
- Network or server problems show a red error message.
- Works on phones and desktops, in light and dark mode.

Each question sent from the page calls Cohere (the same cost as an API call).

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Chat interface (HTML page). |
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
over 1000-character `question`) · `429` request limit reached (see
[Deploy to Render](#deploy-to-render)) · `500` unexpected server error (safe hand-off body).

## Deploy to Render

Create a **Web Service** from this GitHub repository with:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Branch | `phase-1-retrieval` |
| Root directory | *(leave blank: repository root)* |
| Build command | `pip install -e .` |
| Start command | `uvicorn support_agent.api:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

Environment variables (Render dashboard → your service → **Environment**):

| Variable | Required | Value |
| --- | --- | --- |
| `COHERE_API_KEY` | Yes | Your Cohere API key. |
| `PYTHON_VERSION` | Yes | `3.10.11` (also pinned in `.python-version`). |
| `COHERE_CHAT_MODEL` | No | Chat model; default `command-a-03-2025`. |
| `CHAT_RATE_LIMIT_PER_MINUTE` | No | Max `/chat` requests per minute, all visitors combined; default `20`. |
| `CHAT_RATE_LIMIT_PER_DAY` | No | Max `/chat` requests per 24 hours, all visitors combined; default `300`. |

> **Never commit API keys.** Set `COHERE_API_KEY` only in Render's Environment
> settings. `.env` is git-ignored and is for local use only.

Notes:

- The build uses `pip install -e .` (editable) on purpose: the app reads
  `data/kb.json` from the repository folder.
- When a limit is reached, `/chat` returns **HTTP 429** with a `Retry-After`
  header, and the page shows a "try again later" message. `/`, `/health` and
  `/docs` are never limited.
- The rate limiter is **in memory**: counts reset whenever the service restarts
  or redeploys and aren't shared between instances. It's a cost guard for a
  demo, not production-grade rate limiting.
- On Render's free plan, the service sleeps after 15 minutes without traffic,
  and the next visit takes about a minute to wake it. Open the site shortly
  before a demo.

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
