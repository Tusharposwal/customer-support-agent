# Customer Support AI Agent

An AI customer-support agent that answers questions from a small knowledge base
(`data/kb.json`) using Cohere, cites the articles it used, and hands off to a
human when the articles don't contain the answer.

## Setup (Windows, PowerShell)

Requires **Python 3.12** (the project runs on 3.12.16; see `.python-version`).

```powershell
python -m venv .venv   # run with Python 3.12, e.g. py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" -c constraints.txt
Copy-Item .env.example .env   # then put your Cohere API key in .env
```

`-c constraints.txt` installs the exact package versions that are tested in
CI and deployed on Render (see [Dependencies](#dependencies)).

`.env` settings:

| Variable | Required | Purpose |
| --- | --- | --- |
| `COHERE_API_KEY` | Yes | Cohere API key (never commit `.env`). |
| `COHERE_CHAT_MODEL` | No | Chat model; defaults to `command-a-03-2025`. |
| `COHERE_RERANK_ENABLED` | No | `true`/`false` (also `1`/`0`, `yes`/`no`). Turns on optional reranking; default `false`. |
| `COHERE_RERANK_MODEL` | No | Rerank model; default `rerank-v5.0-pro`. |
| `COHERE_RERANK_MIN_SCORE` | No | Number from 0 to 1. Articles below it are dropped; if none remain, the question is handed off (`below_rerank_threshold`). Default: unset (re-order only). No production value has been chosen yet. |
| `COHERE_RERANK_TIMEOUT_SECONDS` | No | Rerank request timeout; default `5`. |
| `COHERE_CHAT_TIMEOUT_SECONDS` | No | Most time the answer-generation step may take, including one retry; default `25` (max `120`). |
| `COHERE_EMBED_TIMEOUT_SECONDS` | No | Same for each embedding request; default `10` (max `120`). |

An invalid rerank or timeout setting (for example `COHERE_RERANK_ENABLED=ture`)
makes `/chat` hand off with `reason="configuration_error"` rather than silently
ignoring it.

### Cohere errors and retries

Chat and embedding requests never use the SDK's own retries or its
300-second default timeout. A rate limit (429), a server error (500, 502, 503,
504) or a connection that could not be opened is retried **once**, after about
a second, and only if enough of the time budget is left. Timeouts, connections
lost after the request may have been sent, authentication errors and invalid
requests are not retried. If a request still fails, the customer gets
the standard hand-off message (`llm_error` or `retrieval_error`); provider
error text never reaches the customer or the logs. Details:
`docs/LEARNING_NOTES.md`.

### Optional reranking

With `COHERE_RERANK_ENABLED=true`, the hybrid-retrieved articles are re-ordered
by Cohere Rerank before the answer is generated (one extra API request per
question; 0.44–1.30 s in our experiments). Answer generation and citation checking
are unchanged. If the rerank request fails or times out, the agent logs a
warning and carries on with the hybrid articles (it does not retry). See
`docs/LEARNING_NOTES.md` for the design and the offline evaluation.

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
| Liveness check | http://127.0.0.1:8000/health |
| Readiness check | http://127.0.0.1:8000/ready |

When the server starts it checks its configuration (API key present, settings
valid, `data/kb.json` loads with at least one article). If anything is wrong
it **refuses to start** and prints which setting is wrong (never its value),
for example:

```
support_agent.startup.StartupError: The support agent cannot start: COHERE_RERANK_ENABLED must be true or false
ERROR:    Application startup failed. Exiting.
```

After the checks pass, the server creates its Cohere clients (no request is
sent; the knowledge base is embedded on the first question) and only then
reports ready. Stop the server with `Ctrl+C`: it reports not ready, lets
running questions finish, then closes the Cohere connections.

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
| GET | `/health` | **Liveness**: the process is up. Always `200 {"status": "ok"}`. Does not call Cohere. |
| GET | `/ready` | **Readiness**: `200 {"status": "ready"}` once startup has completed (configuration validated, rate limiter and Cohere clients created), otherwise `503 {"status": "not_ready"}` (still starting, or shutting down). Does not call Cohere or check that Cohere accepts the key. |
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

Every response has an `X-Request-ID` header. Send your own (up to 64
letters, digits, `.`, `_`, `:` or `-`) to correlate a request with the server
logs; anything else is replaced by a generated ID.

### Logs

App log lines go to stderr (Render shows them under **Logs**) with time,
level, logger and request ID. Each `/chat` request ends with exactly one
outcome line:

```
2026-11-17 10:15:02,123 INFO support_agent.api [3f2a9c...] chat_request status=200 escalate=true reason=insufficient_evidence sources=0 duration_ms=2310.4
2026-11-17 10:15:09,870 WARNING support_agent.api [8b01d2...] chat_request status=429 escalate=- reason=rate_limited sources=0 duration_ms=0.4 limit=minute
```

Logs never contain the question, the answer, source URLs, API keys or
Cohere's error messages. Uvicorn's own startup and access lines are unchanged.

## Deploy to Render

Create a **Web Service** from this GitHub repository with:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Branch | `phase-1-retrieval` |
| Root directory | *(leave blank: repository root)* |
| Build command | `pip install -e . -c constraints.txt` |
| Start command | `uvicorn support_agent.api:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/ready` *(see the notes below: the live service uses `/health` until this is changed in the dashboard)* |

Environment variables (Render dashboard → your service → **Environment**):

| Variable | Required | Value |
| --- | --- | --- |
| `COHERE_API_KEY` | Yes | Your Cohere API key. |
| `PYTHON_VERSION` | Yes | `3.12.16` (also pinned in `.python-version`). Render uses this variable before `.python-version`, so it must be changed here to change the runtime. |
| `COHERE_CHAT_MODEL` | No | Chat model; default `command-a-03-2025`. |
| `CHAT_RATE_LIMIT_PER_MINUTE` | No | Max `/chat` requests per minute, all visitors combined; default `20`. |
| `CHAT_RATE_LIMIT_PER_DAY` | No | Max `/chat` requests per 24 hours, all visitors combined; default `300`. |
| `COHERE_RERANK_*` | No | Optional reranking, off by default; see the settings table in [Setup](#setup-windows-powershell). |
| `COHERE_CHAT_TIMEOUT_SECONDS`, `COHERE_EMBED_TIMEOUT_SECONDS` | No | Request time budgets; defaults `25` and `10`. |
| `LOG_LEVEL` | No | App log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; default `INFO`. Anything else stops startup. |

> **Never commit API keys.** Set `COHERE_API_KEY` only in Render's Environment
> settings. `.env` is git-ignored and is for local use only.

Notes:

- **Health check path: use `/ready`.** Render uses the health check to decide
  whether a new deploy is healthy before switching traffic to it. `/health`
  only proves the process is running; `/ready` returns 200 only after startup
  has completed: configuration validated, rate limiter and Cohere clients
  created. A misconfigured deploy already fails to start, so it never replaces
  the running version; pointing the health check at `/ready` makes Render also
  wait until the app is fully set up. Changing the
  path is a manual dashboard step (Settings → Health Check Path).
- `/ready` checks that `COHERE_API_KEY` is **set**, not that Cohere accepts it,
  and doesn't depend on Cohere being up: a Cohere outage doesn't make the
  service "not ready" (questions are handed off instead).
- The build uses `pip install -e .` (editable) on purpose: the app reads
  `data/kb.json` from the repository folder.
- **`-c constraints.txt` pins every package version**, so each deploy installs
  exactly what CI tested. The live service keeps its old build command
  (`pip install -e .`, newest versions in range) until the build command is
  changed in the dashboard (Settings → Build Command).
- When a limit is reached, `/chat` returns **HTTP 429** with a `Retry-After`
  header, and the page shows a "try again later" message. `/`, `/health`,
  `/ready` and `/docs` are never limited.
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
pytest
```

All tests run offline except `tests/test_semantic_kb.py`, which calls the
Cohere API and is skipped unless the `RUN_LIVE_COHERE_TESTS` environment
variable is `1`.

**CI:** GitHub Actions (`.github/workflows/tests.yml`) runs the offline suite
on every push and pull request, on Ubuntu 24.04 with Python 3.12.16 (from
`.python-version`). It installs with the same `constraints.txt` and uses no
secrets; live Cohere tests stay skipped.

## Dependencies

- `pyproject.toml` lists the direct dependencies with **compatible ranges**.
- `constraints.txt` pins the **exact version of every package**: direct
  dependencies and all of theirs, for runtime and tests. Every install
  (local, CI, Render) uses it with `-c constraints.txt`.
- `tests/test_dependencies.py` fails if a pin is missing, outside the range in
  `pyproject.toml`, or different from what is installed.

To change a version: edit `constraints.txt` (and the range in
`pyproject.toml` if needed), reinstall with
`pip install -e ".[dev]" -c constraints.txt`, and run `pytest`.

Offline evaluation of the rerank integration (no API calls; original and
holdout questions reported separately):

```powershell
python evals/evaluate_rerank_integration.py
```

## Docs

- `docs/LEARNING_NOTES.md`: how each part works, in plain language.
- `docs/PROJECT_STATUS.md`: goals and phases.
