# Customer Support AI Agent

An AI customer-support agent that answers questions from a small knowledge base
(`data/kb.json`, 34 fictional support articles in 9 categories) using Cohere,
cites the articles it used, and hands off to a human when the articles don't
contain the answer. Conversations are kept, so customers can ask follow-up
questions ("What if I bought it 35 days ago?").

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
| `COHERE_REWRITE_MODEL` | No | Model that rewrites follow-up questions; defaults to the chat model. |
| `COHERE_REWRITE_TIMEOUT_SECONDS` | No | Time budget for rewriting a follow-up, including one retry; default `8` (max `120`). |
| `CONVERSATION_DB_PATH` | No | SQLite file for conversations; default `var/conversations.sqlite3` (git-ignored). Created at startup. |

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

After the checks pass, the server opens (or creates) the conversation
database, removes conversations older than 30 days, creates its Cohere
clients (no request is sent; the knowledge base is embedded on the first
question) and only then reports ready. If the database can't be opened, it
refuses to start. Stop the server with `Ctrl+C`: it reports not ready, lets
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
- Follow-up questions continue the same conversation. Reloading the tab shows
  the conversation again; **New conversation** starts a fresh one. The
  conversation ID is kept in `sessionStorage`, so closing the tab forgets it.
- Works on phones and desktops, in light and dark mode.

Each question sent from the page calls Cohere (the same cost as an API call);
a follow-up question makes one extra, small chat call to rewrite it.

### Conversations and follow-up questions

- The first `/chat` request (no `conversation_id`) starts a conversation: the
  server makes a random 128-bit ID and returns it in the response. Later
  requests send it back to ask follow-ups.
- **Follow-ups are rewritten for search only.** "What if I bought it 35 days
  ago?" after "How can I return my order?" is searched as a standalone
  question like "Can I return an order 35 days after purchase?". The answer is
  still written **only from the retrieved articles**, with the usual citation
  checks; earlier assistant replies are never treated as facts. The first
  message of a conversation never makes a rewrite call.
- If a follow-up can't be understood ("Can I do that too?" with nothing to
  refer to), the assistant asks the customer to say more
  (`reason="needs_clarification"`, `escalate=false`, no sources).
- If rewriting fails or its output fails the checks (bad JSON, too long, a
  number the customer never wrote), the message is used as written.
- Each question and its reply are saved together **before** the reply is
  returned. If saving fails, the customer gets a `503` and no answer, so the
  page never shows a message the conversation didn't keep.
- Limits: **40 messages** (20 questions) per conversation, after which `/chat`
  returns `409` and the customer starts a new conversation (nothing is
  dropped). Conversations are deleted **30 days** after their last message.
- **The conversation ID is the only key:** anyone who has it can read that
  conversation. It is never written to the app's logs (they show a short
  one-way hash), and Uvicorn's access log shows `/conversations/<redacted>`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Chat interface (HTML page). |
| GET | `/health` | **Liveness**: the process is up. Always `200 {"status": "ok"}`. Does not call Cohere. |
| GET | `/ready` | **Readiness**: `200 {"status": "ready"}` once startup has completed (configuration validated, rate limiter, conversation database and Cohere clients ready), otherwise `503 {"status": "not_ready"}` (still starting, or shutting down). Does not call Cohere or check that Cohere accepts the key. |
| POST | `/chat` | Answers a customer question; optional `conversation_id` for follow-ups. |
| GET | `/conversations/{id}` | The messages of a conversation (for restoring the page). `404` if unknown or expired. Does not call Cohere. |

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
  "reason": "answered",
  "conversation_id": "3f9c0a6e1b7d4c2e8a5f0d1c2b3a4e5f"
}
```

A follow-up sends the ID back:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"question": "How long is the link valid?", "conversation_id": "3f9c0a6e1b7d4c2e8a5f0d1c2b3a4e5f"}'
```

When the agent can't answer from the knowledge base, `escalate` is `true`,
`sources` is empty, and `answer` is a fixed hand-off message.

Responses: `200` answer, hand-off or clarification request · `404` unknown or
expired conversation · `409` conversation full · `422` invalid request
(missing, empty or over 1000-character `question`, or a malformed
`conversation_id`) · `429` request limit reached (see
[Deploy to Render](#deploy-to-render)) · `503` starting up, or conversations
can't be saved right now (nothing was answered) · `500` unexpected server
error (safe hand-off body, saved to the conversation).

Every response has an `X-Request-ID` header. Send your own (up to 64
letters, digits, `.`, `_`, `:` or `-`) to correlate a request with the server
logs; anything else is replaced by a generated ID.

### Logs

App log lines go to stderr (Render shows them under **Logs**) with time,
level, logger and request ID. Each `/chat` request ends with exactly one
outcome line:

```
2026-11-17 10:15:02,123 INFO support_agent.api [3f2a9c...] chat_request status=200 escalate=true reason=insufficient_evidence sources=0 duration_ms=2310.4 conversation=9e1e76cb0822 turn=2 rewrite=rewritten
2026-11-17 10:15:09,870 WARNING support_agent.api [8b01d2...] chat_request status=429 escalate=- reason=rate_limited sources=0 duration_ms=0.4 conversation=5bcdb9f8f613 limit=minute
```

`conversation` is a one-way hash of the conversation ID (never the ID),
`turn` the question's number in the conversation, and `rewrite` what happened
to it: `not_needed` (first message), `rewritten`, `unchanged`,
`needs_clarification` or `failed` (used as written).

Logs never contain the question, the answer, rewritten queries, source URLs,
conversation IDs, API keys or Cohere's error messages. Uvicorn's own startup
and access lines are unchanged, except that conversation IDs in access-log
paths are replaced by `<redacted>`.

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
| `COHERE_REWRITE_MODEL`, `COHERE_REWRITE_TIMEOUT_SECONDS` | No | Follow-up rewriting; defaults: the chat model and `8`. |
| `CONVERSATION_DB_PATH` | No | Not needed on the free plan (the default works; see the storage note below). |
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
- **Conversations are temporary on the free plan.** The SQLite file lives on
  the service's own disk, which Render wipes on every redeploy, restart and
  spin-down. All conversations then disappear: a customer who returns to an
  open tab sees "Your previous conversation is no longer available" and
  continues in a new conversation. Keeping conversations across restarts
  would need a paid plan with a persistent disk (single instance only) or an
  external database; neither is set up.

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
variable is `1`. Each test uses its own temporary conversation database.
`tests/test_frontend.py` runs the chat page script against a fake DOM with
Node.js (`tests/js/app_smoke.js`, no npm packages) and is skipped if Node.js
isn't installed.

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

## Evaluation

Current knowledge base (KB v2, 77 questions in `evals/support_eval_v2.json`).
Retrieval and answer quality are reported separately:

```powershell
python evals/evaluate_kb_v2.py                 # retrieval, keyword, offline
python evals/evaluate_kb_v2.py --k 5           # other cut-off
python evals/evaluate_kb_v2.py --retriever hybrid                      # LIVE: Cohere embed calls
python evals/evaluate_kb_v2.py --record-answers evals/results/run1.json  # LIVE: embed + chat calls
python evals/evaluate_kb_v2.py --answers evals/results/run1.json         # score answers, offline
```

Follow-up questions (32 short conversations in `evals/conversation_eval.json`:
follow-ups, standalone questions, topic switches and unclear messages):

```powershell
python evals/evaluate_conversations.py                                     # offline
python evals/evaluate_conversations.py --record-rewrites evals/results/rewrites_2026-11-20.json  # LIVE
python evals/evaluate_conversations.py --rewrites evals/results/rewrites_2026-11-20.json         # offline
```

Recording is controlled: it takes the key only from the `COHERE_API_KEY`
environment variable (it never reads `.env`), makes **at most one request per
case with no retries** (32 requests for 32 cases), records each request's
latency and any failure, and refuses to overwrite an existing file. Scoring a
recorded file is offline and also prints the same cases with the last message,
joined messages and hand-written questions, for comparison.

The offline run compares keyword retrieval using the last message alone, all
customer messages joined, and a hand-written standalone question. That shows
how much context matters for retrieval; it does not measure the rewriter,
which needs a recorded (live) run.

Historical experiments (semantic calibration, rerank, rerank integration) were
run on the original 3-article KB. They read a frozen copy,
`evals/fixtures/kb_v1.json`, so their saved results stay reproducible offline:

```powershell
python evals/evaluate_rerank_integration.py
```

## Docs

- `docs/LEARNING_NOTES.md`: how each part works, in plain language.
- `docs/PROJECT_STATUS.md`: goals and phases.
