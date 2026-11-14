# Learning Notes

## Knowledge-base loading and validation

### How the knowledge base is loaded

The support articles live in `data/kb.json` as a JSON list. Each article looks like this:

```json
{"id": "KB-001", "title": "Reset your password", "text": "...", "url": "https://..."}
```

`load_articles()` in `src/support_agent/kb.py` opens that file, parses it with
`json.load`, validates it, and returns a list of Python dictionaries. Keyword
retrieval (`retrieve()`) and semantic retrieval (`semantic_kb.py`) both call it.

### Why validation is necessary

Everything downstream trusts the article data. Without checks, a typo in
`kb.json` (a missing `url`, an empty `text`, a repeated `id`) would not fail
when the file is loaded. It would surface later as a confusing `KeyError` deep
in retrieval, or worse, as an answer citing a blank or wrong source. Validation
makes bad data fail **early, loudly, and with a message that says where the
problem is**.

### How the validation works

`load_articles()` raises a `ValueError` with a clear message when:

| Check | Example message |
| --- | --- |
| File is not valid JSON | `Knowledge base ... is not valid JSON: Expecting value (line 3, column 5)` |
| Top level is not a list | `... must contain a JSON list of articles, got dict` |
| An article is not an object | `..., article #1 must be a JSON object, got str` |
| A required field is missing | `..., article #0 is missing required field(s): url` |
| A field is not a string | `..., article #0 field 'id' must be a string, got int` |
| A field is empty or only spaces | `..., article #0 field 'text' must not be empty` |
| Two articles share an id | `..., article #2 has duplicate id 'KB-001'` |

Required fields are listed once in `REQUIRED_FIELDS = ("id", "title", "text", "url")`.
Extra fields are allowed, so new fields can be added to articles later without
changing the validator.

### Files that implement it

- `src/support_agent/kb.py`: `load_articles()` and `_validate_articles()`.
- `tests/test_kb.py`: offline tests that write small temporary KB files with
  pytest's `tmp_path` fixture and check that each bad case raises the right error.

### Technical decisions (in plain language)

- **Plain Python checks instead of a library (such as Pydantic).** There are
  only four string fields. A short function is easier to read and adds no
  dependency. Pydantic can come later for the LLM's structured output, where
  it is more valuable.
- **One error type (`ValueError`).** Callers only need to catch one kind of
  error, and the message explains the details.
- **Invalid JSON is re-raised as `ValueError` using `from error`.** The message
  includes the file path and line/column, and the original error stays
  attached for debugging.
- **Optional `path` argument.** `load_articles()` still reads `data/kb.json` by
  default, so existing code is unchanged. Tests pass a temporary file, so they
  never touch the real data and never call an external API.
- **Stop at the first problem.** The validator stops at the first error it
  finds. That keeps the code simple, and the knowledge base is small.
- **Article numbers start at 0** (`article #0` is the first item), matching
  Python list indexes.

### Known issue (not caused by validation)

`test_email_change_is_not_supported` fails on the committed baseline as well.
"How do I change my email address?" matches KB-001 because the password article
mentions "email address" (score 0.667 ≥ `MIN_SCORE` 0.30). Fixing this is a
retrieval change and is tracked separately.

## Grounded answer generation (Phase 2, Task 1)

### What it does

`generate_answer(question, articles, client)` in `src/support_agent/answer.py`
takes a customer question plus the articles found by retrieval. It asks Cohere
Chat to write a reply **using only those articles** and returns a `SupportAnswer`.
If it cannot produce a trustworthy, cited answer, it hands the question to a
human instead.

### Why Cohere `documents` and citations

Instead of pasting articles into the prompt, we pass them in Cohere's
`documents` parameter, each with our article ID:

```python
{"id": "KB-001", "data": {"title": "...", "text": "...", "url": "..."}}
```

Cohere then returns **citations**: which parts of the answer came from which
document ID (`response.message.citations[].sources[].id`). That gives us a
mechanical check that the answer is based on our KB, and lets us list the
exact article links as sources.

Trade-off: Cohere does not allow JSON output mode (`response_format`) together
with `documents`, so the model returns plain text and **Python** builds the
structured result from the text and the citations.

### The `SupportAnswer` structure

| Field | Meaning |
| --- | --- |
| `answer` | Text shown to the customer. |
| `sources` | Cited articles: `{"id", "title", "url"}`. Empty when escalating. |
| `escalate` | `True` means a human should take over. |
| `reason` | Short machine-readable label, e.g. `answered`, `no_citations`. |

### How escalation works

The customer sees a fixed, safe hand-off message (`HANDOFF_MESSAGE`). They
never see a guessed answer or an internal error. The `reason` records why:

| Situation | `reason` | API called? |
| --- | --- | --- |
| Empty question | `empty_question` | No |
| No articles retrieved | `no_articles` | No |
| API/network error | `llm_error` | Yes |
| Response has no usable text | `malformed_response` | Yes |
| Answer cut off by the token limit | `incomplete_response` | Yes |
| Model replied `INSUFFICIENT_EVIDENCE` | `insufficient_evidence` | Yes |
| No citation points to a supplied article | `no_citations` | Yes |

Citation IDs we did not supply are ignored, so a made-up ID can never become a source.

### Design decisions

- **The client is passed in.** `generate_answer` never creates a Cohere client
  itself. Tests pass a fake client, so they run offline, cost nothing, and need
  no API key.
- **Fake responses use the real SDK classes** (`V2ChatResponse`, `Citation`,
  `DocumentSource`, ...), so tests match the real response shape.
- **`temperature=0`** for consistent, less creative answers. **`max_tokens=300`**
  caps cost and answer length.
- **Model is configurable** with `COHERE_CHAT_MODEL` (default `command-a-03-2025`,
  a name taken from the installed SDK). It still needs one live check.

### Files

- `src/support_agent/answer.py`: `SupportAnswer`, `generate_answer`, prompt, hand-off message.
- `tests/test_answer.py`: offline tests with a fake client.
- `.env.example`: documents `COHERE_CHAT_MODEL`.

## The support agent: end-to-end flow (Phase 2, Task 2)

### The complete request flow

```
customer question
  → answer_question(question)                      src/support_agent/agent.py
      1. empty question?           → hand off ("empty_question")
      2. get Cohere client         → hand off if COHERE_API_KEY missing ("configuration_error")
      3. retrieve_hybrid(question) → up to 3 articles      (hybrid_kb.py, unchanged)
         fails?                    → hand off ("retrieval_error")
         nothing found?            → hand off ("no_articles"), no chat call
      4. generate_answer(question, articles, client)       (answer.py)
         → cited answer, or hand off (see table above)
  → SupportAnswer(answer, sources, escalate, reason)
```

Try it from the command line (this makes real Cohere calls):

```
python -m support_agent.agent "How do I reset my password?"
```

### How retrieval connects to answer generation

The agent passes the list returned by `retrieve_hybrid` straight to
`generate_answer`. Retrieval decides **which** articles the model may see.
Answer generation decides **whether** they actually answer the question. Extra
fields added by retrieval (such as `score`) are ignored, because only `id`,
`title`, `text` and `url` are sent to Cohere.

Hybrid retrieval almost always returns something, because semantic search ranks
every article. So in practice, the model and its citations are what catch
irrelevant matches, by replying `INSUFFICIENT_EVIDENCE` or by citing nothing.

### What `answer_question` does

It is a small coordinator. It contains no retrieval or prompting logic of its own.
It runs the steps above in order and turns every failure into a safe hand-off.
`retriever`, `client` and `generator` are optional keyword arguments. Real
components are used by default, and tests pass fakes, so the whole workflow
can be tested offline.

### Errors and escalation

- Every failure returns the same `HANDOFF_MESSAGE` with `escalate=True`.
  Exception messages (which might contain keys or internal details) are never
  shown to the customer. Only the exception **type** is written to the log.
- `reason` tells developers which step failed: `configuration_error`,
  `retrieval_error`, `no_articles`, `agent_error`, plus the reasons from
  `generate_answer`.
- The API key is checked **before** retrieval. Semantic retrieval also needs
  it, so a missing key is reported clearly as `configuration_error`.

### Lazy initialization (why nothing happens at import)

`semantic_kb.py` creates a Cohere client the moment it is imported. To keep
`import support_agent.agent` free of side effects (fast, works without a key,
safe in tests), the agent:

- imports `hybrid_kb` **inside** the function that retrieves, and
- creates its chat client in `_default_client()`, which loads `.env`, reads
  `COHERE_API_KEY`, and is cached with `lru_cache` so the client is built once.

A test runs a fresh Python process to confirm importing the agent loads neither
`cohere` nor `semantic_kb`.

### Files

- `src/support_agent/agent.py`: `answer_question`, lazy client, command-line entry point.
- `tests/test_agent.py`: offline workflow tests with a fake retriever and client.

## The HTTP API (Phase 3)

### FastAPI's role

Until now the agent could only be used from Python or the command line.
FastAPI turns it into a **web service**: any website, app or tool can send an
HTTP request and get JSON back. FastAPI does the web plumbing (routing URLs,
reading JSON, checking input, writing JSON). **Uvicorn** is the server program
that listens on a port and passes requests to FastAPI. The agent logic itself
is unchanged: the API calls `answer_question()` and nothing else.

### Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Returns `{"status": "ok"}`. Hosting platforms use it to check the server is alive, so it never calls Cohere (no cost, no dependency on Cohere being up). |
| POST | `/chat` | Body `{"question": "..."}`. Returns `answer`, `sources`, `escalate`, `reason`. |

FastAPI also generates interactive docs at `/docs` from the code.

### Request/response flow

```
client ── POST /chat {"question": "..."} ──► Uvicorn ──► FastAPI
  FastAPI checks the body against ChatRequest (Pydantic)
     invalid → 422 with details of what is wrong (agent never called)
  chat() → agent.answer_question(question)   (retrieval + Cohere, as before)
     SupportAnswer → ChatResponse → 200 JSON
     unexpected exception → 500 JSON with the safe hand-off message
```

### Validation with Pydantic

`ChatRequest` declares `question` as a string that is trimmed and must be
1–1000 characters. FastAPI rejects anything else with **HTTP 422**: a missing
field, `""`, only spaces, a number or `null`, more than 1000 characters, or
broken JSON. The length limit stops one request from sending a huge (and
expensive) prompt. `ChatResponse` describes the output, so the response shape
is checked too and appears in `/docs`.

### Status codes and error safety

- `200`: the agent answered **or** handed off. A hand-off is a normal result,
  not an error, so the client checks `escalate`.
- `422`: the client sent a bad request.
- `500`: something unexpected broke. The body has the same shape as a normal
  answer (hand-off message, `reason: "server_error"`), so a chat UI can still
  display it. Exception messages and stack traces are never sent. Only the
  exception type is logged on the server.

### Design decisions

- **`def`, not `async def`, for `/chat`.** The Cohere SDK calls are blocking.
  FastAPI runs plain `def` endpoints in a thread pool, so one slow Cohere call
  doesn't freeze the other requests.
- **The API calls `agent.answer_question` through the module** (`agent.answer_question(...)`),
  so tests can swap in a fake agent with `monkeypatch`. API tests use
  FastAPI's `TestClient` and make no Cohere calls.
- **No Cohere work at startup.** Importing the API doesn't create a client.
  The first `/chat` request does, so `/health` works even before the key is set.
- **`HOST`/`PORT` environment variables** in `python -m support_agent.api`.
  Most hosting platforms tell the app which port to use through `PORT`.

### Files

- `src/support_agent/api.py`: FastAPI app, request/response models, entry point.
- `tests/test_api.py`: endpoint tests with a mocked agent.
- `pyproject.toml`: adds `fastapi` and `uvicorn` (and `httpx` for tests).
