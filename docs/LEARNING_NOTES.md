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

**Resolved** in V2 Phase 2. See "Keyword title-match rule" at the end of this file.

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
| API/network error, or generation stopped with `ERROR`/`TIMEOUT` | `llm_error` | Yes |
| Unknown finish reason, or no usable text | `malformed_response` | Yes |
| Answer cut off by the token limit | `incomplete_response` | Yes |
| Model replied `INSUFFICIENT_EVIDENCE` (any case, spacing or markdown escaping) | `insufficient_evidence` | Yes |
| A citation points to an article we did not supply | `invalid_citations` | Yes |
| No citations | `no_citations` | Yes |

A citation of an ID we did not supply rejects the **whole** answer (updated
in V2 Phase 3, Milestone 1; it used to be dropped silently). See "Agent
decision reliability" below.

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
| GET | `/ready` | `200` once startup has completed (validation passed; since Milestone 5 also the rate limiter and Cohere clients created), else `503`. See "Startup validation and readiness" and "Startup and shutdown lifecycle" near the end of these notes. |
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
- **No Cohere work at import.** Importing the API doesn't create a client.
  (At the time the first `/chat` request did. Since V2 Phase 4 the server
  refuses to start without the key, and since Milestone 5 it creates the
  clients at startup; see "Startup and shutdown lifecycle".)
- **`HOST`/`PORT` environment variables** in `python -m support_agent.api`.
  Most hosting platforms tell the app which port to use through `PORT`.

### Files

- `src/support_agent/api.py`: FastAPI app, request/response models, entry point.
- `tests/test_api.py`: endpoint tests with a mocked agent.
- `pyproject.toml`: adds `fastapi` and `uvicorn` (and `httpx` for tests).

## The chat web interface

### How the browser talks to FastAPI

```
Browser opens http://127.0.0.1:8000/
  GET /                  → FastAPI returns static/index.html
  GET /static/styles.css → page styling
  GET /static/app.js     → page logic
Customer presses Enter
  app.js: fetch("/chat", POST, {"question": "..."})   ← same server, same /chat as the API
  FastAPI → answer_question() → JSON {answer, sources, escalate, reason}
  app.js builds the message bubble from that JSON
```

The page is just another client of `POST /chat`, exactly like `curl` or
`/docs`. There is no extra backend logic for the UI. Because the page and the
API come from the **same server (same origin)**, the browser allows the
`fetch` without any CORS configuration.

### What `app.js` does

1. Ignores empty messages and shows a hint instead.
2. Shows the customer's message, a loading indicator, and disables the
   input and Send button so a question can't be sent twice.
3. Sends `POST /chat` with JSON, and gives up after 60 seconds.
4. Shows the result:
   - `escalate: false` → answer plus "Sources" links;
   - `escalate: true` → amber "Handed off to a human agent" card;
   - `422` → "Please enter a question of up to 1000 characters";
   - network error, timeout or unreadable response → red error message.
     A `500` still carries the safe hand-off body, so it's shown as a hand-off.

### Safe rendering (why no `innerHTML`)

Questions come from customers and answers come from a model, so both are
**untrusted text**. If they were inserted with `innerHTML`, text like
`<img src=x onerror=...>` would become real HTML and could run code
(an XSS attack). `app.js` builds every element with `document.createElement`
and sets text with `textContent`, so such text is displayed literally. Source
links are only created for `http://` or `https://` URLs (blocking
`javascript:` links), and open in a new tab with `rel="noopener noreferrer"`.

### Serving the files

`api.py` mounts the `static/` folder at `/static` (`StaticFiles`) and adds
`GET /`, which returns `index.html`. `/` is hidden from the API schema, so
`/docs` only lists the real API endpoints. `pyproject.toml` lists `static/*`
as package data so the files ship with the package.

### Files

- `src/support_agent/static/index.html`: page structure.
- `src/support_agent/static/styles.css`: layout, colours (light/dark), mobile rules.
- `src/support_agent/static/app.js`: sending questions and showing answers.
- `src/support_agent/api.py`: `GET /` and the `/static` mount.
- `tests/test_api.py`: page, assets, endpoint wiring and the "no innerHTML" check.

## Rate limiting `/chat` (deployment)

### Why

Every `/chat` request costs money (one Cohere embed call plus one chat call).
Once the site is public, anyone or any script could send thousands of
questions. The limiter caps the total spend.

### How it works

`src/support_agent/rate_limit.py` keeps a list (a `deque`) of the times of
recently accepted requests. For each new `/chat` request:

1. Drop timestamps older than 24 hours.
2. If 300 or more remain, reject with the **"day"** limit.
3. If 20 or more are from the last 60 seconds, reject with the **"minute"** limit.
4. Otherwise record the time and allow the request.

These are **sliding windows**: "the last 60 seconds" and "the last 24 hours"
measured from now, not calendar minutes or days. Rejected requests aren't
recorded, so hammering the endpoint doesn't extend the wait. The list never
holds more than the daily limit (300 timestamps), so memory stays tiny.

In `api.py`, `chat()` calls `chat_limiter.check()` **after** request
validation and **before** calling the agent:

- allowed → normal flow;
- limited → **HTTP 429** with a friendly `detail` message and a `Retry-After`
  header (seconds until a slot frees up). The agent and Cohere are not called.

Invalid requests (422) never reach the limiter, so they don't use up quota.
`/`, `/health` and `/docs` aren't limited, so Render's health check keeps
passing even when chat is at its limit. `app.js` shows the server's message
with a "Please try again later" label.

Limits come from `CHAT_RATE_LIMIT_PER_MINUTE` (default 20) and
`CHAT_RATE_LIMIT_PER_DAY` (default 300). An invalid value such as `0` or
`abc` stops the app at startup with a clear error, rather than silently
running with no protection.

### Design decisions

- **Global, not per user.** Behind Render's proxy, a visitor's IP comes from
  a header that scripts can fake, so per-IP limits are easy to dodge. A global
  cap reliably bounds cost. The trade-off is that one heavy user can use up
  the quota for everyone.
- **Monotonic clock** (`time.monotonic`) so changes to the system clock can't
  break the windows. Tests pass a fake clock, so they can "jump" 24 hours instantly.
- **A lock** (`threading.Lock`), because FastAPI handles `def` endpoints in
  several threads at once, and two requests must not both take the last slot.
- **No new dependency.** The standard library is enough.

### Limitations (why this is demo-grade)

- Counts live in memory: a restart, redeploy or free-plan sleep resets them.
- Each server process has its own counter. Running several workers or
  instances multiplies the real limit.
- A production system would use a shared store (for example Redis), per-user
  limits tied to authentication, and a spending limit on the provider account.

### Files

- `src/support_agent/rate_limit.py`: `RateLimiter`, `limiter_from_env()`.
- `src/support_agent/api.py`: 429 handling in `/chat`.
- `src/support_agent/static/app.js`: 429 message for customers.
- `tests/test_rate_limit.py`: limiter unit tests with a fake clock.
- `tests/test_api.py`: endpoint-level limit tests.
- `.python-version`: pins the Python version for Render and CI (3.10.11 at the time; 3.12.16 since V2 Phase 4, Milestone 4).

## Keyword title-match rule (V2 Phase 2: RAG improvement)

### The problem

"How do I change my email address?" returned KB-001 (Reset your password).
After stop words are removed, the question becomes `{change, email, address}`.
KB-001's body ends with "...sent to your registered **email address**", so it
scored 2/3 = 0.667, which was higher than many correct matches (0.5).

Keyword retrieval caused this. Hybrid RRF then made it worse: a keyword
rank-1 article always scores at least `1/61 + 1/63`, more than any article
found only by semantic search (at most `1/61`). So KB-001 was always first
in hybrid results.

Raising `MIN_SCORE` could not fix it: correct matches such as "Can I get a
refund?", "How can I track my shipment?" and "Where is my order?" score only 0.5.

### The fix

`retrieve()` in `src/support_agent/kb.py` now keeps an article only if
**both** are true:

1. its score is at least `MIN_SCORE` (unchanged, 0.30), and
2. at least one question word appears in the article's **title**.

A title says what an article is *about*. Body text often mentions other
things only in passing.

### How it was measured

The dataset (`evals/retrieval_queries.json`) was expanded first, from 10 to 15
questions. The baseline was recorded on that expanded set before `kb.py` was
changed. The new questions are:

- Answerable: "Where do I enter my order number for a refund?" (KB-002)
- Unsupported: update my email, delete my account, change my phone number,
  change my shipping address

Commands (offline, no Cohere calls):

```
python evals/evaluate_retrieval.py
python -m pytest tests/test_kb.py
```

| Metric (keyword, 15 questions) | Before | After |
| --- | --- | --- |
| Overall accuracy | 8/15 (53%) | 13/15 (87%) |
| Recall@3 (9 answerable) | 7/9 (78%) | 7/9 (78%) |
| Top-1 accuracy | 6/9 | 7/9 |
| Abstention accuracy (6 unsupported) | 1/6 (17%) | 6/6 (100%) |
| `tests/test_kb.py` | 37 passed, 6 failed | 43 passed |

Top-1 is counted by hand from the eval output, because `evaluate_retrieval.py`
does not print it.

### Every changed result

| Question | Before | After | Why |
| --- | --- | --- | --- |
| How do I change my email address? | KB-001 | none | "email", "address" are only in KB-001's body |
| How do I update my email? | KB-001 | none | "email" is only in KB-001's body |
| How do I delete my account? | KB-003 | none | "account" is only in KB-003's body |
| How do I change my phone number? | KB-002 | none | "number" is only in KB-002's body ("order number") |
| How do I change my shipping address? | KB-001, KB-003 | none | "address" and "shipping" are only in body text |
| Where is my order? | KB-002, KB-003 | KB-003 | KB-002 mentions "order" only in its body. Before, the 0.5 tie was broken by file order. |

Unchanged results include password, forgot password, login, refund, track
shipment, the refund order-number question and Mars. Every one of the
correct matches already shared a word with its article's title.

### Trade-offs and known limits

- **Body-only matches are dropped.** A relevant question that matches only
  body words (for example "What if I miss the 30 days?" vs KB-002) now gets
  no keyword result. No current dataset question depends on this. In hybrid,
  semantic search can still find such articles.
- **Titles now matter for retrieval.** A new article needs a title that uses
  the words customers actually type.
- **Still a false positive:** "Can I cancel my order?" returns KB-003,
  because "order" is in its title ("Track an order"). The rule only looks for
  one shared title word, not for the question's intent.
- **Still not found:** "I want my money back for a purchase" and "My package
  has not arrived". These need synonyms, which is the job of semantic search.
- **Hybrid still never abstains.** `retrieve_semantic` has no similarity
  threshold, so it returns all 3 articles for every question. This change
  removes the misleading keyword #1 for the email question, but the LLM
  citation check is still the final safety net. A calibrated semantic
  threshold is the next step, and it needs live Cohere scores.

## Semantic threshold calibration (V2 Phase 2, Milestone 2)

### Why

`retrieve_semantic()` has no minimum similarity. It always returns the top 3,
which here means **every** article, so hybrid retrieval never returns "no
match". Before adding a threshold, we measured whether cosine scores can
tell answerable questions apart from unsupported ones at all.

### Method

- `evals/calibrate_semantic.py --generate` made **exactly two** Cohere embed
  calls, counted and capped in code:
  1. the 3 articles (`embed-v4.0`, `search_document`), via the production
     `semantic_kb._embedded_articles()`;
  2. all 17 dataset questions in one batch (`embed-v4.0`, `search_query`).
- The vectors (1536 dimensions) are saved in
  `evals/fixtures/semantic_embeddings.json` with the model, input types,
  a KB-text hash and a question-list hash. If the KB or the questions change,
  the fixture is rejected and must be regenerated. Labels (`expected_id`) are
  read live, so relabelling needs no new calls. `--generate` refuses to
  overwrite an existing fixture.
- `python evals/calibrate_semantic.py` then runs **offline**. It never imports
  `semantic_kb`, because that module needs an API key at import time.
- For each question:
  - answerable → the **expected article's** score (a threshold must keep it);
  - unsupported → the **highest** article score (a threshold must cut it).
- Every observed score is tried as a threshold (accept if score ≥ threshold):
  - **false rejection** = answerable questions whose expected article is cut;
  - **false acceptance** = unsupported questions where any article passes;
  - **wrong article** = answerable questions where a non-expected article passes.

Two questions were added to the dataset: "Can I cancel my order?"
(unsupported) and "How do I check refund eligibility?" (KB-002).

### Findings (embed-v4.0, 17 questions)

**There is no clean separation.** The lowest answerable score is 0.2550
("My package has not arrived" → KB-003). The highest unsupported score is
0.4576 ("How do I change my email address?" → KB-001). They overlap by 0.2026.

Account-type questions (email, phone, delete account) are semantically close
to the password article. The email question scores 0.4576 against KB-001,
which is higher than 6 of the 10 correct matches. Embeddings measure topic
similarity. They cannot tell that this article does not answer the
question.

| Threshold | False reject | False accept | Wrong article |
| --- | --- | --- | --- |
| 0.0719 (current behaviour) | 0/10 | 7/7 | 10/10 |
| 0.2550 | 0/10 | 6/7 | 1/10 |
| 0.3211 | 2/10 | 4/7 | 1/10 |
| 0.3503 | 4/10 | 4/7 | 0/10 |
| 0.4386 | 4/10 | 1/7 | 0/10 |
| 0.4665 | 6/10 | 0/7 | 0/10 |

(Full table: run the script.)

- With **zero false rejections**, a threshold can only reject the Mars
  question.
- Rejecting **all** unsupported questions also rejects 6 of 10 good questions,
  including "Can I get a refund?" (0.2632) and the paraphrases that only
  semantic search finds ("money back" 0.3496, "package not arrived" 0.2550).
- The top-two margin does not separate the groups either. "Package not
  arrived" has margin 0.0274, while the email question has 0.2419.

### Decision status

No threshold was implemented. The data does not support a single cut-off
that protects answerable questions while removing unsupported ones. The only
zero-false-reject range, (0.0719, 0.2550], is defined by a single
nonsense question (Mars), which is too little evidence for a rule.

### Testing and cost safety

- `tests/test_calibrate_semantic.py`: offline tests for scoring, metrics,
  separation, fixture validation, and generation with a fake client
  (exactly 2 calls, refuses to overwrite). Offline tests block `cohere` and
  `semantic_kb` imports to prove that no call is possible.
- `tests/test_semantic_kb.py` (live) now runs only with
  `RUN_LIVE_COHERE_TESTS=1`, so plain `pytest` makes no API calls.

## Cohere rerank experiment (V2 Phase 2, Milestone 3)

### Question

Cosine similarity could not separate answerable from unsupported questions.
Can a **reranker** do better? A reranker reads the question and each article
together and scores how relevant the article is to that question. Embeddings
compare two separately computed vectors, so they mostly measure topic
similarity. This was an experiment. Production retrieval is unchanged.

### Method

- `evals/rerank_experiment.py`: `--dry-run` shows the plan, `--run` makes the
  calls, and with no flag it prints the offline report.
- Each question was sent once to `ClientV2.rerank(model, query, documents)`
  with all 3 articles formatted as `Title: <title>\nText: <text>` (no URL).
  The SDK silently retries 429/5xx responses twice by default, so every call
  passes `request_options={"max_retries": 0}`. One call in the code is
  exactly one request.
- The model comes from `COHERE_RERANK_MODEL` (default `rerank-v5.0-pro`). The
  approved fallback `rerank-v3.5` is used only if the primary returns a
  400/404 error that is about the model, and only before any result was saved.
  Every other error stops the run.
- Results go to `evals/fixtures/rerank_results.json`. The file stores the
  model, document format, KB/question/dataset hashes, and per question: the
  returned indices mapped to article IDs, the relevance scores, billed units
  and latency. Each result is saved immediately (atomic write). A re-run only
  requests missing questions, never overwrites a saved result, refuses
  invalid or out-of-date files instead of re-requesting, and is capped at 17
  successful calls.
- Comparison, all offline, using the same metric definitions as the
  semantic calibration:
  - **Keyword**: `kb.retrieve`.
  - **Semantic**: cosine scores from the saved embedding fixture.
  - **Hybrid**: the real `hybrid_kb.retrieve_hybrid`, with `retrieve_semantic`
    replaced by a version that reads the saved vectors (no API key, no client).
  - **Rerank**: the saved scores.

### Run

17 successful calls, 0 failed. `rerank-v5.0-pro` was available, so the
fallback was not used. The API returned no `billed_units.search_units`
(stored as `null`). Latency: average 601 ms, median 591 ms, maximum 1299 ms
(the first call, which includes connection setup), minimum 441 ms.

### Results (17 questions: 10 answerable, 7 unsupported)

| Method (no threshold) | Top-1 | Recall@3 | False accept | Abstain | Wrong article |
| --- | --- | --- | --- | --- | --- |
| Keyword | 8/10 | 8/10 | 1/7 | 6/7 | 0/10 |
| Semantic | 10/10 | 10/10 | 7/7 | 0/7 | 10/10 |
| Hybrid | 10/10 | 10/10 | 7/7 | 0/7 | 10/10 |
| Rerank | 10/10 | 10/10 | 7/7 | 0/7 | 10/10 |

Without a threshold, reranking only reorders, and the order was already
correct. Its value is in the **scores**:

| Rerank threshold | False reject | False accept | Wrong article |
| --- | --- | --- | --- |
| 0.0008 (accept all) | 0/10 | 7/7 | 10/10 |
| 0.0900 | 0/10 | 2/7 | 4/10 |
| **0.4232** | **0/10** | **1/7** | **0/10** |
| 0.4720 | 1/10 | 1/7 | 0/10 |
| 0.6457 | 2/10 | 0/7 | 0/10 |

- Lowest answerable score: **0.4232** ("My package has not arrived" → KB-003).
- Highest unsupported score: **0.4746** ("Can I cancel my order?" → KB-002,
  the refund article).
- **No clean separation** (overlap 0.0514), but it comes from that **one**
  question. The other six unsupported questions all score **≤ 0.0900**,
  including every email/account question (email change 0.0412).
- Any threshold in **(0.0900, 0.4232]** gives 0/10 false rejections,
  1/7 false acceptances and 0/10 wrong-article acceptances on this data. The
  best semantic threshold with zero false rejections still accepted 6/7.

### Conclusion and limits

- Reranking separates these questions far better than cosine similarity.
  The email-change case that started this phase is now clearly rejected.
- A rerank threshold looks promising, but **no value has been chosen**.
  17 questions are too few to fix one. The range was found on the same data
  it would be judged on.
- "Can I cancel my order?" still scores high against the refund article.
  The reranker treats cancellation and refunds as related. That case would
  still rely on the LLM citation check.
- Scores belong to `rerank-v5.0-pro` and this document format. Changing
  either means running the experiment again.
- In production, reranking would add one extra request (about 0.45–0.75 s
  here) to every customer question.

## Rerank holdout dataset (V2 Phase 2, Milestone 4)

### Why a holdout set

The rerank score range that looked promising, (0.0900, 0.4232], was found by
looking at the **same** 17 questions it would be judged on. Any range found
that way fits those particular questions, partly by luck. A **holdout set**
is a separate group of questions that played no part in that analysis. If
the pattern still holds on questions it has never "seen", we can trust it
more. If it does not, we have avoided shipping a threshold that only worked
on our own examples.

### The dataset

`evals/rerank_holdout_queries.json` uses the same schema as
`retrieval_queries.json` (`query`, `expected_id`) and has 20 new questions:

- **10 answerable** (4 × KB-001, 3 × KB-002, 3 × KB-003), written as
  paraphrases and everyday customer wording, not copies of the original
  questions.
- **10 unsupported** (`null`): cancellation, refund details the article does
  not cover (refund method and payout time), changing email / phone /
  shipping address, closing the account, two-factor authentication, changing
  a username, and one unrelated question.

Each label was checked against the article text, not against shared
keywords. For example, "How many days does it take for a refund to reach
my bank account?" mentions refunds, but KB-002 only covers the 30-day request
window, so it is labelled unsupported.

### Rules that keep it independent

- The questions were written and labelled **without** looking at rerank
  scores, and no question was added or removed because of a score.
- The original 17-question dataset and `evals/fixtures/rerank_results.json`
  are unchanged. A test checks that the original questions still match the
  hash stored in the saved results.
- Holdout results are saved **separately** from the original experiment, in
  `evals/fixtures/rerank_holdout_results.json` (see below). No threshold was
  chosen from them.
- `tests/test_rerank_holdout.py` checks the structure: 20 questions, 10/10
  split, valid article IDs covering all three articles, no duplicates and no
  overlap with the original dataset (ignoring case and punctuation).

### Holdout run

`python evals/rerank_experiment.py --run --dataset holdout` made exactly the
approved 20 rerank calls (20 successful, 0 failed). It used the same settings
as Milestone 3: `rerank-v5.0-pro` (no fallback needed), the `title-text-v1`
document format, and SDK retries off. Its separate call limit is 20. The
results file is marked `"dataset": "holdout"`, so it can't be mixed up with
the original results.

- Billed search units: not returned (`null`), as before.
- Latency: average 550.7 ms, median 473.8 ms, minimum 438.4 ms, maximum 926.8 ms.

### Holdout results (20 questions: 10 answerable, 10 unsupported)

| Method (no threshold) | Top-1 | Recall@3 | False accept | Abstain | Wrong article |
| --- | --- | --- | --- | --- | --- |
| Keyword | 3/10 | 3/10 | 1/10 | 9/10 | 0/10 |
| Rerank | 10/10 | 10/10 | 10/10 | 0/10 | 10/10 |

- Lowest answerable score: **0.4116** ("I keep typing the wrong password and
  can't get in" → KB-001).
- Highest unsupported score: **0.2924** ("I want to cancel an order I placed
  this morning" → KB-002).
- On this set there is a **clean separation**: any threshold in
  **(0.2924, 0.4116]** (gap 0.1193) gives 0 false rejections, 0 false
  acceptances and 0 wrong-article acceptances.
- The three highest unsupported scores are all refund or cancellation
  questions scored against KB-002 (0.1644–0.2924). Every account question
  (email, phone, address, closing the account, 2FA, username) scored
  ≤ 0.0819.
- Keyword retrieval finds only 3/10 of these paraphrased answerable questions,
  which shows why the semantic and rerank signals matter.

### What the two runs mean together

- The two sets **disagree** on where a threshold could go. On the original
  set, "Can I cancel my order?" scores 0.4746, which is above the holdout
  set's entire clean range. So no single value works on both sets.
- The holdout set was looked at once this run was made, so it can no longer
  serve as independent validation. The two sets were not combined to pick a
  value.
- Cancellation and refund-adjacent questions are the weak spot. Any future
  validation set should include more of them.

## Optional reranking in the agent (V2 Phase 2, Milestone 5)

### What changed

Reranking is now an **optional step** inside `answer_question()`. It is **off
by default**; with it off, the agent behaves exactly as before.

```
answer_question(question)
  1. empty question?               → hand off "empty_question"
  2. client + rerank settings      → hand off "configuration_error" if invalid
  3. hybrid retrieval              → hand off "retrieval_error" / "no_articles"
  4. rerank (only if enabled)
       success                     → articles re-ordered (and filtered, if a minimum score is set)
       nothing ≥ minimum score     → hand off "below_rerank_threshold"
       API error / timeout / bad response → log a warning, keep the hybrid articles
  5. generate_answer(...)          → unchanged, including the citation check
```

### Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `COHERE_RERANK_ENABLED` | `false` | `true/1/yes` or `false/0/no`. Anything else is an error. |
| `COHERE_RERANK_MODEL` | `rerank-v5.0-pro` | Model sent to Cohere Rerank. |
| `COHERE_RERANK_MIN_SCORE` | unset | 0–1. Unset means "re-order only, never remove". **No production value has been chosen.** |
| `COHERE_RERANK_TIMEOUT_SECONDS` | `5` | Must be greater than 0. |

A blank value means "use the default" (same rule as the rate-limit settings).
An **invalid** value is not ignored: the agent hands off with
`configuration_error` and logs it. Otherwise a typo such as `ture` would
silently leave reranking off while you think it is on.

### Files

- `src/support_agent/rerank.py`: `RerankConfig`, `config_from_env()`,
  `document_text()` and `rerank_articles()`. It never creates a Cohere client
  (the agent passes its existing one), so importing it is free.
- `src/support_agent/agent.py`: two new optional arguments, `reranker` and
  `rerank_config` (fakes can be passed in tests, like `retriever`), and the
  step shown above.
- `answer.py`, `hybrid_kb.py`, `semantic_kb.py`, `api.py` and the web page are
  unchanged. The web page only looks at `escalate`, so the new reason code
  needs no frontend change.

### How `rerank_articles()` works

1. Sends the question and one document per article to `client.rerank()`. The
   document text is `"Title: <title>\nText: <text>"`, exactly the format used
   in the experiments, so the saved scores apply. A test checks this.
2. Uses `request_options={"max_retries": 0, "timeout": <seconds>}`. This was
   checked in the installed SDK (cohere 7.3.0): `RequestOptions` has `timeout`
   (seconds, float) and `max_retries`. Without them the SDK waits up to 300 s
   and silently retries 429/5xx/408/409 and connection errors twice.
3. Checks the response: every article must appear exactly once with a valid
   index and a finite numeric score; otherwise it raises `RerankError`.
4. Returns **copies** of the articles, highest score first, with every
   original field kept and a `rerank_score` added. With a minimum score,
   articles below it are dropped (a score equal to the minimum is kept).

`generate_answer` only sends `id`, `title`, `text` and `url` to the chat model,
so the extra field is ignored. Because citations are only accepted for the
articles passed in, **an article removed by the minimum score can never be
cited**.

### Failure policy: fail open

If the rerank request fails (API error, rate limit, timeout, malformed
response), the agent **continues with the original hybrid articles in their
original order**. Why:

- Reranking is an extra improvement, not the only safety check. The LLM
  citation check still runs on the hybrid articles, which is the same
  protection the agent had before reranking existed.
- Turning an outage into a hand-off would report "we can't answer this" when
  the real problem is infrastructure. That would hide the outage and look like
  an unsupported question.
- Cost of this choice: while rerank is down, the extra filtering is lost.
  The warning log makes that visible.

`below_rerank_threshold` is used **only** when reranking succeeded and every
article scored below the configured minimum.

Logs contain the exception type and article IDs only: never the question, an
API response body or a key. Customers only ever see an answer or the fixed
hand-off message.

### Offline evaluation

`python evals/evaluate_rerank_integration.py` sends every question through the
**real** `answer_question()`:

- the real `hybrid_kb.retrieve_hybrid`, with semantic scores computed from
  saved embeddings;
- a replay client whose `rerank()` returns the saved Cohere scores. If the
  replay cannot reproduce a saved result it stops with an error, so the
  agent's fail-open behaviour can't hide a broken evaluation;
- a recording generator in place of the LLM.

The holdout questions needed query embeddings for the hybrid step. These were
created with **one** batched embed call (`--embed-holdout`, 20 questions,
`embed-v4.0`, `search_query`, retries off) and saved in
`evals/fixtures/semantic_holdout_embeddings.json`. That file also holds hashes
of the knowledge base and the questions. Article embeddings are reused from
`semantic_embeddings.json`. No chat or rerank calls were made.

Metrics count the articles that would reach answer generation:

- "false reject": an answerable question handed off;
- "false accept": an unsupported question that still reaches the LLM;
- "wrong article": a non-expected article passed along with an answerable question.

**Original (17 questions: 10 answerable, 7 unsupported)**

| Configuration | Top-1 | False reject | False accept | Wrong article |
| --- | --- | --- | --- | --- |
| Hybrid only | 10/10 | 0/10 | 7/7 | 10/10 |
| Hybrid + rerank | 10/10 | 0/10 | 7/7 | 10/10 |
| min 0.10 | 10/10 | 0/10 | 1/7 | 4/10 |
| min 0.20 | 10/10 | 0/10 | 1/7 | 1/10 |
| min 0.30 | 10/10 | 0/10 | 1/7 | 0/10 |
| min 0.40 | 10/10 | 0/10 | 1/7 | 0/10 |
| min 0.50 | 8/10 | 2/10 | 0/7 | 0/10 |

**Holdout (20 questions: 10 answerable, 10 unsupported)**

| Configuration | Top-1 | False reject | False accept | Wrong article |
| --- | --- | --- | --- | --- |
| Hybrid only | 10/10 | 0/10 | 10/10 | 10/10 |
| Hybrid + rerank | 10/10 | 0/10 | 10/10 | 10/10 |
| min 0.10 | 10/10 | 0/10 | 3/10 | 1/10 |
| min 0.20 | 10/10 | 0/10 | 2/10 | 0/10 |
| min 0.30 | 10/10 | 0/10 | 0/10 | 0/10 |
| min 0.40 | 10/10 | 0/10 | 0/10 | 0/10 |
| min 0.50 | 9/10 | 1/10 | 0/10 | 0/10 |

What this shows:

- Hybrid retrieval always passes all 3 articles, so reranking **without** a
  minimum score only changes the order. Top-1 is already 10/10 on both sets,
  so a reorder-only setup changes the article order but not these metrics.
- Any filtering benefit comes from a minimum score, and no single grid value
  is clean on both sets. The original set's "cancel my order" question
  (0.4746) is still let through at 0.40. On the holdout set, 0.50 already
  rejects an answerable question.

### What this does not show

- The thresholds are a **fixed illustrative grid** (0.10–0.50), not tuned
  values. All 37 questions have been looked at during development, so these
  numbers **do not validate any threshold**. A production value should be
  written down in advance and then tested on a new, unseen question set with
  more refund and cancellation cases.
- The LLM and its citation check are not run in this evaluation. They are a
  second safety gate on top of these numbers.
- Billing units were not returned by the API, so the cost per rerank request
  is unknown. Each allowed `/chat` request makes at most one rerank call, so
  the existing rate limiter also limits rerank usage.
- Adds one request per question: 0.44–1.30 s measured across both experiment
  runs (median about 0.47–0.59 s), capped by the timeout. The chat and embedding calls still use the SDK's 300 s default
  timeout; that is outside this change. (Fixed later: see "External API errors
  and retries (V2 Phase 3, Milestone 2)".)

## Agent decision reliability (V2 Phase 3, Milestone 1)

### The three possible decisions

For every question the agent must do exactly one of:

1. **Answer**, only from knowledge-base articles it was given, with citations
   to those articles and nothing else.
2. **Hand off: not supported.** The evidence is missing or insufficient.
3. **Hand off: failure.** A system or configuration problem prevented safe
   processing.

The customer sees the same fixed `HANDOFF_MESSAGE` for both kinds of hand-off.
`reason` tells them apart for developers and logs.

### Problems found

Every outcome was traced with offline fakes. Seven of them showed the
customer something that should have been handed off:

| Situation | Before | Now |
| --- | --- | --- |
| Chat finished with `ERROR` or `TIMEOUT` but returned cited partial text | **answered** | `llm_error` |
| Chat response had no finish reason (or `TOOL_CALL`, or an unknown value) | **answered** | `malformed_response` |
| Model cited a supplied article **and** an article we never supplied | **answered** (unknown ID silently dropped) | `invalid_citations` |
| Model wrote `INSUFFICIENT\_EVIDENCE` (markdown), `Insufficient evidence.` etc. and cited something | **answered, showing that text** | `insufficient_evidence` |
| Same marker variants without citations | `no_citations` (wrong reason) | `insufficient_evidence` |
| A custom or buggy generator returned a source that was never supplied | **answered** | `invalid_citations` |
| A generator hand-off carried its own text (e.g. an error message) | **shown to the customer** | standard hand-off message |
| Retrieval returned an article missing a field | `llm_error` (wrong layer) | `retrieval_error` |
| Retrieval returned two articles with the same ID | **answered** | `retrieval_error` |

### What changed

**`answer.py` (`generate_answer`)**

- **Finish reason is checked first.**
  - `COMPLETE` and `STOP_SEQUENCE` are the only normal endings.
  - `MAX_TOKENS` gives `incomplete_response`.
  - `ERROR` and `TIMEOUT` give `llm_error` (Cohere failed).
  - Anything else, including a missing value, gives `malformed_response`.

  Text from an unfinished generation is never shown.
- **The insufficient-evidence marker is matched loosely**: any case, spaces,
  hyphens, underscores or markdown escaping. Wrongly handing off is safer than
  wrongly answering, so loose matching errs in the safe direction.
- **Any citation of an unsupplied document rejects the whole answer**
  (`invalid_citations`). Dropping only that source would still show the text
  it was attached to, and that text isn't grounded in our articles.
  `no_citations` now only means "the model cited nothing".
- **Cohere documents are built before the API call.** A malformed article is
  now a data error, not something that looks like an LLM failure.

**`agent.py` (`answer_question`)**

- **Retrieved articles are checked before use.** There must be a list. Every
  article must have non-empty `id`, `title`, `text` and `url` strings, and
  IDs must be unique. Anything else is `retrieval_error`, with no chat call.
  Extra fields such as scores are fine.
- **Reranker output must come from the retrieved articles.** The output has
  to be valid articles whose IDs were all retrieved. Otherwise it's treated
  as a rerank failure: fail open to the hybrid articles, as before.
- **A final gate on every generator result.** It doesn't depend on
  `answer.py`, so it also protects against custom generators and future bugs.
  - **A hand-off** always becomes the standard message with no sources. The
    reason is kept; a missing reason becomes `agent_error`.
  - **An answer** needs non-empty text (otherwise `malformed_response`) and
    at least one source (otherwise `no_citations`).
  - **Every source** must match a supplied article by ID, title and URL
    (otherwise `invalid_citations`).
  - **Anything that isn't a `SupportAnswer`** is `agent_error`.

**Unchanged:**
- the `SupportAnswer` contract;
- `HANDOFF_MESSAGE`;
- `api.py` and the frontend (which only reads `escalate`);
- reranking behaviour and its off-by-default setting;
- the rerank evaluation results.

The evaluation's recording generator now returns a well-formed placeholder
answer, so the new final gate accepts it.

### The full decision flow

```
answer_question(question)
  empty question                               → empty_question          (input)
  client / settings unavailable or invalid     → configuration_error     (failure)
  retrieval raised                             → retrieval_error         (failure)
  nothing retrieved                            → no_articles             (not supported)
  retrieved articles malformed                 → retrieval_error         (failure)
  rerank enabled:
    request fails / bad output                 → keep hybrid articles (fail open, logged)
    nothing ≥ minimum score                    → below_rerank_threshold  (not supported)
  generate_answer:
    chat API error                             → llm_error               (failure)
    finish ERROR / TIMEOUT                     → llm_error               (failure)
    finish MAX_TOKENS                          → incomplete_response     (failure)
    other finish reason, or no text            → malformed_response      (failure)
    INSUFFICIENT_EVIDENCE (any variant)        → insufficient_evidence   (not supported)
    cites an unsupplied document               → invalid_citations       (failure)
    cites nothing                              → no_citations            (not supported)
    generator raised                           → agent_error             (failure)
  final gate on the result                     → invalid_citations / no_citations /
                                                 malformed_response / agent_error
  otherwise                                    → answered
  (api.py: any exception escaping the agent    → HTTP 500, server_error)
```

`invalid_citations` counts as a failure because the response doesn't match
what we sent. `no_citations` counts as "not supported" because the model
declined to ground an answer.

### Tests

- `tests/test_agent_decisions.py` (new, no `cohere` import):
  - a table test that runs all 13 outcomes above through `answer_question`
    with the real `generate_answer`, and checks that no secret appears in the
    answer or the logs;
  - malformed retrieval results;
  - invalid reranker output;
  - every rule of the final gate.
- `tests/test_answer.py`: finish reasons, marker variants, citations without
  an ID, and malformed articles raising before any API call. Two tests now
  expect `invalid_citations` instead of the old "drop the unknown ID"
  behaviour.
- Test fakes that "answered" without sources were given a real source (they
  would otherwise be correctly rejected by the final gate).

### Remaining limitations

- **Citation coverage isn't measured.** An answer with one valid citation
  could still contain uncited sentences. Cohere citations cover phrases, not
  whole answers, and choosing a coverage threshold needs real data.
- **`llm_error` covers every chat failure**: invalid key, rate limit, outage
  and timeout. The log has the exception type, but the reason doesn't
  distinguish them.
- **Chat and embedding calls still use the SDK's 300-second timeout and 2
  automatic retries.** That's error handling, outside this milestone.
  (Fixed in Milestone 2, next section.)
- **Hand-off decisions aren't logged with their reason** in one consistent
  place yet (observability).
- **Loose marker matching** means a genuine answer containing the words
  "insufficient evidence" would be handed off. This is the deliberately safe
  direction.

## External API errors and retries (V2 Phase 3, Milestone 2)

Each question makes up to three kinds of Cohere request:

| Request | Where | When |
| --- | --- | --- |
| Embed the question (`search_query`) | `semantic_kb.retrieve_semantic` | every question |
| Embed all articles (`search_document`) | `semantic_kb._embedded_articles` | first question per process, and again after an article's text changes (cached after success) |
| Rerank | `rerank.rerank_articles` | only if reranking is enabled |
| Chat | `answer.generate_answer` | every question that reaches answer generation |

### What the SDK did on its own (checked in cohere 7.3.0)

- **Timeout: 300 seconds** per request when none is given.
- **Up to 2 automatic retries**, invisible to us, for 429, 408, 409, every 5xx
  (even 501 "not implemented") and dropped connections.
- **Before retrying a 429 it waits for the `Retry-After` header, up to
  60 seconds each time.**
- A read timeout is not retried.
- The SDK's error text (`str(ApiError)`) includes the response headers and
  body, so it must never be logged.

So in the worst case one chat request could keep a customer waiting for
minutes (3 × 300 s plus waits). The only request with explicit settings was
rerank (`timeout=5`, `max_retries=0`), added in Phase 2.

### The policy (`src/support_agent/cohere_calls.py`)

Every chat and embed request now passes
`request_options={"timeout": <seconds left>, "max_retries": 0}`, both
verified in the installed SDK's `RequestOptions`. The SDK never retries by
itself; one small function, `call_with_retries`, decides:

| Failure | Category | Retried? | Why |
| --- | --- | --- | --- |
| 429 | `rate_limit` | once, if `Retry-After` ≤ 2 s (or absent: wait 1 s) | usually short-lived; a long wait is worse than a hand-off |
| 500, 502, 503, 504 | `server_error` | once, after 1 s | usually temporary |
| connection could not be opened: `httpx.ConnectError`, `httpx.ConnectTimeout`, `ConnectionRefusedError` | `connect_error` | once, after 1 s | the request was never sent |
| open connection failed: `httpx.ReadError`, `WriteError`, `CloseError`, `RemoteProtocolError`; `ConnectionResetError`, `ConnectionAbortedError`, `BrokenPipeError` | `connection_lost` | no | Cohere may already have processed the request |
| read/write/pool timeout, 408 | `timeout` | no | the time budget is already used up |
| 401, 403, 498 | `auth_error` | no, logged as ERROR | a bad key won't fix itself |
| other 4xx | `invalid_request` | no, logged as ERROR | a bug in our request |
| unreadable 2xx response | `malformed_response` | no | |
| anything else (including 501) | `unknown` | no | |

**Time budget.** `COHERE_CHAT_TIMEOUT_SECONDS` (default 25) and
`COHERE_EMBED_TIMEOUT_SECONDS` (default 10) are the most time the whole call
may take, *including* a retry and the wait before it. The retry only gets the
time that is left, and it only happens if at least half the budget remains
after the wait. Example (chat, 25 s): a 503 after 3 s → wait 1 s → retry with
21 s. A 503 after 13 s → no retry (only 11 s would be left).

Worst case for one question is therefore about 10 s (query embedding) +
10 s (article embeddings, first question only) + 5 s (rerank, if enabled) +
25 s (chat) = 50 s, instead of many minutes. Values must be above 0 and at
most 120; an invalid value makes the agent hand off with
`configuration_error` before any request.

**Duplicate requests.** A retry after a rate limit or a connection that
could not be opened cannot duplicate work: Cohere never processed the first
request. After a 5xx it might have, so the worst case is one extra billed
request; a chat or embed request has no other side effect. Timeouts are not
retried partly for this reason: the request may still be running on Cohere's
side.

**Which connection failures are safe to retry** (checked in httpx 0.28.1 /
httpcore 1.0.9, and by experiment). An experiment pointed the real
`cohere.ClientV2` at fake servers on 127.0.0.1 (no Cohere traffic), and the
server recorded whether it had received the whole request:

| What the server did | Exception | Server had the full request? |
| --- | --- | --- |
| nothing listening (refused) | `ConnectError` (about 2 s on Windows), or `ConnectTimeout` if the timeout is shorter | never connected |
| reset or closed before reading | `ReadError` | no |
| read the request, then reset | `ReadError` | **yes** |
| read the request, then closed | `RemoteProtocolError` | **yes** |
| read the request, sent half a response | `RemoteProtocolError` | **yes** |
| read the request, never answered | `ReadTimeout` | **yes** |

- `ConnectError` and `ConnectTimeout` come only from httpcore's socket
  backend, while opening the TCP connection or doing the TLS handshake, before
  any HTTP bytes are written. They reliably mean "not sent", so they are
  retried.
- `ReadError` appears both when the server never saw the request and when it
  had the whole request. The exception can't tell us which, so these
  failures are not retried.
- httpcore even swallows a `WriteError` during the request write and tries
  to read a response, so a failed send also surfaces as a read error.
- Cohere has no idempotency key to make a repeat safe.
- The SDK would normally retry `RemoteProtocolError` itself; `max_retries=0`
  turns that off, and a test with the real client over an in-memory transport
  checks it.

**No leaks.** Failures are raised as `CohereCallError`, whose message is only
"Cohere chat request failed: rate_limit (attempts: 2)". It is raised
`from None`, so the provider's exception (with headers and body) isn't
attached to tracebacks. Each failed attempt is logged with the operation,
attempt number, category, exception type and status code, never the
message.

### What the customer sees

Unchanged. Any chat failure still hands off with `reason="llm_error"`, and
any embedding failure with `reason="retrieval_error"`, using the standard
hand-off message. The `SupportAnswer` contract, `api.py` and the web page are
unchanged. The categories are in the logs, for the operator, rather than in
the customer response.

### Other fixes

- **Short embedding responses were accepted silently.** If Cohere returned
  fewer vectors than articles, `zip()` dropped the extra articles and the
  result was cached for the life of the process. `_embed` now checks there is
  exactly one non-empty vector per text and raises
  `CohereCallError("embed", "malformed_response")` otherwise. A failure is
  never cached, so the next question tries again.
- Reranking is unchanged (`timeout` from `COHERE_RERANK_TIMEOUT_SECONDS`,
  no retry, fail open).

### Tests

- `tests/test_cohere_calls.py` (106 tests, offline). The errors are the
  installed SDK's real exception classes (`TooManyRequestsError`,
  `UnauthorizedError`, `ServiceUnavailableError`, `ParsingError`, ...) and
  httpx's (`ReadTimeout`, `ConnectError`, ...), raised by fake clients. They
  cover:
  - classification of 32 error types, including "never sent" connection
    failures (retried) versus "maybe processed" ones (not retried);
  - the real `cohere.ClientV2` over an in-memory `httpx.MockTransport`: the
    request reaches the transport twice for `ConnectError`/`ConnectTimeout`
    and once for `ReadError`, `WriteError`, `CloseError` and
    `RemoteProtocolError`;
  - which errors are retried, and never more than once;
  - `Retry-After` handling;
  - the time budget (with a fake clock), including a worst-case test where
    every attempt uses its full timeout: the total is exactly the budget;
  - no secret in the error, the logs or the customer answer;
  - timeout settings and their validation (`configuration_error`);
  - `generate_answer` and `answer_question` with failing chat clients;
  - `semantic_kb` imported with a fake `cohere` module and a no-op `dotenv`,
    so `.env` is never read and no real client exists: request options,
    recovery, failures and the short-response fix.
- `tests/conftest.py` replaces the retry wait with a no-op in every test, so
  no test really sleeps.

### Remaining limitations

- **The timeout is per network phase, not wall-clock.** httpx applies it to
  connecting and to each wait for data; a non-streaming response sends
  nothing until it's done, so in practice it bounds the request, but a server
  that trickles bytes slowly could take longer.
- **Retries are per process and per request.** There is no circuit breaker:
  during a long outage every question still tries (and retries) once.
- **Hand-off reasons don't distinguish the category.** A rate limit and a bad
  key both reach the customer as `llm_error` (or `retrieval_error`). This is
  on purpose (contract unchanged); the logs have the category.
- **`Retry-After` is only read as a number of seconds**; a date is treated as
  "not given" (wait 1 s).
- Evaluation scripts that call `semantic_kb.client.embed` directly are
  unchanged: `calibrate_semantic.py` (question embeddings) still uses the SDK's
  timeout and retries, and `evaluate_rerank_integration.py --embed-holdout`
  uses `max_retries=0` with the SDK's timeout. They are manual, one-off tools. If a retry
  happens inside `_embedded_articles()` during calibration, it counts toward
  that script's embed-call limit (which then refuses rather than spending
  more).

## Agent pipeline reliability (V2 Phase 3, Milestone 3)

### The pipeline, stage by stage

| # | Stage | Receives → returns | On failure |
| --- | --- | --- | --- |
| 0 | `api.py` `/chat` | JSON → `ChatRequest` (stripped, 1–1000 chars) | 422; any exception from the agent → 500 `server_error` hand-off |
| 1 | Input check | question → continue | `empty_question` |
| 2 | Configuration | client, rerank settings, timeouts | `configuration_error` |
| 3 | Retrieval: `retrieve_hybrid` = `kb.retrieve` (keyword; re-reads `data/kb.json`) + `retrieve_semantic` (embeds the question; compares with article embeddings) → RRF fusion | question → list of article dicts | exception → `retrieval_error`; nothing → `no_articles`; wrong type / missing fields / duplicate IDs → `retrieval_error` |
| 4 | Rerank (optional) | articles → re-ordered subset | exception or invalid output → **fail open** (keep stage-3 articles); everything below the minimum score → `below_rerank_threshold` |
| 5 | `generate_answer` | question + articles → `SupportAnswer` | chat failure → `llm_error`; unfinished → `incomplete_response`; unreadable → `malformed_response`; marker → `insufficient_evidence`; citations → `invalid_citations` / `no_citations` |
| 6 | Final check `_checked_result` | `SupportAnswer` → `SupportAnswer` | any inconsistency → hand-off (see below) |

### What the audit found

Each suspicion was confirmed with an offline probe before any change.

| Problem | Before | Now |
| --- | --- | --- |
| A chat response whose citations, sources, document ID or finish reason had an unexpected type (e.g. `citations=5`) | `TypeError` escaped `generate_answer` → `agent_error` (the same kind of failure as "no text", which gives `malformed_response`) | `malformed_response` |
| A generator result whose `sources` is not a list (e.g. `5`) | **`answer_question` itself raised `TypeError`** (CLI traceback; API 500) | `invalid_citations` |
| An answer (`escalate=False`) labelled `reason="llm_error"` or `None` | passed through to the customer | `agent_error` |
| `escalate=None` (or any non-bool) | passed through; the API's response model then failed → 500 | `agent_error` |
| A reranker returning the same IDs with changed title, text or URL | changed text sent to the answer generator | fails open to the retrieved articles |
| **Editing `data/kb.json` while the app runs** | keyword retrieval saw the new file, but semantic retrieval kept the first-loaded articles (old text **and URL**) until restart | both halves see the current file; articles are embedded again only when their text changes |
| Empty knowledge base | an embed request with an empty list was still sent | no document embedding request; `no_articles` |

**Already handled correctly (left as they were):**
- A failure in configuration, retrieval, reranking or generation gives a hand-off, never an exception.
- A retrieval failure never becomes an empty "successful" answer: an exception gives `retrieval_error`, an empty result gives `no_articles`, and the chat model is not called in either case.
- A broken `kb.json` (invalid JSON, missing file, missing fields, duplicate IDs) gives `retrieval_error`. Only the exception type is logged, not the message, which contains the file path.
- Reranking fails open to valid hybrid results.
- Logs carry only exception types and categories.
- `api.py` turns anything unexpected into a 500 response with the standard hand-off body.

### Changes

- `answer.py`: reading the chat response (`_read_response`) is isolated. Any
  error while reading it is `malformed_response`. An unexpected finish reason
  is logged shortened to 40 characters.
- `agent.py` final check:
  - `escalate` must be a bool;
  - an answer must have `reason="answered"`;
  - `sources` must be a list or tuple;
  - if the check itself crashes, the result is an `agent_error` hand-off,
    never an unchecked answer.
- `agent.py` rerank boundary: every reranked article must equal a retrieved
  one in `id`, `title`, `text` and `url`. Extra fields such as `rerank_score`
  are fine.
- `semantic_kb.py`: the `lru_cache` is replaced by a cache keyed on the
  embedded text (title + text of every article). Articles are re-read on every
  question, as keyword retrieval already did. A URL-only edit uses the cached
  vectors with the new URL; a text edit costs one new embed request.
- `evals/evaluate_rerank_integration.py`: its recording generator now returns
  `reason="answered"`, which the final check requires, and its tally maps that
  back to "sent to generator". The output is byte-identical.

The customer-facing contract, the hand-off message, the retry policy and the
rerank settings are unchanged.

### Tests

`tests/test_agent_pipeline.py` (44 tests, offline) covers:
- an unexpected exception at each stage, with a fake secret, a file path and
  "Traceback" never reaching the answer or the logs;
- retrieval returning nothing or the wrong type;
- unreadable chat responses;
- every new final-check rule, including a result that crashes the check;
- rerankers that change articles, and valid re-ordering;
- the **real hybrid retrieval** against a temporary `kb.json` with a fake Cohere
  SDK: a broken KB, an embedding outage, edits being picked up, when articles
  are embedded again, and an empty KB.

The fake-SDK `semantic_kb` fixture moved from `tests/test_cohere_calls.py`
to `tests/conftest.py` so both files share it.

### Remaining limitations

- **An embedding outage stops every question.** Keyword retrieval works
  without Cohere, but hybrid retrieval currently fails as a whole
  (`retrieval_error`). Falling back to keyword-only results would be a
  behaviour change; it is not done here.
- **The client passed to `answer_question` isn't used for embeddings.**
  `semantic_kb` has its own client, created at import. A test or caller that
  passes a fake client but uses the default retriever still gets the real
  embedding client.
- **One question reads `kb.json` twice** (keyword and semantic). An edit
  landing between the two reads could mix versions for that one question.
  The next question is consistent.
- The final check doesn't look for the "insufficient evidence" marker itself.
  `generate_answer` does; a custom generator would have to as well.

## Startup validation and readiness (V2 Phase 4, Milestone 1)

**The problem.** `/health` always said "ok", and it was Render's health check.
A deploy with no API key, a typo in a setting or a broken `kb.json` started
normally and replaced the working version; then every question was handed off
(`configuration_error` or `retrieval_error`) while monitoring stayed green.

### How it works

```
uvicorn starts --> import api.py --> lifespan startup
                                      startup.check_startup()
                                        load_dotenv()      (local .env; real env vars win)
                                        find_problems()    (every check, all problems collected)
                                      problems? --yes--> raise StartupError
                                                         Uvicorn: "Application startup failed. Exiting." (exit code 3)
                                         no
                                      app.state.ready = True --> serve requests
                                      (shutdown) app.state.ready = False
```

`find_problems()` (`src/support_agent/startup.py`) **reuses the validators the
running app already uses**, so every rule exists in one place:

| What | Checked by (existing code) |
| --- | --- |
| `COHERE_API_KEY` present and not blank | `agent.api_key_from_env` (now also used by `_default_client`) |
| `COHERE_RERANK_ENABLED / _MODEL / _MIN_SCORE / _TIMEOUT_SECONDS` | `rerank.config_from_env` |
| `COHERE_CHAT_TIMEOUT_SECONDS`, `COHERE_EMBED_TIMEOUT_SECONDS` | `cohere_calls.chat_timeout` / `embed_timeout` |
| `CHAT_RATE_LIMIT_PER_MINUTE / _PER_DAY` | `rate_limit.limiter_from_env` |
| `LOG_LEVEL` (added in Milestone 2) | `log_config.log_level_from_env` |
| `data/kb.json` exists, is valid JSON, every article valid | `kb.load_articles` |
| ...and has at least one article | new rule: an empty KB can never answer anything |

So the same functions can check a given set of settings (tests pass a plain
dict), `limiter_from_env` gained the optional `environ` argument that the
rerank and timeout readers already had.

**The `/chat` rate limiter is created at startup, after `.env` is loaded.**
It used to be created when `api.py` was imported, which is before anything
loads `.env`, so a *valid* `CHAT_RATE_LIMIT_*` in a local `.env` was silently
ignored while an *invalid* one (now validated after loading `.env`) stopped
startup. Now the lifespan runs `check_startup()` (one `load_dotenv()`, then
validation) and only then `chat_limiter = limiter_from_env()`, so the limiter
uses exactly the settings that were validated. On Render nothing changes:
real environment variables are already set, and python-dotenv never
overrides them. Until startup has run, `chat_limiter` is `None` and `/chat`
returns `503` without calling the agent, so Cohere can never be called
without a limit.

### `/health` versus `/ready`

| | `/health` (liveness) | `/ready` (readiness) |
| --- | --- | --- |
| Question it answers | Is the process running? | Has startup completed? (Validation passed; since Milestone 5 also the limiter and Cohere clients created.) |
| Response | always `200 {"status":"ok"}` | `200 {"status":"ready"}` or `503 {"status":"not_ready"}` |
| Work per request | none | reads one flag (`app.state.ready`) |

`/ready` returns 503 before startup finishes, after shutdown begins, and when
the app is served without its lifecycle (for example a `TestClient` used
without `with`).

### Design decisions

- **Fail startup, don't just report "not ready".** A misconfigured process
  that stays up can still receive traffic. Exiting makes the deploy fail on
  Render, so the previous working version keeps serving. `/ready` is the
  second line of defence for the time before startup completes.
- **Validate once, at startup.** Environment variables can't change while a
  process runs, and `kb.json` only changes with a deploy, so re-checking on
  every probe would only add work. A probe is instant and never reads files.
- **No Cohere call, no Cohere import.** (The validation step. Since
  Milestone 5 the clients are created right after it, still without any
  request; see "Startup and shutdown lifecycle".) The check confirms the key is *set*,
  not *valid*. Calling Cohere from a readiness probe would cost money on every
  probe, and a Cohere outage would mark the service not ready (and could make
  the platform restart it), which doesn't help: the app already hands off
  safely when Cohere fails.
- **Report every problem at once,** so one failed deploy shows the whole list.
- **No values in errors.** Messages name the setting and the rule
  ("`COHERE_RERANK_ENABLED must be true or false`"), never its value. The
  existing validators used to add `got '<value>'`; that was removed, because a
  key pasted into the wrong variable would have been printed in the deploy
  log. Unexpected exceptions are reported by type only, and `StartupError` is
  raised outside any `except` block, so no chained traceback repeats an
  earlier message.
- **`.env` is loaded first** (as the running app already does), so local
  development behaves like production. In tests, an autouse fixture in
  `tests/conftest.py` replaces `startup.load_dotenv`, so the real `.env` is
  never read.

### Tests (`tests/test_startup.py`, 69)

Valid configuration (defaults, non-defaults, blank values), missing or blank
key, each invalid rerank, timeout and rate-limit setting, missing, malformed
and empty knowledge base, the shipped `kb.json`, all problems reported
together, no setting value or key in any error or log, the app lifecycle
(ready after startup, startup failing on bad settings or a broken KB, not
ready before startup and after shutdown), probes doing no work, a minimal
`/ready` body, and no Cohere import or network access during startup or
probes (`cohere` made unimportable, host-name lookup and connections made to
fail). Rate limits from a temporary `.env` file (parsed by the real
python-dotenv, never the project's `.env`): valid values applied and enforced
on `/chat`, blank means default, real environment variables win, invalid
values fail startup without showing the value, importing `api.py` doesn't read
them, and `/chat` returns 503 before startup.

### Remaining limitations

- **The key's validity isn't checked.** A wrong or revoked key passes startup;
  questions then hand off (`llm_error` / `retrieval_error`). Checking it needs
  a Cohere call, which this design avoids on purpose.
- **Model names aren't checked** (`COHERE_CHAT_MODEL`, `COHERE_RERANK_MODEL`)
  for the same reason.
- **Readiness isn't re-evaluated** after startup. A Cohere outage or an
  exhausted quota doesn't show in `/ready` (by design, see above).
- **Render's health check path is still `/health`** until it is changed in
  the dashboard.

## Logging and request-level observability (V2 Phase 4, Milestone 2)

### What the audit found

- **App logs weren't configured under Uvicorn.** Uvicorn's logging setup only
  covers its own `uvicorn.*` loggers. `support_agent.*` records went up to the
  root logger, which has no handler, so Python's "last resort" handler printed
  WARNING and ERROR as bare messages (no time, level or source) and **dropped
  INFO** (for example the `below_rerank_threshold` line).
- **No per-request outcome.** The access log says `POST /chat 200` for both
  answers and hand-offs. Expected hand-offs logged nothing; there was no
  request ID and no duration.
- **No way to connect the lines of one request** (agent, retry, rerank).
- **Sensitive data was already handled well** (verified, kept): questions,
  answers and provider error text are never logged; errors are logged as an
  exception type or a category such as `rate_limit (chat)`. The one
  provider-supplied value, an unknown `finish_reason`, is logged as a
  truncated `repr`. Uvicorn's access log includes the client IP address; that
  is Uvicorn's standard behaviour and was left unchanged.

### Configuration (`src/support_agent/log_config.py`)

- `LOG_LEVEL` (default `INFO`, any case) sets the level of the
  `support_agent` logger. It is validated with the other settings at startup
  (`startup.find_problems`); anything else stops startup with
  `LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL`, without
  the value.
- At startup, after validation, `configure_logging()` gives the
  `support_agent` logger **one** stderr handler with the format
  `%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s`.
- **Only our logger is configured.** The root logger and Uvicorn's loggers are
  untouched, so Uvicorn's lines look exactly as before. `propagate = False`
  means that if a host or tool ever adds a root handler, our lines still
  appear once, not twice. Calling it again (a restart in the same process)
  replaces the handler instead of adding a second one.
- **Why INFO by default:** the per-request outcome line is INFO, and it's the
  main thing to read in Render's logs. `WARNING` keeps only problems.

### Request lifecycle

```
request ─► RequestContextMiddleware (plain ASGI middleware, every HTTP request)
             request_id = valid client X-Request-ID, or uuid4().hex
             request_id_var.set(request_id)     (contextvar: visible to every log line,
                                                 including the worker thread running /chat)
             start = clock()
           ─► FastAPI ─► /chat endpoint
                 records its outcome in the request scope:
                   503 not_ready · 429 rate_limited (+ limit) ·
                   200 reason/escalate/sources · 500 server_error (+ error type)
           ◄─ response: X-Request-ID header added
           finally (path /chat only): ONE outcome line, level by status:
             INFO (2xx, 4xx) · WARNING (429, 503) · ERROR (other 5xx)
```

- **Client-supplied IDs** are used only if they are 1–64 characters of
  letters, digits, `.`, `_`, `:` or `-`. Anything else (spaces, quotes, `<`,
  non-ASCII, longer) is replaced, so a header can't inject text or line breaks
  into the logs.
- **Rejected before the agent runs:** a 422 (validation) or 405 never reaches
  the endpoint, so the middleware derives the reason from the status
  (`invalid_request`, `method_not_allowed`). 429 and 503 are recorded by the
  endpoint.
- **Exactly once:** only the middleware writes the outcome. The old separate
  "rate limit reached" warning and "unexpected error in /chat" error were
  folded into the outcome line (`limit=minute`, `error=RuntimeError`).
- **Unexpected exceptions:** the endpoint already turns agent failures into a
  safe 500. If something outside that `try` fails (the limiter, response
  validation), the middleware sends the same safe hand-off body with
  `X-Request-ID` and logs one ERROR outcome with only the exception type. If
  the response had already started, it can't be replaced; the exception is
  re-raised to Uvicorn after logging.
- **Safe values only:** `reason` is logged only if it looks like an identifier
  (`[a-z0-9_]{1,40}`), otherwise as `other`. The error is a type name only.
  The question, answer, source titles and URLs are never read by the logging
  code.

### Example lines

```
2026-11-17 23:36:06,572 INFO support_agent.api [-] Startup checks passed; ready for requests
2026-11-17 10:15:02,123 INFO support_agent.api [3f2a9c...] chat_request status=200 escalate=false reason=answered sources=2 duration_ms=2310.4
2026-11-17 10:15:04,010 WARNING support_agent.agent [b71e40...] Retrieval failed: rate_limit (embed)
2026-11-17 10:15:04,011 INFO support_agent.api [b71e40...] chat_request status=200 escalate=true reason=retrieval_error sources=0 duration_ms=1102.7
2026-11-17 10:15:09,870 WARNING support_agent.api [8b01d2...] chat_request status=429 escalate=- reason=rate_limited sources=0 duration_ms=0.4 limit=minute
2026-11-17 10:16:00,001 ERROR support_agent.api [c0ffee...] chat_request status=500 escalate=true reason=server_error sources=0 duration_ms=12.0 error=RuntimeError
```

### Tests (`tests/test_logging.py`, 59)

`LOG_LEVEL` parsing and its value-free error (also through startup), the line
format, the level actually applied, one handler and untouched root and Uvicorn
loggers, request IDs generated, unique, returned and used in logs, valid and
unsafe client IDs, the header never duplicated, other endpoints getting an ID
but no outcome line, outcomes for answers, nine hand-off reasons, 422, 405,
429 (minute and day), 503, agent exceptions and exceptions outside the agent,
exact duration with a fake clock, an odd `reason` logged as `other`, agent
log lines carrying the request ID from the worker thread, and the question,
answer, URL, key and provider message never appearing in the logs (checked
both through `caplog` and in the formatted output of a real startup).

`tests/conftest.py` restores the `support_agent` logger after each test,
because startup sets `propagate = False`, which would otherwise hide later
tests' records from `caplog`.

### Remaining limitations

- **Plain `key=value` text, not JSON.** Easy to read and search in Render's
  log view; a log platform that parses JSON would need a JSON formatter.
- **No metrics.** Answer rate, hand-off rate and latency percentiles have to
  be worked out from the lines; there is no dashboard or alerting.
- **The request ID isn't shown to customers** or included in the response
  body; it is only in the `X-Request-ID` header.
- **Uvicorn's access line and our outcome line are separate** (the access
  line has no request ID). They appear next to each other.
- **The `python -m support_agent.agent` CLI keeps its own simple
  `logging.basicConfig(level=WARNING)` setup.**

## Reproducible builds and CI (V2 Phase 4, Milestone 3)

### What the audit found

- `pyproject.toml` had only ranges (`cohere>=7.3,<8`, `fastapi>=0.115,<1`,
  `uvicorn>=0.30,<1`, `python-dotenv>=1.2`) and there was no lock file. Every
  Render build installed whatever was newest in those ranges *that day*, and
  nothing tested that combination first. The retry code depends on SDK
  details (`status_code`, `request_options`, httpx error classes), so a minor
  release could change behaviour quietly.
- No CI: `.github/` didn't exist. Tests only ran when someone ran them locally.
- The local `.venv` had 42 packages. Following each package's own metadata
  from the project showed that **all 42 are actually needed** (none left over
  from experiments), with no conflicts, and that Linux needs the same set minus
  `colorama` (required by pytest and click only on Windows). Those exact
  versions pass the full suite and `pip check`, and every one has a Linux
  CPython 3.10 wheel (checked with `pip download --platform manylinux...`), so
  Render and CI never compile anything.

### Approach: a constraints file

`constraints.txt` lists `name==version` for every package (direct and
indirect, runtime and test). It is passed with `-c`:

```
Render:  pip install -e . -c constraints.txt
Dev/CI:  pip install -e ".[dev]" -c constraints.txt
```

A **constraints** file (not a requirements file) only fixes the version of a
package *if* something asks for it. So one file serves every case: Render
installs only the runtime packages, dev and CI also get pytest, and
`colorama` is installed only on Windows.

Why not the alternatives:

- **`pip freeze > requirements.txt`**: forces every listed package to be
  installed, including test tools on Render and `colorama` on Linux, and
  loses the "what do we actually depend on" view that `pyproject.toml` gives.
- **pip-tools / uv / Poetry lock files**: would add a new tool to the
  project and to Render's build. The constraints file needs only pip.
- **Exact pins in `pyproject.toml`**: would pin only direct dependencies; the
  ~35 indirect ones would still float.

`pyproject.toml` keeps its ranges (what the code is compatible with);
`constraints.txt` says exactly what is installed. `tests/test_dependencies.py`
keeps them in step:

- every line is an exact `==` pin, listed once;
- every direct dependency is pinned, inside its `pyproject.toml` range;
- every installed dependency of the project (followed through package
  metadata) is pinned, so adding a dependency without pinning it fails;
- installed versions equal the pins (in CI this proves the pinned set was
  installed); the README and the workflow both use `-c constraints.txt`.

### CI (`.github/workflows/tests.yml`)

- Runs on every push and pull request; a newer run on the same branch
  cancels the older one.
- `ubuntu-24.04`, not `ubuntu-latest`: Python 3.10.11 builds are published for
  Ubuntu 24.04 but not 26.04, which `ubuntu-latest` will move to.
- Python from `.python-version` (3.10.11, the same as Render's
  `PYTHON_VERSION`).
- Installs with the same constraints file, prints `pip check` and the
  installed versions, then runs `python -m pytest -q` (pytest finds `src` and
  `tests` through `pytest.ini`).
- **No secrets:** read-only token (`permissions: contents: read`, checkout
  without saved credentials), no Cohere key, `RUN_LIVE_COHERE_TESTS=0`. The
  offline tests need neither: the app's tests use fakes, and startup tests
  never read `.env`.

### Validation done

- Clean install in a fresh Python 3.10.11 venv, from a copy of only the files
  Git would ship (no `.env`), with `COHERE_API_KEY` unset: installed versions
  identical to `constraints.txt` and to the existing `.venv` (no upgrades),
  `pip check` clean, full suite passing, `data/kb.json` found through the
  editable install.
- A runtime-only install (Render's command) started with Uvicorn and a fake
  key: `/ready` returned 200. Startup makes no Cohere call.

### Remaining limitations

- **Render still uses the old build command** until it is changed in the
  dashboard to `pip install -e . -c constraints.txt`.
- **No hashes.** Pins fix versions, not file contents; `--require-hashes`
  would also protect against a tampered package but needs hashes for every
  platform's wheel.
- **Build tools and pip aren't pinned.** `setuptools>=64` (used to build the
  editable install) comes from PyPI at install time, and pip itself is
  whatever the Python install includes. Neither changes the installed app
  packages.
- **CI runs on Linux only;** local development is on Windows (the same
  versions, apart from `colorama`).
- **Updating versions is manual:** edit `constraints.txt`, reinstall, run the
  tests. Nothing proposes updates automatically.
- **The action versions (`checkout@v6`, `setup-python@v6`) are major-version
  tags, not commit SHAs.** setup-python v7 removed end-of-life Python versions
  and 3.10 reached end-of-life in October 2026, so v6 is used until the
  Python upgrade in Milestone 4.

## Python 3.12 runtime (V2 Phase 4, Milestone 4)

**Why:** Python 3.10 reached end-of-life on 2026-10-01 (no more security
fixes). 3.12 receives security fixes until October 2028. The project now
targets **3.12.16**, the newest 3.12 release (October 2026).

### Compatibility audit

- **Dependencies:** every pinned version publishes CPython 3.12 wheels for
  Linux and Windows (checked by downloading all 41 Linux and 42 Windows wheels
  for `cp312`), and none declares a `Requires-Python` that excludes 3.12. No
  version had to change.
- **Platform-specific pins:** on 3.12, `tomli` and `exceptiongroup` are no
  longer installed (pytest, anyio and huggingface_hub request them only below
  Python 3.11). They stay in `constraints.txt`, which only applies a pin when
  a package is requested, so a 3.10 environment still installs the same set.
  `colorama` is still Windows-only.
- **Code and tests:** nothing relies on 3.10 (no `distutils`, `imp`,
  `asyncore`, removed `unittest` aliases or similar). The one version branch,
  `tomllib` with a `tomli` fallback in `tests/test_dependencies.py`, works on
  both.
- **CI:** `actions/setup-python@v6` installs 3.12.16 from `.python-version`.
- **Render:** the `PYTHON_VERSION` environment variable takes precedence over
  `.python-version`, so Render keeps running 3.10.11 until the dashboard
  variable is changed to `3.12.16`.

### How it was tested

No Python 3.12 was installed on the development machine (python.org's last
3.12 Windows installer is 3.12.10; later 3.12 releases are source-only). The
check used Astral's self-contained `python-build-standalone` CPython 3.12.16
build (release `20261110`), unpacked into a temporary folder after its
SHA-256 matched the publisher's `SHA256SUMS` and GitHub's recorded digest.
The system Python and the project's `.venv` were not touched.

In a fresh 3.12.16 venv, from a copy of only the files Git would ship (no
`.env`) and with no Cohere key:

- `pip install -e ".[dev]" -c constraints.txt`: 40 packages, every one at its
  pinned version; `pip check` clean.
- All modules import; importing the API still doesn't load `cohere`.
- `data/kb.json` is found through the editable install (3 articles), and
  startup validation reports no problems.
- Full offline suite: 640 passed, 1 skipped (the live Cohere test), with
  Python warnings shown: none reported.
- Render-style install (`pip install -e . -c constraints.txt`, 36 packages,
  no pytest) started under Uvicorn with a fake key: `/health` and `/ready`
  returned 200, the outcome log line was formatted correctly, and an invalid
  `LOG_LEVEL` stopped startup with the usual value-free error.

### Remaining limitations

- **Render still uses 3.10.11** until `PYTHON_VERSION` is changed to `3.12.16`
  in the dashboard. Change it together with the build command from
  Milestone 3, then check the deploy log and `/ready`.
- **The local `.venv` is still Python 3.10.11.** It keeps working (the
  constraints still cover 3.10), but development should move to a 3.12 venv.
- **Local testing used a Windows build;** CI is the first run on Linux 3.12.16.
- **`requires-python` in `pyproject.toml` is still `>=3.10`.** Raising it to
  `>=3.12` would stop 3.10 installs, so it is left for when 3.10 is no longer
  needed.
- `actions/setup-python` stays at `@v6`. v7 now also supports this Python
  version, but changing it isn't needed for this milestone.

## Startup and shutdown lifecycle (V2 Phase 4, Milestone 5)

### What the audit found

| Question | Before this milestone |
| --- | --- |
| When is the chat client created? | On the first `/chat`: `agent._default_client()` (cached with `lru_cache`) loads `.env`, checks the key, and creates `cohere.ClientV2`. |
| When is the embedding client created? | Also on the first `/chat`: `_hybrid_retrieve` imports `semantic_kb`, which creates its own `ClientV2` at import. |
| When is the knowledge base embedded? | On the first question that reaches semantic retrieval (one embed request), then cached until the articles change. |
| Do imports do expensive work? | Importing the API doesn't load the Cohere SDK. `import cohere` itself is cheap (2–21 ms); the SDK loads its modules when `ClientV2` is first used. |
| What needs closing? | Each `ClientV2` owns one `httpx.Client` (a connection pool). The SDK closes it when a client is used with `with`. Its default thread pool is shared by the whole SDK, isn't used by our calls, and isn't ours to close. |
| Startup failing halfway? | Validation ran first and nothing was created yet, so there was nothing to clean up. |
| In-flight requests at shutdown? | Uvicorn stops accepting connections, waits for running requests to finish (no time limit unless `--timeout-graceful-shutdown` is set), then runs the app's shutdown. Nothing was closed: the OS freed the connections at exit. |

### Measured cost

Fresh processes, real SDK, fake key, nothing sent (checked with an audit hook):

| | Python 3.10.11 | Python 3.12.16 |
| --- | --- | --- |
| Before: startup to ready | 25–142 ms | 35–55 ms |
| Before: client setup paid by the first question | 554–1023 ms (5.6 s once, on a cold disk) | 869–1302 ms |
| After: startup to ready | 599–637 ms | 919–1096 ms |
| After: client setup paid by the first question | 0 ms | 0 ms |

The first question still embeds the knowledge base (one Cohere request): that
is real API work, so it stays lazy.

### Design

**Startup** (`lifespan` in `api.py`):

```
validate settings (no SDK) -> configure logging -> rate limiter
  -> create chat client (agent._default_client) and embedding client (import semantic_kb)
  -> ready = True
```

- Creating the clients sends nothing; it only moves about a second of SDK
  loading from the first customer's question to startup, before `/ready`
  turns true and before Uvicorn opens the port. On Render's free plan, a
  request that wakes the service waits for startup either way, so that
  request is no slower; every other first request after a deploy or restart
  is faster.
- **The knowledge base is not embedded at startup.** That would be a Cohere
  request on every boot and wake-up (cost, and a dependency on Cohere being up
  just to start).
- A client that can't be created now fails startup (a broken SDK install
  shows up in the deploy, not on a customer's question).

**Shutdown:**

```
ready = False -> wait (up to 10 s) for /chat requests still inside the agent
  -> close the chat client (and clear its cache) -> close the embedding client
```

- `close_client()` (`cohere_calls.py`) uses the SDK's own `with` support
  (`__exit__`), which closes the client's httpx pool. No new resource
  abstraction: two existing clients, two close calls.
- **Not while in use:** `/chat` counts requests inside the agent. Normally
  Uvicorn has already waited for them, so the count is 0. If Uvicorn gave up
  on a request (only with `--timeout-graceful-shutdown`, whose worker thread
  keeps running), shutdown waits up to 10 seconds and otherwise leaves the
  clients open with a warning, rather than breaking that request. The OS
  frees the connections when the process exits.
- **Partial startup failure:** client creation runs inside the same
  `try/finally` as shutdown, so anything already created is closed.
- **Errors aren't hidden:** each close is wrapped; a failure is logged as a
  warning with the exception type only, and the original startup or server
  error is what propagates.
- **Readiness** is set to false before anything is closed.

### Tests (`tests/test_lifecycle.py`, 15)

Startup order (validation, logging, limiter, clients, then ready), clients
created at startup with no request and no knowledge-base embedding, no
clients when validation fails, the first real `/chat` reusing the startup
clients, both clients closed after shutdown with the cache cleared, readiness
false while closing, partial startup failure closing the chat client and
re-raising the original error, a failing close neither hiding the startup
error nor leaking its message, a failing close at shutdown logged not raised,
shutdown waiting for a running chat before closing, clients left open (with a
warning) when a chat never finishes, the in-progress count returning to zero
after an error, closing without startup, and `close_client` against the real
installed SDK (its httpx pool ends up closed; no request sent).

Test changes: startup tests now use the fake Cohere SDK from `conftest.py`
(the `semantic_kb` fixture), because startup creates clients by design.
"Startup does not import cohere" became "validation does not import cohere".
`conftest.py` also stops `agent.load_dotenv` reading `.env` and clears the
cached chat client between tests; `FakeEmbedClient` records closing.

Also checked: a real `uvicorn.Server` with the real SDK and a fake key, shut
down the way SIGTERM does it: `/ready` 200 before, then not ready, both httpx
pools closed and the chat-client cache cleared. Full suite on Python 3.10.11
and 3.12.16: 655 passed, 1 skipped.

### Remaining limitations

- **Startup is about 0.6–1.1 s slower** (the SDK loading moved there).
- **The first question still waits for the knowledge-base embedding**
  (one Cohere request), by design.
- **`semantic_kb` can't embed after it is closed.** A server process shuts
  down once, so this only matters if the same process starts the app again
  (tests use fakes; `uvicorn --reload` starts a new process).
- **Hard kills skip cleanup.** If the platform kills the process (for example
  after its shutdown grace period), nothing is closed; the OS frees the
  connections.
- **The 10-second wait is fixed,** not a setting.

### Follow-up fixes (Phase 4 final audit)

- **One key rule for every client.** `semantic_kb` used to read
  `COHERE_API_KEY` itself, untrimmed, while startup and the chat client used
  `agent.api_key_from_env()` (trimmed). A key with a pasted newline would
  have passed startup and worked for chat, but broken every embedding call
  while `/ready` said 200. Now `semantic_kb` uses `api_key_from_env()` too, so
  both clients get the same normalized key, and a missing key raises the same
  `ConfigurationError("COHERE_API_KEY is not set")` (no longer "missing from
  .env": on Render the key comes from the environment). `agent` imports
  `semantic_kb` only inside a function, so this creates no import cycle.
- **Tests synchronised with events, not sleeps.** Two shutdown tests assumed a
  worker thread had started after `time.sleep(0.05)`, which can fail on a slow
  CI machine. Now the worker sets a `threading.Event` once it is inside the
  request, and the "waits" test learns that shutdown has started waiting
  through a stand-in for the api module's `asyncio.sleep`. Every wait has a
  5-second limit, so a broken test fails instead of hanging.
- **Readiness wording** now says what `/ready` actually means since
  Milestone 5: startup has completed (validated, limiter and clients created).

## Knowledge base v2 and evaluation dataset (Phase 5, Milestone 1)

### Why

Three articles cannot show whether retrieval picks the *right* article among
similar ones, and 37 questions about three topics cannot show much about
answer quality. This milestone grows the KB to a realistic size and adds a
question set that is labelled for **expected behaviour**, not just for an
article ID.

### The knowledge base (`data/kb.json`)

34 original, fictional articles (the store and its policies are invented):

| Category | Articles |
| --- | --- |
| `account` | 5 (email, phone, delete account, two-factor, guest checkout) |
| `password_login` | 4 (reset, change, locked account, reset email missing) |
| `orders_shipping` | 4 (tracking, options/costs, destinations, change address) |
| `returns_refunds` | 4 (eligibility, returning, non-returnable items, refund timing) |
| `payments_billing` | 6 (methods, declined, charged twice, invoices, promo codes, gift cards) |
| `cancellations` | 2 (orders, Plus membership) |
| `delivery_problems` | 3 (late, delivered-not-received, damaged/wrong/missing) |
| `product_info` | 3 (warranty, restock alerts, price drop) |
| `technical` | 3 (app/website, checkout errors, missing emails) |

- **Schema.** Same four required fields plus an optional `category`.
  `load_articles()` already allowed extra fields, and nothing in the app reads
  `category` (only the evaluation does), so retrieval, the agent and the API
  are unchanged. Semantic search still embeds only title and text.
- **KB-001..003 kept their ID, title, URL and original sentences**; new
  sentences were only appended. Nothing that was true before became untrue.
- **One policy sheet.** Every number was decided once, then written into each
  article that mentions it: 30 days *of purchase* for refunds, returns and
  shipped-order returns; 7 days *of delivery* to report damaged or wrong
  items; 14 days to report delivered-not-received; Processing lasts about
  1 hour; pending charges clear in 3-5 business days; order confirmation
  within 15 minutes; passwords of at least 10 characters. A test
  (`test_shared_policy_numbers_are_consistent`) fails if these drift apart.
- **Titles use customer words**, because keyword retrieval only keeps
  articles that share a word with the title (V2 Phase 2 rule).

### Keeping the historical experiments reproducible

Every saved fixture (semantic embeddings, rerank results, holdout embeddings)
stores a SHA-256 of the KB it was made from, and the two v1 question sets are
labelled for three articles ("change my email" is *unsupported* there; it is
now answerable). Replacing `data/kb.json` would have made those fixtures fail
their hash check and those labels wrong.

- `evals/fixtures/kb_v1.json` is a **byte-identical copy** of the old KB.
  A test checks its hash against all four fixtures.
- `calibrate_semantic.historical_kb()` points `support_agent.kb.KB_PATH` at
  that copy inside a `with` block. Both `load_articles()` and keyword
  `retrieve()` read `KB_PATH` when called, so the real hybrid retriever used
  by the replay also sees v1.
- The historical scripts (`calibrate_semantic`, `rerank_experiment`,
  `evaluate_rerank_integration`, `evaluate_retrieval`, `compare_retrieval`)
  run inside that block. Their offline output is identical to the numbers
  recorded in the sections above. No fixture or v1 question file changed.
- `tests/test_kb.py`'s keyword tests and the real-fixture tests also read the
  v1 copy, so they still test the same behaviour as before.

### The dataset (`evals/support_eval_v2.json`)

77 questions: **51 answerable, 16 unsupported, 10 ambiguous**. None repeats a
v1 question (checked ignoring case and punctuation).

| Field | Used for | Meaning |
| --- | --- | --- |
| `id` | all | `V2-001` ... |
| `label` | all | `answerable`, `unsupported` or `ambiguous` (expected behaviour) |
| `category` | all | one of the 9 KB categories, or `none` |
| `expected_ids` | answerable | articles **needed** to answer (2 for multi-intent questions) |
| `also_relevant_ids` | answerable | other articles that also contain the answer |
| `answer_facts` | answerable | key facts; each is a list of acceptable spellings |
| `related_ids` | unsupported, ambiguous | topical articles that do **not** settle the question |
| `note` | unsupported, ambiguous | why the label is what it is |
| `tags` | all | edge-case and scope tags (below) |

- **Unsupported** questions carry exactly one scope tag: `out_of_domain` (3,
  including one prompt-injection attempt), `not_covered` (5: a support topic
  with no article, e.g. gift wrapping) or `near_miss` (8: an article covers
  the topic but not the detail, e.g. customs duties for Canada). A test
  checks that the KB never mentions those topics, so the labels stay true
  when articles are edited.
- **Ambiguous** questions ("Can I cancel?", "My code doesn't work") have
  several plausible articles with different answers. The right behaviour is
  to ask or hand off, not to pick one. The agent cannot ask follow-up
  questions yet, so answered ambiguous questions are listed for manual review
  instead of being scored automatically.
- **Edge cases are tags, not a fourth label**, because an edge case still has
  an expected behaviour: `boundary` (8, e.g. day 31 of a 30-day window),
  `paraphrase`, `informal`, `typo`, `negation`, `multi_intent`, `short`,
  `adversarial`.
- `evaluate_kb_v2.validate_cases()` rejects unknown article IDs, missing or
  extra fields, duplicate IDs or queries, a multi-intent question with one
  expected ID, unsupported questions without a scope tag or note, and similar
  labelling mistakes. `validate_kb()` adds ID format, category, and unique
  title and URL checks on top of `load_articles()`.

### Metrics (`evals/evaluate_kb_v2.py`)

**Retrieval**: which articles would reach the LLM. Default K = 3, as in
production. For answerable questions:

- *Expected article @K*: at least one expected article in the top K.
- *Recall@K*: share of expected articles in the top K (averaged).
- *Precision@K*: relevant articles in the top K / K. With one relevant
  article the maximum is 0.33 at K = 3. *Precision of returned* divides by
  the number of articles actually returned instead.
- *Top-1* and *MRR@K* count expected or also-relevant articles.
- Unsupported: share for which **no** article was returned, split by scope.

**Answers**: scored from a recorded run of the real `answer_question()`.

- *Answered* vs *handed off* (a hand-off of an answerable question is a false
  hand-off).
- *Valid citations*: every source exists in the KB with the same title and URL.
- *Supporting article cited*: at least one source is an expected or
  also-relevant article.
- *Key facts*: each `answer_facts` group found as whole words (case, dash and
  apostrophe differences ignored). This is a **proxy**: it misses correct
  paraphrases and cannot see added wrong claims.
- *Correct (proxy)*: answered, supporting article cited and all key facts
  present.
- Unsupported: abstained vs answered. Ambiguous: handed off vs answered
  (answered ones are listed for review).

A recorded run stores hashes of the questions and the KB; scoring refuses a
file made from a different version of either.

### Results so far (keyword retrieval, K = 3, offline)

| Answerable (51) | Value |
| --- | --- |
| Top-1 relevant | 31/51 (61%) |
| Expected article in top 3 | 38/51 (75%) |
| Recall@3 | 0.745 |
| Precision@3 / precision of returned | 0.288 / 0.601 |
| MRR@3 | 0.676 |
| No article returned | 8/51 |

Unsupported: nothing returned for 11/16 (out-of-domain 3/3, not covered
2/5, near miss 6/8). Weakest categories: returns/refunds (2/8) and
orders/shipping (3/7). Typical misses: no stemming ("shipping" vs "Where we
ship"), synonyms ("send it back" vs "Return an item") and stop-word gaps
("Where is my order?" now ranks "Where we ship" first). K = 5 gives the
same hit rate, recall and MRR: the misses are articles keyword retrieval
does not return at all, not articles ranked 4th or 5th.

**Not measured yet:** semantic and hybrid retrieval (they need embedding
calls for 34 articles and 77 questions) and every answer metric (it needs a
recorded run, i.e. chat calls). This milestone makes no claim about answer
quality. A bigger dataset gives more reliable numbers; it does not make the
agent better.

### Limitations

- The KB and the questions were written by the same author in the same
  session, so wording overlap is probably higher than with real customer
  questions and retrieval numbers are probably optimistic. Article titles
  were **not** changed after seeing the results above.
- Labels were checked against the article text by one person; there is no
  second annotator.
- On the larger KB, keyword retrieval reorders some v1 questions (e.g. "I
  forgot my password and cannot sign in" now ranks "Account locked" above
  "Reset your password"; the right article is still in the top 3). The
  production path is hybrid, so this matters most when semantic search is
  weak.
- The semantic and rerank threshold work was done on v1 and does not
  transfer to v2 without new measurements.

## Conversations and follow-up questions (Phase 5, Milestone 2)

### What changed

Before this milestone every `/chat` request stood alone: "What if I bought it
35 days ago?" reached retrieval with nothing to say what "it" was. Now a
conversation is stored, and a follow-up is rewritten into a standalone
question **for retrieval only**.

```
Browser ──POST /chat {question, conversation_id?}──> api.chat
  1. load the conversation (404 unknown/expired, 409 full, 503 storage down)
  2. rate limit (429)
  3. agent.answer_question(question, history=..., trace=...)
       a. no history  -> no rewrite call (first message: exactly as before)
          history     -> query_rewrite: one small chat call -> standalone query
       b. needs_clarification -> fixed "tell me more" reply, stop
       c. retrieve + rerank with the standalone query
       d. generate from the articles; the model sees the original message
          plus the standalone query, never earlier replies
       e. final gate (_checked_result): citations checked as before
  4. save question + reply together (one transaction), then respond
```

### Storage (`conversation_store.py`)

- SQLite from the standard library (no new dependency). Two tables,
  `conversations` and `messages`; the rewritten query is its own column, so
  the customer's text is stored unchanged.
- **Thread safety:** `/chat` runs in worker threads, so each operation opens
  its own connection and closes it. Writes use `BEGIN IMMEDIATE` and WAL mode;
  SQLite serialises writers. A test runs 12 threads against one conversation
  and checks the message limit holds exactly.
- **Limits:** 40 messages per conversation, checked inside the write
  transaction. A full conversation is refused (409); nothing is dropped or
  moved to another conversation.
- **Retention:** 30 days after the last message. Expired conversations count
  as missing at once; cleanup (startup, then at most hourly) only deletes
  conversations idle that long, and every write re-checks expiry in its
  transaction, so cleanup can't remove one that is in use.
- **IDs:** 128 random bits (`secrets.token_hex(16)`), always made by the
  server. A client can't create a conversation with an ID of its choosing
  (unknown IDs are 404), and anything that isn't 32 lowercase hex characters
  is rejected (422) before any lookup. The ID is a bearer secret: logs carry
  a 12-character SHA-256 prefix instead, and a filter on Uvicorn's access log
  replaces `/conversations/<id>` with `/conversations/<redacted>`.

### Failing safely

- No answer without saving: a new conversation pings the database first; a
  follow-up reads its history first. Either failing returns 503 before any
  Cohere call. If the final save fails, the generated answer is **discarded**
  and the customer gets 503 ("your message was not sent"), so the page never
  shows a reply the conversation doesn't contain. The cost: an occasional
  wasted Cohere call.
- An agent crash is saved as the usual safe hand-off and returned as 500.
- Startup fails if the database can't be opened, so `/ready` never reports a
  service that can't save.

### Rewriting (`query_rewrite.py`)

- One Cohere chat call, temperature 0, JSON output, `COHERE_REWRITE_TIMEOUT_SECONDS`
  (default 8) with the shared retry policy from `cohere_calls`.
- Input: the last 6 messages, each cut to 500 characters, at most 3,000 in
  total. This only bounds the prompt; nothing is removed from storage.
  Assistant replies are labelled "context only, may be wrong".
- The output is checked: correct JSON shape, at most 300 characters, and **no
  number the customer never wrote** (stops "30 days" from an assistant reply,
  or an invented "$49", becoming part of the search). Anything unusable, and
  any API failure, falls back to the message as written; a rewrite problem is
  never a reason to refuse an answer.
- The rewrite is never evidence. The answer model gets the original message
  plus the rewrite labelled "not a source of facts", and still must cite the
  retrieved articles.

### The clarification outcome

`needs_clarification` is a new, non-escalated reply with a fixed message and
no sources. The final gate accepts it **only** when the agent itself produced
it after the rewriter asked for it (`clarification_allowed=True`), and only
in exactly that form. A generator returning `reason="needs_clarification"` is
treated as an error (hand-off), so the new outcome can't be used to get an
uncited answer past the citation checks.

### Frontend

The ID is kept in `sessionStorage` (one tab, forgotten when it closes). On
load the page calls `GET /conversations/{id}` and redraws the messages with
`textContent`. If the conversation is gone (404, e.g. after a Render
restart) or can't be loaded, the page starts fresh instead of continuing a
conversation the customer can't see. On 404/409/503 after sending, the
question is put back in the box because it wasn't saved. A Node-based test
(`tests/js/app_smoke.js`) runs the real script against a fake DOM that throws
on any `innerHTML`-style insertion, and checks that restored `<script>` and
`<img>` text stays text and `javascript:` URLs never become links.

### Evaluation (`evals/conversation_eval.json`, `evaluate_conversations.py`)

32 conversations: 20 follow-ups, 4 standalone questions, 3 topic switches, 5
that should get a clarification request. Each answerable case has a
hand-written standalone question, the articles it should find, required terms
and (for topic switches) terms from the old topic that must not leak in. The
dataset check runs every reference through the production output check, so a
guard that rejected good rewrites would fail the tests.

Offline result (keyword retrieval, Expected@3, 27 answerable cases):

| Query used | All | Follow-ups | Standalone | Topic switch |
| --- | --- | --- | --- | --- |
| Last message alone | 10/27 (37%) | 4/20 | 4/4 | 2/3 |
| All customer messages joined | 23/27 (85%) | 17/20 | 3/4 | 3/3 |
| Hand-written standalone question | 24/27 (89%) | 17/20 | 4/4 | 3/3 |

What this shows: context matters a lot for follow-ups. It does **not** show
that the rewriter works: the third row is written by hand, an upper bound.
Simply joining customer messages is almost as good for keyword retrieval on
this small set, but it hurt a standalone question; semantic retrieval and
the rewriter itself are not measured (both need live Cohere calls:
`--record-rewrites`, then `--rewrites`).

### Remaining limitations

- **Render free plan:** the database is on the service's temporary disk, so
  all conversations disappear on every redeploy, restart and spin-down.
  The page handles this (starts a new conversation), but history is lost.
- The rewrite prompt and `response_format` JSON schema have not been tried
  against the live API (no live calls in this milestone). If the API rejected
  the request, every follow-up would fall back to the message as written:
  safe, but no better than before.
- Whoever has a conversation ID can read the conversation; there are no
  accounts. The ID is in the URL of `GET /conversations/{id}`, so a proxy in
  front of the app (including Render's own request logs, if any) could record
  it; only this app's and Uvicorn's logs are redacted.
- One extra chat call per follow-up (up to 8 s more in the worst case); first
  messages cost the same as before.
- The dataset was written by one person, together with the KB; the numbers
  are probably optimistic and small.
