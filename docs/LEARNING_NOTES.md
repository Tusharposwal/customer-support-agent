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
- `.python-version`: pins Python 3.10.11 for Render.

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
