# Milestone 3.3 live smoke test: 2026-11-21

**Current status: NOT VERIFIED. Both attempts failed authentication (HTTP
401).** The tool-calling cycle has never run against the real API, so
Milestone 3.3 is verified offline only. Neither attempt shows a defect in
the code: Cohere refused the key before looking at the request.

| Attempt | Requests sent | Result |
| --- | --- | --- |
| 1 | 2 of 4 | 401 on both questions |
| 2 | 1 of 4 | 401 on question 1; stopped, question 2 not sent |

Total outbound Cohere requests across both attempts: **3**, all rejected at
authentication. No retries, and no embedding, rerank or rewrite requests.

## Attempt 2 (latest)

Same harness, setup, questions and limits as attempt 1 (below), plus two
changes made only in the throwaway script:

- **Stop rule:** after a question whose first request is rejected, or that
  hits an authentication or invalid-request error, no further question is
  sent. Attempt 1 had wasted a request here.
- **More detail recorded:** the structure of the second request's echoed
  assistant message (keys, call IDs, whether a tool plan is present), never
  the arguments.

Pre-flight: `COHERE_API_KEY` was present and non-empty in the process
environment. Presence was checked only; the value was not printed or
inspected, so whether it differs from attempt 1's key is unknown. `.env` was
not read.

| | Question 1: status |
| --- | --- |
| First request | messages `system, user`; documents KB-034, KB-003, KB-011; tools `get_order_status`, `check_refund_eligibility`, `request_order_cancellation`; no `tool_choice`; timeout 25 s; SDK retries 0 |
| Cohere response | **401 Unauthorized** (`UnauthorizedError`) after 0.69 s |
| Application result | safe hand-off, `reason="llm_error"`, no sources, no tool run |
| Question 2 | **not sent** (stop rule) |

**Outbound requests: 1.** Nothing about tool calling could be checked: the
first request was refused at authentication.

### Still unverified after both attempts

1. Whether Cohere accepts `documents` and `tools` in one request.
2. Whether the model returns `finish_reason="TOOL_CALL"` with the expected
   tool.
3. Whether the second request (echoed assistant tool-call message and tool
   messages, `tool_choice="NONE"`) is accepted.
4. The format of tool citation IDs.
5. Whether article and tool citations pass validation together.

### What is needed to finish

A Cohere API key that Cohere accepts, in the environment the tools inherit.
That means checking or replacing the Windows user environment variable, then
restarting VS Code. Then a new approved run of at most 4 requests.

---

## Attempt 1 (historical)

**Conclusion of attempt 1: FAIL (authentication).** Cohere rejected both requests with
HTTP 401 Unauthorized, so the tool-calling cycle never ran against the real
API. None of the compatibility questions (tool-call finish reason, second
request shape, tool citation ID format) could be answered. The failure is
the API key, not a demonstrated incompatibility in the code.

### Setup

- Script: a throwaway harness outside the repository that calls the real
  `agent.answer_question`. No application code or tests were changed.
- Client: `recording_client()` from `evals/evaluate_conversations.py`, using
  `COHERE_API_KEY` from the process environment only. `load_dotenv` and
  `find_dotenv` were replaced with functions that raise, so `.env` could not
  be read.
- Request cap: a wrapper allowed at most 4 `chat` requests and refused a
  fifth before sending it. Embedding, rerank and any other client method
  were blocked.
- Retries: off for this process only (`cohere_calls.MAX_RETRIES = 0`).
  Every request also carried the SDK option `max_retries: 0`.
- Retrieval: keyword (`kb.retrieve`), so no embedding requests. Reranking
  was off. There was no conversation history, so no rewrite request.
- Data: fictional order ORD-1003 (shipped) with its matching demo email from
  `data/demo_orders.json`. The email is redacted from this report.

### Results

| | Question 1: status | Question 2: return + policy |
| --- | --- | --- |
| Question (email redacted) | "What is the status of my order ORD-1003? My email is [email]." | "Order ORD-1003, email [email]: can I return it after it arrives, and what is your return policy?" |
| Expected tool | `get_order_status` | `check_refund_eligibility` |
| Tools offered | yes (order number present) | yes |
| Articles sent as `documents` | KB-034, KB-003, KB-011 | KB-024, KB-015, KB-034 |
| First request | messages `system, user`; 3 articles; 3 tools; no `tool_choice`; timeout 25 s; SDK retries 0 | same shape |
| Cohere response | **401 Unauthorized** (`UnauthorizedError`) after 0.67 s | **401 Unauthorized** after 0.35 s |
| Finish reason | none (rejected) | none (rejected) |
| Tool requested | not reached | not reached |
| Second request (tool messages, `tool_choice="NONE"`) | not reached | not reached |
| Tool citation ID format | not observed | not observed |
| Citation validation (article / tool / together) | not reached | not reached |
| Application result | safe hand-off, `reason="llm_error"`, no sources | same |

**Outbound requests: 2 of the allowed 4.** Retries: 0. No embedding, rerank
or rewrite requests. No tool ran (`tool_calls: 0`).

### What this run did confirm

- The application builds the planned first request: the system and user
  messages, the retrieved articles as `documents`, the three tool
  definitions, no `tool_choice`, a 25 s budget and SDK retries off. Cohere
  rejected the request at authentication. A 401 says nothing about whether
  the body would have been accepted.
- An authentication failure is handled safely. Error type `auth_error`: no
  retry, the standard hand-off message, and no error text from Cohere shown
  or logged. Each log line names only the category and status.
- The request cap and the retry switch-off worked.

### Not verified (still open)

1. Whether Cohere accepts `documents` and `tools` together in one request.
2. Whether the model returns `finish_reason="TOOL_CALL"` for these questions.
3. Whether the second request accepts the echoed assistant tool-call message
   and the tool messages.
4. The format of tool citation IDs. The code accepts our document ID, the
   call ID, or `<call ID>:<n>`.
5. Whether article and tool citations pass validation together.

### Errors and limitations

- **401 Unauthorized on both requests.** `COHERE_API_KEY` is set in the
  environment (presence checked only) but Cohere does not accept it. Likely
  causes: a mistyped, revoked or expired key, or a key for a different
  account or environment. The key was not printed, inspected or compared
  with anything, and `.env` was not opened.
- The harness sent Question 2 after Question 1 had already failed
  authentication. That was not a retry, and it stayed within the cap, but it
  was predictably wasted. A rerun should stop after the first `auth_error`.
- Rerunning needs a valid key in the environment inherited by the tools, and
  new approval. The 2 requests already used count against nothing else.
