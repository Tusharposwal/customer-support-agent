# Customer Support Agent — Project Status

## 1. Project Goal

Build a production-oriented AI customer support agent that answers repetitive, well-documented questions using Retrieval-Augmented Generation (RAG), provides citations, and escalates unsupported or risky questions to a human.

The project is part of the AI Engineering: Zero to Production course and is intended as a resume portfolio project.

## 2. Core Architecture

1. Knowledge-base ingestion and retrieval (keyword + semantic hybrid, optional rerank).
2. LLM-generated answers grounded in retrieved articles (Cohere).
3. Citation validation and a final safety gate.
4. Human escalation when evidence is missing or something fails, recorded as internal hand-off records.
5. Read-only order tools over fictional demo data, plus confirmed cancellations.
6. Security controls: order ownership checks, secret IDs, no sensitive data in logs.
7. Automated tests and agent evaluations.
8. Logging, documentation, and deployment (Render).

## 3. Safety Principles

- Retrieved evidence should support customer-facing answers.
- Unsupported questions should be escalated rather than answered speculatively.
- Customer identity must come from trusted application authentication, not arbitrary model arguments. (Current demo: ownership = order number + email written by the customer; not real authentication.)
- Sensitive or irreversible actions require appropriate authorization and approval. (Cancellation: only the customer's explicit confirmation through its own endpoint.)
- Pydantic validates output structure but does not prove factual correctness.
- Evaluation must measure answer quality and escalation behavior separately.
- Never claim a human has been contacted: no human-support system is connected.

## 4. Implementation Phases

1. RAG foundation. Done.
2. Structured answers and safety gate. Done.
3. Agent tools and mock customer data. Done (Phase 5, Milestone 3).
4. Testing and evaluation. Done (offline; live parts listed below).
5. Security hardening and deployment. Done (Render; free plan).

## 5. Development Environment

- Operating system: Windows
- Editor: VS Code
- Python: 3.12.16 for deployment and CI; the full test suite is also run on 3.10 locally.
- Git: 2.50.1
- Virtual environment: `.venv`
- Application source: `src/support_agent/`

## 6. Current Progress

Phase 5 Milestones 1–5 are complete and approved. Implementation and offline evaluation are complete; the work is awaiting GitHub commit/push and Render deployment.

- [x] Milestone 1–2: conversation memory (SQLite) and follow-up rewriting.
- [x] Milestone 3.1: fictional demo orders and refund policy in Python.
- [x] Milestone 3.2: tool registry with validation, grounding and ownership checks.
- [x] Milestone 3.3: Cohere tool-call cycle (offline only; live smoke test blocked by HTTP 401).
- [x] Milestone 3.4: persistent cancellation with explicit confirmation.
- [x] Milestone 3.5: cancellation confirmation on the page, restored after reload.
- [x] Milestone 3.6: tool-selection evaluation (offline; scripted decisions and a keyword baseline).
- [x] Milestone 4: internal hand-off records and reply feedback.
- [x] Milestone 5: final reliability review, three fixes, end-to-end tests.

Still unverified live: the tool-call cycle against Cohere (both live smoke-test attempts failed authentication with HTTP 401; see `evals/results/tool_smoke_2026-11-21.md`), hybrid retrieval and answer quality on KB v2, and real tool-selection accuracy. See `docs/LEARNING_NOTES.md`, "Final reliability review".

Known deployment limitation: on Render's free plan, the SQLite database lives on the service's own disk and is wiped on every redeploy, restart and spin-down. Conversations, hand-off records, feedback ratings and cancellations are lost each time. Keeping them would need a paid plan with a persistent disk or an external database; neither is set up.

## 7. Working Agreement

Build incrementally. Explain important concepts before implementation. Test components before integrating them. Use course code as a learning reference when relevant, rather than importing unrelated examples wholesale.

Maintain this document as the project evolves. Record completed work, key decisions, known limitations, and the exact next task.

## 8. Current Next Step

Commit and push the Phase 5 work to GitHub, then deploy to Render (also update the Render dashboard: health check path `/ready` and build command `pip install -e . -c constraints.txt`). Then: the Project Mastery walkthrough. Separately, when a valid Cohere key is available and approved: the Milestone 3.3 live smoke test.
