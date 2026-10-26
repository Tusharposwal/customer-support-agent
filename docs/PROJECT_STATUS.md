# Customer Support Agent — Project Status

## 1. Project Goal

Build a production-oriented AI customer support agent that answers repetitive, well-documented questions using Retrieval-Augmented Generation (RAG), provides citations, and escalates unsupported or risky questions to a human.

The project is part of the AI Engineering: Zero to Production course and is intended as a resume portfolio project.

## 2. Core Architecture

1. Knowledge-base ingestion and retrieval.
2. LLM-generated structured answers.
3. Citation and confidence validation.
4. Human escalation when evidence or policy requirements are not met.
5. Tools for order lookup and ticket operations.
6. Security controls, including customer-data isolation.
7. Automated tests and agent evaluations.
8. Logging, documentation, and deployment.

## 3. Safety Principles

- Retrieved evidence should support customer-facing answers.
- Unsupported questions should be escalated rather than answered speculatively.
- Customer identity must come from trusted application authentication, not arbitrary model arguments.
- Sensitive or irreversible actions require appropriate authorization and approval.
- Pydantic validates output structure but does not prove factual correctness.
- Evaluation must measure answer quality and escalation behavior separately.

## 4. Planned Implementation Phases

1. RAG foundation.
2. Structured answers and safety gate.
3. Agent tools and mock customer data.
4. Testing and evaluation.
5. Security hardening and deployment.

The sequence may change as implementation requirements become clearer.

## 5. Development Environment

- Operating system: Windows
- Editor: VS Code
- Python: 3.10.11
- Git: 2.50.1
- Virtual environment: `.venv`
- Application source: `src/support_agent/`

Python dependency compatibility must be checked before selecting libraries.

## 6. Current Progress

- [x] Python and Git verified.
- [x] Project directory created.
- [x] Virtual environment created and activated.
- [x] Initial folder structure created.
- [x] `.gitignore` created.
- [x] Git repository initialized.
- [ ] Project README completed.
- [ ] Dependencies selected and installed.
- [ ] RAG foundation implemented.
- [ ] Tests and evaluations implemented.
- [ ] Security and deployment completed.

## 7. Working Agreement

Build incrementally. Explain important concepts before implementation. Test components before integrating them. Use course code as a learning reference when relevant, rather than importing unrelated examples wholesale.

Maintain this document as the project evolves. Record completed work, key decisions, known limitations, and the exact next task.

## 8. Current Next Step

Complete the README and dependency strategy, then implement the smallest testable RAG foundation.
