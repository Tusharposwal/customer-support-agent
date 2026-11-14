"""HTTP API for the customer support agent.

Run locally:
    uvicorn support_agent.api:app --reload
or:
    python -m support_agent.api
"""

import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from support_agent import agent
from support_agent.answer import HANDOFF_MESSAGE

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1000
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Customer Support Agent API",
    description="Answers customer questions from the knowledge base, or hands off to a human.",
    version="0.1.0",
)

# The chat page's CSS and JavaScript, served at /static/...
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    # Leading/trailing spaces are removed first, so "   " counts as empty.
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH),
    ]


class Source(BaseModel):
    id: str
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    escalate: bool
    reason: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The customer-facing chat page. It calls POST /chat from the browser."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness check. Does not call Cohere."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a customer question using the existing support agent."""
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the
    # blocking Cohere calls do not freeze the server.
    try:
        result = agent.answer_question(request.question)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            escalate=result.escalate,
            reason=result.reason,
        )
    except Exception as error:
        # Never send exception details or stack traces to the customer.
        logger.error("Unexpected error in /chat: %s", type(error).__name__)
        safe = ChatResponse(
            answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason="server_error"
        )
        return JSONResponse(status_code=500, content=safe.model_dump())


def main() -> None:
    """Start the API server. HOST and PORT can be set as environment variables."""
    import uvicorn

    uvicorn.run(
        "support_agent.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
