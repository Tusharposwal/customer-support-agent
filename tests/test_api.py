"""Offline API tests. The agent is mocked, so no Cohere API calls are made."""

import pytest
from fastapi.testclient import TestClient

from support_agent import agent
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer
from support_agent.api import MAX_QUESTION_LENGTH, app

client = TestClient(app)

GROUNDED = SupportAnswer(
    answer="Open the sign-in page and select Forgot password.",
    sources=[
        {
            "id": "KB-001",
            "title": "Reset your password",
            "url": "https://support.example.com/password-reset",
        }
    ],
    escalate=False,
    reason="answered",
)

HANDOFF = SupportAnswer(
    answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason="insufficient_evidence"
)


@pytest.fixture
def fake_agent(monkeypatch):
    """Replace answer_question with a fake that records questions."""

    class FakeAgent:
        def __init__(self):
            self.questions = []
            self.result = GROUNDED
            self.error = None

        def __call__(self, question):
            self.questions.append(question)
            if self.error:
                raise self.error
            return self.result

    fake = FakeAgent()
    monkeypatch.setattr(agent, "answer_question", fake)
    return fake


def test_health_does_not_call_agent(fake_agent):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_agent.questions == []


def test_successful_grounded_answer(fake_agent):
    response = client.post("/chat", json={"question": "How do I reset my password?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Open the sign-in page and select Forgot password.",
        "sources": [
            {
                "id": "KB-001",
                "title": "Reset your password",
                "url": "https://support.example.com/password-reset",
            }
        ],
        "escalate": False,
        "reason": "answered",
    }
    assert fake_agent.questions == ["How do I reset my password?"]


def test_question_whitespace_is_trimmed(fake_agent):
    client.post("/chat", json={"question": "  How do I reset my password?\n"})

    assert fake_agent.questions == ["How do I reset my password?"]


def test_human_escalation(fake_agent):
    fake_agent.result = HANDOFF

    response = client.post("/chat", json={"question": "How do I change my email address?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": HANDOFF_MESSAGE,
        "sources": [],
        "escalate": True,
        "reason": "insufficient_evidence",
    }


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   \n\t"},
        {"question": None},
        {"question": 123},
        {"question": ["How do I reset my password?"]},
        {"message": "How do I reset my password?"},
        {"question": "x" * (MAX_QUESTION_LENGTH + 1)},
    ],
    ids=["missing", "empty", "whitespace", "null", "number", "list", "wrong-field", "too-long"],
)
def test_invalid_questions_return_422(fake_agent, body):
    response = client.post("/chat", json=body)

    assert response.status_code == 422
    assert fake_agent.questions == []


def test_non_json_body_returns_422(fake_agent):
    response = client.post(
        "/chat", content="not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert fake_agent.questions == []


def test_unexpected_agent_error_is_safe(fake_agent):
    fake_agent.error = RuntimeError("COHERE_API_KEY=sk-secret-value at line 42")

    response = client.post("/chat", json={"question": "How do I reset my password?"})

    assert response.status_code == 500
    assert response.json() == {
        "answer": HANDOFF_MESSAGE,
        "sources": [],
        "escalate": True,
        "reason": "server_error",
    }
    assert "secret" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


def test_response_structure(fake_agent):
    body = client.post("/chat", json={"question": "How do I reset my password?"}).json()

    assert set(body) == {"answer", "sources", "escalate", "reason"}
    assert isinstance(body["answer"], str)
    assert isinstance(body["escalate"], bool)
    assert isinstance(body["reason"], str)
    assert all(set(source) == {"id", "title", "url"} for source in body["sources"])


def test_openapi_schema_lists_endpoints():
    paths = client.get("/openapi.json").json()["paths"]

    assert "/health" in paths and "get" in paths["/health"]
    assert "/chat" in paths and "post" in paths["/chat"]


# --- Web interface ---------------------------------------------------------


def test_home_page_loads_without_calling_agent(fake_agent):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>AI Customer Support Assistant</title>" in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="question"' in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'href="/static/styles.css"' in response.text
    assert fake_agent.questions == []


def test_page_assets_are_served():
    script = client.get("/static/app.js")
    styles = client.get("/static/styles.css")

    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")


def test_frontend_posts_json_to_chat_endpoint():
    script = client.get("/static/app.js").text

    assert 'const CHAT_URL = "/chat";' in script
    assert "fetch(CHAT_URL" in script
    assert 'method: "POST"' in script
    assert '"Content-Type": "application/json"' in script
    assert "JSON.stringify({ question })" in script


def test_frontend_never_uses_innerhtml():
    script = client.get("/static/app.js").text

    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert "textContent" in script


def test_home_page_is_not_in_api_schema():
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths


def test_existing_routes_still_work(fake_agent):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert client.post("/chat", json={"question": "Hi"}).status_code == 200
    assert client.get("/static/missing.js").status_code == 404
