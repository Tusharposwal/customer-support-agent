"""Offline API tests. The agent is mocked, so no Cohere API calls are made."""

import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer
from support_agent.api import MAX_QUESTION_LENGTH, app
from support_agent.rate_limit import RateLimiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Give every test its own default-sized limiter, so tests don't share quota."""
    limiter = RateLimiter(per_minute=20, per_day=300)
    monkeypatch.setattr(api, "chat_limiter", limiter)
    return limiter

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


# --- Rate limiting -----------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def use_limiter(monkeypatch, per_minute, per_day):
    clock = FakeClock()
    monkeypatch.setattr(
        api, "chat_limiter", RateLimiter(per_minute=per_minute, per_day=per_day, clock=clock)
    )
    return clock


def ask():
    return client.post("/chat", json={"question": "How do I reset my password?"})


def test_requests_under_the_limit_succeed(fake_agent, monkeypatch):
    use_limiter(monkeypatch, per_minute=3, per_day=10)

    assert [ask().status_code for _ in range(3)] == [200, 200, 200]
    assert len(fake_agent.questions) == 3


def test_minute_limit_returns_429(fake_agent, monkeypatch):
    use_limiter(monkeypatch, per_minute=2, per_day=10)
    ask()
    ask()

    response = ask()

    assert response.status_code == 429
    assert response.json() == {"detail": api.LIMIT_MESSAGES["minute"]}
    assert response.headers["Retry-After"] == "60"
    assert len(fake_agent.questions) == 2  # The agent (and Cohere) was not called.


def test_minute_limit_resets_after_window(fake_agent, monkeypatch):
    clock = use_limiter(monkeypatch, per_minute=1, per_day=10)
    ask()
    assert ask().status_code == 429

    clock.now += 60

    assert ask().status_code == 200


def test_day_limit_returns_429(fake_agent, monkeypatch):
    clock = use_limiter(monkeypatch, per_minute=10, per_day=2)
    ask()
    ask()
    clock.now += 3600  # An hour later the minute window is clear, the day is not.

    response = ask()

    assert response.status_code == 429
    assert response.json() == {"detail": api.LIMIT_MESSAGES["day"]}
    assert response.headers["Retry-After"] == str(24 * 3600 - 3600)
    assert len(fake_agent.questions) == 2


def test_day_limit_resets_after_24_hours(fake_agent, monkeypatch):
    clock = use_limiter(monkeypatch, per_minute=10, per_day=1)
    ask()
    assert ask().status_code == 429

    clock.now += 24 * 3600

    assert ask().status_code == 200


def test_health_and_page_work_when_chat_is_limited(fake_agent, monkeypatch):
    use_limiter(monkeypatch, per_minute=1, per_day=1)
    ask()
    assert ask().status_code == 429

    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_invalid_requests_do_not_use_quota(fake_agent, monkeypatch):
    use_limiter(monkeypatch, per_minute=1, per_day=1)

    for _ in range(5):
        assert client.post("/chat", json={"question": ""}).status_code == 422

    assert ask().status_code == 200


def test_rate_limit_message_is_safe(fake_agent, monkeypatch):
    use_limiter(monkeypatch, per_minute=1, per_day=10)
    ask()

    body = ask().text

    assert "Traceback" not in body
    assert "COHERE" not in body


def test_frontend_handles_429():
    script = client.get("/static/app.js").text

    assert "response.status === 429" in script
    assert "data.detail" in script
