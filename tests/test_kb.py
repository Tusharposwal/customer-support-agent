from support_agent.kb import load_articles, retrieve


def test_load_articles():
    articles = load_articles()
    assert len(articles) == 3


def test_password_query():
    results = retrieve("How do I reset my password?")
    assert results
    assert results[0]["id"] == "KB-001"


def test_refund_query():
    results = retrieve("Can I get a refund?")
    assert results
    assert results[0]["id"] == "KB-002"


def test_order_tracking_query():
    results = retrieve("How can I track my shipment?")
    assert results
    assert results[0]["id"] == "KB-003"


def test_unrelated_query():
    results = retrieve("What is the weather on Mars?")
    assert results == []


def test_login_problem():
    results = retrieve("I can't log into my account")
    assert results
    assert results[0]["id"] == "KB-001"


def test_email_change_is_not_supported():
    results = retrieve("How do I change my email address?")
    assert results == []
