from support_agent.semantic_kb import retrieve_semantic


def test_password_query():
    results = retrieve_semantic("How do I reset my password?")
    assert results
    assert results[0]["id"] == "KB-001"


def test_login_problem():
    results = retrieve_semantic("I can't log into my account")
    assert results
    assert results[0]["id"] == "KB-001"


def test_package_not_arrived():
    results = retrieve_semantic("My package has not arrived")
    assert results
    assert results[0]["id"] == "KB-003"


def test_refund_query():
    results = retrieve_semantic("I want my money back for a purchase")
    assert results
    assert results[0]["id"] == "KB-002"


def test_empty_query():
    assert retrieve_semantic("   ") == []
