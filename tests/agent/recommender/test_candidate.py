"""Tests for candidate.py helpers."""

from src.agent.recommender.nodes.candidate import _build_query


def test_build_query_uses_manifesto_only_when_query_is_blank():
    result = _build_query(manifesto="prefers quiet cafes", user_query="")
    assert "prefers quiet cafes" in result
    # No leading user-query segment when query is blank.
    assert not result.startswith("|")


def test_build_query_places_user_query_first():
    """The user's explicit query should sit at the *start* of the search string
    so the embedding model weights it more heavily than the manifesto."""
    manifesto = "loves seafood and outdoor seating"
    user_query = "quiet place to read"

    result = _build_query(manifesto, user_query)

    # User query appears, and appears before the manifesto excerpt.
    assert user_query in result
    assert result.index(user_query) < result.index(manifesto)


def test_build_query_collapses_manifesto_newlines():
    """Manifestos can be multi-paragraph; the query is a single line."""
    manifesto = "line one\nline two\nline three"
    result = _build_query(manifesto, "")
    assert "\n" not in result


def test_build_query_prefers_last_user_message_over_query():
    messages = [
        {"role": "user", "content": "Show me cafes"},
        {"role": "assistant", "content": "Here are some options..."},
        {"role": "user", "content": "outdoor seating please"},
    ]
    result = _build_query("loves quiet spots", "original query", messages)
    assert "outdoor seating" in result
    assert "original query" not in result


def test_build_query_falls_back_to_query_when_no_messages():
    result = _build_query("loves quiet spots", "sushi", [])
    assert "sushi" in result


def test_build_query_skips_non_user_messages():
    """Only user-role messages should drive the effective query."""
    messages = [
        {"role": "assistant", "content": "Here are some options..."},
    ]
    result = _build_query("loves quiet spots", "sushi", messages)
    assert "sushi" in result
    assert "Here are some options" not in result
