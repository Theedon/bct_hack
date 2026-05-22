"""Tests for profiler.py — multi-turn conversation history injection."""

import src.agent.recommender.nodes.profiler as profiler_mod

_BASE_STATE = {
    "user_id": "u1",
    "user_name": "Ada",
    "user_review_count": 10,
    "average_stars": 4.0,
    "user_elite_count": 0,
    "user_fans": 2,
    "query": "",
    "k": 5,
}


class _StubLLM:
    """Captures the HumanMessage content passed to the LLM."""

    def __init__(self):
        self.last_human_content = None
        self.last_system_content = None

    def invoke(self, msgs):
        self.last_system_content = msgs[0].content
        self.last_human_content = msgs[1].content
        return type("R", (), {"content": "stub manifesto"})()


def test_profiler_injects_history_into_prompt(monkeypatch):
    stub = _StubLLM()
    monkeypatch.setattr(profiler_mod, "_llm", stub)
    monkeypatch.setattr(profiler_mod, "_fetch_user_reviews", lambda uid: ([], []))

    state = {
        **_BASE_STATE,
        "messages": [
            {"role": "user", "content": "Show me quiet cafes"},
            {"role": "assistant", "content": "Here are 5 cafes..."},
            {"role": "user", "content": "Do any have outdoor seating?"},
        ],
    }
    profiler_mod.profiler(state)

    assert "Conversation History" in stub.last_human_content
    assert "outdoor seating" in stub.last_human_content
    assert "Show me quiet cafes" in stub.last_human_content


def test_profiler_no_history_prompt_unchanged(monkeypatch):
    stub = _StubLLM()
    monkeypatch.setattr(profiler_mod, "_llm", stub)
    monkeypatch.setattr(profiler_mod, "_fetch_user_reviews", lambda uid: ([], []))

    state = {**_BASE_STATE, "messages": []}
    profiler_mod.profiler(state)

    assert "Conversation History" not in stub.last_human_content


def test_profiler_missing_messages_key_treated_as_empty(monkeypatch):
    stub = _StubLLM()
    monkeypatch.setattr(profiler_mod, "_llm", stub)
    monkeypatch.setattr(profiler_mod, "_fetch_user_reviews", lambda uid: ([], []))

    state = {**_BASE_STATE}  # no 'messages' key at all
    profiler_mod.profiler(state)

    assert "Conversation History" not in stub.last_human_content


def test_format_history_formats_roles():
    result = profiler_mod._format_history(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    )
    assert result == "User: hello\nAssistant: world"


def test_profiler_injects_nigerian_mode(monkeypatch):
    stub = _StubLLM()
    monkeypatch.setattr(profiler_mod, "_llm", stub)
    monkeypatch.setattr(profiler_mod, "_fetch_user_reviews", lambda uid: ([], []))

    state = {**_BASE_STATE, "nigerian_mode": True}
    profiler_mod.profiler(state)

    assert "Frame the manifesto in a Nigerian context" in stub.last_system_content
