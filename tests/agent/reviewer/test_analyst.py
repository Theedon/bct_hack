"""Tests for analyst.py helpers.

Demonstrated patterns:
- A plain test function (`test_*`) that asserts on a returned value.
- `@pytest.mark.parametrize` to run the same body against multiple inputs.
- Why this matters: `_sample_reviews` has three branches (all-low, all-high,
  mixed). Parametrize lets us cover all three with one function body.
"""

import pytest

from src.agent.reviewer.nodes.analyst import _sample_reviews


def test_sample_reviews_returns_all_when_n_exceeds_input():
    """If you ask for 10 reviews from a list of 3, you get back at most 3."""
    reviews = [
        {"stars": 4, "text": "x"},
        {"stars": 5, "text": "y"},
        {"stars": 2, "text": "z"},
    ]
    result = _sample_reviews(reviews, n=10)
    assert len(result) == 3


@pytest.mark.parametrize(
    "stars_in_bucket",
    [
        [1, 2, 3],  # all low — no high reviews
        [4, 4, 5, 5],  # all high — no low reviews
    ],
    ids=["all_low", "all_high"],
)
def test_sample_reviews_single_bucket_uses_plain_random(stars_in_bucket):
    """When one bucket is empty, falls back to a uniform random sample."""
    reviews = [{"stars": s, "text": "x"} for s in stars_in_bucket]
    result = _sample_reviews(reviews, n=2)
    assert len(result) == 2
    # Every returned review must come from the original list.
    assert all(r in reviews for r in result)


def test_sample_reviews_mixed_preserves_both_buckets(mixed_reviews):
    """When both buckets exist, the sample contains at least one of each.

    `mixed_reviews` is a fixture defined in `tests/conftest.py`. Pytest sees
    the parameter name and wires it in automatically — no import needed.
    """
    result = _sample_reviews(mixed_reviews, n=4)

    low = [r for r in result if r["stars"] <= 3]
    high = [r for r in result if r["stars"] > 3]

    assert len(low) >= 1, "expected at least one low-star review in sample"
    assert len(high) >= 1, "expected at least one high-star review in sample"
    assert len(result) == 4


class _StubLLM:
    def __init__(self):
        self.last_system_content = None

    def invoke(self, msgs):
        self.last_system_content = msgs[0].content
        return type("R", (), {"content": "stub manifesto"})()


def test_analyst_injects_nigerian_mode(monkeypatch):
    import src.agent.reviewer.nodes.analyst as analyst_mod

    stub = _StubLLM()
    monkeypatch.setattr(analyst_mod, "_llm", stub)
    monkeypatch.setattr(analyst_mod, "_fetch_user_reviews", lambda uid: [])

    state = {
        "user_id": "u",
        "user_name": "U",
        "user_review_count": 0,
        "user_elite_count": 0,
        "user_fans": 0,
        "average_stars": 4.0,
        "nigerian_mode": True,
    }
    analyst_mod.analyst(state)
    assert "Frame the persona in a Nigerian context" in stub.last_system_content


def test_analyst_no_nigerian_mode(monkeypatch):
    import src.agent.reviewer.nodes.analyst as analyst_mod

    stub = _StubLLM()
    monkeypatch.setattr(analyst_mod, "_llm", stub)
    monkeypatch.setattr(analyst_mod, "_fetch_user_reviews", lambda uid: [])

    state = {
        "user_id": "u",
        "user_name": "U",
        "user_review_count": 0,
        "user_elite_count": 0,
        "user_fans": 0,
        "average_stars": 4.0,
    }
    analyst_mod.analyst(state)
    assert "Nigerian context" not in stub.last_system_content
