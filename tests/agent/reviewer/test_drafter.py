"""Tests for drafter.py helpers.

Demonstrated patterns:
- Test behaviour, not implementation. We assert "higher-scored fragments come
  first," not "the score is exactly 2." That way internal scoring tweaks don't
  break the test.
- Test the boring base case (empty input → empty output) too. It's the cheapest
  test you'll ever write and it catches the dumbest crashes.
"""

from src.agent.reviewer.nodes.drafter import _avg_word_count, _extract_anchors


def test_extract_anchors_returns_empty_when_no_stylistic_fragments():
    reviews = [{"text": "The food was decent and the service okay."}]
    assert _extract_anchors(reviews) == []


def test_extract_anchors_prefers_exclamation_fragments():
    """Fragments ending in `!` should rank above bland ones."""
    reviews = [
        {"text": "Honestly amazing food. Must try this!"},
    ]
    anchors = _extract_anchors(reviews, max_anchors=5)
    assert anchors, "expected at least one anchor"
    # The first anchor should be the strongest fragment (ends in '!').
    assert anchors[0].endswith("!")


def test_extract_anchors_dedupes_case_insensitively():
    """If the same fragment appears in two reviews with different casing,
    it should only be returned once."""
    reviews = [
        {"text": "I loved it so much"},
        {"text": "i loved it so much"},
    ]
    anchors = _extract_anchors(reviews)
    lowered = [a.lower() for a in anchors]
    assert len(lowered) == len(set(lowered))


def test_avg_word_count_returns_default_for_empty_input():
    """When there are no reviews to measure, fall back to a sensible default."""
    assert _avg_word_count([]) == 80


def test_avg_word_count_rounds_to_integer():
    reviews = [{"text": "one two three"}, {"text": "one two three four"}]
    # mean of 3 and 4 is 3.5, round() → 4 (banker's rounding in Python).
    assert _avg_word_count(reviews) == 4


def test_build_system_prompt_includes_nigerian_mode():
    from src.agent.reviewer.nodes.drafter import _build_system_prompt

    prompt = _build_system_prompt(80, [], 4.0, "Restaurants", nigerian_mode=True)
    assert "Nigerian Context: Use Nigerian English style rules" in prompt


def test_build_system_prompt_excludes_nigerian_mode_by_default():
    from src.agent.reviewer.nodes.drafter import _build_system_prompt

    prompt = _build_system_prompt(80, [], 4.0, "Restaurants")
    assert "Nigerian Context: Use Nigerian English style rules" not in prompt
