"""Tests for the ranker node — Pattern 3: mocking an LLM.

The ranker calls an LLM (`_llm.invoke(...)`) and then uses its output to look
up candidates by index. The LLM call itself is slow, costs money, and is
non-deterministic — so we replace it with a stub that returns exactly what we
want, then assert on the *plumbing around it* (index → candidate mapping).

This is the same bug we caught manually in development: the LLM used to
return business_ids and would hallucinate them. We now use `candidate_number`
(1-based index). These tests lock that behaviour in.

Demonstrated patterns:
- `monkeypatch` — pytest's built-in fixture for temporarily replacing
  attributes. Here we swap the module-level `_llm` for a fake. After the test
  finishes, pytest restores the original automatically.
- Building a tiny stand-in class instead of pulling in a mocking library.
"""

import pytest

from src.agent.recommender.nodes import ranker as ranker_mod
from src.agent.recommender.nodes.ranker import (
    RankedItem,
    RankerOutput,
    ranker,
)


class _StubLLM:
    """A fake LLM that returns whatever RankerOutput we pre-load it with."""

    def __init__(self, output: RankerOutput):
        self._output = output

    def invoke(self, _messages):
        return self._output


def _state_with_candidates(candidates):
    return {
        "user_id": "u",
        "user_name": "U",
        "user_review_count": 0,
        "average_stars": 0.0,
        "user_elite_count": 0,
        "user_fans": 0,
        "query": "",
        "k": 2,
        "user_manifesto": "test manifesto",
        "cold_start": False,
        "visited_business_ids": [],
        "candidates": candidates,
        "recommendations": [],
        "reasoning_log": "",
    }


def test_ranker_maps_candidate_number_to_business_id(monkeypatch, sample_candidates):
    """When the LLM says candidate_number=2, we should return biz-002."""
    fake_output = RankerOutput(
        reasoning="picked the two best",
        ranked=[
            RankedItem(candidate_number=2, score=0.9, rationale="fits because Y"),
            RankedItem(candidate_number=3, score=0.8, rationale="fits because Z"),
        ],
    )
    monkeypatch.setattr(ranker_mod, "_llm", _StubLLM(fake_output))

    result = ranker(_state_with_candidates(sample_candidates))

    recs = result["recommendations"]
    assert len(recs) == 2
    # candidate_number=2 → candidates[1] in the fixture (biz-002).
    assert recs[0]["business_id"] == "biz-002"
    assert recs[0]["biz_name"] == "Loud Bar"
    assert recs[1]["business_id"] == "biz-003"


def test_ranker_drops_out_of_range_candidate_numbers(monkeypatch, sample_candidates):
    """If the LLM returns an index past the candidate list, we drop it
    instead of crashing or hallucinating a value."""
    fake_output = RankerOutput(
        reasoning="one valid, one invalid",
        ranked=[
            RankedItem(candidate_number=1, score=0.9, rationale="ok"),
            RankedItem(candidate_number=99, score=0.5, rationale="oops"),
        ],
    )
    monkeypatch.setattr(ranker_mod, "_llm", _StubLLM(fake_output))

    result = ranker(_state_with_candidates(sample_candidates))

    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["business_id"] == "biz-001"


def test_ranker_returns_empty_when_no_candidates(monkeypatch):
    """No candidates → no LLM call, no recommendations.

    We don't even set up a stub LLM here because the ranker should
    short-circuit before calling it.
    """
    result = ranker(_state_with_candidates(candidates=[]))
    assert result["recommendations"] == []
    assert "No candidates" in result["reasoning_log"]
