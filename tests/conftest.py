"""Shared pytest fixtures.

A `conftest.py` file is automatically loaded by pytest — any `@pytest.fixture`
defined here is available to every test in this directory (and subdirectories)
without needing an import. Think of it as "test setup that's discoverable by
convention."
"""

import random
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _deterministic_random():
    """Seed Python's `random` module before every test.

    `autouse=True` means this runs implicitly — you don't need to ask for it in
    your test signature. This keeps tests that depend on `random.sample` (e.g.
    the stratified sampler) deterministic so they don't flake.
    """
    random.seed(42)


@pytest.fixture
def mixed_reviews() -> list[dict[str, Any]]:
    """A small review list with both low (<=3) and high (>3) star buckets."""
    return [
        {
            "stars": 1,
            "text": "Awful",
            "biz_name": "A",
            "categories": "X",
            "business_id": "a",
        },
        {
            "stars": 2,
            "text": "Meh",
            "biz_name": "B",
            "categories": "X",
            "business_id": "b",
        },
        {
            "stars": 3,
            "text": "Okay",
            "biz_name": "C",
            "categories": "X",
            "business_id": "c",
        },
        {
            "stars": 4,
            "text": "Good",
            "biz_name": "D",
            "categories": "Y",
            "business_id": "d",
        },
        {
            "stars": 5,
            "text": "Loved it!",
            "biz_name": "E",
            "categories": "Y",
            "business_id": "e",
        },
    ]


@pytest.fixture
def sample_candidates() -> list[dict[str, Any]]:
    """Three candidate businesses shaped like the real `candidate` node output."""
    return [
        {
            "business_id": "biz-001",
            "biz_name": "Quiet Cafe",
            "categories": "Coffee & Tea",
            "biz_attributes_clean": "WiFi: free, Noise: quiet",
            "biz_stars": 4.5,
            "avg_user_stars": 4.2,
            "review_count": 30,
        },
        {
            "business_id": "biz-002",
            "biz_name": "Loud Bar",
            "categories": "Bars, Nightlife",
            "biz_attributes_clean": "Noise: very loud",
            "biz_stars": 3.8,
            "avg_user_stars": 3.5,
            "review_count": 120,
        },
        {
            "business_id": "biz-003",
            "biz_name": "Library Cafe",
            "categories": "Coffee & Tea, Books",
            "biz_attributes_clean": "WiFi: free, Noise: very quiet",
            "biz_stars": 4.8,
            "avg_user_stars": 4.7,
            "review_count": 45,
        },
    ]
