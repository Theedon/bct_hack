"""Tests for the business ingest script.

Demonstrated patterns:
- Testing a DataFrame transformation end-to-end with a hand-rolled small
  DataFrame. Cheap, fast, and reads like a spec.
"""

import pandas as pd

from scripts.ingest_businesses import SNIPPET_CHARS, _top_snippets, build_documents


def test_top_snippets_picks_shortest_and_truncates():
    texts = [
        "short one",
        "this is a much, much longer review that should not be the first pick",
        "tiny",
        "",  # filtered out (empty after strip)
        "   ",  # filtered out (whitespace only)
        "medium length review here",
    ]
    snippets = _top_snippets(texts, n=3)
    # 'tiny' is shortest, then 'short one', then 'medium length review here'.
    assert snippets == ["tiny", "short one", "medium length review here"]


def test_top_snippets_truncates_to_80_chars():
    long_text = "x" * 200
    snippets = _top_snippets([long_text], n=1)
    assert len(snippets[0]) == SNIPPET_CHARS


def test_build_documents_groups_by_business_id():
    """Three reviews of two businesses should produce two documents."""
    df = pd.DataFrame(
        [
            {
                "business_id": "biz-A",
                "biz_name": "Alpha",
                "categories": "Cafe",
                "biz_attributes_clean": "WiFi: free",
                "biz_stars": 4.5,
                "stars_review": 5.0,
                "text": "loved it",
            },
            {
                "business_id": "biz-A",
                "biz_name": "Alpha",
                "categories": "Cafe",
                "biz_attributes_clean": "WiFi: free",
                "biz_stars": 4.5,
                "stars_review": 3.0,
                "text": "meh",
            },
            {
                "business_id": "biz-B",
                "biz_name": "Beta",
                "categories": "Bar",
                "biz_attributes_clean": "Noise: loud",
                "biz_stars": 3.8,
                "stars_review": 4.0,
                "text": "fine",
            },
        ]
    )

    docs = build_documents(df)

    assert len(docs) == 2

    by_id = {d.metadata["business_id"]: d for d in docs}

    alpha = by_id["biz-A"]
    assert alpha.metadata["biz_name"] == "Alpha"
    assert alpha.metadata["review_count"] == 2
    # avg of 5.0 and 3.0
    assert alpha.metadata["avg_user_stars"] == 4.0
    # page_content carries the searchable signal
    assert "Alpha" in alpha.page_content
    assert "Cafe" in alpha.page_content

    beta = by_id["biz-B"]
    assert beta.metadata["review_count"] == 1
    assert beta.metadata["avg_user_stars"] == 4.0
