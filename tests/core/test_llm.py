import pytest
from src.core.llm import clean_content


def test_clean_content_string():
    assert clean_content("hello world") == "hello world"


def test_clean_content_list_of_strings():
    assert clean_content(["hello", " ", "world"]) == "hello world"


def test_clean_content_list_of_dicts():
    content = [{"text": "hello"}, {"text": " "}, {"text": "world"}]
    assert clean_content(content) == "hello world"


def test_clean_content_mixed_list():
    content = [
        "hello",
        {"text": " "},
        {"type": "image_url", "image_url": "http://..."},
        {"text": "world"},
    ]
    assert clean_content(content) == "hello world"


def test_clean_content_other_types():
    assert clean_content(123) == "123"
    assert clean_content(None) == "None"
