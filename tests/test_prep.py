import numpy as np
import pandas as pd
import pytest

from src.prep import count_hype, count_words, parse_bathrooms_text, parse_price, strip_html


def test_parse_price_handles_symbols_and_missing():
    out = parse_price(pd.Series(["$1,234.50", "$75,000.00", None, "$10.00"]))
    assert out.iloc[0] == 1234.5
    assert out.iloc[1] == 75000.0
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == 10.0


@pytest.mark.parametrize("text, expected", [
    ("1 bath", (1.0, False, False)),
    ("1.5 shared baths", (1.5, True, False)),
    ("1 private bath", (1.0, False, True)),
    ("Half-bath", (0.5, False, False)),
    ("Shared half-bath", (0.5, True, False)),
    ("Private half-bath", (0.5, False, True)),
    ("No bathroom", (0.0, False, False)),
    ("0 shared baths", (0.0, True, False)),
    ("30 baths", (30.0, False, False)),
])
def test_parse_bathrooms_text(text, expected):
    assert parse_bathrooms_text(text) == expected


def test_parse_bathrooms_text_missing():
    value, shared, private = parse_bathrooms_text(np.nan)
    assert np.isnan(value) and not shared and not private


def test_count_hype_whole_words_only():
    text = "Stunning flat, PERFECTLY placed; luxury-style but not imperfect or unperfected."
    assert count_hype(text) == 3          # stunning, perfectly, luxury (not imperfect)
    assert count_hype("") == 0
    assert count_hype("world-class views and five-star sheets") == 2


def test_strip_html_and_count_words():
    cleaned = strip_html("Bright room<br /><br />Close to the Tube &amp; parks")
    assert "<" not in cleaned and "&amp;" not in cleaned
    assert count_words(cleaned) == 7   # '&' is not a word
    assert strip_html(None) == ""
