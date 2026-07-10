"""Tests for box_office.utils.format_helpers.safe_format."""

from box_office.utils.format_helpers import safe_format


def test_float_formats_normally():
    assert safe_format(0.87234, ".4f") == "0.8723"


def test_int_formats_normally():
    assert safe_format(42, ".0f") == "42"


def test_string_sentinel_returns_fallback():
    assert safe_format("N/A", ".4f") == "N/A"


def test_none_returns_fallback():
    assert safe_format(None, ".4f", fallback="--") == "--"


def test_non_numeric_list_returns_fallback():
    assert safe_format([1, 2, 3], ".4f") == "N/A"


def test_default_fallback_is_na():
    assert safe_format("oops", ".2%") == "N/A"


def test_custom_fallback_used_on_failure():
    assert safe_format("oops", ".4f", fallback="missing") == "missing"


def test_percent_format_succeeds():
    assert safe_format(0.5, ".2%") == "50.00%"
