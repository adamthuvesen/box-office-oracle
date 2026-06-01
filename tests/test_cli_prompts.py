"""Tests for ``box_office.utils.cli_prompts.confirm``.

Covers the matrix the spec promises:
- TTY=True with explicit y/n responses calls ``input`` and parses correctly.
- TTY=False auto-answers the supplied default without calling ``input``.
- ``non_interactive=True`` overrides the TTY check (e.g. operator passed
  ``--yes`` from an interactive shell during testing).
"""

from __future__ import annotations

import logging
from unittest.mock import patch


from box_office.utils.cli_prompts import confirm


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input", return_value="yes")
def test_tty_yes_returns_true(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ") is True
    mock_input.assert_called_once_with("delete? ")


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input", return_value="no")
def test_tty_no_returns_false(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ", default=True) is False


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input", return_value="Y")
def test_tty_uppercase_yes_returns_true(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ") is True


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input", return_value="")
def test_tty_empty_falls_back_to_default(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ", default=False) is False
    assert confirm("delete? ", default=True) is True


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input", return_value="maybe")
def test_tty_unknown_response_falls_back_to_default(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ", default=False) is False
    assert confirm("delete? ", default=True) is True


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input")
def test_non_tty_default_false_returns_false_without_input(
    mock_input, mock_stdin, caplog
):
    mock_stdin.isatty.return_value = False
    with caplog.at_level(logging.INFO, logger="box_office.utils.cli_prompts"):
        assert confirm("delete? ", default=False) is False
    mock_input.assert_not_called()
    assert any("non-interactive" in r.message for r in caplog.records)


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input")
def test_non_tty_default_true_returns_true_without_input(mock_input, mock_stdin):
    mock_stdin.isatty.return_value = False
    assert confirm("proceed? ", default=True) is True
    mock_input.assert_not_called()


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input")
def test_non_interactive_flag_overrides_tty(mock_input, mock_stdin):
    # Even when stdin is a TTY, an explicit ``non_interactive=True`` (caller
    # passed ``--yes``) bypasses the prompt.
    mock_stdin.isatty.return_value = True
    assert confirm("delete? ", default=True, non_interactive=True) is True
    assert confirm("delete? ", default=False, non_interactive=True) is False
    mock_input.assert_not_called()


@patch("box_office.utils.cli_prompts.sys.stdin")
@patch("builtins.input")
def test_non_tty_logs_auto_answer(mock_input, mock_stdin, caplog):
    mock_stdin.isatty.return_value = False
    with caplog.at_level(logging.INFO, logger="box_office.utils.cli_prompts"):
        confirm("delete X? ", default=False)
    messages = " ".join(r.message for r in caplog.records)
    assert "non-interactive" in messages
    assert "--yes" in messages
