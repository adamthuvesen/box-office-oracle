"""Shared confirmation prompt helper for destructive CLI scripts.

Wraps ``input()`` so cron / Prefect / CI / SageMaker pipeline runs (no TTY
on stdin) never block waiting on a human. In non-interactive contexts the
helper logs a single clear line and returns the caller's ``default`` instead
of blocking.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def confirm(
    prompt: str,
    *,
    default: bool = False,
    non_interactive: bool = False,
) -> bool:
    """Ask the operator to confirm a destructive action.

    Args:
        prompt: Text shown to the operator (caller supplies any "(y/N)" hint).
        default: Value returned for non-interactive runs and for empty /
            unrecognized responses in interactive runs.
        non_interactive: When True (e.g. caller passed ``--yes``), skip the
            ``input()`` call and return ``default`` immediately.

    Returns:
        True if the operator answered yes, False otherwise. In non-interactive
        contexts, returns ``default`` after logging the auto-answer.
    """
    is_tty = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False

    if non_interactive or not is_tty:
        logger.info(
            "non-interactive run; auto-answering %s; pass --yes to override "
            "(prompt was: %r)",
            default,
            prompt,
        )
        return default

    response = input(prompt).strip().lower()
    if response in {"y", "yes"}:
        return True
    if response in {"n", "no"}:
        return False
    return default
