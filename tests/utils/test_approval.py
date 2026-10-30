"""The approval gate cannot be swallowed by a generic retry loop."""

import pytest

from box_office.utils.approval import ApprovalRequired, require_approval


def test_refusal_is_not_an_exception_subclass():
    assert issubclass(ApprovalRequired, SystemExit)
    assert not issubclass(ApprovalRequired, Exception)


def test_except_exception_does_not_swallow_the_refusal():
    with pytest.raises(ApprovalRequired):
        try:
            require_approval("spending", approved=False)
        except Exception:
            pytest.fail("a generic except Exception caught the refusal")


def test_refusal_names_the_action():
    with pytest.raises(ApprovalRequired, match="launching a job"):
        require_approval("launching a job", approved=False)


def test_approved_passes():
    require_approval("launching a job", approved=True)
