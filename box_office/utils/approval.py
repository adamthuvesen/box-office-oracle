"""Approval gate for functions that spend money or write to production."""


class ApprovalRequired(SystemExit):
    """Raised when a spending function runs without approval.

    A ``SystemExit`` subclass, so an ``except Exception`` retry loop cannot swallow it.
    """


def require_approval(action: str, approved: bool) -> None:
    if not approved:
        raise ApprovalRequired(
            f"refused: {action} spends money or writes to production. "
            "Pass --approved (approved=True in code)."
        )
