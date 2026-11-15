"""Approval gate for functions that spend money or write to production."""

# Every refusal message starts with this; scripts/build_training_image.sh prints it too.
REFUSAL_TOKEN = "refused:"


class ApprovalRequired(SystemExit):
    """Raised when a spending function runs without approval.

    A ``SystemExit`` subclass, so an ``except Exception`` retry loop cannot swallow it.
    """


def require_approval(action: str, approved: bool) -> None:
    if not approved:
        raise ApprovalRequired(
            f"{REFUSAL_TOKEN} {action} spends money or writes to production. "
            "Pass --approved (approved=True in code)."
        )
