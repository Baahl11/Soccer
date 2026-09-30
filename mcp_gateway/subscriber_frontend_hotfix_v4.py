from __future__ import annotations


def install(subscriber_app) -> None:
    """Retired compatibility shim.

    The subscriber UI is moving to a clean preview-first frontend shell.  Keep this
    installer deliberately inert so legacy string-replacement patches cannot mutate
    the rendered HTML or hide persisted markets while the new frontend is built.
    Canonical model logic, thresholds, gates, provider budget, strict-close rules,
    and production BET logic are untouched.
    """
    return
