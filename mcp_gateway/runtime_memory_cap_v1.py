"""Bound the upstream provider phase for small-memory canonical schedulers.

The platform's existing memory guard correctly aborts a worker near cgroup
OOM. This fail-closed admission budget prevents the *same upstream phase*
from performing ~50 calls on a 512MiB service before expensive payload
assembly. No model, decision, price, or calibration math is modified.

The resource cap is only a LOWER bound on allowed work relative to the
existing quota-derived cap. Unused matches are deferred, never fabricated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

VERSION = "SOCCER_SCHEDULER_LOW_MEMORY_UPSTREAM_BUDGET_v1.0"
SMALL_INSTANCE_MAX_BYTES = 768 * 1024 * 1024
MIN_VALID_CGROUP_BYTES = 128 * 1024 * 1024
DEFAULT_SMALL_INSTANCE_UPSTREAM_CALLS = 28
MIN_SAFE_CALLS = 12
MAX_SAFE_CALLS = 32


def cgroup_memory_limit_bytes() -> int | None:
    """Read actual cgroup memory limit, excluding pseudo-unlimited values."""
    for file in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(file).read_text(encoding="utf-8").strip()
            if not raw.isdigit():
                continue
            value = int(raw)
            if MIN_VALID_CGROUP_BYTES <= value < (1 << 50):
                return value
        except (OSError, ValueError):
            continue
    return None


def safe_upstream_phase_cap(
    existing_cap: int, *, cgroup_limit_bytes: int | None,
    small_instance_cap: int = DEFAULT_SMALL_INSTANCE_UPSTREAM_CALLS,
) -> dict[str, Any]:
    """Apply at most one restrictive cap; never enlarge upstream allowance."""
    if type(existing_cap) is not int or existing_cap < 1:
        raise ValueError("INVALID_EXISTING_UPSTREAM_CAP")
    if type(small_instance_cap) is not int or not MIN_SAFE_CALLS <= small_instance_cap <= MAX_SAFE_CALLS:
        raise ValueError("INVALID_SMALL_INSTANCE_UPSTREAM_CAP")
    small = (type(cgroup_limit_bytes) is int
             and MIN_VALID_CGROUP_BYTES <= cgroup_limit_bytes <= SMALL_INSTANCE_MAX_BYTES)
    cap = min(existing_cap, small_instance_cap) if small else existing_cap
    return {
        "version": VERSION,
        "applied": cap < existing_cap,
        "cgroup_limit_bytes": cgroup_limit_bytes if type(cgroup_limit_bytes) is int else None,
        "small_instance": bool(small),
        "existing_upstream_cap": existing_cap,
        "effective_upstream_cap": cap,
        "deferred_extra_upstream_calls_max": max(0, existing_cap - cap),
        "provider_budget_increased": False,
        "sport_model_weights_changed": False,
        "market_gates_changed": False,
        "source": "CGROUP_MEMORY_LIMIT",
        "policy": "LOW_MEMORY_UPSTREAM_ONLY; ZERO_PROVIDER_BUDGET_INCREASE; "
                  "DEFER_BEFORE_OOM; NO_SYNTHETIC_EVENTS",
    }
