from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
path = root / 'mcp_gateway' / 'automation_v126.py'
text = path.read_text(encoding='utf-8')

old_import = "from mcp_gateway import automation_v4 as v4\n"
new_import = "from mcp_gateway import automation_v4 as v4\nfrom mcp_gateway import automation_v90 as v90\n"
if new_import not in text:
    if old_import not in text:
        raise SystemExit('import anchor missing')
    text = text.replace(old_import, new_import, 1)

old_release = '''def _release_reserved_price_phase_cap() -> dict[str, Any]:
    """Release the v208 low-water cap only at the reserved price phase boundary.

    v208 correctly prevented later upstream layers from silently reopening the
    reserved cap, but that low-water mark must not survive into the explicitly
    reserved post-upstream price resolver phase. The global elastic cap has
    already been restored by v123 before `_fetch_fixture_odds` is called.

    This function never raises the declared global cap and never adds budget;
    it only lets the reserved portion of that same cap become usable by the
    component it was reserved for.
    """
    try:
        declared_global_cap = max(1, int(v2.MAX_API_CALLS_PER_TICK))
    except (TypeError, ValueError):
        declared_global_cap = 1
    try:
        calls_at_release = max(0, int(v2._API_CALLS_THIS_TICK or 0))
    except (TypeError, ValueError):
        calls_at_release = 0

    previous = v4._MONOTONIC_TICK_CAP
    previous_cap = _as_int(previous, declared_global_cap) if previous is not None else None
    released = previous is not None and declared_global_cap > int(previous)
    if released:
        # The global elastic cap is authoritative at this explicit phase
        # boundary. It must already be >= used calls; clamp defensively.
        v4._MONOTONIC_TICK_CAP = max(calls_at_release, declared_global_cap)

    return {
        "released": bool(released),
        "previous_low_water_cap": previous_cap,
        "declared_global_cap": declared_global_cap,
        "effective_cap_after_release": _as_int(v4._MONOTONIC_TICK_CAP, declared_global_cap),
        "provider_calls_at_release": calls_at_release,
        "provider_budget_changed": False,
        "provider_requests_added": 0,
        "phase": "POST_UPSTREAM_RESERVED_PRICE_RESOLUTION",
    }
'''
new_release = '''def _quota_aware_global_cap() -> int:
    """Return the existing elastic global cap from the latest verified quota."""
    remaining = v2._LAST_DAILY_REMAINING
    if remaining is not None:
        try:
            global_cap, _ = v90._elastic_request_cap(int(remaining))
            return max(1, int(global_cap))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(v2.MAX_API_CALLS_PER_TICK))
    except (TypeError, ValueError):
        return 1


def _release_reserved_price_phase_cap() -> dict[str, Any]:
    """Release only to the already-existing quota-aware global cap."""
    declared_global_cap = _quota_aware_global_cap()
    try:
        calls_at_release = max(0, int(v2._API_CALLS_THIS_TICK or 0))
    except (TypeError, ValueError):
        calls_at_release = 0

    previous = v4._MONOTONIC_TICK_CAP
    previous_cap = _as_int(previous, declared_global_cap) if previous is not None else None
    released = previous is not None and declared_global_cap > int(previous)
    if released:
        # This does not increase the provider policy budget. It only restores
        # the reserved slice inside the same quota-derived global cap.
        v4._MONOTONIC_TICK_CAP = max(calls_at_release, declared_global_cap)

    return {
        "released": bool(released),
        "previous_low_water_cap": previous_cap,
        "declared_global_cap": declared_global_cap,
        "quota_remaining_basis": v2._LAST_DAILY_REMAINING,
        "effective_cap_after_release": _as_int(v4._MONOTONIC_TICK_CAP, declared_global_cap),
        "provider_calls_at_release": calls_at_release,
        "provider_budget_changed": False,
        "provider_requests_added": 0,
        "phase": "POST_UPSTREAM_RESERVED_PRICE_RESOLUTION",
    }
'''
if old_release not in text:
    raise SystemExit('release anchor missing')
text = text.replace(old_release, new_release, 1)

old_lock = '''    def phase_locked_effective_tick_cap() -> int:
        try:
            configured = max(1, int(v2.MAX_API_CALLS_PER_TICK))
        except (TypeError, ValueError):
            configured = 1

        if phase_lock["active"]:
            current = max(1, _as_int(phase_lock.get("low_water_cap"), configured))
            tightened = min(current, configured)
            if tightened < current:
                phase_lock["tighten_count"] = int(phase_lock.get("tighten_count") or 0) + 1
            phase_lock["low_water_cap"] = tightened

            # Call the canonical v4 low-water implementation with a temporary
            # configured cap that cannot exceed the explicit upstream phase lock.
            original_configured = v2.MAX_API_CALLS_PER_TICK
            v2.MAX_API_CALLS_PER_TICK = tightened
            try:
                return original_effective_tick_cap()
            finally:
                v2.MAX_API_CALLS_PER_TICK = original_configured

        return original_effective_tick_cap()
'''
new_lock = '''    def phase_locked_effective_tick_cap() -> int:
        try:
            configured = max(1, int(v2.MAX_API_CALLS_PER_TICK))
        except (TypeError, ValueError):
            configured = 1

        if phase_lock["active"]:
            current = max(1, _as_int(phase_lock.get("low_water_cap"), configured))
            quota_global_cap = _quota_aware_global_cap()
            quota_upstream_cap, quota_reserved_calls = v123._reserve_from_elastic_cap(quota_global_cap)
            tightened = min(current, configured, max(1, int(quota_upstream_cap)))
            if tightened < current:
                phase_lock["tighten_count"] = int(phase_lock.get("tighten_count") or 0) + 1
            phase_lock["low_water_cap"] = tightened
            phase_lock["quota_global_cap"] = int(quota_global_cap)
            phase_lock["quota_upstream_cap"] = int(quota_upstream_cap)
            phase_lock["reserved_calls"] = int(quota_reserved_calls)
            phase_lock["quota_remaining_basis"] = v2._LAST_DAILY_REMAINING

            # Call the canonical v4 low-water implementation with a temporary
            # configured cap that cannot exceed the quota-aware reserved cap.
            original_configured = v2.MAX_API_CALLS_PER_TICK
            v2.MAX_API_CALLS_PER_TICK = tightened
            try:
                effective = int(original_effective_tick_cap())
                phase_lock["low_water_cap"] = min(tightened, effective)
                return int(phase_lock["low_water_cap"])
            finally:
                v2.MAX_API_CALLS_PER_TICK = original_configured

        return original_effective_tick_cap()
'''
if old_lock not in text:
    raise SystemExit('lock anchor missing')
text = text.replace(old_lock, new_lock, 1)

text = text.replace('v209.4 UPSTREAM CAP PRESEED/LOCK + RESERVED-PHASE RELEASE:', 'v216.5 QUOTA-AWARE UPSTREAM CAP LOCK + RESERVED-PHASE RELEASE:', 1)
path.write_text(text, encoding='utf-8')
