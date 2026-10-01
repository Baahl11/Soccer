from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
path = root / 'mcp_gateway' / 'automation_v126.py'
text = path.read_text(encoding='utf-8')

old = '''            current = max(1, _as_int(phase_lock.get("low_water_cap"), configured))
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
'''
new = '''            current = max(1, _as_int(phase_lock.get("low_water_cap"), configured))
            if v2._LAST_DAILY_REMAINING is None:
                # Before the first verified quota response, `configured` may
                # already be the reserved upstream cap installed by v90/v6.
                # Never reserve a second time from that already-reserved value.
                quota_global_cap = None
                quota_upstream_cap = configured
                quota_reserved_calls = _as_int(phase_lock.get("reserved_calls"), 0)
            else:
                quota_global_cap = _quota_aware_global_cap()
                quota_upstream_cap, quota_reserved_calls = v123._reserve_from_elastic_cap(quota_global_cap)

            tightened = min(current, configured, max(1, int(quota_upstream_cap)))
            if tightened < current:
                phase_lock["tighten_count"] = int(phase_lock.get("tighten_count") or 0) + 1
            phase_lock["low_water_cap"] = tightened
            if quota_global_cap is not None:
                phase_lock["quota_global_cap"] = int(quota_global_cap)
                phase_lock["quota_upstream_cap"] = int(quota_upstream_cap)
                phase_lock["reserved_calls"] = int(quota_reserved_calls)
                phase_lock["quota_remaining_basis"] = v2._LAST_DAILY_REMAINING
'''
if old not in text:
    raise SystemExit('patched lock anchor missing')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
