from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
path = root / 'mcp_gateway' / 'clv_postgres_v4.py'
text = path.read_text()

replacements = [
    (
'''    future_kickoffs: list[datetime] = []
    for fixture_id in signal_without_true_clv:
''',
'''    future_kickoffs: list[datetime] = []
    already_kicked_off_fixture_ids: list[int] = []
    pending_future_fixture_ids: list[int] = []
    for fixture_id in signal_without_true_clv:
'''),
    (
'''        if delta.total_seconds() <= 0:
            timing["already_kicked_off"] += 1
            continue
        future_kickoffs.append(kickoff)
''',
'''        if delta.total_seconds() <= 0:
            timing["already_kicked_off"] += 1
            already_kicked_off_fixture_ids.append(int(fixture_id))
            continue
        future_kickoffs.append(kickoff)
        pending_future_fixture_ids.append(int(fixture_id))
'''),
    (
'''        "pending_timing": timing,
        "pending_future_fixtures": len(future_kickoffs),
        "next_pending_kickoff": min(future_kickoffs).isoformat() if future_kickoffs else None,
''',
'''        "pending_timing": timing,
        "already_kicked_off_fixture_ids": sorted(already_kicked_off_fixture_ids),
        "pending_future_fixture_ids": sorted(pending_future_fixture_ids),
        "pending_future_fixtures": len(future_kickoffs),
        "next_pending_kickoff": min(future_kickoffs).isoformat() if future_kickoffs else None,
'''),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'anchor not found: {old.splitlines()[0]}')
    text = text.replace(old, new, 1)

path.write_text(text)
