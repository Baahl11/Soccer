from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
path = root / 'mcp_gateway' / 'clv_postgres_v4.py'
text = path.read_text(encoding='utf-8')

needle1 = """        team_totals_modeled_signal_kickoffs: dict[int, datetime] = {}\n        for signal in derivative_signals:\n"""
replacement1 = """        team_totals_modeled_signal_kickoffs: dict[int, datetime] = {}\n        team_totals_directed_audit_signals: list[dict[str, Any]] = []\n        for signal in derivative_signals:\n"""
if needle1 not in text:
    raise SystemExit('anchor1 missing')
text = text.replace(needle1, replacement1, 1)

needle2 = """            team_totals_modeled_signal_kickoffs[int(fixture_id)] = kickoff\n        derivative_family_counts = Counter(\n"""
replacement2 = """            team_totals_modeled_signal_kickoffs[int(fixture_id)] = kickoff\n            team_totals_directed_audit_signals.append({\n                'fixture_id': int(fixture_id),\n                'generated_at': signal.get('generated_at'),\n                'kickoff': kickoff,\n                'market_candidate': dict(signal.get('market_candidate') or {}),\n            })\n        derivative_family_counts = Counter(\n"""
if needle2 not in text:
    raise SystemExit('anchor2 missing')
text = text.replace(needle2, replacement2, 1)

needle3 = """        fixture_ids = sorted({int(row[\"fixture_id\"]) for row in signals if row.get(\"fixture_id\") is not None})\n"""
replacement3 = """        fixture_ids = sorted(\n            {int(row['fixture_id']) for row in signals if row.get('fixture_id') is not None}\n            | {int(row['fixture_id']) for row in team_totals_directed_audit_signals if row.get('fixture_id') is not None}\n        )\n"""
if needle3 not in text:
    raise SystemExit('anchor3 missing')
text = text.replace(needle3, replacement3, 1)

needle4 = """    comparable = [row for row in tracked if row.get(\"probability_comparable_same_line\")]\n"""
insert4 = """    # Research-only directed Team Totals audit. This does not append to tracked rows,\n    # does not change family counts, and does not create synthetic closes. It only\n    # classifies already-kicked-off Team Totals signals using the exact same strict\n    # snapshot/provider/selection semantics as the canonical matcher above.\n    now_utc = datetime.now(timezone.utc)\n    team_totals_directed_close_audit: dict[str, set[int]] = defaultdict(set)\n    for audit_signal in team_totals_directed_audit_signals:\n        fixture_id = int(audit_signal['fixture_id'])\n        generated_at = audit_signal.get('generated_at')\n        kickoff = audit_signal.get('kickoff')\n        candidate = audit_signal.get('market_candidate') or {}\n        if not isinstance(generated_at, datetime) or not isinstance(kickoff, datetime):\n            team_totals_directed_close_audit['MISSING_TIMESTAMPS'].add(fixture_id)\n            continue\n        if kickoff > now_utc:\n            continue\n        later_market_snapshots = [\n            snap\n            for snap in snapshots_by_fixture.get(fixture_id, [])\n            if snap.get('captured_at') is not None\n            and generated_at < snap['captured_at'] < kickoff\n            and _norm(snap.get('market')) == _norm(candidate.get('market'))\n        ]\n        if not later_market_snapshots:\n            team_totals_directed_close_audit['NO_LATER_PREKICKOFF_MARKET_SNAPSHOT'].add(fixture_id)\n            continue\n        strict_candidates = [\n            snap for snap in later_market_snapshots\n            if _is_strictly_later_provider_quote(snap, generated_at)\n        ]\n        if not strict_candidates:\n            team_totals_directed_close_audit['NO_LATER_PROVIDER_UPDATE'].add(fixture_id)\n            continue\n        entry_line = _num(candidate.get('line'))\n        if entry_line is None:\n            entry_line = _line_from_selection(candidate.get('selection'))\n        exact_match = False\n        selection_match = False\n        for snap in strict_candidates:\n            values = snap.get('values') if isinstance(snap.get('values'), list) else []\n            fair, price = _group_fair_probability(values, candidate.get('selection'), entry_line)\n            if fair is not None:\n                exact_match = True\n                break\n            close_line, close_price = _closing_line_candidate(values, candidate.get('selection'))\n            if close_line is not None or close_price is not None:\n                selection_match = True\n        if exact_match:\n            team_totals_directed_close_audit['STRICT_EXACT_CLOSE_EXISTS'].add(fixture_id)\n        elif selection_match:\n            team_totals_directed_close_audit['SELECTION_MATCH_LINE_MOVED'].add(fixture_id)\n        else:\n            team_totals_directed_close_audit['NO_SELECTION_MATCH_AT_CLOSE'].add(fixture_id)\n\n    comparable = [row for row in tracked if row.get(\"probability_comparable_same_line\")]\n"""
if needle4 not in text:
    raise SystemExit('anchor4 missing')
text = text.replace(needle4, insert4, 1)

needle5 = """        \"team_totals_skip_fixture_counts\": {\n            reason: len(fixture_ids)\n            for reason, fixture_ids in sorted(team_totals_skip_fixture_ids.items())\n        },\n"""
replacement5 = """        \"team_totals_skip_fixture_counts\": {\n            reason: len(fixture_ids)\n            for reason, fixture_ids in sorted(team_totals_skip_fixture_ids.items())\n        },\n        \"team_totals_directed_close_audit\": {\n            reason: sorted(fixture_ids)\n            for reason, fixture_ids in sorted(team_totals_directed_close_audit.items())\n        },\n        \"team_totals_directed_close_audit_counts\": {\n            reason: len(fixture_ids)\n            for reason, fixture_ids in sorted(team_totals_directed_close_audit.items())\n        },\n"""
if needle5 not in text:
    raise SystemExit('anchor5 missing')
text = text.replace(needle5, replacement5, 1)

path.write_text(text, encoding='utf-8')
