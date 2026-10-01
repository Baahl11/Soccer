from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
path = root / 'mcp_gateway' / 'clv_postgres_v4.py'
text = path.read_text()

replacements = [
    (
'''    skip_reason_market_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skip_reason_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    ft_totals_unpriced_research_placeholders_ignored = 0
''',
'''    skip_reason_market_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skip_reason_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    team_totals_skip_fixture_ids: dict[str, set[int]] = defaultdict(set)
    ft_totals_unpriced_research_placeholders_ignored = 0
'''),
    (
'''            if not later_market_snapshots:
                reasons["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"] += 1
                skip_reason_market_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][family] += 1
                continue
''',
'''            if not later_market_snapshots:
                reasons["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"] += 1
                skip_reason_market_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][family] += 1
                if family in {"TEAM_TOTALS", "HOME_TT", "AWAY_TT"}:
                    team_totals_skip_fixture_ids["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"].add(fixture_id)
                continue
'''),
    (
'''            if not candidates:
                reasons["NO_LATER_PROVIDER_UPDATE"] += 1
                skip_reason_market_counts["NO_LATER_PROVIDER_UPDATE"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PROVIDER_UPDATE"][family] += 1
                continue
''',
'''            if not candidates:
                reasons["NO_LATER_PROVIDER_UPDATE"] += 1
                skip_reason_market_counts["NO_LATER_PROVIDER_UPDATE"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PROVIDER_UPDATE"][family] += 1
                if family in {"TEAM_TOTALS", "HOME_TT", "AWAY_TT"}:
                    team_totals_skip_fixture_ids["NO_LATER_PROVIDER_UPDATE"].add(fixture_id)
                continue
'''),
    (
'''        "signal_source_counts": dict(sorted(signal_source_counts.items())),
        "team_totals_maturation_funnel": team_totals_maturation_funnel,
        "skip_reasons": dict(sorted(reasons.items())),
''',
'''        "signal_source_counts": dict(sorted(signal_source_counts.items())),
        "team_totals_maturation_funnel": team_totals_maturation_funnel,
        "team_totals_skip_fixture_ids": {
            reason: sorted(fixture_ids)[:100]
            for reason, fixture_ids in sorted(team_totals_skip_fixture_ids.items())
        },
        "team_totals_skip_fixture_counts": {
            reason: len(fixture_ids)
            for reason, fixture_ids in sorted(team_totals_skip_fixture_ids.items())
        },
        "skip_reasons": dict(sorted(reasons.items())),
'''),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'anchor not found: {old.splitlines()[0]}')
    text = text.replace(old, new, 1)

path.write_text(text)
