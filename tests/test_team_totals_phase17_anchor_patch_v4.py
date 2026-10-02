from datetime import datetime, timezone

from mcp_gateway import team_totals_phase17_anchor_patch_v4 as patch


def _signal(*, fixture_id: int, at: str, market: str, selection: str, line: float, source: str = "DERIVATIVE_INTELLIGENCE:team_totals_intelligence"):
    return {
        "fixture_id": fixture_id,
        "generated_at": datetime.fromisoformat(at).replace(tzinfo=timezone.utc),
        "signal_source": source,
        "market_candidate": {
            "market": market,
            "selection": selection,
            "line": line,
            "decimal_price": 1.91,
        },
    }


def test_collapses_recycled_team_totals_to_oldest_exact_signal_only():
    newest = _signal(
        fixture_id=1549849,
        at="2026-10-02T00:40:00",
        market="Total - Home",
        selection="OVER",
        line=1.5,
    )
    oldest = _signal(
        fixture_id=1549849,
        at="2026-10-01T22:00:00",
        market="Total - Home",
        selection="Over 1.5",
        line=1.5,
    )
    different_line = _signal(
        fixture_id=1549849,
        at="2026-10-01T22:05:00",
        market="Total - Home",
        selection="OVER",
        line=2.5,
    )
    other_family = _signal(
        fixture_id=999,
        at="2026-10-02T00:45:00",
        market="Goals Over/Under First Half",
        selection="Over",
        line=1.5,
        source="DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence",
    )

    collapsed = patch.collapse_oldest_exact_team_totals(
        [newest, other_family, different_line, oldest]
    )

    team_totals = [
        row
        for row in collapsed
        if row["signal_source"] == "DERIVATIVE_INTELLIGENCE:team_totals_intelligence"
    ]
    assert len(team_totals) == 2

    line_15 = next(row for row in team_totals if row["market_candidate"]["line"] == 1.5)
    assert line_15["generated_at"] == oldest["generated_at"]

    line_25 = next(row for row in team_totals if row["market_candidate"]["line"] == 2.5)
    assert line_25["generated_at"] == different_line["generated_at"]

    assert other_family in collapsed


def test_exact_key_normalizes_over_under_wording_but_keeps_home_away_separate():
    over_plain = _signal(
        fixture_id=1,
        at="2026-10-01T20:00:00",
        market="Total - Home",
        selection="OVER",
        line=1.5,
    )
    over_with_line = _signal(
        fixture_id=1,
        at="2026-10-01T21:00:00",
        market="Total - Home",
        selection="Over 1.5",
        line=1.5,
    )
    away = _signal(
        fixture_id=1,
        at="2026-10-01T20:30:00",
        market="Total - Away",
        selection="OVER",
        line=1.5,
    )

    collapsed = patch.collapse_oldest_exact_team_totals([over_with_line, away, over_plain])
    assert len(collapsed) == 2
    assert over_plain in collapsed
    assert over_with_line not in collapsed
    assert away in collapsed


def test_recent_cap_independent_slice_replaces_rows_without_increasing_max_rows():
    raw_old_tt = _signal(
        fixture_id=1,
        at="2026-10-02T02:00:00",
        market="Total - Home",
        selection="OVER",
        line=1.5,
    )
    raw_other = _signal(
        fixture_id=20,
        at="2026-10-02T03:00:00",
        market="Goals Over/Under First Half",
        selection="Over",
        line=1.5,
        source="DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence",
    )
    raw_tail = _signal(
        fixture_id=21,
        at="2026-10-02T03:10:00",
        market="Goals Over/Under - Second Half",
        selection="Under",
        line=1.5,
        source="DERIVATIVE_INTELLIGENCE:two_h_goals_intelligence",
    )
    missing_recent = _signal(
        fixture_id=1549849,
        at="2026-10-01T22:00:00",
        market="Total - Home",
        selection="OVER",
        line=1.5,
    )

    merged = patch.merge_bounded_recent_team_totals(
        [raw_old_tt, raw_other, raw_tail],
        [missing_recent],
        max_rows=3,
    )

    assert len(merged) == 3
    assert merged[0] is missing_recent
    assert raw_other in merged
    assert raw_tail not in merged


def test_recent_slice_replaces_recycled_same_exact_key_with_oldest_recent_anchor():
    raw_recycled = _signal(
        fixture_id=1490463,
        at="2026-10-02T01:10:00",
        market="Total - Away",
        selection="UNDER",
        line=2.5,
    )
    recent_oldest = _signal(
        fixture_id=1490463,
        at="2026-10-01T22:10:00",
        market="Total - Away",
        selection="Under 2.5",
        line=2.5,
    )
    other_family = _signal(
        fixture_id=30,
        at="2026-10-02T01:20:00",
        market="Goals Over/Under First Half",
        selection="Over",
        line=1.5,
        source="DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence",
    )

    merged = patch.merge_bounded_recent_team_totals(
        [raw_recycled, other_family],
        [recent_oldest],
        max_rows=2,
    )

    assert len(merged) == 2
    assert recent_oldest in merged
    assert raw_recycled not in merged
    assert other_family in merged
