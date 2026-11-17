"""Bounded, read-only coverage of canonical FT corners evidence; not OOS scores."""
from collections import Counter
from datetime import datetime, timedelta, timezone

from mcp_gateway.formation_postgres_audit import _formations

BASELINE_END = datetime(2026, 9, 23, 6, tzinfo=timezone.utc)
LIMIT = 128
LINEUP_LIMIT = 4

SQL = """
WITH candidates AS MATERIALIZED (
    SELECT fixture_id, kickoff, league_id, home_team_id, away_team_id
    FROM soccer_fixtures
    WHERE kickoff >= %s AND kickoff < %s AND status = 'FT'
    ORDER BY kickoff DESC, fixture_id DESC LIMIT %s
)
SELECT f.kickoff, f.home_team_id, f.away_team_id, COALESCE(l.snapshots, '[]'::jsonb),
       EXISTS (
         SELECT 1 FROM soccer_fe11_pregame_baselines b
         JOIN soccer_fe11_verified_corner_snapshots c ON c.fixture_id = b.fixture_id
         WHERE b.fixture_id = f.fixture_id AND b.kickoff = f.kickoff
           AND b.league_id = f.league_id AND b.home_team_id = f.home_team_id
           AND b.away_team_id = f.away_team_id AND c.kickoff = f.kickoff
           AND c.observed_at > f.kickoff AND c.observed_at <= %s
           AND c.period = 'FT_90' AND c.source = 'API_FOOTBALL_FIXTURE_STATISTICS'
           AND c.source_response_verified IS TRUE
       )
FROM candidates f
LEFT JOIN LATERAL (
    SELECT jsonb_agg(s.payload) AS snapshots
    FROM (
        SELECT payload FROM soccer_lineup_snapshots
        WHERE fixture_id = f.fixture_id AND both_xi_confirmed IS TRUE
          AND captured_at < f.kickoff AND stage IS DISTINCT FROM 'POSTGAME'
        ORDER BY captured_at DESC, snapshot_id DESC LIMIT %s
    ) s
) l ON TRUE
ORDER BY f.kickoff DESC, f.fixture_id DESC
"""


def bucket(home, away, snapshots, truth):
    formation = False
    for payload in snapshots:
        if not isinstance(payload, dict):
            continue
        try:
            h, a, _ = _formations(payload, home, away)
        except (ValueError, TypeError):
            continue
        if h and a and home != away:
            formation = True
            break
    if formation and truth:
        return "paired"
    if formation:
        return "formation_only"
    if truth:
        return "official_ft_only"
    return "neither"


def inventory(cur, now):
    # Independent seven-day bounded samples preserve older evidence and show
    # whether recent weeks are actually feeding the pipeline. No model grading.
    windows = []
    end = now
    for _ in range(8):
        start = max(BASELINE_END, end - timedelta(days=7))
        if start >= end:
            break
        with cur.connection.transaction():
            cur.execute(SQL, (start, end, LIMIT + 1, now, LINEUP_LIMIT))
            rows = cur.fetchall()
        counts = Counter(bucket(home, away, snaps, truth)
                         for _, home, away, snaps, truth in rows[:LIMIT])
        windows.append({"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                        "sampled": min(len(rows), LIMIT), "truncated": len(rows) > LIMIT,
                        **{k: counts[k] for k in ("paired", "formation_only", "official_ft_only", "neither")}})
        end = start
    return windows
