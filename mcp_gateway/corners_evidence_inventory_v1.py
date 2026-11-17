"""Bounded, read-only coverage of canonical FT corners evidence; not OOS scores."""
from collections import Counter
from datetime import datetime, timedelta, timezone

from mcp_gateway.formation_postgres_audit import _formations

BASELINE_END = datetime(2026, 9, 23, 6, tzinfo=timezone.utc)
LIMIT = 128
LINEUP_LIMIT = 4
OBSERVATION_LIMIT = 4

SQL = """
WITH candidates AS MATERIALIZED (
    SELECT fixture_id, kickoff, league_id, home_team_id, away_team_id
    FROM soccer_fixtures
    WHERE kickoff >= %s AND kickoff < %s AND status = 'FT'
    ORDER BY kickoff DESC, fixture_id DESC LIMIT %s
)
SELECT f.fixture_id, f.kickoff, f.league_id, f.home_team_id, f.away_team_id,
       COALESCE(l.snapshots, '[]'::jsonb),
       EXISTS (
         SELECT 1 FROM soccer_fe11_pregame_baselines b
         JOIN soccer_fe11_verified_corner_snapshots c ON c.fixture_id = b.fixture_id
         WHERE b.fixture_id = f.fixture_id AND b.kickoff = f.kickoff
           AND b.league_id = f.league_id AND b.home_team_id = f.home_team_id
           AND b.away_team_id = f.away_team_id AND c.kickoff = f.kickoff
           AND c.observed_at > f.kickoff AND c.observed_at <= %s
           AND c.period = 'FT_90' AND c.source = 'API_FOOTBALL_FIXTURE_STATISTICS'
           AND c.source_response_verified IS TRUE
       ), COALESCE(o.snapshots, '[]'::jsonb)
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
LEFT JOIN LATERAL (
    SELECT jsonb_agg(s.payload) AS snapshots
    FROM (
        SELECT payload FROM soccer_refresh_events
        WHERE fixture_id = f.fixture_id
          AND event_type = 'FIXTURE_OBSERVATION_RESEARCH'
          AND stage = 'FIXTURE_OBSERVATION' AND bet_eligible IS FALSE
          AND generated_at > f.kickoff AND generated_at <= %s
        ORDER BY generated_at DESC, event_id DESC LIMIT %s
    ) s
) o ON TRUE
ORDER BY f.kickoff DESC, f.fixture_id DESC
"""


def verified_observation_corners(payloads, fixture_id, kickoff, league, home, away):
    """Use original persisted provider observations; never infer missing stats."""
    from mcp_gateway.fe11_postgame_corners_v1 import parse_verified_ft_corners

    counts = set()
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("event_type") != "FIXTURE_OBSERVATION_RESEARCH":
            continue
        if payload.get("stage") != "FIXTURE_OBSERVATION" or payload.get("bet_eligible") is not False:
            continue
        fx, sport = payload.get("fixture"), payload.get("sporting")
        if not isinstance(fx, dict) or not isinstance(sport, dict):
            continue
        identity = {"fixture_id": fixture_id, "league_id": league,
                    "home_team_id": home, "away_team_id": away}
        if any(type(fx.get(k)) is not int or fx[k] != v for k, v in identity.items()):
            continue
        if fx.get("status") != "FT" or sport.get("collection_scope") != "POSTGAME_OBSERVATION":
            continue
        try:
            stamp = datetime.fromisoformat(str(fx.get("kickoff")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp != kickoff:
            continue
        result = parse_verified_ft_corners(sport.get("fixture_statistics"), home, away)
        if result is not None:
            counts.add((result["home"], result["away"]))
    return next(iter(counts)) if len(counts) == 1 else None


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
            cur.execute(SQL, (start, end, LIMIT + 1, now, LINEUP_LIMIT, now, OBSERVATION_LIMIT))
            rows = cur.fetchall()
        counts = Counter()
        added_pairs = added_truth = 0
        for fid, kickoff, league, home, away, snaps, fe11_truth, observations in rows[:LIMIT]:
            # FE11 remains authoritative when present. This bridge only fills absent truth.
            extra = not fe11_truth and verified_observation_corners(
                observations, fid, kickoff, league, home, away) is not None
            state = bucket(home, away, snaps, fe11_truth or extra)
            counts[state] += 1
            added_truth += int(extra)
            added_pairs += int(extra and state == "paired")
        windows.append({"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                        "sampled": min(len(rows), LIMIT), "truncated": len(rows) > LIMIT,
                        "recovered_observation_truth": added_truth, "recovered_observation_pairs": added_pairs,
                        **{k: counts[k] for k in ("paired", "formation_only", "official_ft_only", "neither")}})
        end = start
    return windows
