"""Bounded coverage, not model grading or bookmaker settlement."""
from collections import Counter
from datetime import datetime, timedelta

from mcp_gateway import cards_intelligence_live, corners_intelligence, red_cards_intelligence
from mcp_gateway.fe11_postgame_corners_v1 import parse_verified_ft_corners

FAMILIES = ("FT_CORNERS", "TEAM_CORNERS", "FT_YELLOWS", "TEAM_YELLOWS", "ANY_RED")
STATES = ("both", "prediction_only", "outcome_only", "neither")
LIMIT = 32
EVENTS = 8
FIELDS = ("corners_intelligence", "cards_intelligence_live", "team_cards_intelligence", "red_cards_intelligence")

SQL = """
WITH sample AS MATERIALIZED (
  SELECT fixture_id, league_id, season, home_team_id, away_team_id, kickoff
  FROM soccer_fixtures
  WHERE kickoff >= %s AND kickoff < %s AND status = 'FT'
  ORDER BY kickoff DESC, fixture_id DESC LIMIT %s
)
SELECT s.fixture_id, s.league_id, s.season, s.home_team_id, s.away_team_id, s.kickoff,
       COALESCE(p.snapshots, '[]'::jsonb), COALESCE(o.snapshots, '[]'::jsonb), COALESCE(c.counts, '[]'::jsonb)
FROM sample s
LEFT JOIN LATERAL (
  SELECT jsonb_agg(jsonb_build_object('stage', stage, 'event_type', event_type,
    'generated_at', generated_at, 'payload', payload) ORDER BY generated_at DESC, event_id DESC) AS snapshots
  FROM (
    SELECT stage, event_type, generated_at, event_id, payload FROM soccer_refresh_events
    WHERE fixture_id=s.fixture_id AND generated_at < s.kickoff AND event_type='SOCCER_REFRESH'
      AND payload ?| %s::text[]
    ORDER BY generated_at DESC, event_id DESC LIMIT %s
  ) rows
) p ON TRUE
LEFT JOIN LATERAL (
  SELECT jsonb_agg(payload ORDER BY generated_at DESC, event_id DESC) AS snapshots
  FROM (
    SELECT generated_at, event_id, payload FROM soccer_refresh_events
    WHERE fixture_id=s.fixture_id AND generated_at > s.kickoff AND generated_at <= %s
      AND event_type='FIXTURE_OBSERVATION_RESEARCH' AND stage='FIXTURE_OBSERVATION'
      AND bet_eligible IS FALSE
      AND payload #>> '{sporting,collection_scope}'='POSTGAME_OBSERVATION'
    ORDER BY generated_at DESC, event_id DESC LIMIT %s
  ) rows
) o ON TRUE
LEFT JOIN LATERAL (
  SELECT jsonb_agg(jsonb_build_array(c.home_corners, c.away_corners)) AS counts
  FROM (
    SELECT c.home_corners, c.away_corners FROM soccer_fe11_verified_corner_snapshots c
    JOIN soccer_fe11_pregame_baselines b ON b.fixture_id=c.fixture_id
    WHERE c.fixture_id=s.fixture_id AND c.kickoff=s.kickoff AND b.kickoff=s.kickoff
      AND b.league_id=s.league_id AND b.home_team_id=s.home_team_id AND b.away_team_id=s.away_team_id
      AND c.period='FT_90' AND c.source='API_FOOTBALL_FIXTURE_STATISTICS'
      AND c.source_response_verified IS TRUE AND c.bet_eligible IS FALSE
      AND c.observed_at > s.kickoff AND c.observed_at <= %s
    LIMIT %s
  ) c
) c ON TRUE
"""


def card_counts(raw, home, away, metric):
    if not isinstance(raw, list) or len(raw) != 2 or home == away:
        return None
    counts = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("team"), dict):
            return None
        team = item["team"].get("id")
        stats = item.get("statistics")
        if type(team) is not int or team not in (home, away) or team in counts or not isinstance(stats, list):
            return None
        entries = [s.get("value") for s in stats if isinstance(s, dict) and s.get("type") == metric]
        if len(entries) != 1 or type(entries[0]) is not int or not 0 <= entries[0] <= 100:
            return None
        counts[team] = entries[0]
    return counts[home], counts[away]


def outcomes(observations, fixture, fe11=()):
    """Provider FT counts only. Missing, conflicting, or truncated is not zero."""
    seen = {metric: set() for metric in ("corners", "yellow", "red")}
    if len(observations) > EVENTS:
        return set()
    home, away = fixture["home_team_id"], fixture["away_team_id"]
    for payload in observations:
        if not isinstance(payload, dict):
            continue
        fx, sport = payload.get("fixture"), payload.get("sporting")
        if not isinstance(fx, dict) or not isinstance(sport, dict):
            continue
        if (payload.get("event_type") != "FIXTURE_OBSERVATION_RESEARCH"
                or payload.get("stage") != "FIXTURE_OBSERVATION" or payload.get("bet_eligible") is not False
                or fx.get("status") != "FT" or sport.get("collection_scope") != "POSTGAME_OBSERVATION"):
            continue
        if any(type(fx.get(k)) is not int or fx[k] != fixture[k]
               for k in ("fixture_id", "league_id", "home_team_id", "away_team_id")):
            continue
        try:
            if datetime.fromisoformat(str(fx.get("kickoff")).replace("Z", "+00:00")) != datetime.fromisoformat(fixture["kickoff"]):
                continue
        except ValueError:
            continue
        raw = sport.get("fixture_statistics")
        corners = parse_verified_ft_corners(raw, home, away)
        if corners is not None:
            seen["corners"].add((corners["home"], corners["away"]))
        for key, metric in (("yellow", "Yellow Cards"), ("red", "Red Cards")):
            cards = card_counts(raw, home, away, metric)
            if cards is not None:
                seen[key].add(cards)
    if len(fe11) > EVENTS:
        seen["corners"].clear()
    else:
        seen["corners"].update(tuple(pair) for pair in fe11)
    return {metric for metric, values in seen.items() if len(values) == 1}


def inventory(cur, now):
    cur.execute(SQL, (now - timedelta(days=14), now, LIMIT + 1, list(FIELDS), EVENTS, now, EVENTS + 1, now, EVENTS + 1))
    rows = cur.fetchall()
    counts = {f: Counter() for f in FAMILIES}
    truncated = 0
    missing = []
    for fid, league, season, home, away, kickoff, events, obs, fe11 in rows[:LIMIT]:
        fixture = dict(fixture_id=fid, league_id=league, season=season, home_team_id=home,
                       away_team_id=away, kickoff=kickoff.isoformat())
        corners = corners_intelligence.saved_match_rates(events, fixture)["status"] == "PERSISTED_RESEARCH_RATES"
        yellow = cards_intelligence_live.saved_match_rates(events, fixture)["status"] == "PERSISTED_RESEARCH_RATES"
        team_yellow = cards_intelligence_live.saved_match_rates(events, fixture, by_team=True)["status"] == "PERSISTED_RESEARCH_RATES"
        red = red_cards_intelligence.saved_match_probability(events, fixture)["status"] == "PERSISTED_RESEARCH_PROJECTION"
        truth = outcomes(obs, fixture, fe11)
        gaps = []
        truncated += int(len(obs) > EVENTS)
        for family, pred, metric in zip(FAMILIES, (corners, corners, yellow, team_yellow, red),
                                        ("corners", "corners", "yellow", "yellow", "red")):
            observed = metric in truth
            if pred and not observed:
                gaps.append(family)
            counts[family]["both" if pred and observed else "prediction_only" if pred else
                           "outcome_only" if observed else "neither"] += 1
        if gaps:
            missing.append({"fixture_id": fid, "families": gaps, "final_observation_rows": len(obs),
                            "alternate_corner_rows": len(fe11)})
    return {"prediction_only_targets": missing, "sampled": min(len(rows), LIMIT), "sample_truncated": len(rows) > LIMIT,
            "observation_truncated": truncated,
            "families": {f: {k: counts[f][k] for k in STATES} for f in FAMILIES}}
