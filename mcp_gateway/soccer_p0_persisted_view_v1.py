"""Owner-only, pre-kickoff SOCCER research previews from verified persisted sources.

Neither source authorizes public redisplay, BET, or commercial predictions.
The canonical research source may be emitted by the runtime SPORT_PREVIEW or
by the isolated Git-attested frozen SPORT capture. This module makes no API
provider calls and never recalculates a historical projection.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import persistence
from mcp_gateway import soccer_p0_sport_preview_v1 as p0
from mcp_gateway import isolated_sport_frozen_pg_v1 as frozen

MAX_FIXTURE_IDS = 160
MIN_SPLIT_SAMPLE = 5


def asof_isolated_preview(record: Any, *, now: datetime) -> dict[str, Any] | None:
    """Recheck an immutable frozen record; never project probabilities here."""
    if not isinstance(record, dict):
        return None
    # Full source, provenance, fixed model, lineups-unverified and timestamp
    # validation are enforced by the exact existing frozen-import contract.
    try:
        frozen.validated_row(record)
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
    captured = frozen._time(record.get("captured_at_utc"))
    kickoff = frozen._time(record.get("kickoff"))
    current = now.astimezone(timezone.utc)
    if not captured or not kickoff:
        return None
    if not captured <= current < kickoff - timedelta(minutes=5):
        return None
    if captured > kickoff - timedelta(minutes=45):
        return None
    raw = record["raw_sport_projection"]
    sample = raw.get("sample") if isinstance(raw.get("sample"), dict) else {}
    min_n = sample.get("minimum_split_sample")
    if type(min_n) is not int or min_n < MIN_SPLIT_SAMPLE:
        return None
    keys = (
        "raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
        "raw_btts_yes_prob", "raw_over_2_5_prob",
    )
    probs = {k: round(float(raw[k]), 6) for k in keys}
    return {
        "fixture_id": record["fixture_id"],
        "kickoff_utc": kickoff.isoformat(),
        "captured_at_utc": captured.isoformat(),
        "sport_source": frozen.SOURCE,
        "model_version": raw.get("model_version"),
        "projection_type": "LIMITED_GOAL_RATE_BASELINE_NOT_XG",
        "min_split_sample": min_n,
        "raw_probabilities": probs,
        "model_validation": "NOT_VERIFIED_OOS",
        "source_updated_at": "NOT_VERIFIED",
        "market_queried": False,
        "bet_eligible": False,
        "direct_executable_quote_verified": False,
        "commercial_display_allowed": False,
    }


def _read_frozen_rows(cur: Any, fixture_ids: list[int], current: datetime) -> list[tuple[int, Any]]:
    """Fixture ID, team IDs, date and kickoff must agree with canonical registry."""
    cur.execute(
        """
        SELECT DISTINCT ON (f.fixture_id) f.fixture_id, s.raw_projection
        FROM soccer_isolated_sport_frozen_research_v1 s
        JOIN soccer_fixtures f
          ON f.fixture_id = s.fixture_id
         AND f.kickoff = s.kickoff
         AND f.home_team_id = s.home_team_id
         AND f.away_team_id = s.away_team_id
        WHERE s.fixture_id = ANY(%s)
          AND s.source = %s
          AND s.projection_model = %s
          AND s.bet_eligible = FALSE
          AND s.production_promotion_allowed = FALSE
          AND s.source_generated_at <= %s
          AND s.source_generated_at <= s.kickoff - INTERVAL '45 minutes'
          AND s.kickoff > %s + INTERVAL '5 minutes'
          AND (s.kickoff AT TIME ZONE 'America/Mexico_City')::date =
              (%s AT TIME ZONE 'America/Mexico_City')::date
        ORDER BY f.fixture_id, s.source_generated_at DESC
        LIMIT %s
        """,
        (fixture_ids, frozen.SOURCE, frozen.MODEL, current, current, current, MAX_FIXTURE_IDS),
    )
    return list(cur.fetchall())


def read_previews(fixture_ids: list[int], *, now: datetime | None = None) -> dict[str, Any]:
    verified_ids = sorted({
        fid for fid in fixture_ids if type(fid) is int and 0 < fid < 10**12
    })[:MAX_FIXTURE_IDS]
    if not verified_ids:
        return {"status": "NO_SLATE_FIXTURES", "rows": [], "attempted_count": 0}
    current = now or datetime.now(timezone.utc)
    original: list[tuple[int, Any]] = []
    isolated: list[tuple[int, Any]] = []
    frozen_status = "SOURCE_NOT_VERIFIED"
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (e.fixture_id)
                  e.fixture_id, e.payload
                FROM soccer_refresh_events e
                JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
                WHERE e.fixture_id=ANY(%s)
                  AND e.event_type='SOCCER_REFRESH'
                  AND e.stage='SPORT_PREVIEW'
                  AND e.generated_at >= %s - INTERVAL '36 hours'
                  AND e.generated_at < f.kickoff - INTERVAL '5 minutes'
                  AND f.kickoff > %s + INTERVAL '5 minutes'
                  AND (f.kickoff AT TIME ZONE 'America/Mexico_City')::date =
                      (%s AT TIME ZONE 'America/Mexico_City')::date
                ORDER BY e.fixture_id, e.generated_at DESC
                LIMIT %s
                """,
                (verified_ids, current, current, current, MAX_FIXTURE_IDS),
            )
            original = list(cur.fetchall())
            try:
                isolated = _read_frozen_rows(cur, verified_ids, current)
                frozen_status = "FROZEN_RESEARCH_READ"
            except Exception:
                # A missing research table must never break the original
                # SPORT_PREVIEW path; report its absence, never synthesize rows.
                frozen_status = "ISOLATED_FROZEN_READ_NOT_VERIFIED"

    forecasts: dict[int, dict[str, Any]] = {}
    for fid, payload in original:
        if not isinstance(payload, dict):
            continue
        derived = p0.asof_research_projection(payload, now=current)
        if derived is not None and derived["fixture_id"] == fid:
            forecasts[fid] = derived
    for fid, payload in isolated:
        if fid in forecasts:
            continue
        derived = asof_isolated_preview(payload, now=current)
        if derived is not None and derived["fixture_id"] == fid:
            forecasts[fid] = derived
    rows = sorted(forecasts.values(), key=lambda v: (v["kickoff_utc"], v["fixture_id"]))
    return {
        "status": "PREMATCH_LIMITED_RESEARCH" if rows else "NO_VERIFIED_PREMATCH_RAW",
        "rows": rows,
        "attempted_count": len(original) + len(isolated),
        "forecast_count": len(rows),
        "isolated_frozen_read_status": frozen_status,
        "provider_requests_added": 0,
        "direct_executable_price_verified": False,
        "market_queried": False,
        "bet_eligible": False,
        "source_redisplay_rights_verified": False,
        "model_oos_calibrated": False,
        "commercial_display_allowed": False,
    }
