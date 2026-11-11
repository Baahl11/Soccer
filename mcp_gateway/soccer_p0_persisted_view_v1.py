"""Read prospective soccer SPORT previews from authenticated local PostgreSQL only.

No provider calls, odds, commercial recommendations or third-party raw data.
Only OWNER/ADMIN contract attaches this research while redisplay rights remain
unverified. Snapshot timing and mathematical ranges rechecked on reads.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from mcp_gateway import persistence
from mcp_gateway import soccer_p0_sport_preview_v1 as p0

MAX_FIXTURE_IDS=160


def read_previews(fixture_ids: list[int], *, now: datetime|None=None) -> dict[str,Any]:
    verified_ids=sorted({fid for fid in fixture_ids if type(fid) is int and 0<fid<10**12})[:MAX_FIXTURE_IDS]
    if not verified_ids:
        return {"status":"NO_SLATE_FIXTURES","rows":[],"attempted_count":0}
    current=now or datetime.now(timezone.utc)
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
              SELECT DISTINCT ON (e.fixture_id)
                e.fixture_id, e.payload
              FROM soccer_refresh_events e
              JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
              WHERE e.fixture_id=ANY(%s)
                AND e.event_type='SOCCER_REFRESH'
                AND e.stage='SPORT_PREVIEW'
                AND e.generated_at >= NOW() - INTERVAL '36 hours'
                AND e.generated_at < f.kickoff - INTERVAL '5 minutes'
                AND f.kickoff > NOW() + INTERVAL '5 minutes'
                AND (f.kickoff AT TIME ZONE 'America/Mexico_City')::date =
                    (NOW() AT TIME ZONE 'America/Mexico_City')::date
              ORDER BY e.fixture_id, e.generated_at DESC
              LIMIT %s
            """,(verified_ids,MAX_FIXTURE_IDS))
            records=cur.fetchall()
    forecasts=[]
    for fid,payload in records:
        if not isinstance(payload,dict):continue
        derived=p0.asof_research_projection(payload,now=current)
        if derived is None or derived["fixture_id"]!=fid:continue
        forecasts.append(derived)
    return {
        "status":"PREMATCH_LIMITED_RESEARCH" if forecasts else "NO_VERIFIED_PREMATCH_RAW",
        "rows":forecasts,
        "attempted_count":len(records),
        "forecast_count":len(forecasts),
        "provider_requests_added":0,
        "direct_executable_price_verified":False,
        "market_queried":False,
        "bet_eligible":False,
        "source_redisplay_rights_verified":False,
        "model_oos_calibrated":False,
        "commercial_display_allowed":False,
    }
