"""Bounded SPORT-FIRST SGP/prop research capture into Render's existing market cache.

Never computes a custom same-game-parlay price, places bets or changes weights.
Called ONLY by main-branch GitHub OIDC workflow; max two API-Football calls/run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from mcp_gateway import persistence, price_resolver_v4, subscriber_research_desk_v1

BOOKMAKER_ID = 8
MARKET_IDS = {7, 24, 25, 49, 92, 240, 241, 269, 270}
MAX_FIXTURES_PER_RUN = 2
API_BASE = "https://v3.football.api-sports.io"

SPORT_SHORTLIST_SQL = """
WITH latest AS (
 SELECT DISTINCT ON (f.fixture_id)
  f.fixture_id,f.kickoff,r.run_timestamp
 FROM soccer_model_runs r
 JOIN soccer_fixtures f ON f.fixture_id=r.fixture_id
 WHERE f.status='NS'
   AND f.kickoff > now() AND f.kickoff < now()+interval '30 hours'
   AND r.raw_projection IS NOT NULL
   AND r.run_timestamp < f.kickoff
   AND r.run_timestamp >= now()-interval '3 days'
 ORDER BY f.fixture_id,r.run_timestamp DESC
)
SELECT fixture_id,kickoff FROM latest ORDER BY kickoff,fixture_id LIMIT 16
"""


def _shortlist() -> list[int]:
    if not persistence.persistence_configured():
        raise RuntimeError("RENDER_PERSISTENCE_NOT_CONFIGURED")
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(SPORT_SHORTLIST_SQL)
                candidates = [(int(row[0]), row[1]) for row in cur.fetchall()]
    source_ids: list[int] = []
    # Archived Supabase quotes qualify ONLY when an independent prior raw
    # sporting model already produced a specific joint probability.
    try:
        feed = subscriber_research_desk_v1._query_research_feed()
        raw = [
            row for row in feed.get("sgp_rows", [])
            if row.get("offer_type") == "BOOK_QUOTED_COMPOSITE"
            and row.get("raw_joint_probability") is not None
            and row.get("fixture_id") is not None
            and row.get("kickoff_utc") is not None
        ]
        raw.sort(key=lambda r: str(r.get("kickoff_utc")))
        source_ids = list(dict.fromkeys(int(row["fixture_id"]) for row in raw))
    except Exception:
        pass
    # Prefer one independent canonical modeled fixture and one research-only
    # independently modeled fixture when both exist.
    ordered: list[int] = []
    if candidates:
        ordered.append(candidates[0][0])
    if source_ids:
        ordered.append(source_ids[0])
    ordered.extend(fid for fid, _ in candidates[1:])
    ordered.extend(source_ids[1:])
    ordered = list(dict.fromkeys(ordered))
    if not ordered:
        return []
    # Independently recheck all candidates against Render's fixture registry;
    # never trust a raw pilot ID without a future NS kickoff.
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("""
                  SELECT fixture_id FROM soccer_fixtures
                  WHERE fixture_id=ANY(%s)
                    AND status='NS' AND kickoff > now()
                    AND kickoff < now() + interval '30 hours'
                  """, (ordered[:16],))
                allowed = {int(row[0]) for row in cur.fetchall()}
    return [fid for fid in ordered if fid in allowed][:MAX_FIXTURES_PER_RUN]


def _capture() -> dict[str, Any]:
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        return {"status":"SOURCE_API_KEY_NOT_VERIFIED","api_requests":0,"snapshots_written":0}
    targets = _shortlist()
    if not targets:
        return {"status":"NO_PRIOR_SPORTING_SHORTLIST","api_requests":0,"snapshots_written":0}
    now = datetime.now(timezone.utc)
    calls = 0
    snapshots = 0
    statuses = []
    # Two separate API calls, no pagination / market probing or line shopping.
    with httpx.Client(timeout=10.0,follow_redirects=False) as client:
        for fixture_id in targets:
            try:
                resp = client.get(f"{API_BASE}/odds",
                     params={"fixture":fixture_id,"bookmaker":BOOKMAKER_ID,"page":1},
                     headers={"x-apisports-key":api_key})
                calls += 1
                if resp.status_code == 429:
                    statuses.append({"fixture_id":fixture_id,"status":"RATE_LIMITED"})
                    break
                resp.raise_for_status()
                payload=resp.json()
                markets=[
                    m for m in price_resolver_v4.normalize_api_response(payload)
                    if m.get("bookmaker_id")==BOOKMAKER_ID
                    and m.get("market_id") in MARKET_IDS
                    and m.get("fixture_id")==fixture_id
                ]
                written=0
                with persistence._connect() as conn:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            for m in markets:
                                updated=subscriber_research_desk_v1._as_utc_datetime(m.get("provider_update"))
                                if (updated is None or updated < now-timedelta(hours=12)
                                      or updated > now+timedelta(minutes=5)):
                                    continue
                                cur.execute("""
                                  SELECT 1 FROM soccer_market_snapshots
                                  WHERE fixture_id=%s AND bookmaker_id=%s AND market_id=%s
                                    AND provider_update=%s
                                    AND captured_at >= now()-interval '4 hours'
                                  LIMIT 1
                                """,(fixture_id,BOOKMAKER_ID,m["market_id"],updated))
                                if cur.fetchone() is not None:
                                    continue
                                cur.execute("""
                                  INSERT INTO soccer_market_snapshots
                                   (fixture_id,captured_at,stage,bookmaker_id,
                                    bookmaker,market_id,market,values,provider_update)
                                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                                """,(
                                    fixture_id,now,"SGP_SPORT_FIRST_RESEARCH",
                                    BOOKMAKER_ID,"Bet365",m["market_id"],m["market"],
                                    json.dumps(m.get("values") or []),updated
                                ))
                                written+=1
                snapshots+=written
                statuses.append({"fixture_id":fixture_id,"status":"SAVED_RESEARCH_MARKETS",
                                 "markets_returned":len(markets),"snapshots_written":written})
            except Exception:
                statuses.append({"fixture_id":fixture_id,"status":"SOURCE_OR_PERSISTENCE_NOT_VERIFIED"})
    return {"status":"RESEARCH_CAPTURE_COMPLETED","api_requests":calls,
            "snapshots_written":snapshots,"bookmaker_id":BOOKMAKER_ID,
            "bet_eligible":False,"sgp_betslip_price_verified":False,
            "provider_requests_cap":MAX_FIXTURES_PER_RUN,
            "fixtures":statuses}
