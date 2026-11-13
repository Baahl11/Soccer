"""Independent pregame player shots/SOT research joined to book quotes *after* sport.

SPORT FIRST: read frozen T-minus player-shots distributions by official player
ID from canonical Postgres refresh events. Those distributions were computed
from historical player trends, role/minutes and opponent history without any
sportsbook price in their calculation.

MARKET SECOND: optional exact-ID, exact-operator, half-line comparison; a
bookmaker selection ending "- 3" cannot prove its settlement operator.
No prices are used to choose players, rank model results or compute model P.
No provider calls, SQL writes, wagers, stake, or promotion permissions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
import time
from typing import Any

VERSION = "FE11_PLAYER_SPORT_FIRST_OFFER_BRIDGE_v1.0"
OOS_MODEL_VERSION = "SOCCER_PLAYER_PROPS_OOS_V4_1.1.1"
MIN_PLAYED_MATCHES = 8
MIN_OBSERVED_MINUTES = 360
MIN_OOS_REVIEW_PLAYER_GAMES = 500
MIN_OOS_ACTIONABLE_PLAYER_GAMES = 2000
MAX_UPCOMING_FIXTURES = 24
MAX_RESEARCH_PROFILES = 48
SUPPORTED_FAMILIES = ("SHOTS", "SOT")

# Bounded fixtures first, without inspecting prices. LATERAL indexed event
# lookup avoids loading massive JSON rows from all 4,000+ scanned fixtures.
FROZEN_MODEL_SQL = """
WITH upcoming AS (
    SELECT f.fixture_id, f.kickoff, f.home_team_id, f.away_team_id,
           f.home_team, f.away_team
    FROM soccer_fixtures f
    WHERE f.status = 'NS'
      AND f.kickoff > now()
      AND f.kickoff <= now() + interval '36 hours'
    ORDER BY f.kickoff ASC, f.fixture_id ASC
    LIMIT %s
)
SELECT f.fixture_id, f.kickoff, f.home_team_id, f.away_team_id,
       f.home_team, f.away_team, e.generated_at, e.stage,
       e.payload->'player_shots_intelligence' AS shots,
       e.payload->'player_sot_intelligence' AS sot
FROM upcoming f
JOIN LATERAL (
    SELECT r.generated_at, r.stage, r.payload
    FROM soccer_refresh_events r
    WHERE r.fixture_id = f.fixture_id
      AND r.generated_at < f.kickoff
      AND r.generated_at <= now()
      AND r.stage IN ('T-10','T-20','T-30','T-40')
      AND (r.payload ? 'player_shots_intelligence'
           OR r.payload ? 'player_sot_intelligence')
    ORDER BY CASE r.stage
        WHEN 'T-10' THEN 4 WHEN 'T-20' THEN 3
        WHEN 'T-30' THEN 2 WHEN 'T-40' THEN 1 ELSE 0 END DESC,
        r.generated_at DESC
    LIMIT 1
) e ON TRUE
ORDER BY f.kickoff ASC, f.fixture_id ASC
"""


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    if not isinstance(value, str):
        return None
    try:
        out = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return out.astimezone(timezone.utc) if out.tzinfo else None
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    if type(value) is bool or value is None or value == "":
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _id(value: Any) -> int | None:
    if type(value) is int and 0 < value < 10**13:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        v = int(value)
        return v if 0 < v < 10**13 else None
    return None


def _model_lines(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        line, over, under = (
            _num(entry.get("line")), _num(entry.get("p_over")),
            _num(entry.get("p_under")),
        )
        if (line is None or over is None or under is None
                or line not in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
                or not 0 <= over <= 1 or not 0 <= under <= 1
                or abs(over + under - 1.0) > 0.0001):
            continue
        result.append({"line": line, "p_over_raw": over, "p_under_raw": under})
    return result


def sport_profiles(rows: list[dict[str, Any]], *, now_utc: datetime) -> dict[str, Any]:
    """Do NOT receive odds or market quotes; output rank is kickoff + player ID."""
    now = _dt(now_utc)
    if not now or not isinstance(rows, list):
        raise ValueError("UNVERIFIED_MODEL_SOURCE")
    profiles: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    fixtures = set()

    def block(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for record in rows[:MAX_UPCOMING_FIXTURES]:
        if not isinstance(record, dict):
            block("SNAPSHOT_NOT_OBJECT")
            continue
        fixture_id, home_id, away_id = (
            _id(record.get("fixture_id")), _id(record.get("home_team_id")),
            _id(record.get("away_team_id")),
        )
        kick, captured = _dt(record.get("kickoff")), _dt(record.get("generated_at"))
        if (fixture_id is None or home_id is None or away_id is None or home_id == away_id
                or not kick or not captured or not (captured < kick and captured <= now < kick)
                or record.get("stage") not in ("T-10","T-20","T-30","T-40")):
            block("NOT_VERIFIED_ASOF_FIXTURE")
            continue
        fixtures.add(fixture_id)
        for family, source, status, rate_key, mean_key in (
            ("SHOTS", "shots", "LIVE_RESEARCH_SHOTS_DISTRIBUTION",
             "base_posterior_mean_shots_per90", "expected_shots"),
            ("SOT", "sot", "LIVE_RESEARCH_SOT_DISTRIBUTION",
             "base_posterior_mean_sot_per90", "expected_sot"),
        ):
            intel = record.get(source)
            if not isinstance(intel, dict) or intel.get("status") not in (
                "LIVE_RESEARCH_SHOTS_PROPS", "LIVE_RESEARCH_SOT_PROPS"
            ):
                continue
            if _id(intel.get("fixture_id")) != fixture_id:
                block("MODEL_FIXTURE_MISMATCH")
                continue
            player_list = intel.get("players")
            if not isinstance(player_list, list):
                block("MODEL_PLAYER_LIST_MISSING")
                continue
            for player in player_list:
                if not isinstance(player, dict) or player.get("status") != status:
                    continue
                pid, team_id, opp_id = (
                    _id(player.get("player_id")), _id(player.get("team_id")),
                    _id(player.get("opponent_team_id")),
                )
                if (pid is None or team_id not in (home_id, away_id)
                        or opp_id not in (home_id, away_id) or team_id == opp_id
                        or player.get("confirmed_starter") is not True):
                    block("MODEL_PLAYER_ID_XI_NOT_VERIFIED")
                    continue
                reg_ts = _dt(player.get("registry_generated_at_utc"))
                n = player.get("historical_played_matches_last20")
                minutes = _num(player.get("historical_observed_minutes_last20"))
                rate = _num(player.get(rate_key))
                expected = _num(player.get(mean_key))
                future_minutes = _num(player.get("expected_minutes_if_confirmed_starter"))
                opp = player.get("opponent_adjustment")
                opp = opp if isinstance(opp, dict) else {}
                opponent_verified = (
                    opp.get("status") in (
                        "SHRUNK_OPPONENT_SHOT_SUPPRESSION",
                        "SHRUNK_OPPONENT_SOT_SUPPRESSION",
                    ) and type(opp.get("opponent_last10_n")) is int
                    and opp["opponent_last10_n"] >= 5
                    and _num(opp.get("factor")) is not None
                )
                blockers = []
                if not reg_ts or not (reg_ts <= captured):
                    blockers.append("PLAYER_REGISTRY_ASOF_NOT_VERIFIED")
                if type(n) is not int or n < MIN_PLAYED_MATCHES:
                    blockers.append("HISTORICAL_PLAYER_SAMPLE_INSUFFICIENT")
                if minutes is None or minutes < MIN_OBSERVED_MINUTES:
                    blockers.append("HISTORICAL_PLAYER_MINUTES_INSUFFICIENT")
                if rate is None or rate < 0 or expected is None or expected < 0:
                    blockers.append("RAW_SHOTS_RATE_NOT_VERIFIED")
                if future_minutes is None or not 0 < future_minutes <= 105:
                    blockers.append("EXPECTED_MINUTES_NOT_VERIFIED")
                if not opponent_verified:
                    blockers.append("OPPONENT_CONTEXT_NOT_VERIFIED")
                lines = _model_lines(player.get("lines"))
                if not lines:
                    blockers.append("MODEL_LINE_DISTRIBUTION_NOT_VERIFIED")
                if blockers:
                    block(blockers[0])
                    continue
                profiles.append({
                    "fixture_id": fixture_id, "player_id": pid,
                    "team_id": team_id, "opponent_team_id": opp_id,
                    "player": str(player.get("player") or "")[:90],
                    "family": family, "kickoff_utc": kick.isoformat(),
                    "home_team": str(record.get("home_team") or "")[:90],
                    "away_team": str(record.get("away_team") or "")[:90],
                    "pregame_snapshot_at_utc": captured.isoformat(),
                    "registry_generated_at_utc": reg_ts.isoformat(),
                    "stage": record["stage"],
                    "historical_played_matches": n,
                    "historical_observed_minutes": round(minutes, 1),
                    "historical_rate_per90": round(rate, 4),
                    "expected_minutes": round(future_minutes, 1),
                    "projected_count": round(expected, 4),
                    "opponent_adjustment_factor": round(float(opp["factor"]), 4),
                    "lines": lines,
                    "model_family": "GAMMA_POISSON_NEGATIVE_BINOMIAL",
                    "source": "CANONICAL_PREMATCH_FROZEN_PLAYER_MODEL",
                    "classification": "RESEARCH_ONLY",
                    "oos_validation_complete": False,
                    "bet_eligible": False,
                })
    profiles.sort(key=lambda r: (r["kickoff_utc"], r["fixture_id"], r["player_id"], r["family"]))
    return {
        "version": VERSION,
        "status": "VERIFIED_SPORTING_INPUTS_RESEARCH_ONLY" if profiles
                  else "NO_SPORTING_PROFILE_WITH_FULL_PROVENANCE",
        "source_fixture_count": len(fixtures),
        "verified_pregame_sport_models": len(profiles),
        "unverified_input_reasons": excluded,
        "rows": profiles[:MAX_RESEARCH_PROFILES],
        "provider_requests_added": 0,
        "odds_used_to_choose_players": False,
        "bet_eligible": False,
    }


def _exact_market(offer: dict[str, Any]) -> tuple[str, str, float] | None:
    # Only explicit 'Over 2.5'/'Under 2.5' operators with a half-line are
    # unambiguous. 'Player Name - 3' is NOT verified as 3+ at this boundary.
    if offer.get("line_basis") != "EXPLICIT_OVER_UNDER":
        return None
    family = offer.get("player_market_family")
    if family not in SUPPORTED_FAMILIES:
        return None
    selection = str(offer.get("selection") or "")
    match = re.search(r"(?i)\b(over|under)\s+(\d+\.[05])\b", selection)
    if not match:
        return None
    line = _num(offer.get("parsed_line"))
    parsed = float(match.group(2))
    if line is None or parsed != line:
        return None
    return family, match.group(1).upper(), line


def match_offers_to_sport(
    offers: list[dict[str, Any]], sport: dict[str, Any],
    *, now_utc: datetime,
) -> dict[str, Any]:
    """Price never enters model selection; every offer gets a blocker audit."""
    now = _dt(now_utc)
    if not now or not isinstance(offers, list) or not isinstance(sport, dict):
        raise ValueError("INVALID_OFFER_AUDIT_SOURCE")
    model_by_key = {}
    for row in sport.get("rows") or []:
        if not isinstance(row, dict):
            continue
        model_by_key[(row.get("fixture_id"),row.get("player_id"),row.get("family"))] = row
    outcomes = []
    all_blockers: dict[str, int] = {}
    for quote in offers[:320]:
        if not isinstance(quote, dict) or quote.get("offer_type") != "PLAYER_COMPONENT_QUOTE":
            continue
        blockers = []
        fixture_id, pid, team_id = (_id(quote.get("fixture_id")),
                                    _id(quote.get("player_id")),
                                    _id(quote.get("team_id")))
        matched = _exact_market(quote)
        price = _num(quote.get("offered_price_decimal"))
        updated = _dt(quote.get("provider_updated_at"))
        captured = _dt(quote.get("captured_at"))
        if not pid or quote.get("xi_alignment_status") != "MATCHED_CONFIRMED_XI":
            blockers.append("PLAYER_ID_NOT_CONFIRMED_BY_OFFICIAL_XI")
        if not matched:
            blockers.append("EXACT_MARKET_OPERATOR_OR_HALF_LINE_NOT_VERIFIED")
        if quote.get("bookmaker_id") != 8 or not price or price <= 1:
            blockers.append("BOOKMAKER_PRICE_NOT_VERIFIED")
        if (not updated or not captured or updated > now + timedelta(minutes=5)
                or captured > now + timedelta(minutes=5)
                or updated < now - timedelta(minutes=90)
                or captured < now - timedelta(minutes=60)):
            blockers.append("QUOTE_NOT_VERIFIED_CURRENT")
        family = matched[0] if matched else quote.get("player_market_family")
        model = model_by_key.get((fixture_id,pid,family))
        if not model:
            blockers.append("INDEPENDENT_ASOF_PLAYER_MODEL_NOT_FOUND")
        elif team_id != model["team_id"]:
            blockers.append("PLAYER_TEAM_ID_MISMATCH")
        if model and updated and _dt(model["pregame_snapshot_at_utc"]) > updated:
            blockers.append("MODEL_NOT_PRIOR_TO_BOOK_PRICE")
        if blockers:
            for b in blockers:
                all_blockers[b] = all_blockers.get(b,0)+1
            outcomes.append({"status": "NOT_VERIFIED", "blockers": blockers})
            continue
        _, direction, line = matched
        model_line = next((x for x in model["lines"] if x["line"] == line), None)
        if not model_line:
            b = "EXACT_LINE_NOT_IN_FROZEN_MODEL"
            all_blockers[b] = all_blockers.get(b,0)+1
            outcomes.append({"status":"NOT_VERIFIED","blockers":[b]})
            continue
        p = model_line["p_over_raw"] if direction=="OVER" else model_line["p_under_raw"]
        outcomes.append({
            "status":"RAW_MODEL_LINKED_RESEARCH_ONLY_OOS_PENDING",
            "fixture_id":fixture_id,"player_id":pid,"family":family,
            "market_line":line,"operator":direction,
            "raw_model_probability":p,
            "model_snapshot_at_utc":model["pregame_snapshot_at_utc"],
            "line_source":"PROVIDER_EXPLICIT_OPERATOR",
            "book_price_decimal_observed":price,
            "break_even_probability":round(1/price,6),
            "oos_validation_complete":False,
            "ev_authorized":False,"bet_eligible":False,
            "note":"NO_ACTIONABLE_EDGE_UNTIL_OOS_AND_AVAILABILITY_REVIEW",
        })
    return {
        "status":"RESEARCH_ONLY_PLAYER_OFFER_AUDIT",
        "source_player_offer_count":len(outcomes),
        "raw_model_exact_matches":sum(x["status"]=="RAW_MODEL_LINKED_RESEARCH_ONLY_OOS_PENDING" for x in outcomes),
        "unverified_offer_count":sum(x["status"]=="NOT_VERIFIED" for x in outcomes),
        "unverified_reasons":all_blockers,
        "aligned_research_offers":[x for x in outcomes if x["status"].startswith("RAW_MODEL_LINKED")][:20],
        "bet_eligible":False, "ev_authorized":False,
        "provider_requests_added":0,"odds_used_to_choose_players":False,
    }


OOS_STATE_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/analysis/player_props_oos_v4_report.json"
)
_OOS_READ_MEMO: dict[str, Any] = {"expires_at": 0.0, "payload": None}


def get_oos_readiness(*, now_utc: datetime | None = None) -> dict[str, Any]:
    """Read published OOS-only result, never treat 500 N as model approval.

    Real historical, finalized outcomes are produced by the existing protected
    OIDC PostgreSQL OOS workflow. Cached 15 min, source freshness <=72 h;
    missing/stale/malformed reports keep the gate CLOSED.
    """
    now = _dt(now_utc or datetime.now(timezone.utc))
    if not now:
        raise ValueError("INVALID_OOS_AUDIT_TIME")
    clock = time.monotonic()
    source = _OOS_READ_MEMO["payload"] if clock < _OOS_READ_MEMO["expires_at"] else None
    if source is None:
        try:
            import httpx
            response=httpx.get(OOS_STATE_URL, timeout=5, follow_redirects=True)
            response.raise_for_status()
            source=response.json()
            if not isinstance(source,dict):
                source=None
        except Exception:
            source=None
        _OOS_READ_MEMO["payload"]=source
        _OOS_READ_MEMO["expires_at"]=clock+900
    good = isinstance(source,dict) and source.get("model_version")==OOS_MODEL_VERSION
    ts=_dt(source.get("generated_at_utc")) if good else None
    fresh=bool(ts and timedelta(0)<=now-ts<=timedelta(hours=72))
    families={}
    for family in SUPPORTED_FAMILIES:
        row=(source.get("families") or {}).get(family) if good else None
        row=row if isinstance(row,dict) else {}
        n=row.get("unique_player_fixtures")
        n=n if type(n) is int and n>=0 else None
        brier=_num(row.get("brier_score"))
        loss=_num(row.get("log_loss"))
        bins=row.get("binary_probability_rows")
        bins=bins if type(bins) is int and bins>=0 else None
        ready_for_review=bool(
            fresh and n is not None and n>=MIN_OOS_REVIEW_PLAYER_GAMES
            and bins is not None and bins>=MIN_OOS_REVIEW_PLAYER_GAMES
            and brier is not None and loss is not None
            and row.get("oos_validation_complete") is True
        )
        families[family]={
            "unique_player_fixtures":n,
            "binary_probability_rows":bins,
            "minimum_review_sample":MIN_OOS_REVIEW_PLAYER_GAMES,
            "minimum_actionable_review_sample":MIN_OOS_ACTIONABLE_PLAYER_GAMES,
            "brier_score":brier if brier is not None and brier>=0 else None,
            "log_loss":loss if loss is not None and loss>=0 else None,
            "ready_for_manual_oos_review":ready_for_review,
            "calibration_approved":False,
            "status":("REVIEW_SAMPLE_READY_NOT_APPROVED" if ready_for_review
                      else "COLLECTING_OR_UNVERIFIED_OOS"),
        }
    return {
        "status":"VERIFIED_RECENT_OOS_SOURCE_FOR_REVIEW" if fresh
                 else "NOT_VERIFIED_CURRENT_OOS_REPORT",
        "source_model_version":OOS_MODEL_VERSION if good else None,
        "source_generated_at_utc":ts.isoformat() if ts else None,
        "families":families,
        "manual_performance_review_required":True,
        "production_promotion_allowed":False,
        "provider_requests_added":0,
        "bet_eligible":False,
    }


def load_frozen_models() -> list[dict[str, Any]]:
    """Owner-only bounded read. Does not import heavy canonical model modules."""
    from mcp_gateway import persistence
    if not persistence.persistence_configured():
        return []
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(FROZEN_MODEL_SQL, (MAX_UPCOMING_FIXTURES,))
                keys=[d.name for d in cur.description]
                return [dict(zip(keys,row)) for row in cur.fetchall()]


def build_live_research(offers: list[dict[str, Any]]) -> dict[str, Any]:
    """Own sport-first read before optional market link; never creates a pick."""
    now=datetime.now(timezone.utc)
    rows=load_frozen_models()
    sport=sport_profiles(rows, now_utc=now)
    compared=match_offers_to_sport(offers, sport, now_utc=now)
    oos=get_oos_readiness(now_utc=now)
    return {
        "status":"SPORT_FIRST_RESEARCH_ONLY",
        "model_version":VERSION,
        "sport":sport,
        "market_alignment":compared,
        "oos":oos,
        "oos_readiness_status":oos["status"],
        "oos_review_minimum_player_games":MIN_OOS_REVIEW_PLAYER_GAMES,
        "oos_actionable_review_minimum_player_games":MIN_OOS_ACTIONABLE_PLAYER_GAMES,
        "production_promotion_allowed":False,
        "bet_eligible":False,
        "provider_requests_added":0,
    }


def saved_match_profiles(events: Any, fixture: dict[str, Any], *, now_utc: datetime) -> dict[str, Any]:
    """Use the same verified sport-first gate on the match's existing refresh rows."""
    keys = ("fixture_id", "home_team_id", "away_team_id", "league_id", "season")
    kickoff = _dt(fixture.get("kickoff"))
    now = _dt(now_utc)
    records = []
    if (isinstance(events, list) and kickoff and now and now < kickoff
            and all(type(fixture.get(k)) is int and _id(fixture[k]) is not None for k in keys)
            and fixture["home_team_id"] != fixture["away_team_id"]):
        for entry in events[:12]:
            if not isinstance(entry, dict) or entry.get("event_type") != "SOCCER_REFRESH":
                continue
            e = entry.get("payload")
            if not isinstance(e, dict) or e.get("stage") != entry.get("stage") or e.get("event_type") != "SOCCER_REFRESH":
                continue
            fx = e.get("fixture")
            if (not isinstance(fx, dict) or fx.get("status") != "NS"
                    or any(fx.get(k) != fixture[k] for k in keys) or _dt(fx.get("kickoff")) != kickoff):
                continue
            captured = _dt(entry.get("generated_at"))
            if captured is None or captured > now or captured >= kickoff:
                continue
            records.append({**{k: fixture[k] for k in keys},
                "home_team": fixture.get("home_team"), "away_team": fixture.get("away_team"),
                "kickoff": kickoff, "generated_at": captured, "stage": entry.get("stage"),
                "shots": e.get("player_shots_intelligence"),
                "sot": e.get("player_sot_intelligence")})
    output, seen, reasons = [], set(), {}
    for record in sorted(records, key=lambda row: row["generated_at"], reverse=True):
        result = sport_profiles([record], now_utc=now_utc)
        for reason, count in result["unverified_input_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        for row in result["rows"]:
            key = (row["player_id"], row["family"])
            if key not in seen and len(output) < MAX_RESEARCH_PROFILES:
                output.append(row)
                seen.add(key)
    return {
        "status": "VERIFIED_SPORTING_INPUTS_RESEARCH_ONLY" if output else "NOT_VERIFIED",
        "source": "POSTGRES_REFRESH_EVENTS_PLAYER_SHOTS_SOT",
        "fixture_id": fixture.get("fixture_id"),
        "rows": sorted(output, key=lambda row: (row["team_id"], row["player_id"], row["family"])),
        "unverified_input_reasons": reasons,
        "reason": None if output else ("NO_PLAYER_PROFILE_PASSED_PROVENANCE" if records else "NO_MATCHED_PREGAME_PLAYER_SNAPSHOT"),
        "bet_eligible": False,
    }
