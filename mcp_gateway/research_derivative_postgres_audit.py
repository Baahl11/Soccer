from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.4.0"
MODEL_VERSION = "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.4.0"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def classify_market(market: Any) -> str | None:
    name = _norm(market)
    if not name:
        return None

    aggregate_player_shot_markets = (
        "home player shots total",
        "away player shots total",
        "player shots total - home",
        "player shots total - away",
    )
    aggregate_player_sot_markets = (
        "home player shots on target total",
        "away player shots on target total",
        "player shots on target total - home",
        "player shots on target total - away",
    )
    if any(token in name for token in aggregate_player_sot_markets):
        return "TEAM_SOT"
    if any(token in name for token in aggregate_player_shot_markets):
        return "TEAM_SHOTS"

    player_card_tokens = (
        "player cards",
        "player card",
        "player booked",
        "player booking",
        "player yellow card",
        "player yellow cards",
        "to be booked",
        "to be carded",
    )
    if any(token in name for token in player_card_tokens):
        return "PLAYER_CARDS"

    if "score or assist" in name or "score/assist" in name:
        return None

    if "first goal scorer" in name:
        return "GOALSCORER_FIRST"
    if "last goal scorer" in name:
        return "GOALSCORER_LAST"
    if any(token in name for token in ("anytime goal scorer", "anytime goalscorer", "player to score")):
        return "GOALSCORER_ANYTIME"
    if "goal scorer" in name or "goalscorer" in name:
        return "GOALSCORER_OTHER"

    if "shots on target - player" in name or "player shots on target" in name:
        return "SOT"
    if "player shots" in name or "player shot" in name:
        return "SHOTS"
    if "goalkeeper saves" in name or "keeper saves" in name or "gk saves" in name:
        return "GK_SAVES"
    if "player assists" in name or "player assist" in name:
        return "ASSISTS"

    card_tokens = (
        "cards over/under",
        "card over/under",
        "total cards",
        "total yellow cards",
        "yellow cards",
        "red card",
        "team cards",
        "booking points",
        "bookings",
        "cards asian handicap",
        "cards european handicap",
        "first card received",
    )
    if name == "rcard" or any(token in name for token in card_tokens):
        return "CARDS"

    compact = re.sub(r"[^a-z0-9]+", "", name)
    if "shotontarget" in compact or "shotongoal" in compact:
        return "TEAM_SOT"
    if "totalshots" in compact:
        return "TEAM_SHOTS"
    return None


def _payload(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [row for row in parsed if isinstance(row, dict)] if isinstance(parsed, list) else []
    return []


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def value_line(value: dict[str, Any]) -> float | None:
    for key in ("line", "parsed_line", "handicap"):
        parsed = _num(value.get(key))
        if parsed is not None:
            return parsed
    raw = value.get("raw_selection")
    if raw is None:
        raw = value.get("selection")
    if raw is None:
        raw = value.get("value")
    text = str(raw or "")

    match = re.search(
        r"\b(?:over|under)\s+([+-]?\d+(?:\.\d+)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _num(match.group(1))

    # Historical API-Football Player Props can be encoded as "Player - N",
    # where N means N+ events. The equivalent decimal line is N-0.5.
    threshold_match = re.match(r"^.+?\s+-\s+(\d+)\s*$", text.strip())
    if threshold_match:
        threshold = _num(threshold_match.group(1))
        if threshold is not None and threshold >= 1:
            return threshold - 0.5
    return None


def _price(value: dict[str, Any]) -> float | None:
    for key in ("decimal_price", "price", "odd"):
        parsed = _num(value.get(key))
        if parsed is not None and parsed > 1.0:
            return parsed
    return None


PLAYER_PROP_FAMILIES = {
    "PLAYER_CARDS",
    "SHOTS",
    "SOT",
    "GOALSCORER_ANYTIME",
    "GOALSCORER_FIRST",
    "GOALSCORER_LAST",
    "GOALSCORER_OTHER",
    "ASSISTS",
    "GK_SAVES",
}


def _norm_player_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _confirmed_starters(lineup_payload: Any) -> list[dict[str, Any]]:
    payload = lineup_payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict) or payload.get("both_xi_confirmed") is not True:
        return []

    starters: list[dict[str, Any]] = []
    for team in payload.get("teams") or []:
        if not isinstance(team, dict):
            continue
        for player in team.get("starters") or []:
            if not isinstance(player, dict) or not player.get("name"):
                continue
            starters.append({
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "team_id": team.get("team_id"),
                "team": team.get("team"),
                "position": player.get("pos"),
            })
    return starters


def align_value_to_confirmed_xi(
    value: dict[str, Any],
    *,
    lineup_payload: Any,
    family: str,
) -> dict[str, Any]:
    out = dict(value)
    raw = value.get("raw_selection")
    if raw is None:
        raw = value.get("selection")
    if raw is None:
        raw = value.get("value")

    normalized = _norm_player_text(raw)
    explicit_player_id = value.get("player_id")
    explicit_player_name = _norm_player_text(value.get("player_name") or value.get("player"))
    starters = _confirmed_starters(lineup_payload)
    matches: list[dict[str, Any]] = []

    if not starters:
        out["xi_alignment_status"] = "NO_CONFIRMED_XI_AT_QUOTE"
        return out

    if explicit_player_id is not None:
        matches = [
            starter for starter in starters
            if str(starter.get("player_id")) == str(explicit_player_id)
        ]
    elif explicit_player_name:
        matches = [
            starter for starter in starters
            if _norm_player_text(starter.get("player_name")) == explicit_player_name
        ]
    else:
        if not normalized:
            out["xi_alignment_status"] = "SELECTION_MISSING"
            return out
        padded = f" {normalized} "
        for starter in starters:
            player_name = _norm_player_text(starter.get("player_name"))
            if player_name and f" {player_name} " in padded:
                matches.append(starter)
    if len(matches) == 0:
        out["xi_alignment_status"] = "PLAYER_NOT_MATCHED_TO_CONFIRMED_XI"
        return out
    if len(matches) > 1:
        out["xi_alignment_status"] = "AMBIGUOUS_CONFIRMED_XI_MATCH"
        return out

    starter = matches[0]
    if family == "GK_SAVES" and str(starter.get("position") or "").upper() != "G":
        out["xi_alignment_status"] = "MATCHED_NON_GOALKEEPER"
        return out

    out.update({
        "xi_alignment_status": "MATCHED_CONFIRMED_XI",
        "player_id": starter.get("player_id"),
        "player_name": starter.get("player_name"),
        "team_id": starter.get("team_id"),
        "team": starter.get("team"),
        "position": starter.get("position"),
        "confirmed_starter": True,
    })
    return out


def summarize_rows(rows: Iterable[dict[str, Any]], *, lookback_days: int) -> dict[str, Any]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples: list[dict[str, Any]] = []
    candidate_rows = 0
    market_name_counts: Counter[str] = Counter()
    unclassified_market_name_counts: Counter[str] = Counter()
    unclassified_samples: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_rows += 1
        market_name = str(row.get("market") or "UNKNOWN")
        market_name_counts[market_name] += 1
        family = classify_market(row.get("market"))
        if family is None:
            unclassified_market_name_counts[market_name] += 1
            if len(unclassified_samples) < 50:
                unclassified_samples.append({
                    "fixture_id": row.get("fixture_id"),
                    "captured_at": row.get("captured_at"),
                    "stage": row.get("stage"),
                    "bookmaker": row.get("bookmaker"),
                    "market": row.get("market"),
                    "provider_update": row.get("provider_update"),
                    "pre_kickoff": bool(row.get("pre_kickoff")),
                    "confirmed_xi_before_market": bool(row.get("confirmed_xi_before_market")),
                    "values": _payload(row.get("values"))[:6],
                })
            continue
        values = _payload(row.get("values"))
        line_values = sum(1 for value in values if value_line(value) is not None)
        priced_values = sum(1 for value in values if _price(value) is not None)
        aligned_values = [
            align_value_to_confirmed_xi(
                value,
                lineup_payload=row.get("confirmed_lineup_payload"),
                family=family,
            )
            for value in values
        ] if family in PLAYER_PROP_FAMILIES else []
        xi_aligned_values = [
            value for value in aligned_values
            if value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI"
        ]
        xi_aligned_line_values = sum(
            1 for value in xi_aligned_values if value_line(value) is not None
        )
        xi_aligned_priced_values = sum(
            1 for value in xi_aligned_values if _price(value) is not None
        )
        normalized = {
            **row,
            "family": family,
            "value_count": len(values),
            "line_value_count": line_values,
            "priced_value_count": priced_values,
            "xi_aligned_value_count": len(xi_aligned_values),
            "xi_aligned_line_value_count": xi_aligned_line_values,
            "xi_aligned_priced_value_count": xi_aligned_priced_values,
            "xi_alignment_values": aligned_values,
        }
        family_rows[family].append(normalized)
        if len(samples) < 50:
            samples.append({
                "fixture_id": row.get("fixture_id"),
                "captured_at": row.get("captured_at"),
                "stage": row.get("stage"),
                "bookmaker": row.get("bookmaker"),
                "market": row.get("market"),
                "family": family,
                "provider_update": row.get("provider_update"),
                "confirmed_xi_before_market": bool(row.get("confirmed_xi_before_market")),
                "pre_kickoff": bool(row.get("pre_kickoff")),
                "value_count": len(values),
                "line_value_count": line_values,
                "xi_aligned_value_count": len(xi_aligned_values),
                "xi_aligned_priced_value_count": xi_aligned_priced_values,
                "xi_alignment_values": aligned_values[:6],
            })

    summaries: dict[str, Any] = {}
    all_fixtures: set[int] = set()
    all_confirmed: set[int] = set()
    all_rows = 0
    all_line_values = 0
    all_xi_aligned_fixtures: set[int] = set()

    for family in (
        "CARDS",
        "PLAYER_CARDS",
        "SHOTS",
        "SOT",
        "GOALSCORER_ANYTIME",
        "GOALSCORER_FIRST",
        "GOALSCORER_LAST",
        "GOALSCORER_OTHER",
        "ASSISTS",
        "GK_SAVES",
        "TEAM_SHOTS",
        "TEAM_SOT",
    ):
        items = family_rows.get(family, [])
        fixtures = {int(row["fixture_id"]) for row in items if row.get("fixture_id") is not None}
        prekickoff_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None and bool(row.get("pre_kickoff"))
        }
        confirmed_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None
            and bool(row.get("confirmed_xi_before_market"))
            and bool(row.get("pre_kickoff"))
        }
        provider_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None and row.get("provider_update") is not None
        }
        xi_aligned_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None
            and bool(row.get("pre_kickoff"))
            and int(row.get("xi_aligned_priced_value_count") or 0) > 0
        }
        stages = Counter(str(row.get("stage") or "UNKNOWN") for row in items)
        bookmakers = {str(row.get("bookmaker")) for row in items if row.get("bookmaker")}
        line_values = sum(int(row.get("line_value_count") or 0) for row in items)
        priced_values = sum(int(row.get("priced_value_count") or 0) for row in items)
        xi_aligned_values = sum(int(row.get("xi_aligned_value_count") or 0) for row in items)
        xi_aligned_line_values = sum(int(row.get("xi_aligned_line_value_count") or 0) for row in items)
        xi_aligned_priced_values = sum(int(row.get("xi_aligned_priced_value_count") or 0) for row in items)

        summaries[family] = {
            "market_snapshot_rows": len(items),
            "unique_fixtures": len(fixtures),
            "pre_kickoff_unique_fixtures": len(prekickoff_fixtures),
            "provider_update_unique_fixtures": len(provider_fixtures),
            "confirmed_xi_pre_kickoff_unique_fixtures": len(confirmed_fixtures),
            "bookmaker_count": len(bookmakers),
            "value_rows": sum(int(row.get("value_count") or 0) for row in items),
            "priced_value_rows": priced_values,
            "exact_line_value_rows": line_values,
            "xi_aligned_value_rows": xi_aligned_values,
            "xi_aligned_priced_value_rows": xi_aligned_priced_values,
            "xi_aligned_exact_line_value_rows": xi_aligned_line_values,
            "confirmed_xi_player_aligned_unique_fixtures": len(xi_aligned_fixtures),
            "stages": dict(sorted(stages.items())),
            "exact_observed_market_history_materialized": bool(items),
            "confirmed_xi_overlap_materialized": bool(confirmed_fixtures),
            "player_xi_alignment_materialized": bool(xi_aligned_fixtures),
        }
        all_fixtures.update(fixtures)
        all_confirmed.update(confirmed_fixtures)
        all_xi_aligned_fixtures.update(xi_aligned_fixtures)
        all_rows += len(items)
        all_line_values += line_values

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_DERIVATIVE_MARKET_AUDIT",
        "lookback_days": int(lookback_days),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "decision_weight": 0.0,
        "candidate_rows": candidate_rows,
        "classified_rows": all_rows,
        "unclassified_rows": candidate_rows - all_rows,
        "market_snapshot_rows": all_rows,
        "unique_fixtures": len(all_fixtures),
        "confirmed_xi_pre_kickoff_unique_fixtures": len(all_confirmed),
        "confirmed_xi_player_aligned_unique_fixtures": len(all_xi_aligned_fixtures),
        "exact_line_value_rows": all_line_values,
        "market_name_counts": dict(market_name_counts.most_common(100)),
        "unclassified_market_name_counts": dict(unclassified_market_name_counts.most_common(100)),
        "families": summaries,
        "sample_rows": samples,
        "unclassified_sample_rows": unclassified_samples,
        "policy": (
            "POSTGRES_READ_ONLY; PREMATCH MARKET SNAPSHOTS ONLY FOR OOS READINESS; "
            "CONFIRMED_XI MUST EXIST AT_OR_BEFORE MARKET CAPTURE; PLAYER PROP VALUES MUST ALIGN "
            "TO EXACT CONFIRMED STARTERS; NO PROVIDER CALLS; MARKET HISTORY DOES NOT IMPLY "
            "MODEL VALIDATION OR PRODUCTION PROMOTION"
        ),
    }


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_rows = max(100, min(int(max_rows), 200000))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "lookback_days": lookback_days,
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
            "families": {},
            "sample_rows": [],
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    m.fixture_id,
                    m.captured_at,
                    m.stage,
                    m.bookmaker_id,
                    m.bookmaker,
                    m.market_id,
                    m.market,
                    m.values,
                    m.provider_update,
                    f.kickoff,
                    CASE
                        WHEN f.kickoff IS NOT NULL AND m.captured_at < f.kickoff THEN TRUE
                        ELSE FALSE
                    END AS pre_kickoff,
                    (confirmed_lineup.payload IS NOT NULL) AS confirmed_xi_before_market,
                    confirmed_lineup.payload AS confirmed_lineup_payload
                FROM soccer_market_snapshots m
                LEFT JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                LEFT JOIN LATERAL (
                    SELECT l.payload
                    FROM soccer_lineup_snapshots l
                    WHERE l.fixture_id = m.fixture_id
                      AND l.captured_at <= m.captured_at
                      AND l.both_xi_confirmed IS TRUE
                    ORDER BY l.captured_at DESC
                    LIMIT 1
                ) confirmed_lineup ON TRUE
                WHERE m.captured_at >= NOW() - (%s * INTERVAL '1 day')
                  AND (
                    LOWER(COALESCE(m.market, '')) LIKE '%%card%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%booking%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%shot%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%gk save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%goalscorer%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%scorer%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%assist%%'
                  )
                ORDER BY m.captured_at DESC
                LIMIT %s
                """,
                (lookback_days, max_rows),
            )
            columns = [desc.name for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    for row in rows:
        for key in ("captured_at", "provider_update", "kickoff"):
            value = row.get(key)
            if isinstance(value, datetime):
                row[key] = value.astimezone(timezone.utc).isoformat()

    report = summarize_rows(rows, lookback_days=lookback_days)
    report["rows_scanned_from_postgres"] = len(rows)
    report["max_rows"] = max_rows
    return report
