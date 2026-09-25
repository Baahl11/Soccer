from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base
from mcp_gateway import player_props_oos_postgres_v4 as oos
from mcp_gateway import research_derivative_postgres_audit as derivative_audit

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PHASE15_COVERAGE_AUDIT_V4_1.1.0"

FAMILY_CONFIG = oos.FAMILY_CONFIG


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("event_payload")
    return payload if isinstance(payload, dict) else row


def _modelable_player_ids(event: dict[str, Any], family: str) -> set[str]:
    config = FAMILY_CONFIG[family]
    intel = event.get(config["intel_key"])
    if not isinstance(intel, dict):
        return set()
    out: set[str] = set()
    for player in intel.get(config["rows_key"]) or []:
        if not isinstance(player, dict) or player.get("player_id") is None:
            continue
        if family != "GK_SAVES" and player.get("confirmed_starter") is not True:
            continue
        modelable = False
        if config["mode"] == "LINES":
            modelable = any(
                isinstance(line, dict)
                and _num(line.get("line")) is not None
                and _num(line.get("p_over")) is not None
                for line in (player.get("lines") or [])
            )
        else:
            probability = _num(player.get(config.get("probability_key")))
            modelable = probability is not None and 0.0 <= probability <= 1.0
        if modelable:
            out.add(str(player["player_id"]))
    return out


def _postgame_player_ids(event: dict[str, Any]) -> set[str]:
    return set(oos._postgame_players(event).keys())


def _family_market_rows(event: dict[str, Any], family: str) -> list[dict[str, Any]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    rows: list[dict[str, Any]] = []
    for row in market.get("research_cards_props_markets") or []:
        if not isinstance(row, dict):
            continue
        row_family = row.get("research_subfamily") or derivative_audit.classify_market(row.get("market"))
        if row_family == family:
            rows.append(row)
    return rows


def _aligned_quoted_player_ids(event: dict[str, Any], family: str) -> set[str]:
    lineup = event.get("lineups") if isinstance(event.get("lineups"), dict) else None
    out: set[str] = set()
    for market_row in _family_market_rows(event, family):
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            aligned = (
                dict(value)
                if value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI"
                else derivative_audit.align_value_to_confirmed_xi(
                    value,
                    lineup_payload=lineup,
                    family=family,
                )
            )
            if aligned.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI" and aligned.get("player_id") is not None:
                out.add(str(aligned["player_id"]))
    return out


def classify_fixture_family(
    pregame_row: dict[str, Any],
    *,
    family: str,
    postgame_row: dict[str, Any] | None,
    finalized_result_exists: bool,
) -> dict[str, Any]:
    event = _event_payload(pregame_row)
    lineup = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    modelable_ids = _modelable_player_ids(event, family)
    market_rows = _family_market_rows(event, family)
    quoted_ids = _aligned_quoted_player_ids(event, family)
    model_price_overlap = modelable_ids & quoted_ids

    post_event = _event_payload(postgame_row) if isinstance(postgame_row, dict) else {}
    post_ids = _postgame_player_ids(post_event)
    result_overlap = modelable_ids & post_ids

    if not modelable_ids:
        oos_reason = "NO_PREGAME_MODEL_SIGNAL"
        oos_recoverability = "FUTURE_CAPTURE_ONLY"
    elif postgame_row is None:
        oos_reason = "POSTGAME_EVENT_MISSING"
        oos_recoverability = (
            "PROVIDER_BACKFILL_CANDIDATE"
            if finalized_result_exists and bool(coverage.get("statistics_players"))
            else "FUTURE_CAPTURE_OR_PROVIDER_UNAVAILABLE"
        )
    elif not post_ids:
        oos_reason = "POSTGAME_PLAYER_STATS_MISSING"
        oos_recoverability = (
            "PROVIDER_BACKFILL_CANDIDATE"
            if bool(coverage.get("statistics_players"))
            else "PROVIDER_PLAYER_STATS_NOT_COVERED"
        )
    elif not result_overlap:
        oos_reason = "PLAYER_ID_OVERLAP_MISSING"
        oos_recoverability = "RECONCILIATION_CANDIDATE"
    else:
        oos_reason = "READY_OOS"
        oos_recoverability = "ALREADY_MATERIALIZED"

    if not modelable_ids:
        clv_reason = "NO_PREGAME_MODEL_SIGNAL"
        clv_recoverability = "FUTURE_CAPTURE_ONLY"
    elif not market_rows:
        clv_reason = "PLAYER_PROP_MARKET_NOT_CAPTURED"
        clv_recoverability = "FUTURE_CAPTURE_ONLY"
    elif lineup.get("both_xi_confirmed") is not True:
        clv_reason = "XI_NOT_CONFIRMED_AT_QUOTE"
        clv_recoverability = "FUTURE_CAPTURE_ONLY"
    elif not quoted_ids:
        clv_reason = "PLAYER_PRICE_NOT_XI_ALIGNED"
        clv_recoverability = "RECONCILIATION_CANDIDATE"
    elif not model_price_overlap:
        clv_reason = "MODEL_PRICE_PLAYER_MISMATCH"
        clv_recoverability = "RECONCILIATION_CANDIDATE"
    else:
        clv_reason = "ENTRY_SIGNAL_READY"
        clv_recoverability = "NEEDS_LATER_CLOSE_OR_ALREADY_TRACKABLE"

    return {
        "family": family,
        "modelable_player_count": len(modelable_ids),
        "postgame_player_count": len(post_ids),
        "oos_player_overlap_count": len(result_overlap),
        "market_row_count": len(market_rows),
        "xi_aligned_quoted_player_count": len(quoted_ids),
        "model_price_player_overlap_count": len(model_price_overlap),
        "both_xi_confirmed": bool(lineup.get("both_xi_confirmed")),
        "statistics_players_coverage": bool(coverage.get("statistics_players")),
        "finalized_result_exists": bool(finalized_result_exists),
        "oos_reason": oos_reason,
        "oos_recoverability": oos_recoverability,
        "clv_reason": clv_reason,
        "clv_recoverability": clv_recoverability,
    }


def build_audit(
    pregame_rows: Iterable[dict[str, Any]],
    postgame_rows: Iterable[dict[str, Any]],
    finalized_fixture_ids: set[int] | None = None,
) -> dict[str, Any]:
    finalized_fixture_ids = finalized_fixture_ids or set()
    canonical = oos.choose_canonical_pregame_events(pregame_rows)
    postgame = oos._latest_postgame_by_fixture(postgame_rows)

    fixture_rows: list[dict[str, Any]] = []
    oos_reasons: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_CONFIG}
    clv_reasons: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_CONFIG}
    oos_recovery: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_CONFIG}
    clv_recovery: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_CONFIG}

    for row in canonical:
        event = _event_payload(row)
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = row.get("fixture_id") or fixture.get("fixture_id")
        if fixture_id is None:
            continue
        fid = int(fixture_id)
        family_rows = {}
        for family in FAMILY_CONFIG:
            audit = classify_fixture_family(
                row,
                family=family,
                postgame_row=postgame.get(fid),
                finalized_result_exists=fid in finalized_fixture_ids,
            )
            family_rows[family] = audit
            oos_reasons[family][audit["oos_reason"]] += 1
            clv_reasons[family][audit["clv_reason"]] += 1
            oos_recovery[family][audit["oos_recoverability"]] += 1
            clv_recovery[family][audit["clv_recoverability"]] += 1

        fixture_rows.append({
            "fixture_id": fid,
            "stage": row.get("stage") or event.get("stage"),
            "generated_at": str(row.get("generated_at") or ""),
            "kickoff": str(row.get("kickoff") or fixture.get("kickoff") or ""),
            "has_postgame_event": fid in postgame,
            "has_finalized_result": fid in finalized_fixture_ids,
            "families": family_rows,
        })

    families: dict[str, Any] = {}
    for family in FAMILY_CONFIG:
        families[family] = {
            "canonical_fixtures": len(fixture_rows),
            "oos_reasons": dict(sorted(oos_reasons[family].items())),
            "clv_reasons": dict(sorted(clv_reasons[family].items())),
            "oos_recoverability": dict(sorted(oos_recovery[family].items())),
            "clv_recoverability": dict(sorted(clv_recovery[family].items())),
            "oos_ready_fixtures": oos_reasons[family]["READY_OOS"],
            "oos_provider_backfill_candidates": oos_recovery[family]["PROVIDER_BACKFILL_CANDIDATE"],
            "oos_reconciliation_candidates": oos_recovery[family]["RECONCILIATION_CANDIDATE"],
            "clv_entry_ready_fixtures": clv_reasons[family]["ENTRY_SIGNAL_READY"],
            "clv_reconciliation_candidates": clv_recovery[family]["RECONCILIATION_CANDIDATE"],
            "future_capture_only_oos": oos_recovery[family]["FUTURE_CAPTURE_ONLY"],
            "future_capture_only_clv": clv_recovery[family]["FUTURE_CAPTURE_ONLY"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PHASE15_COVERAGE_RECONCILIATION_AUDIT",
        "canonical_pregame_fixtures": len(fixture_rows),
        "postgame_overlap_fixtures": sum(1 for row in fixture_rows if row["has_postgame_event"]),
        "finalized_result_fixtures": sum(1 for row in fixture_rows if row["has_finalized_result"]),
        "families": families,
        "fixtures": fixture_rows,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "AUDIT ONLY. DISTINGUISH HISTORICALLY RECOVERABLE EVIDENCE FROM DATA THAT NEVER EXISTED "
            "AT THE ORIGINAL PREDICTION POINT. DO NOT BACKFILL PREGAME MODEL SIGNALS OR XI/PRICES "
            "RETROACTIVELY. PROVIDER BACKFILL MAY ONLY TARGET FINALIZED PLAYER OUTCOMES WHEN THE "
            "ORIGINAL PREGAME MODEL SIGNAL ALREADY EXISTS."
        ),
    }


def _load_rows(conn, *, lookback_days: int, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.fixture_id, e.generated_at, e.stage, e.payload AS event_payload, f.kickoff
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND e.stage IN ('T-40','T-30','T-20','T-10')
            ORDER BY e.fixture_id, e.generated_at
            LIMIT %s
            """,
            (cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        pregame = [dict(zip(columns, row)) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT e.fixture_id, e.generated_at, e.stage, e.payload AS event_payload, f.kickoff
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.stage IN ('POSTGAME','POSTGAME_BACKFILL')
            ORDER BY e.fixture_id, e.generated_at
            LIMIT %s
            """,
            (cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        postgame = [dict(zip(columns, row)) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT fixture_id
            FROM soccer_results
            WHERE graded_at >= %s
            """,
            (cutoff,),
        )
        finalized = {int(row[0]) for row in cur.fetchall() if row and row[0] is not None}

    return pregame, postgame, finalized


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_rows = max(100, min(int(max_rows), 200000))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "families": {},
            "fixtures": [],
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        pregame, postgame, finalized = _load_rows(
            conn,
            lookback_days=lookback_days,
            max_rows=max_rows,
        )

    report = build_audit(pregame, postgame, finalized)
    report["lookback_days"] = lookback_days
    report["pregame_event_rows_loaded"] = len(pregame)
    report["postgame_event_rows_loaded"] = len(postgame)
    return report
