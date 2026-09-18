from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v33 as v33
from mcp_gateway import quote_freshness as qf

MODEL_VERSION = v33.MODEL_VERSION
AUTOMATION_VERSION = "3.10.0"
REGISTRY_CACHE_KEY = "_galaxy_candidate_registry_v1"
REGISTRY_TTL = timedelta(hours=14)
MAX_ACTIVE_SAME_GAME = 20
MAX_ACTIVE_MULTI = 20


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        out = value
    else:
        try:
            out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out.astimezone(dt_timezone.utc)


def _sgp_key(candidate: dict[str, Any]) -> str:
    legs = []
    for leg in candidate.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        legs.append(
            (
                str(leg.get("family") or ""),
                str(leg.get("selection") or ""),
                str(leg.get("line") if leg.get("line") is not None else ""),
            )
        )
    legs.sort()
    return f"SGP:{candidate.get('fixture_id')}:{repr(legs)}"


def _multi_key(candidate: dict[str, Any]) -> str:
    legs = []
    for leg in candidate.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        legs.append(
            (
                str(leg.get("fixture_id") or ""),
                str(leg.get("family") or ""),
                str(leg.get("selection") or ""),
                str(leg.get("line") if leg.get("line") is not None else ""),
            )
        )
    legs.sort()
    return f"MULTI:{candidate.get('bookmaker')}:{repr(legs)}"


def _candidate_future(candidate: dict[str, Any], now: datetime) -> bool:
    if candidate.get("type") == "SAME_GAME_PARLAY":
        ko = _dt(candidate.get("kickoff"))
        return bool(ko and ko > now)
    legs = candidate.get("legs") or []
    kickoffs = [_dt(leg.get("kickoff")) for leg in legs if isinstance(leg, dict)]
    kickoffs = [x for x in kickoffs if x is not None]
    return bool(kickoffs) and min(kickoffs) > now



def _candidate_price_fresh(candidate: dict[str, Any], now: datetime) -> bool:
    return qf.candidate_price_fresh(
        candidate,
        now,
        qf.DEFAULT_MAX_AGE_MINUTES,
    )


def _candidate_fixture_ids(candidate: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    fid = candidate.get("fixture_id")
    try:
        if fid is not None:
            out.add(int(fid))
    except (TypeError, ValueError):
        pass
    for leg in candidate.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        try:
            if leg.get("fixture_id") is not None:
                out.add(int(leg.get("fixture_id")))
        except (TypeError, ValueError):
            pass
    return out


def _load_registry(now: datetime) -> dict[str, Any]:
    cached = base._cache_get("sport_shortlist", REGISTRY_CACHE_KEY, REGISTRY_TTL, now)
    if not isinstance(cached, dict):
        cached = {}
    same = cached.get("same_game") if isinstance(cached.get("same_game"), dict) else {}
    multi = cached.get("multi_match") if isinstance(cached.get("multi_match"), dict) else {}
    return {"schema_version": "1.0.0", "same_game": dict(same), "multi_match": dict(multi)}


def _persist_registry(registry: dict[str, Any], now: datetime) -> None:
    base._cache_set("sport_shortlist", REGISTRY_CACHE_KEY, registry, now)


def _rank_sgp(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if candidate.get("all_leg_models_actionable") else 0.0,
        1.0 if candidate.get("component_price_reference") else 0.0,
        float(candidate.get("joint_model_probability") or 0.0),
    )


def _rank_multi(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if candidate.get("all_leg_models_actionable") else 0.0,
        float(candidate.get("probability_edge_vs_component_product_pp") or 0.0),
        float(candidate.get("conservative_joint_probability") or 0.0),
    )


def _rolling_builder(payload: dict[str, Any]) -> dict[str, Any]:
    now = _dt(payload.get("generated_at_utc")) or datetime.now(dt_timezone.utc)
    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    current_sgp = [dict(x) for x in builder.get("same_game_candidates") or [] if isinstance(x, dict)]
    current_multi = [dict(x) for x in builder.get("multi_match_candidates") or [] if isinstance(x, dict)]

    due_fixture_ids: set[int] = set()
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        try:
            fid = int(fx.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        due_fixture_ids.add(fid)

    registry = _load_registry(now)
    same = registry["same_game"]
    multi = registry["multi_match"]

    # Expire started matches and invalidate old candidates whenever a fixture is
    # actively re-evaluated. The current tick is authoritative for that fixture.
    for key, row in list(same.items()):
        candidate = row.get("candidate") if isinstance(row, dict) else None
        if (
            not isinstance(candidate, dict)
            or not _candidate_future(candidate, now)
            or not _candidate_price_fresh(candidate, now)
            or (_candidate_fixture_ids(candidate) & due_fixture_ids)
        ):
            same.pop(key, None)
    for key, row in list(multi.items()):
        candidate = row.get("candidate") if isinstance(row, dict) else None
        if (
            not isinstance(candidate, dict)
            or not _candidate_future(candidate, now)
            or not _candidate_price_fresh(candidate, now)
            or (_candidate_fixture_ids(candidate) & due_fixture_ids)
        ):
            multi.pop(key, None)

    stamp = payload.get("generated_at_local") or payload.get("generated_at_utc")
    for candidate in current_sgp:
        candidate["rolling_registry_last_updated"] = stamp
        candidate["rolling_registry_status"] = "ACTIVE_UNTIL_REEVALUATED_QUOTE_STALE_OR_KICKOFF"
        same[_sgp_key(candidate)] = {"updated_at": now.timestamp(), "candidate": candidate}
    for candidate in current_multi:
        candidate["rolling_registry_last_updated"] = stamp
        candidate["rolling_registry_status"] = "ACTIVE_UNTIL_REEVALUATED_QUOTE_STALE_OR_FIRST_KICKOFF"
        multi[_multi_key(candidate)] = {"updated_at": now.timestamp(), "candidate": candidate}

    active_sgp = [row.get("candidate") for row in same.values() if isinstance(row, dict) and isinstance(row.get("candidate"), dict)]
    active_multi = [row.get("candidate") for row in multi.values() if isinstance(row, dict) and isinstance(row.get("candidate"), dict)]
    active_sgp = [x for x in active_sgp if _candidate_future(x, now) and _candidate_price_fresh(x, now)]
    active_multi = [x for x in active_multi if _candidate_future(x, now) and _candidate_price_fresh(x, now)]
    active_sgp.sort(key=_rank_sgp, reverse=True)
    active_multi.sort(key=_rank_multi, reverse=True)
    active_sgp = active_sgp[:MAX_ACTIVE_SAME_GAME]
    active_multi = active_multi[:MAX_ACTIVE_MULTI]

    registry["same_game"] = {_sgp_key(x): {"updated_at": now.timestamp(), "candidate": x} for x in active_sgp}
    registry["multi_match"] = {_multi_key(x): {"updated_at": now.timestamp(), "candidate": x} for x in active_multi}
    registry["updated_at_utc"] = now.isoformat()
    registry["policy"] = "ROLLING_CANDIDATES_PERSIST_BETWEEN_10_MIN_TICKS_ONLY_WHILE_PROVIDER_UPDATE_IS_FRESH; CURRENT_FIXTURE_REEVALUATION_REPLACES_PRIOR_CANDIDATES; STALE_OR_MISSING_PROVIDER_TIMESTAMP_EXPIRES; KICKOFF_EXPIRES; NO_EXTRA_PROVIDER_REQUESTS"
    _persist_registry(registry, now)

    out = dict(builder)
    out["current_tick_same_game_candidate_count"] = len(current_sgp)
    out["current_tick_multi_match_candidate_count"] = len(current_multi)
    out["current_tick_candidate_count"] = len(current_sgp) + len(current_multi)
    out["same_game_candidates"] = active_sgp
    out["multi_match_candidates"] = active_multi
    out["same_game_candidate_count"] = len(active_sgp)
    out["multi_match_candidate_count"] = len(active_multi)
    out["candidate_count"] = len(active_sgp) + len(active_multi)
    out["rolling_registry"] = {
        "schema_version": "1.0.0",
        "active": True,
        "same_game_active": len(active_sgp),
        "multi_match_active": len(active_multi),
        "current_tick_same_game": len(current_sgp),
        "current_tick_multi_match": len(current_multi),
        "invalidated_fixture_count_this_tick": len(due_fixture_ids),
        "updated_at_utc": now.isoformat(),
        "policy": registry["policy"],
    }
    return out


async def run_tick() -> dict[str, Any]:
    payload = await v33.run_tick()
    current_tick_count = int(payload.get("galaxy_builder_candidate_count_this_tick") or 0)
    payload["galaxy_builder"] = _rolling_builder(payload)
    payload["galaxy_builder_candidate_count_this_tick"] = current_tick_count
    payload["galaxy_builder_active_candidate_count"] = int(payload["galaxy_builder"].get("candidate_count") or 0)
    payload["galaxy_builder_rolling_registry_policy"] = (
        "DURABLE_VIA_SHORTLIST_STATE; ACTIVE_CANDIDATES_SURVIVE_NON_DUE_TICKS; "
        "REEVALUATION_REPLACES_PRIOR_FIXTURE_CANDIDATES; PROVIDER_UPDATE_FRESHNESS_EXPIRES; "
        "CAPTURE_TIME_DOES_NOT_REFRESH_QUOTES; KICKOFF_EXPIRES; ZERO_EXTRA_PROVIDER_REQUESTS"
    )
    payload["shortlist_state"] = v6.export_shortlist_state()
    payload["shortlist_state_count"] = len(payload["shortlist_state"])
    payload["v310_provider_requests_added"] = 0
    payload["v310_model_weights_changed"] = False
    payload["v310_canonical_bet_logic_changed"] = False
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
