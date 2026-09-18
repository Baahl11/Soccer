from __future__ import annotations

from typing import Any

RESEARCH_ONLY_STAGES = {"EARLY_RESEARCH", "T-90", "T-60", "T-30"}
ACTIONABLE_STAGES = {"T-40", "T-20", "T-10"}
MARKET_RESEARCH_STAGES = {"EARLY_RESEARCH", "T-90"}
MIN_ACTIONABLE_AVAILABILITY = 0.85


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shortlisted(event: dict[str, Any]) -> bool:
    shortlist = event.get("sporting_shortlist")
    return isinstance(shortlist, dict) and bool(shortlist.get("shortlisted"))


def _xi_gk_verified(event: dict[str, Any]) -> bool:
    lineup = event.get("lineups")
    return (
        isinstance(lineup, dict)
        and bool(lineup.get("both_xi_confirmed"))
        and bool(lineup.get("both_goalkeepers_confirmed"))
    )


def _downgrade_to_watch(event: dict[str, Any], reason: str) -> bool:
    if event.get("classification") not in {"BET", "LEAN"}:
        return False
    event["classification"] = "WATCH"
    event["bet_eligible"] = False
    event["stake_units"] = 0.0
    notes = list(event.get("notes") or [])
    if reason not in notes:
        notes.append(reason)
    event["notes"] = notes
    return True


def apply(payload: dict[str, Any]) -> dict[str, int]:
    research_capped = 0
    actionable_released = 0
    actionable_blocked = 0
    early_market_snapshots = 0
    true_t90_market_snapshots = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue

        stage = str(event.get("stage") or "")
        shortlisted = _shortlisted(event)
        provenance = (
            event.get("market_provenance")
            if isinstance(event.get("market_provenance"), dict)
            else {}
        )
        fresh_market = provenance.get("fresh") is True
        availability = _num(event.get("availability_confidence"))
        xi_gk = _xi_gk_verified(event)

        if stage in RESEARCH_ONLY_STAGES:
            reason = (
                f"LIFECYCLE CAP: {stage} is research-only; BET/LEAN cannot be "
                "promoted before the actionable market-validation lifecycle."
            )
            if _downgrade_to_watch(event, reason):
                research_capped += 1

            market_snapshot_allowed = stage in MARKET_RESEARCH_STAGES and shortlisted
            if stage == "EARLY_RESEARCH" and isinstance(event.get("market"), dict):
                early_market_snapshots += 1
            if stage == "T-90" and isinstance(event.get("market"), dict):
                true_t90_market_snapshots += 1

            event["lifecycle_cap"] = {
                "stage": stage,
                "mode": "RESEARCH_ONLY",
                "maximum_classification": "WATCH",
                "bet_allowed": False,
                "lean_allowed": False,
                "market_snapshot_allowed": market_snapshot_allowed,
                "market_snapshot_requires_sport_first_shortlist": True,
                "market_snapshot_purpose": (
                    "PRICE_DISCOVERY_ONLY"
                    if market_snapshot_allowed
                    else "NO_MARKET_OR_NOT_SHORTLISTED"
                ),
                "fresh_market_observed": fresh_market,
                "availability_confidence": availability,
                "xi_gk_verified": xi_gk,
                "cap_release_basis": [
                    "LIFECYCLE_STAGE",
                    "SPORT_FIRST_SHORTLIST",
                    "PROVIDER_UPDATE_QUOTE_FRESHNESS",
                    "AVAILABILITY",
                    "XI_GK_VERIFICATION",
                ],
            }
            continue

        if stage in ACTIONABLE_STAGES:
            blockers: list[str] = []
            if not fresh_market:
                blockers.append("CURRENT_PROVIDER_TIMESTAMPED_MARKET_REQUIRED")
            if availability is None or availability < MIN_ACTIONABLE_AVAILABILITY:
                blockers.append("AVAILABILITY_BELOW_EXISTING_0.85_GATE")
            if not xi_gk:
                blockers.append("XI_GK_NOT_VERIFIED")

            released = not blockers
            if released:
                actionable_released += 1
            else:
                actionable_blocked += 1
                _downgrade_to_watch(
                    event,
                    "LIFECYCLE CAP: actionable stage remains WATCH until existing "
                    "quote-freshness and availability/XI-GK gates are satisfied.",
                )

            event["lifecycle_cap"] = {
                "stage": stage,
                "mode": "CANONICAL_GATES_RELEASED" if released else "ELASTIC_CAP_HELD",
                "maximum_classification": (
                    "CANONICAL_ENGINE_TIER_CAP"
                    if released
                    else "WATCH"
                ),
                "bet_allowed": released,
                "lean_allowed": released,
                "fresh_market_observed": fresh_market,
                "availability_confidence": availability,
                "xi_gk_verified": xi_gk,
                "blockers": blockers,
                "cap_release_basis": [
                    "LIFECYCLE_STAGE",
                    "PROVIDER_UPDATE_QUOTE_FRESHNESS",
                    "EXISTING_AVAILABILITY_0.85_GATE",
                    "XI_GK_VERIFICATION",
                ],
                "canonical_probability_thresholds_changed": False,
                "canonical_model_weights_changed": False,
            }
            continue

        if stage == "CLOSE":
            event["lifecycle_cap"] = {
                "stage": stage,
                "mode": "CLOSING_SNAPSHOT_ONLY",
                "maximum_classification": "CLOSE",
                "new_bet_promotion_allowed": False,
            }
        elif stage == "POSTGAME":
            event["lifecycle_cap"] = {
                "stage": stage,
                "mode": "POSTGAME_GRADING_ONLY",
                "new_bet_promotion_allowed": False,
            }

    return {
        "research_stage_promotions_blocked": research_capped,
        "actionable_stage_caps_released": actionable_released,
        "actionable_stage_caps_held": actionable_blocked,
        "early_research_market_snapshots": early_market_snapshots,
        "true_t90_market_snapshots": true_t90_market_snapshots,
    }
