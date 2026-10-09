from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import httpx

CACHE_TTL_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 3.0

_REPORTS = {
    "1X2": "v4_017_1x2_calibration_validation.json",
    "BTTS": "v4_018_btts_calibration_validation.json",
    "FT Totals": "v4_016_ft_totals_production_validation.json",
    "Team Totals": "v4_019_team_totals_oos_validation.json",
    "1H": "v4_020_1h_oos_validation.json",
    "Corners": "v4_022_corners_oos_validation.json",
    "2H": "v4_021_2h_oos_validation.json",
    "Cards": "phase14_cards_referee_validation.json",
    "Player Props": "phase15_player_props_validation.json",
}
_COLLECTION_KEYS = {
    "1X2": ("1X2",),
    "BTTS": ("BTTS",),
    "FT Totals": ("FT_TOTALS",),
    "Team Totals": ("HOME_TT", "AWAY_TT"),
    "1H": ("1H",),
    "Corners": ("FT_CORNERS", "TEAM_CORNERS"),
    "2H": ("2H",),
    "Cards": ("CARDS",),
    "Player Props": ("SHOTS", "SOT", "GOALSCORER", "ASSISTS", "PLAYER_CARDS", "GK_SAVES"),
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT = 0.0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sum_keys(node: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    found = False
    total = 0
    for key in keys:
        value = _integer(node.get(key))
        if value is None:
            continue
        found = True
        total += value
    return total if found else None


def _state_config() -> tuple[str, str] | None:
    repo = os.getenv("STATE_REPO", "").strip()
    branch = os.getenv("STATE_BRANCH", "").strip()
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/"):]
    if not repo or not branch:
        return None
    return repo, branch


def _blockers(report: dict[str, Any]) -> list[str]:
    values = report.get("blockers")
    return [str(value) for value in values if value is not None] if isinstance(values, list) else []


def _target_from_blockers(blockers: list[str], prefix: str) -> tuple[int | None, int | None]:
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_LT_(\d+)")
    for blocker in blockers:
        match = pattern.search(blocker.upper())
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def _generic_sample(report: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    calibration = _dict(report.get("calibration_sample"))
    n = _integer(calibration.get("n"))
    if n is not None:
        return n, _integer(calibration.get("minimum_n")), "calibration rows"

    canonical = _dict(report.get("canonical_multiclass_oos"))
    temperature = _dict(canonical.get("temperature_scaled"))
    n = _integer(temperature.get("n"))
    if n is not None:
        return n, None, "multiclass OOS rows"

    sample = _dict(report.get("sample"))
    for key in ("model_settled", "oos_rows", "rows", "n"):
        n = _integer(sample.get(key))
        if n is not None:
            return n, None, "validation rows"
    return None, None, None


def _model_evidence(label: str, report: dict[str, Any]) -> dict[str, Any]:
    if label == "Team Totals":
        sample = _dict(report.get("oos_sample"))
        current = _integer(sample.get("evaluated_fixtures"))
        target = _integer(sample.get("minimum_actionable_review_fixtures"))
        return {"current": current, "target": target, "unit": "OOS fixtures", "ready": bool(current is not None and target and current >= target)}

    if label == "1H":
        calibration = _dict(report.get("calibration"))
        challenger = _dict(calibration.get("challenger"))
        current = _integer(calibration.get("n")) or _integer(challenger.get("n"))
        target = _integer(calibration.get("minimum_calibrated_oos"))
        return {"current": current, "target": target, "unit": "calibrated OOS rows", "ready": bool(current is not None and target and current >= target)}

    if label == "Corners":
        ft = _dict(report.get("ft_corners"))
        current = _integer(ft.get("formation_adjusted_evaluations"))
        blocker_current, blocker_target = _target_from_blockers(_blockers(report), "FORMATION_ADJUSTED")
        if current is None:
            current = blocker_current
        target = blocker_target
        return {"current": current, "target": target, "unit": "formation-adjusted fixtures", "ready": bool(current is not None and target and current >= target)}

    if label == "2H":
        sample = _dict(report.get("oos_sample"))
        current = _integer(sample.get("walk_forward_evaluated"))
        target = _integer(sample.get("minimum_required"))
        return {"current": current, "target": target, "unit": "walk-forward OOS", "ready": bool(current is not None and target and current >= target)}

    if label == "Cards":
        yellow = _dict(report.get("yellow_cards"))
        red = _dict(report.get("red_cards"))
        current = _integer(yellow.get("oos_n"))
        target = _integer(yellow.get("minimum_oos"))
        return {
            "current": current,
            "target": target,
            "unit": "yellow-card OOS",
            "ready": bool(current is not None and target and current >= target),
            "secondary": {
                "red_card_oos": _integer(red.get("oos_n")),
                "red_card_market_review_target": _integer(red.get("minimum_market_review")),
                "yellow_referee_adjusted": _integer(yellow.get("referee_adjusted_n")),
                "yellow_referee_target": _integer(yellow.get("minimum_referee_adjusted")),
            },
        }

    if label == "Player Props":
        families = _dict(report.get("prop_families"))
        profiles = []
        oos_rows = []
        for node in families.values():
            if not isinstance(node, dict):
                continue
            value = _integer(node.get("profiles_valid"))
            if value is not None:
                profiles.append(value)
            oos = _dict(node.get("oos_evidence"))
            value = _integer(oos.get("player_game_rows"))
            if value is not None:
                oos_rows.append(value)
        return {
            "current": max(oos_rows) if oos_rows else 0,
            "target": None,
            "unit": "prop OOS rows",
            "ready": any(value > 0 for value in oos_rows),
            "structural_profiles_max": max(profiles) if profiles else None,
        }

    current, target, unit = _generic_sample(report)
    return {"current": current, "target": target, "unit": unit or "validation rows", "ready": current is not None and current > 0}


def _choose_next_gate(label: str, blockers: list[str]) -> str | None:
    if not blockers:
        return None
    priorities = {
        "Cards": ("OBSERVED_MARKET_PRICE_HISTORY_MISSING", "REFEREE_ADJUSTED", "TRUE_CLV"),
        "Player Props": ("OOS_LEDGER_NOT_MATERIALIZED", "OBSERVED_MARKET_PRICE_HISTORY_MISSING", "TRUE_CLV"),
        "Corners": ("FORMATION_ADJUSTED", "TRUE_CLV", "LEAGUE_LIFT"),
        "1H": ("BRIER_NOT_BETTER", "LOG_LOSS_NOT_BETTER", "TRUE_CLV", "LINE_COVERAGE"),
        "2H": ("DOES_NOT_BEAT_BASELINE", "TRUE_CLV", "LIVE_CONTEXT"),
    }
    wanted = priorities.get(label, ("TRUE_CLV",))
    upper = [(blocker, blocker.upper()) for blocker in blockers]
    for needle in wanted:
        for original, normalized in upper:
            if needle in normalized:
                return original
    return blockers[0]


def _stage(label: str, blockers: list[str], priced: int | None, true_clv: int | None, target: int | None, evidence: dict[str, Any]) -> str:
    normalized = " ".join(blockers).upper()
    if label == "Player Props":
        return "OOS + PRICE EVIDENCE BLOCKED" if not evidence.get("ready") else "PRICE EVIDENCE COLLECTION"
    if label == "Cards" and (priced in (None, 0) or "PRICE_HISTORY_MISSING" in normalized):
        return "PRICE + REFEREE EVIDENCE BLOCKED"
    if "DOES_NOT_BEAT_BASELINE" in normalized or "NOT_BETTER_THAN_BASELINE" in normalized:
        return "MODEL REVIEW + CLV COLLECTION"
    if label == "Corners" and not evidence.get("ready"):
        return "FORMATION MATURATION + CLV"
    if target and true_clv is not None and true_clv >= target:
        return "CLV GATE MET · REVIEW BLOCKED" if blockers else "REVIEW READY"
    if priced is not None and priced > 0:
        return "TRUE CLV COLLECTION"
    if evidence.get("current") not in (None, 0):
        return "MARKET EVIDENCE COLLECTION"
    return "EVIDENCE BUILDING"


def _build_family_rows(clv_report: dict[str, Any], reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mapped_counts = _dict(clv_report.get("mapped_family_counts"))
    priced_counts = _dict(clv_report.get("priced_entry_family_counts"))
    clv_counts = _dict(clv_report.get("family_counts"))
    team_funnel = _dict(clv_report.get("team_totals_maturation_funnel"))

    rows: list[dict[str, Any]] = []
    for label in _REPORTS:
        report = _dict(reports.get(label))
        keys = _COLLECTION_KEYS[label]
        mapped = _sum_keys(mapped_counts, keys)
        priced = _sum_keys(priced_counts, keys)
        true_clv = _sum_keys(clv_counts, keys)
        if true_clv is None:
            true_clv = 0 if clv_report else None
        true_clv_node = _dict(report.get("true_clv"))
        target = _integer(true_clv_node.get("minimum_rows"))
        fixtures = _integer(true_clv_node.get("unique_fixtures"))
        blockers = _blockers(report)
        evidence = _model_evidence(label, report)
        modeled_fixtures = _integer(team_funnel.get("modeled_signal_unique_fixtures")) if label == "Team Totals" else None

        rows.append({
            "label": label,
            "stage": _stage(label, blockers, priced, true_clv, target, evidence),
            "model_evidence": evidence,
            "mapped_rows": mapped,
            "priced_rows": priced,
            "modeled_signal_fixtures": modeled_fixtures,
            "true_clv_rows": true_clv,
            "true_clv_target": target,
            "true_clv_fixtures": fixtures,
            "report_status": report.get("status"),
            "next_gate": _choose_next_gate(label, blockers),
            "blockers": blockers,
            "source": _REPORTS[label],
        })
    return rows



# Read-only, market-level presentation inventory. Parent family evidence is never
# silently represented as an independent side/prop sample or a production BET.
_MARKET_INVENTORY = (
    ("1X2", "1X2", "1X2"),
    ("BTTS", "BTTS", "BTTS"),
    ("FT Totals", "FT Totals", "FT_TOTALS"),
    ("Home Team Totals", "Team Totals", "HOME_TT"),
    ("Away Team Totals", "Team Totals", "AWAY_TT"),
    ("1H", "1H", "1H"),
    ("2H", "2H", "2H"),
    ("FT Corners", "Corners", "FT_CORNERS"),
    ("Team Corners", "Corners", "TEAM_CORNERS"),
    ("Yellow Cards", "Cards", "YELLOW_CARDS"),
    ("Red Cards", "Cards", "RED_CARDS"),
    ("Player Shots", "Player Props", "SHOTS"),
    ("Shots on Target", "Player Props", "SOT"),
    ("Anytime Goalscorer", "Player Props", "GOALSCORER"),
    ("Player Assists", "Player Props", "ASSISTS"),
    ("Player Cards", "Player Props", "PLAYER_CARDS"),
    ("Goalkeeper Saves", "Player Props", "GK_SAVES"),
    ("Double Chance", None, "DOUBLE_CHANCE"),
    ("Draw No Bet", None, "DNB"),
    ("Asian Handicap", None, "ASIAN_HANDICAP"),
    ("Correct Score", None, "CORRECT_SCORE"),
)


def _build_market_inventory(
    family_rows: list[dict[str, Any]],
    clv_report: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parents = {str(row.get("label")): row for row in family_rows}
    clv = _dict(clv_report.get("family_counts"))
    mapped = _dict(clv_report.get("mapped_family_counts"))
    priced = _dict(clv_report.get("priced_entry_family_counts"))
    inventory = []

    for label, parent_name, key in _MARKET_INVENTORY:
        parent = parents.get(parent_name or "", {})
        report = _dict(reports.get(parent_name or ""))
        model_evidence = {
            "current": None,
            "target": None,
            "unit": "market-specific OOS sample not independently verified",
            "ready": False,
        }
        if key in {"1X2", "BTTS", "FT_TOTALS", "1H", "2H", "FT_CORNERS"}:
            model_evidence = dict(_dict(parent.get("model_evidence")))
        elif key in {"YELLOW_CARDS", "RED_CARDS"}:
            node = _dict(report.get("yellow_cards" if key == "YELLOW_CARDS" else "red_cards"))
            model_evidence = {
                "current": _integer(node.get("oos_n")),
                "target": _integer(node.get("minimum_oos")),
                "unit": "card OOS observations",
                "ready": False,
            }
            n, target = model_evidence["current"], model_evidence["target"]
            model_evidence["ready"] = bool(n is not None and target and n >= target)
        elif key == "TEAM_CORNERS":
            node = _dict(report.get("team_corners"))
            model_evidence = {
                "current": _integer(node.get("formation_adjusted_evaluations")),
                "target": _integer(node.get("minimum_formation_adjusted")),
                "unit": "team-side formation-adjusted observations",
                "ready": False,
            }
            n, target = model_evidence["current"], model_evidence["target"]
            model_evidence["ready"] = bool(n is not None and target and n >= target)
        elif parent_name == "Player Props":
            families = _dict(report.get("prop_families"))
            node = _dict(families.get(key))
            oos = _dict(node.get("oos_evidence"))
            model_evidence = {
                "current": _integer(oos.get("player_game_rows")),
                "target": _integer(oos.get("minimum_oos_rows")),
                "unit": "player-game OOS rows",
                "ready": False,
            }
            n, target = model_evidence["current"], model_evidence["target"]
            model_evidence["ready"] = bool(n is not None and target and n >= target)

        # Sparse canonical reports cannot prove zero observations in a missing key.
        # Only an explicitly stored numeric 0 is a verified zero.
        clv_n = _integer(clv.get(key))
        mapped_n = _integer(mapped.get(key))
        priced_n = _integer(priced.get(key))
        direct_oos = model_evidence.get("current") is not None
        blockers = []
        if not report:
            blockers.append("SOURCE_REPORT_NOT_VERIFIED")
        if not direct_oos:
            blockers.append("MARKET_OOS_NOT_VERIFIED")
        if priced_n is None:
            blockers.append("MARKET_PRICE_HISTORY_NOT_VERIFIED")
        if clv_n is None:
            blockers.append("MARKET_TRUE_CLV_NOT_VERIFIED")
        for blocker in parent.get("blockers") or []:
            if blocker not in blockers:
                blockers.append(blocker)
        collection_keys = _COLLECTION_KEYS.get(parent_name or "", ())
        has_independent_parent_target = bool(report) and len(collection_keys) == 1 and key in collection_keys
        inventory.append({
            "key": key,
            "label": label,
            "parent_family": parent_name,
            "source": parent.get("source") if report else None,
            "report_status": parent.get("report_status") if report else "NOT VERIFIED",
            "parent_research_stage": parent.get("stage") if report else None,
            "model_evidence": model_evidence,
            "mapped_rows": mapped_n,
            "priced_rows": priced_n,
            "true_clv_rows": clv_n,
            "true_clv_target": _integer(parent.get("true_clv_target")) if has_independent_parent_target else None,
            "next_gate": blockers[0] if blockers else None,
            "blockers": blockers,
            "market_specific_evidence_verified": direct_oos,
            "production_promotion_allowed": False,
            "classification": "RESEARCH_ONLY",
        })
    return inventory

def load_maturity_evidence(*, force: bool = False) -> dict[str, Any]:
    global _CACHE, _CACHE_AT
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE is not None and now - _CACHE_AT < CACHE_TTL_SECONDS:
            return dict(_CACHE)

        config = _state_config()
        if config is None:
            result = {
                "status": "UNAVAILABLE",
                "reason": "STATE_REPO_OR_STATE_BRANCH_NOT_CONFIGURED",
                "families": [],
                "provider_requests_added": 0,
                "canonical_bet_logic_changed": False,
            }
            _CACHE, _CACHE_AT = result, now
            return dict(result)

        repo, branch = config
        base = f"https://raw.githubusercontent.com/{repo}/{branch}/soccer_edge_state/analysis"
        reports: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        clv_report: dict[str, Any] = {}

        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            try:
                response = client.get(f"{base}/clv_v4_postgres_report.json", headers={"Accept": "application/json"})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    clv_report = payload
                else:
                    errors["CLV"] = "unexpected_json_shape"
            except Exception as exc:
                errors["CLV"] = f"{type(exc).__name__}: {str(exc)[:120]}"

            for label, filename in _REPORTS.items():
                try:
                    response = client.get(f"{base}/{filename}", headers={"Accept": "application/json"})
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        reports[label] = payload
                    else:
                        errors[label] = "unexpected_json_shape"
                except Exception as exc:
                    errors[label] = f"{type(exc).__name__}: {str(exc)[:120]}"

        rows = _build_family_rows(clv_report, reports)
        result = {
            "status": "UNAVAILABLE" if not (clv_report or any(reports.values())) else ("PARTIAL" if errors else "OK"),
            "families": rows,
            "market_rows": _build_market_inventory(rows, clv_report, reports),
            "source_model_version": clv_report.get("model_version"),
            "comparable_true_clv_rows": _integer(clv_report.get("comparable_true_clv_rows")),
            "minimum_true_close_rows": _integer(clv_report.get("minimum_true_close_rows")),
            "truth_note": "Maturity is multi-gate. OOS/model evidence, observed market collection and strict True CLV are shown separately; a 0 True-CLV count is not labeled as no evidence when upstream evidence exists.",
            "errors": errors,
            "state_repo": repo,
            "state_branch": branch,
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "production_promotion_allowed": False,
        }
        _CACHE, _CACHE_AT = result, now
        return dict(result)
