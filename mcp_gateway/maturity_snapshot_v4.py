from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from mcp_gateway import maturation_watchdogs_v4

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_MATURITY_SNAPSHOT_V4_1.1.0"
CACHE_TTL_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 3.0

_REPORTS = {
    "one_x_two": "v4_017_1x2_calibration_validation.json",
    "btts": "v4_018_btts_calibration_validation.json",
    "team_totals": "v4_019_team_totals_oos_validation.json",
    "one_h": "v4_020_1h_oos_validation.json",
    "two_h": "v4_021_2h_oos_validation.json",
    "corners": "v4_022_corners_oos_validation.json",
    "cards": "phase14_cards_referee_validation.json",
    "player_props": "phase15_player_props_validation.json",
    "signal_summary": "signal_ledger_summary.json",
    "settlement_coverage": "settlement_coverage_report.json",
    "api_efficiency": "api_efficiency.json",
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT = 0.0
_WATCHDOG_BASELINE: dict[str, Any] | None = None


def _state_config() -> tuple[str, str] | None:
    repo = os.getenv("STATE_REPO", "").strip()
    branch = os.getenv("STATE_BRANCH", "").strip()
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/"):]
    if not repo or not branch:
        return None
    return repo, branch


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fetch_reports(repo: str, branch: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reports: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    base = f"https://raw.githubusercontent.com/{repo}/{branch}/soccer_edge_state/analysis"
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for key, filename in _REPORTS.items():
            try:
                response = client.get(f"{base}/{filename}", headers={"Accept": "application/json"})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    reports[key] = payload
                else:
                    errors[key] = "unexpected_json_shape"
            except Exception as exc:  # dashboard telemetry must never break the product route
                errors[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return reports, errors


def _gate(current: Any, target: Any, *, source: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "current": _int(current),
        "target": _int(target),
        "source": source,
    }
    if extra:
        out.update(extra)
    return out


def _build_summary(reports: dict[str, dict[str, Any]], errors: dict[str, str]) -> dict[str, Any]:
    one_x_two = reports.get("one_x_two", {})
    btts = reports.get("btts", {})
    team_totals = reports.get("team_totals", {})
    one_h = reports.get("one_h", {})
    corners = reports.get("corners", {})
    player_props = reports.get("player_props", {})

    one_x_two_clv = _dict(one_x_two.get("true_clv"))
    btts_clv = _dict(btts.get("true_clv"))
    team_totals_clv = _dict(team_totals.get("true_clv"))
    one_h_clv = _dict(one_h.get("true_clv"))
    corners_ft = _dict(corners.get("ft_corners"))
    props_clv = _dict(player_props.get("true_clv"))
    props_by_family = _dict(props_clv.get("by_family"))

    prop_rows = {
        str(name): _int(_dict(node).get("rows"))
        for name, node in props_by_family.items()
    }
    prop_rows_known = [value for value in prop_rows.values() if value is not None]
    prop_current = min(prop_rows_known) if prop_rows_known else _int(props_clv.get("rows"))

    gates = {
        "1x2_true_clv": _gate(
            one_x_two_clv.get("rows"),
            one_x_two_clv.get("minimum_rows"),
            source=_REPORTS["one_x_two"],
            extra={"unique_fixtures": _int(one_x_two_clv.get("unique_fixtures"))},
        ),
        "btts_true_clv": _gate(
            btts_clv.get("rows"),
            btts_clv.get("minimum_rows"),
            source=_REPORTS["btts"],
            extra={"unique_fixtures": _int(btts_clv.get("unique_fixtures"))},
        ),
        "team_totals_true_clv": _gate(
            team_totals_clv.get("rows"),
            team_totals_clv.get("minimum_rows"),
            source=_REPORTS["team_totals"],
            extra={
                "unique_fixtures": _int(team_totals_clv.get("unique_fixtures")),
                "minimum_unique_fixtures": _int(team_totals_clv.get("minimum_unique_fixtures")),
            },
        ),
        "1h_true_clv": _gate(
            one_h_clv.get("rows"),
            one_h_clv.get("minimum_rows"),
            source=_REPORTS["one_h"],
            extra={"unique_fixtures": _int(one_h_clv.get("unique_fixtures"))},
        ),
        "corners_formation": _gate(
            corners_ft.get("formation_adjusted_evaluations"),
            corners.get("minimum_formation_adjusted") or 100,
            source=_REPORTS["corners"],
        ),
        "player_props_true_clv": _gate(
            prop_current,
            props_clv.get("minimum_rows") or 50,
            source=_REPORTS["player_props"],
            extra={"by_family_rows": prop_rows},
        ),
    }

    loaded = len(reports)
    status = "OK" if loaded == len(_REPORTS) else ("PARTIAL" if loaded else "UNAVAILABLE")
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports_loaded": loaded,
        "reports_expected": len(_REPORTS),
        "errors": errors,
        "gates": gates,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
    }


def load_snapshot(*, force: bool = False) -> dict[str, Any]:
    """Read validation/maturation counters from the persisted state branch.

    The state branch is research telemetry, not a runtime decision source. The
    result is cached in-process so dashboard refreshes do not repeatedly hit
    GitHub. If STATE_REPO/STATE_BRANCH are absent (for example unit tests), the
    function returns an unavailable snapshot without doing network I/O.
    """
    global _CACHE, _CACHE_AT, _WATCHDOG_BASELINE

    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE is not None and now - _CACHE_AT < CACHE_TTL_SECONDS:
            return dict(_CACHE)

        config = _state_config()
        if config is None:
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION,
                "status": "UNAVAILABLE",
                "reason": "STATE_REPO_OR_STATE_BRANCH_NOT_CONFIGURED",
                "gates": {},
                "errors": {},
                "maturation_watchdogs": {
                    "status": "NOT_VERIFIED",
                    "reason": "STATE_REPO_OR_STATE_BRANCH_NOT_CONFIGURED",
                    "watchdogs": {},
                    "provider_requests_added": 0,
                    "production_promotion_allowed": False,
                },
                "provider_requests_added": 0,
                "production_promotion_allowed": False,
            }
        else:
            repo, branch = config
            reports, errors = _fetch_reports(repo, branch)
            snapshot = _build_summary(reports, errors)
            watchdogs, next_baseline = maturation_watchdogs_v4.build_watchdogs(
                reports,
                baseline=_WATCHDOG_BASELINE,
            )
            _WATCHDOG_BASELINE = next_baseline
            snapshot["maturation_watchdogs"] = watchdogs
            snapshot["state_repo"] = repo
            snapshot["state_branch"] = branch

        _CACHE = snapshot
        _CACHE_AT = now
        return dict(snapshot)
