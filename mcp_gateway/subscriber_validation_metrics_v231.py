from __future__ import annotations

import os
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

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT = 0.0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _state_config() -> tuple[str, str] | None:
    repo = os.getenv("STATE_REPO", "").strip()
    branch = os.getenv("STATE_BRANCH", "").strip()
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/"):]
    if not repo or not branch:
        return None
    return repo, branch


def _metric_row(label: str, report: dict[str, Any]) -> dict[str, Any]:
    calibration = _dict(report.get("calibration_sample"))
    canonical = _dict(report.get("canonical_multiclass_oos"))
    canonical_temp = _dict(canonical.get("temperature_scaled"))
    sample = _dict(report.get("sample"))
    true_clv = _dict(report.get("true_clv"))

    brier = (
        _number(calibration.get("brier"))
        if calibration.get("brier") is not None
        else _number(canonical_temp.get("multiclass_brier"))
    )
    if brier is None:
        brier = _number(sample.get("mean_brier_legacy_actionable_signal_probability"))

    log_loss = (
        _number(calibration.get("log_loss"))
        if calibration.get("log_loss") is not None
        else _number(canonical_temp.get("multiclass_log_loss"))
    )
    if log_loss is None:
        log_loss = _number(sample.get("mean_log_loss_legacy_actionable_signal_probability"))

    n = _integer(calibration.get("n"))
    if n is None:
        n = _integer(canonical_temp.get("n"))
    if n is None:
        n = _integer(sample.get("model_settled"))

    settled = _integer(sample.get("commercial_settled"))
    if settled is None:
        settled = _integer(_dict(report.get("settlement_context")).get("settled"))

    return {
        "label": label,
        "status": report.get("status"),
        "model_version": report.get("model_version"),
        "sample_n": n,
        "brier": brier,
        "log_loss": log_loss,
        "ece": _number(calibration.get("ece")),
        "accuracy": _number(calibration.get("top1_accuracy")),
        "settled": settled,
        "true_clv_rows": _integer(true_clv.get("rows")),
        "true_clv_target": _integer(true_clv.get("minimum_rows")),
        "avg_clv_pp": _number(true_clv.get("avg_probability_clv_pp")),
        "blockers": [str(v) for v in report.get("blockers", []) if v is not None]
        if isinstance(report.get("blockers"), list)
        else [],
    }


def load_validation_metrics(*, force: bool = False) -> dict[str, Any]:
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
                "rows": [],
                "provider_requests_added": 0,
            }
            _CACHE, _CACHE_AT = result, now
            return dict(result)

        repo, branch = config
        base = f"https://raw.githubusercontent.com/{repo}/{branch}/soccer_edge_state/analysis"
        rows: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            for label, filename in _REPORTS.items():
                try:
                    response = client.get(f"{base}/{filename}", headers={"Accept": "application/json"})
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        row = _metric_row(label, payload)
                        row["source"] = filename
                        rows.append(row)
                    else:
                        errors[label] = "unexpected_json_shape"
                except Exception as exc:
                    errors[label] = f"{type(exc).__name__}: {str(exc)[:120]}"

        known_clv = [r["avg_clv_pp"] for r in rows if r.get("avg_clv_pp") is not None]
        weighted_pairs = [
            (float(r["avg_clv_pp"]), int(r["true_clv_rows"]))
            for r in rows
            if r.get("avg_clv_pp") is not None and r.get("true_clv_rows") not in (None, 0)
        ]
        weighted_avg_clv = None
        if weighted_pairs:
            total_rows = sum(n for _, n in weighted_pairs)
            weighted_avg_clv = sum(value * n for value, n in weighted_pairs) / total_rows if total_rows else None

        result = {
            "status": "OK" if rows and not errors else ("PARTIAL" if rows else "UNAVAILABLE"),
            "rows": rows,
            "weighted_avg_clv_pp": weighted_avg_clv,
            "families_with_clv": len(known_clv),
            "errors": errors,
            "state_repo": repo,
            "state_branch": branch,
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "production_promotion_allowed": False,
        }
        _CACHE, _CACHE_AT = result, now
        return dict(result)
