from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from mcp_gateway import maturation_baseline_store_v4, maturation_watchdogs_v4

SCHEMA_VERSION = "1.2.0"
MODEL_VERSION = "SOCCER_MATURITY_SNAPSHOT_V4_1.2.0"
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


def _latest_report_commit_timestamp(
    client: httpx.Client,
    repo: str,
    branch: str,
    filename: str,
) -> str | None:
    path = f"soccer_edge_state/analysis/{filename}"
    response = client.get(
        f"https://api.github.com/repos/{repo}/commits",
        params={"path": path, "sha": branch, "per_page": 1},
        headers={"Accept": "application/vnd.github+json"},
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    commit = _dict(rows[0].get("commit"))
    committer = _dict(commit.get("committer"))
    author = _dict(commit.get("author"))
    value = committer.get("date") or author.get("date")
    return str(value) if value else None


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
                    payload = dict(payload)
                    reports[key] = payload
                else:
                    errors[key] = "unexpected_json_shape"
            except Exception as exc:  # dashboard telemetry must never break the product route
                errors[key] = f"{type(exc).__name__}: {str(exc)[:160]}"

        signal = reports.get("signal_summary")
        if isinstance(signal, dict):
            signal.setdefault("first_evidence_at_local", signal.get("first_generated_at_local"))
            signal.setdefault("last_evidence_at_local", signal.get("last_generated_at_local"))
            try:
                updated_at = _latest_report_commit_timestamp(
                    client, repo, branch, _REPORTS["signal_summary"]
                )
                if updated_at:
                    signal["report_updated_at_utc"] = updated_at
            except Exception as exc:
                errors["signal_summary_artifact_timestamp"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return reports, errors


def _gate(current: Any, target: Any, *, source: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"current": _int(current), "target": _int(target), "source": source}
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

    prop_rows = {str(name): _int(_dict(node).get("rows")) for name, node in props_by_family.items()}
    prop_rows_known = [value for value in prop_rows.values() if value is not None]
    prop_current = min(prop_rows_known) if prop_rows_known else _int(props_clv.get("rows"))

    gates = {
        "1x2_true_clv": _gate(one_x_two_clv.get("rows"), one_x_two_clv.get("minimum_rows"), source=_REPORTS["one_x_two"], extra={"unique_fixtures": _int(one_x_two_clv.get("unique_fixtures"))}),
        "btts_true_clv": _gate(btts_clv.get("rows"), btts_clv.get("minimum_rows"), source=_REPORTS["btts"], extra={"unique_fixtures": _int(btts_clv.get("unique_fixtures"))}),
        "team_totals_true_clv": _gate(team_totals_clv.get("rows"), team_totals_clv.get("minimum_rows"), source=_REPORTS["team_totals"], extra={"unique_fixtures": _int(team_totals_clv.get("unique_fixtures")), "minimum_unique_fixtures": _int(team_totals_clv.get("minimum_unique_fixtures"))}),
        "1h_true_clv": _gate(one_h_clv.get("rows"), one_h_clv.get("minimum_rows"), source=_REPORTS["one_h"], extra={"unique_fixtures": _int(one_h_clv.get("unique_fixtures"))}),
        "corners_formation": _gate(corners_ft.get("formation_adjusted_evaluations"), corners.get("minimum_formation_adjusted") or 100, source=_REPORTS["corners"]),
        "player_props_true_clv": _gate(prop_current, props_clv.get("minimum_rows") or 50, source=_REPORTS["player_props"], extra={"by_family_rows": prop_rows}),
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


def _tower_status(current: int | None, target: int | None, watchdog: dict[str, Any] | None = None) -> str:
    watch_status = str(_dict(watchdog).get("status") or "").upper()
    if watch_status == "WATCH":
        return "WATCH"
    if current is None:
        return "NOT_VERIFIED"
    if target is not None and target > 0 and current >= target:
        return "READY"
    if current <= 0:
        return "NO_EVIDENCE"
    return "MATURING"


def _build_maturation_control_tower(
    reports: dict[str, dict[str, Any]],
    watchdog_bundle: dict[str, Any],
    baseline_store: dict[str, Any],
) -> dict[str, Any]:
    watchdogs = _dict(watchdog_bundle.get("watchdogs"))

    def clv_family(key: str, label: str) -> dict[str, Any]:
        report = _dict(reports.get(key))
        clv = _dict(report.get("true_clv"))
        current = _int(clv.get("rows"))
        target = _int(clv.get("minimum_rows"))
        return {
            "key": key,
            "label": label,
            "evidence_kind": "TRUE_CLV",
            "current": current,
            "target": target,
            "unique_fixtures": _int(clv.get("unique_fixtures")),
            "status": _tower_status(current, target),
            "source": _REPORTS[key],
        }

    families = [
        clv_family("one_x_two", "1X2"),
        clv_family("btts", "BTTS"),
        clv_family("team_totals", "Team Totals"),
        clv_family("one_h", "1H"),
    ]

    two_h = clv_family("two_h", "2H")
    two_h_watch = _dict(watchdogs.get("two_h_market_maturation"))
    two_h["status"] = _tower_status(two_h.get("current"), two_h.get("target"), two_h_watch)
    two_h["blocker"] = two_h_watch.get("reason")
    families.append(two_h)

    corners = _dict(reports.get("corners"))
    corners_ft = _dict(corners.get("ft_corners"))
    corners_watch = _dict(watchdogs.get("corners_formation_join"))
    corners_current = _int(corners_ft.get("formation_adjusted_evaluations"))
    corners_target = _int(corners.get("minimum_formation_adjusted")) or 100
    families.append({
        "key": "corners",
        "label": "Corners",
        "evidence_kind": "FORMATION_ADJUSTED",
        "current": corners_current,
        "target": corners_target,
        "unique_fixtures": corners_current,
        "status": _tower_status(corners_current, corners_target, corners_watch),
        "blocker": corners_watch.get("reason"),
        "source": _REPORTS["corners"],
    })

    cards = _dict(reports.get("cards"))
    cards_clv = _dict(cards.get("true_clv"))
    cards_market = _dict(_dict(cards.get("market_evidence")).get("match_cards"))
    cards_watch = _dict(watchdogs.get("cards_settlement"))
    cards_current = _int(cards_clv.get("rows"))
    if cards_current is None:
        cards_current = _int(cards_market.get("priced_value_rows"))
        cards_kind = "PRICED_MARKET_ROWS"
    else:
        cards_kind = "TRUE_CLV"
    cards_target = _int(cards_clv.get("minimum_rows"))
    families.append({
        "key": "cards",
        "label": "Cards",
        "evidence_kind": cards_kind,
        "current": cards_current,
        "target": cards_target,
        "unique_fixtures": _int(cards_market.get("unique_fixtures")),
        "status": _tower_status(cards_current, cards_target, cards_watch),
        "blocker": cards_watch.get("reason"),
        "source": _REPORTS["cards"],
    })

    props = _dict(reports.get("player_props"))
    props_clv = _dict(props.get("true_clv"))
    props_by_family = _dict(props_clv.get("by_family"))
    prop_rows = [_int(_dict(node).get("rows")) for node in props_by_family.values()]
    prop_rows = [value for value in prop_rows if value is not None]
    props_current = min(prop_rows) if prop_rows else _int(props_clv.get("rows"))
    props_target = _int(props_clv.get("minimum_rows")) or 50
    props_watch = _dict(watchdogs.get("player_props_xi_confirmed"))
    families.append({
        "key": "player_props",
        "label": "Player Props",
        "evidence_kind": "TRUE_CLV_PER_FAMILY",
        "current": props_current,
        "target": props_target,
        "unique_fixtures": _int(props_clv.get("unique_fixtures")),
        "status": _tower_status(props_current, props_target, props_watch),
        "blocker": props_watch.get("reason"),
        "source": _REPORTS["player_props"],
    })

    signal_freshness = _dict(watchdogs.get("signal_close_freshness"))
    signal_age = _dict(watchdogs.get("signal_evidence_age"))
    growth = _dict(watchdogs.get("evidence_growth_48h"))
    return {
        "schema_version": "1.0.0",
        "model_version": "SOCCER_MATURATION_CONTROL_TOWER_V4_1.0.0",
        "status": watchdog_bundle.get("status") or "NOT_VERIFIED",
        "families": families,
        "monitoring": {
            "report_freshness": signal_freshness,
            "evidence_age": signal_age,
            "evidence_growth_48h": growth,
            "baseline_store": baseline_store,
        },
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "decision_weight": 0.0,
    }


def load_snapshot(*, force: bool = False) -> dict[str, Any]:
    """Read validation/maturation counters from the persisted state branch.

    State reports and the watchdog baseline are observability-only. They never
    feed production selection, pricing, thresholds, gates or provider budgets.
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
                "maturation_watchdogs": {"status": "NOT_VERIFIED", "reason": "STATE_REPO_OR_STATE_BRANCH_NOT_CONFIGURED", "watchdogs": {}, "provider_requests_added": 0, "production_promotion_allowed": False},
                "maturation_control_tower": {"status": "NOT_VERIFIED", "families": [], "provider_requests_added": 0, "production_promotion_allowed": False},
                "provider_requests_added": 0,
                "production_promotion_allowed": False,
            }
        else:
            repo, branch = config
            reports, errors = _fetch_reports(repo, branch)
            snapshot = _build_summary(reports, errors)

            baseline = _WATCHDOG_BASELINE
            if baseline is None:
                baseline, load_status = maturation_baseline_store_v4.load_baseline()
            else:
                load_status = {
                    "status": "OK",
                    "source": "MEMORY_CACHE+POSTGRES:maturation_watchdog_state",
                    "provider_requests_added": 0,
                    "production_promotion_allowed": False,
                }
            if isinstance(baseline, dict):
                baseline = dict(baseline)
                baseline.setdefault("source", "POSTGRES:maturation_watchdog_state")

            watchdogs, next_baseline = maturation_watchdogs_v4.build_watchdogs(reports, baseline=baseline)
            next_baseline = dict(next_baseline)
            next_baseline["source"] = "POSTGRES:maturation_watchdog_state"
            save_status = maturation_baseline_store_v4.save_baseline(next_baseline)
            _WATCHDOG_BASELINE = next_baseline
            baseline_status = {
                "status": "OK" if save_status.get("status") == "OK" else save_status.get("status"),
                "source": "POSTGRES:maturation_watchdog_state",
                "load": load_status,
                "save": save_status,
                "provider_requests_added": 0,
                "production_promotion_allowed": False,
            }

            snapshot["maturation_watchdogs"] = watchdogs
            snapshot["watchdog_baseline_store"] = baseline_status
            snapshot["maturation_control_tower"] = _build_maturation_control_tower(reports, watchdogs, baseline_status)
            snapshot["state_repo"] = repo
            snapshot["state_branch"] = branch

        _CACHE = snapshot
        _CACHE_AT = now
        return dict(snapshot)
