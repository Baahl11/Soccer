from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.6.0"
MODEL_VERSION = "SOCCER_PRODUCT_DASHBOARD_V4_1.6.0"

DISPLAY_VIEWS = (
    ("team_totals", "Team Totals"),
    ("first_half", "1H"),
    ("second_half", "2H"),
    ("corners", "Corners"),
    ("player_props", "Player Props"),
)

WATCHDOG_LABELS = {
    "signal_close_freshness": "Signal / report freshness",
    "signal_evidence_age": "Signal evidence age",
    "two_h_market_maturation": "2H strict-close evidence",
    "corners_formation_join": "Corners formation joins",
    "cards_settlement": "Cards settlement readiness",
    "player_props_xi_confirmed": "Player Props XI alignment",
    "evidence_growth_48h": "Evidence growth · 48h",
    "provider_requests_without_evidence_growth": "Provider efficiency · 48h",
}

LIVE_CODES = {"1H", "2H", "HT", "ET", "BT", "P", "LIVE", "INT"}
FINISHED_CODES = {"FT", "AET", "PEN"}
POSTPONED_CODES = {"PST", "CANC", "ABD", "SUSP"}


def _esc(value: Any) -> str:
    if value is None:
        return "N/V"
    return html.escape(str(value), quote=True)


def _num(value: Any, fallback: str = "N/V") -> str:
    if value is None:
        return fallback
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _esc(value)


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _status_class(value: Any) -> str:
    text = str(value or "").upper()
    if text == "OK" or any(token in text for token in ("HEALTHY", "LIVE", "READY", "PASS", "TICK_OBSERVED", "CALIBRATED")):
        return "ok"
    if any(token in text for token in ("WATCH", "WAIT", "HOLD", "LOCKED", "RESEARCH", "VALIDATION", "PENDING", "N/V", "NOT_VERIFIED", "STALE", "BLOCKED", "MATURING", "NO_EVIDENCE", "COOLDOWN")):
        return "warn"
    if any(token in text for token in ("ERROR", "FAILED", "DEGRADED", "OFFLINE")):
        return "bad"
    return "neutral"


def _friendly_execution(value: Any) -> str:
    text = str(value or "NOT_VERIFIED").upper()
    mapping = {
        "READY": "Ready",
        "BET": "Ready",
        "WAIT_PRICE": "Price watch",
        "WAIT_FRESH_QUOTE": "Fresh price",
        "WAIT_XI": "Lineup watch",
        "WAIT_GK": "GK watch",
        "WAIT_AVAILABILITY": "Availability",
        "RESEARCH_ONLY": "Research",
        "DEFER_COOLDOWN": "Cooldown",
        "NOT_VERIFIED": "Not verified",
    }
    return mapping.get(text, text.replace("_", " ").title())


def _edge_value(row: dict[str, Any]) -> float | None:
    for key in ("calibrated_edge_pp", "prob_edge_pp", "edge_pp"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _nested(mapping: Any, *keys: str) -> Any:
    node = mapping
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _team_name(row: dict[str, Any], side: str) -> str:
    candidates = (
        row.get(side),
        row.get(f"{side}_team"),
        row.get(f"{side}_name"),
        _nested(row, "teams", side, "name"),
        _nested(row, side, "name"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return side.title()


def _team_id(row: dict[str, Any], side: str) -> int | None:
    candidates = (
        row.get(f"{side}_team_id"),
        row.get(f"{side}_id"),
        _nested(row, "teams", side, "id"),
        _nested(row, side, "id"),
    )
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _team_logo(row: dict[str, Any], side: str) -> str | None:
    candidates = (
        row.get(f"{side}_logo"),
        row.get(f"{side}_team_logo"),
        _nested(row, "teams", side, "logo"),
        _nested(row, side, "logo"),
    )
    for value in candidates:
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    team_id = _team_id(row, side)
    if team_id is not None and team_id > 0:
        return f"https://media.api-sports.io/football/teams/{team_id}.png"
    return None


def _initials(name: str) -> str:
    parts = [part for part in name.replace("-", " ").split() if part]
    if not parts:
        return "FC"
    return "".join(part[0] for part in parts[:2]).upper()


def _team_badge(row: dict[str, Any], side: str) -> str:
    name = _team_name(row, side)
    logo = _team_logo(row, side)
    fallback = _esc(_initials(name))
    image = ""
    if logo:
        image = (
            f"<img src='{_esc(logo)}' alt='' loading='lazy' "
            "onerror=\"this.style.display='none'\">"
        )
    return (
        "<span class='team-badge'>"
        f"<span class='team-initials'>{fallback}</span>{image}"
        "</span>"
    )


def _status_code(row: dict[str, Any]) -> str:
    raw = row.get("status")
    if isinstance(raw, dict):
        raw = raw.get("short") or raw.get("status")
    if not raw:
        raw = row.get("fixture_status") or row.get("stage") or row.get("event_stage")
    return str(raw or "NS").upper()


def _elapsed(row: dict[str, Any]) -> int | None:
    values = (
        row.get("elapsed"),
        row.get("fixture_elapsed"),
        _nested(row, "fixture", "status", "elapsed"),
        _nested(row, "status", "elapsed"),
    )
    for value in values:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            continue
    return None


def _score_pair(row: dict[str, Any]) -> tuple[Any, Any]:
    direct_pairs = (
        (row.get("home_score"), row.get("away_score")),
        (_nested(row, "goals", "home"), _nested(row, "goals", "away")),
        (_nested(row, "score", "current", "home"), _nested(row, "score", "current", "away")),
        (_nested(row, "score", "fulltime", "home"), _nested(row, "score", "fulltime", "away")),
    )
    for home, away in direct_pairs:
        if home is not None or away is not None:
            return home, away
    return None, None


def _match_state(row: dict[str, Any]) -> tuple[str, str]:
    code = _status_code(row)
    minute = _elapsed(row)
    if code in LIVE_CODES:
        if code in {"1H", "2H", "ET", "LIVE"} and minute is not None:
            return f"{minute}'", "live"
        return code, "live"
    if code in FINISHED_CODES:
        return code, "finished"
    if code in POSTPONED_CODES:
        return code, "bad"
    if code in {"NS", "TBD", "PREMATCH", "PRE"}:
        kickoff = row.get("kickoff") or row.get("fixture_date") or row.get("date")
        return (str(kickoff) if kickoff else "PRE"), "scheduled"
    return code, "scheduled"


def _fixture_key(row: dict[str, Any], index: int) -> str:
    fixture_id = row.get("fixture_id") or row.get("match_id") or row.get("id")
    if fixture_id is not None:
        return f"id:{fixture_id}"
    return f"name:{_team_name(row, 'home')}|{_team_name(row, 'away')}|{row.get('league')}|{index}"


def _group_slate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        key = _fixture_key(row, index)
        if key not in groups:
            groups[key] = {"base": row, "markets": []}
            order.append(key)
        groups[key]["markets"].append(row)
        base = groups[key]["base"]
        if _status_code(row) in LIVE_CODES | FINISHED_CODES or _score_pair(row) != (None, None):
            groups[key]["base"] = row
        elif _status_code(base) in {"NS", "PRE", "PREMATCH"} and _status_code(row) not in {"NS", "PRE", "PREMATCH"}:
            groups[key]["base"] = row
    return [groups[key] for key in order]


def _market_line(row: dict[str, Any]) -> str:
    market = row.get("market") or row.get("recommended_market") or row.get("market_family") or "Market"
    selection = row.get("selection") or row.get("pick") or row.get("side")
    line = row.get("line")
    price = row.get("price")
    if price is None and isinstance(row.get("soft_price"), dict):
        price = row["soft_price"].get("american_odds") or row["soft_price"].get("decimal_odds")
    parts = [str(market)]
    detail = " ".join(str(value) for value in (selection, line) if value not in (None, ""))
    if detail:
        parts.append(detail)
    if price not in (None, ""):
        parts.append(f"@ {price}")
    return " · ".join(parts)


def _fixture_card(group: dict[str, Any]) -> str:
    row = group["base"]
    markets = [item for item in group.get("markets", []) if isinstance(item, dict)]
    home = _team_name(row, "home")
    away = _team_name(row, "away")
    status, tone = _match_state(row)
    home_score, away_score = _score_pair(row)
    if home_score is not None or away_score is not None:
        score_html = f"<div class='score mono'>{_esc(home_score)}<span>–</span>{_esc(away_score)}</div>"
    else:
        score_html = "<div class='score score-empty'>vs</div>"
    league = row.get("league") or row.get("league_name") or "League N/V"
    market_html = "".join(
        f"<div class='market-row'><span>{_esc(_market_line(item))}</span>"
        f"<span class='mini-pill {_status_class(item.get('execution_status'))}'>{_esc(_friendly_execution(item.get('execution_status')))}</span></div>"
        for item in markets[:3]
    )
    if len(markets) > 3:
        market_html += f"<div class='more-markets'>+{len(markets) - 3} more verified rows</div>"

    return (
        "<article class='fixture-card'>"
        f"<div class='fixture-head'><span class='league'>{_esc(league)}</span><span class='match-status {tone}'>{_esc(status)}</span></div>"
        "<div class='teams'>"
        f"<div class='team home'>{_team_badge(row, 'home')}<strong>{_esc(home)}</strong></div>"
        f"{score_html}"
        f"<div class='team away'>{_team_badge(row, 'away')}<strong>{_esc(away)}</strong></div>"
        "</div>"
        f"<div class='markets'>{market_html or '<div class=\"muted\">No verified market rows.</div>'}</div>"
        "</article>"
    )


def _slate_section(view: dict[str, Any]) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    groups = _group_slate(rows)
    if not groups:
        body = "<div class='empty'>No active fixtures in the latest persisted tick.</div>"
    else:
        body = "<div class='fixture-grid'>" + "".join(_fixture_card(group) for group in groups) + "</div>"
    total_rows = view.get("total") if isinstance(view, dict) else len(rows)
    return (
        "<section class='panel slate-panel' id='todays_slate'>"
        "<div class='panel-head'><div><div class='eyebrow'>FULL SLATE · PERSISTED DATA</div>"
        "<h2>Today&#x27;s Slate</h2>"
        "<p class='panel-copy'>All active fixtures available in the latest persisted tick. Live/partial states and scores appear only when present in the stored payload.</p>"
        f"</div><span class='count'>{_num(len(groups), '0')} fixtures · {_num(total_rows, '0')} rows</span></div>{body}</section>"
    )


def _signal_card(row: dict[str, Any], *, accent: str = "cyan") -> str:
    edge = _edge_value(row)
    edge_text = "N/V" if edge is None else f"{edge:+.1f} pp"
    price = row.get("price")
    execution = row.get("execution_status")
    stage = row.get("stage") or row.get("event_stage")
    return (
        f"<article class='signal-card {accent}'>"
        "<div class='signal-card-top'>"
        f"<span class='league'>{_esc(row.get('league'))}</span>"
        f"<span class='pill {_status_class(execution)}'>{_esc(_friendly_execution(execution))}</span>"
        "</div>"
        "<div class='signal-fixture'>"
        f"{_team_badge(row, 'home')}<strong>{_esc(_team_name(row, 'home'))}</strong>"
        "<span class='muted'>vs</span>"
        f"{_team_badge(row, 'away')}<strong>{_esc(_team_name(row, 'away'))}</strong>"
        "</div>"
        f"<div class='market-name'>{_esc(row.get('market') or row.get('market_family'))}</div>"
        f"<div class='selection'>{_esc(row.get('selection'))} <strong>{_esc(row.get('line'))}</strong></div>"
        "<div class='signal-stats'>"
        f"<div><span>Price</span><strong class='mono'>{_esc(price)}</strong></div>"
        f"<div><span>Edge</span><strong class='mono positive'>{_esc(edge_text)}</strong></div>"
        f"<div><span>Signal</span><strong>{_esc(row.get('model_signal'))}</strong></div>"
        f"<div><span>Stage</span><strong>{_esc(stage)}</strong></div>"
        "</div>"
        "</article>"
    )


def _signal_grid(view: dict[str, Any], *, accent: str = "cyan", empty: str) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not rows:
        return f"<div class='empty'>{_esc(empty)}</div>"
    return "<div class='signal-grid'>" + "".join(_signal_card(row, accent=accent) for row in rows[:8]) + "</div>"


def _row_html(row: dict[str, Any]) -> str:
    fixture = f"{_esc(_team_name(row, 'home'))} vs {_esc(_team_name(row, 'away'))}"
    edge = _edge_value(row)
    edge_html = "N/V" if edge is None else f"{edge:+.1f} pp"
    return (
        "<tr>"
        f"<td><strong>{fixture}</strong><div class='muted'>{_esc(row.get('league'))}</div></td>"
        f"<td>{_esc(row.get('market') or row.get('market_family'))}<div class='muted'>{_esc(row.get('selection'))} {_esc(row.get('line'))}</div></td>"
        f"<td class='mono'>{_esc(row.get('price'))}</td>"
        f"<td><span class='signal'>{_esc(row.get('model_signal'))}</span></td>"
        f"<td class='edge mono'>{_esc(edge_html)}</td>"
        f"<td><span class='pill {_status_class(row.get('execution_status'))}'>{_esc(_friendly_execution(row.get('execution_status')))}</span></td>"
        "</tr>"
    )


def _view_section(key: str, title: str, view: dict[str, Any]) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    total = int(view.get("total") or 0) if isinstance(view, dict) else 0
    body = "<div class='empty'>No verified rows in the latest persisted tick.</div>"
    if rows:
        body = (
            "<div class='table-wrap'><table><thead><tr><th>Fixture</th><th>Market</th><th>Price</th><th>Signal</th><th>Edge</th><th>Status</th></tr></thead><tbody>"
            + "".join(_row_html(row) for row in rows)
            + "</tbody></table></div>"
        )
    return (
        f"<section class='panel feed-panel' id='{_esc(key)}'>"
        f"<div class='panel-head'><div><div class='eyebrow'>MARKET VIEW</div><h2>{_esc(title)}</h2></div><span class='count'>{total} total</span></div>{body}</section>"
    )


def _metric(label: str, value: Any, sub: Any = None, *, tone: str = "") -> str:
    sub_html = f"<div class='metric-sub'>{_esc(sub)}</div>" if sub is not None else ""
    return f"<div class='metric-card {tone}'><div class='metric-label'>{_esc(label)}</div><div class='metric-value mono'>{_num(value)}</div>{sub_html}</div>"


def _gate_html(gate: dict[str, Any]) -> str:
    current = gate.get("current")
    target = gate.get("target")
    if current is None or target in (None, 0):
        value, width, cls = f"N/V / {_num(target)}", 0, "unknown"
    else:
        width = max(0, min(100, int(float(current) / float(target) * 100)))
        value = f"{_num(current)} / {_num(target)}"
        cls = "complete" if float(current) >= float(target) else "progress"
    return (
        "<div class='gate'>"
        f"<div class='gate-top'><span>{_esc(gate.get('label'))}</span><strong class='mono'>{_esc(value)} {_esc(gate.get('unit'))}</strong></div>"
        f"<div class='track'><div class='fill {cls}' style='width:{width}%'></div></div></div>"
    )


def _maturation_family_html(node: dict[str, Any]) -> str:
    current = node.get("current")
    target = node.get("target")
    status = node.get("status") or "NOT_VERIFIED"
    if current is None:
        ratio = f"N/V / {_num(target)}"
        width = 0
    elif target not in (None, 0):
        ratio = f"{_num(current)} / {_num(target)}"
        width = max(0, min(100, int(float(current) / float(target) * 100)))
    else:
        ratio = _num(current)
        width = 0
    return (
        "<div class='maturity-card'>"
        f"<div class='maturity-top'><div><strong>{_esc(node.get('label'))}</strong><div class='maturity-kind'>{_esc(node.get('evidence_kind'))}</div></div><span class='pill {_status_class(status)}'>{_esc(status)}</span></div>"
        f"<div class='maturity-value mono'>{_esc(ratio)}</div>"
        f"<div class='track'><div class='fill' style='width:{width}%'></div></div>"
        f"<div class='maturity-meta'>{_esc(node.get('blocker') or 'No explicit blocker')}</div>"
        "</div>"
    )


def _watchdog_html(key: str, node: dict[str, Any]) -> str:
    status = node.get("status") or "NOT_VERIFIED"
    return (
        "<div class='watchdog-card'>"
        f"<div class='watchdog-top'><strong>{_esc(WATCHDOG_LABELS.get(key, key.replace('_', ' ').title()))}</strong><span class='pill {_status_class(status)}'>{_esc(status)}</span></div>"
        f"<div class='watchdog-reason'>{_esc(node.get('reason') or 'WATCHDOG_REASON_UNAVAILABLE')}</div>"
        f"<div class='watchdog-source'>{_esc(node.get('source'))}</div>"
        "</div>"
    )


def _phase_html(phase: dict[str, Any]) -> str:
    status = phase.get("status") or "NOT_VERIFIED"
    return (
        "<div class='phase-card'>"
        f"<div class='phase-num'>PHASE {_esc(phase.get('phase'))}</div><div class='phase-name'>{_esc(phase.get('name'))}</div>"
        f"<span class='pill {_status_class(status)}'>{_esc(status)}</span></div>"
    )


def _errors_html(errors: dict[str, Any]) -> str:
    count = int(errors.get("count") or 0) if isinstance(errors, dict) else 0
    rows = errors.get("rows") if isinstance(errors, dict) and isinstance(errors.get("rows"), list) else []
    if count == 0:
        return "<div class='empty-success'><span class='check'>✓</span><div><strong>No current pipeline errors</strong><div class='muted'>Latest persisted tick contains no PIPELINE_ERROR rows.</div></div></div>"
    return "".join(
        "<div class='error-row'>"
        f"<div><strong>{_esc(row.get('home'))} vs {_esc(row.get('away'))}</strong><div class='muted'>{_esc(row.get('league'))}</div></div>"
        f"<div class='error-reason'>{_esc(row.get('reason'))}</div></div>"
        for row in rows if isinstance(row, dict)
    )


def render_dashboard(product_payload: dict[str, Any]) -> str:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    tower = views.get("control_tower") if isinstance(views.get("control_tower"), dict) else {}
    health = tower.get("system_health") if isinstance(tower.get("system_health"), dict) else {}
    pipeline = tower.get("pipeline") if isinstance(tower.get("pipeline"), dict) else {}
    errors = tower.get("errors") if isinstance(tower.get("errors"), dict) else {}
    gates = tower.get("validation_gates") if isinstance(tower.get("validation_gates"), list) else []
    phases = tower.get("phases") if isinstance(tower.get("phases"), list) else []
    maturity = tower.get("maturity_snapshot") if isinstance(tower.get("maturity_snapshot"), dict) else {}
    watchdog_bundle = maturity.get("maturation_watchdogs") if isinstance(maturity.get("maturation_watchdogs"), dict) else {}
    watchdogs = watchdog_bundle.get("watchdogs") if isinstance(watchdog_bundle.get("watchdogs"), dict) else {}
    maturation_tower = maturity.get("maturation_control_tower") if isinstance(maturity.get("maturation_control_tower"), dict) else {}
    maturation_families = maturation_tower.get("families") if isinstance(maturation_tower.get("families"), list) else []

    generated = product_payload.get("generated_at_utc")
    pipeline_version = product_payload.get("pipeline_version")
    status = tower.get("status") or product_payload.get("status")
    slate = views.get("todays_slate") if isinstance(views.get("todays_slate"), dict) else {}
    strong = views.get("strong_sport_signals") if isinstance(views.get("strong_sport_signals"), dict) else {}
    values = views.get("value_plays") if isinstance(views.get("value_plays"), dict) else {}
    waiting_price = views.get("waiting_for_price") if isinstance(views.get("waiting_for_price"), dict) else {}
    waiting_xi = views.get("waiting_for_xi") if isinstance(views.get("waiting_for_xi"), dict) else {}

    slate_rows = slate.get("rows") if isinstance(slate.get("rows"), list) else []
    fixture_groups = _group_slate([row for row in slate_rows if isinstance(row, dict)])
    live_fixture_count = sum(1 for group in fixture_groups if _match_state(group["base"])[1] == "live")

    meta_json = html.escape(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": pipeline_version,
        "generated_at_utc": generated,
        "status": status,
        "production_promotion_allowed": False,
    }, ensure_ascii=False), quote=True)

    feed_sections = "".join(
        _view_section(key, title, views.get(key) if isinstance(views.get(key), dict) else {})
        for key, title in DISPLAY_VIEWS
    )
    gate_rows = "".join(_gate_html(gate) for gate in gates if isinstance(gate, dict))
    phase_rows = "".join(_phase_html(phase) for phase in phases if isinstance(phase, dict))
    watchdog_rows = "".join(_watchdog_html(key, node) for key, node in watchdogs.items() if isinstance(node, dict)) or "<div class='empty'>No watchdog telemetry in the current snapshot.</div>"
    maturation_rows = "".join(_maturation_family_html(node) for node in maturation_families if isinstance(node, dict)) or "<div class='empty'>No maturation family telemetry in the current snapshot.</div>"

    state_note = f"State {_esc(maturity.get('status'))} · {_num(maturity.get('reports_loaded'))}/{_num(maturity.get('reports_expected'))} reports"
    watchdog_note = f"{_num(watchdog_bundle.get('ok_count'), '0')} OK · {_num(watchdog_bundle.get('watch_count'), '0')} watch · {_num(watchdog_bundle.get('not_verified_count'), '0')} N/V"

    operator_metrics = "".join((
        _metric("Fixtures scanned", pipeline.get("fixtures_scanned"), "latest tick"),
        _metric("Due", pipeline.get("due"), "refresh candidates"),
        _metric("Events", pipeline.get("events"), "persisted"),
        _metric("Deep dives", pipeline.get("deep_dives"), "processed"),
        _metric("API calls", pipeline.get("api_calls"), f"cap {_num(pipeline.get('api_call_cap'))}"),
        _metric("API remaining", health.get("api_football_remaining"), health.get("daily_budget_mode")),
    ))

    slate_metrics = "".join((
        _metric("Full slate", len(fixture_groups), f"{_num(slate.get('total'), '0')} market rows"),
        _metric("Live now", live_fixture_count, "stored status"),
        _metric("Strong signals", strong.get("total"), "verified"),
        _metric("Value plays", values.get("total"), "calibrated"),
        _metric("Price watch", waiting_price.get("total"), "awaiting quote"),
        _metric("XI watch", waiting_xi.get("total"), "awaiting lineup"),
    ))

    health_cards = "".join(
        f"<div class='health-card'><span>{_esc(label)}</span><strong class='{_status_class(value)}'>{_esc(value)}</strong></div>"
        for label, value in (
            ("Runtime", health.get("runtime")),
            ("Postgres", health.get("postgres")),
            ("Scheduler", health.get("scheduler")),
            ("API-Football", health.get("api_football")),
            ("Galaxy", health.get("galaxy")),
        )
    )

    scheduler_cards = "".join((
        _metric("Scheduler mode", pipeline.get("scheduler_mode"), pipeline.get("scheduler_schema_version")),
        _metric("Unseen processed", pipeline.get("scheduler_unseen_processed"), "coverage"),
        _metric("Actionable processed", pipeline.get("scheduler_actionable_processed"), "priority"),
        _metric("Starvation", pipeline.get("scheduler_starvation_count"), "fixtures"),
        _metric("TT close candidates", pipeline.get("team_totals_maturation_candidates"), f"{_num(pipeline.get('team_totals_later_quote_refreshes'), '0')} later quotes"),
        _metric("Primary close candidates", pipeline.get("primary_clv_maturation_candidates"), f"{_num(pipeline.get('primary_clv_maturation_refreshed'), '0')} refreshed"),
    ))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Soccer Edge · Control Tower</title>
<style>
:root {{
  --bg:#071017; --panel:#0c171f; --panel2:#101e28; --line:#20323d; --text:#f1f5f9;
  --muted:#8ca0ad; --cyan:#32d6d2; --green:#66d19e; --amber:#f7bd68; --red:#ff7f85;
}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:radial-gradient(circle at 15% -10%,#14313b 0,transparent 38%),var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
a{{color:inherit}} .shell{{max-width:1500px;margin:auto;padding:20px 24px 72px}} .mono{{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace}}
.topbar{{position:sticky;top:0;z-index:20;margin:-20px -24px 22px;padding:14px 24px;background:rgba(7,16,23,.9);backdrop-filter:blur(18px);border-bottom:1px solid rgba(255,255,255,.07);display:flex;gap:18px;align-items:center;justify-content:space-between}}
.brand{{display:flex;align-items:center;gap:12px}} .brand-mark{{width:34px;height:34px;border:1px solid #2f6870;border-radius:11px;display:grid;place-items:center;background:#0d2830;box-shadow:0 0 28px rgba(50,214,210,.12)}} .brand h1{{font-size:15px;margin:0;letter-spacing:.08em;text-transform:uppercase}} .brand small{{display:block;color:var(--muted);font-size:11px}}
.status-cluster{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}} .live-dot{{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(102,209,158,.1)}} .refresh-state{{font-size:12px;color:var(--muted);padding:7px 10px;border:1px solid var(--line);border-radius:999px}}
.hero{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(280px,.6fr);gap:16px;margin-bottom:16px}} .hero-main,.hero-side{{background:linear-gradient(145deg,rgba(16,30,40,.96),rgba(10,20,28,.96));border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 18px 50px rgba(0,0,0,.18)}} .eyebrow{{font-size:11px;letter-spacing:.13em;color:var(--cyan);font-weight:800;text-transform:uppercase}} .hero h2{{font-size:30px;line-height:1.08;margin:10px 0 8px}} .hero p{{color:var(--muted);max-width:760px;margin:0}} .hero-tags{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}
.pill,.mini-pill,.match-status,.count{{display:inline-flex;align-items:center;border-radius:999px;border:1px solid var(--line);padding:5px 9px;font-size:11px;font-weight:750;white-space:nowrap}} .pill.ok,.mini-pill.ok{{background:rgba(102,209,158,.10);border-color:rgba(102,209,158,.28);color:#9ce5bd}} .pill.warn,.mini-pill.warn{{background:rgba(247,189,104,.10);border-color:rgba(247,189,104,.28);color:#ffd28e}} .pill.bad,.mini-pill.bad{{background:rgba(255,127,133,.10);border-color:rgba(255,127,133,.28);color:#ffa5aa}}
.hero-side{{display:grid;align-content:center;gap:12px}} .hero-state{{font-size:24px;font-weight:850}} .hero-meta{{color:var(--muted);font-size:12px}}
.metric-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:0 0 16px}} .metric-card{{background:rgba(12,23,31,.9);border:1px solid var(--line);border-radius:14px;padding:14px;min-width:0}} .metric-label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em}} .metric-value{{font-size:21px;font-weight:850;margin-top:6px;overflow:hidden;text-overflow:ellipsis}} .metric-sub{{color:var(--muted);font-size:11px;margin-top:4px;overflow:hidden;text-overflow:ellipsis}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px;align-items:start}} .main{{display:grid;gap:16px}} .rail{{display:grid;gap:16px;position:sticky;top:84px}} .panel{{background:rgba(12,23,31,.93);border:1px solid var(--line);border-radius:18px;padding:18px;overflow:hidden}} .panel-head{{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:14px}} .panel-head h2{{font-size:18px;margin:4px 0 0}} .panel-copy{{color:var(--muted);font-size:12px;max-width:760px;margin:7px 0 0}} .count{{color:var(--muted);font-weight:650}}
.fixture-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .fixture-card{{background:linear-gradient(155deg,#101f29,#0b151c);border:1px solid #203743;border-radius:16px;padding:14px;min-width:0}} .fixture-head{{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:14px}} .league{{color:var(--muted);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .match-status.live{{color:#9ce5bd;background:rgba(102,209,158,.10);border-color:rgba(102,209,158,.3)}} .match-status.finished{{color:#b7c4cc}} .match-status.bad{{color:#ffa5aa;background:rgba(255,127,133,.08)}} .match-status.scheduled{{color:#98dce0}}
.teams{{display:grid;grid-template-columns:minmax(0,1fr) 66px minmax(0,1fr);gap:10px;align-items:center}} .team{{display:flex;align-items:center;gap:9px;min-width:0}} .team.away{{justify-content:flex-end;text-align:right}} .team strong{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}} .team-badge{{position:relative;width:36px;height:36px;min-width:36px;border-radius:50%;display:grid;place-items:center;background:#172833;border:1px solid #29404b;overflow:hidden}} .team-badge img{{position:absolute;inset:4px;width:28px;height:28px;object-fit:contain}} .team-initials{{font-size:10px;font-weight:850;color:#9ab0bd}} .score{{font-size:22px;font-weight:900;text-align:center;letter-spacing:.02em}} .score span{{color:var(--muted);padding:0 4px}} .score-empty{{font-size:11px;color:var(--muted);text-transform:uppercase}}
.markets{{margin-top:13px;border-top:1px solid rgba(255,255,255,.06);padding-top:10px;display:grid;gap:7px}} .market-row{{display:flex;justify-content:space-between;align-items:center;gap:10px;font-size:11px;color:#c8d4db}} .market-row>span:first-child{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .more-markets{{font-size:11px;color:var(--muted)}}
.signal-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .signal-card{{border:1px solid var(--line);border-radius:15px;padding:14px;background:#0e1b24}} .signal-card-top{{display:flex;justify-content:space-between;gap:8px}} .signal-fixture{{display:flex;gap:7px;align-items:center;margin:12px 0 8px;min-width:0}} .signal-fixture .team-badge{{width:28px;height:28px;min-width:28px}} .signal-fixture .team-badge img{{width:22px;height:22px;inset:3px}} .signal-fixture strong{{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .market-name{{color:var(--muted);font-size:11px}} .selection{{font-size:15px;margin-top:3px}} .signal-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:12px}} .signal-stats div{{background:#0a151c;border-radius:9px;padding:8px}} .signal-stats span{{display:block;color:var(--muted);font-size:9px;text-transform:uppercase}} .signal-stats strong{{font-size:11px}} .positive{{color:#91e8bc}}
.table-wrap{{overflow:auto;border:1px solid rgba(255,255,255,.05);border-radius:12px}} table{{width:100%;border-collapse:collapse;min-width:720px}} th,td{{text-align:left;padding:11px;border-bottom:1px solid rgba(255,255,255,.055);font-size:12px}} th{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.07em;background:#0a151c}} .muted{{color:var(--muted);font-size:11px}} .edge{{color:#95e8bd}}
.health-grid{{display:grid;gap:8px}} .health-card{{display:flex;justify-content:space-between;gap:8px;padding:9px 10px;background:#0a151c;border-radius:10px}} .health-card span{{color:var(--muted)}} .health-card strong.ok{{color:#9ce5bd}} .health-card strong.warn{{color:#ffd28e}} .health-card strong.bad{{color:#ffa5aa}}
.gates{{display:grid;gap:13px}} .gate-top{{display:flex;justify-content:space-between;gap:12px;font-size:11px;margin-bottom:6px}} .gate-top span{{color:#cbd7de}} .track{{height:6px;border-radius:999px;background:#152630;overflow:hidden}} .fill{{height:100%;background:linear-gradient(90deg,#27757a,var(--cyan));border-radius:999px}} .fill.complete{{background:linear-gradient(90deg,#2a7959,var(--green))}} .fill.unknown{{width:0!important}}
.watchdog-grid,.maturity-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}} .watchdog-card,.maturity-card{{border:1px solid var(--line);border-radius:12px;padding:12px;background:#0b171e}} .watchdog-top,.maturity-top{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}} .watchdog-reason,.maturity-meta{{color:#c6d2d9;font-size:11px;margin-top:9px}} .watchdog-source,.maturity-kind{{color:var(--muted);font-size:10px;margin-top:4px}} .maturity-value{{font-size:18px;font-weight:850;margin:12px 0 8px}}
.phase-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}} .phase-card{{background:#0a151c;border:1px solid var(--line);border-radius:11px;padding:10px}} .phase-num{{font-size:9px;color:var(--muted);letter-spacing:.09em}} .phase-name{{font-weight:800;margin:3px 0 8px}} .empty,.empty-success{{padding:18px;border:1px dashed #29404b;border-radius:12px;color:var(--muted)}} .empty-success{{display:flex;gap:10px;align-items:center}} .check{{font-size:22px;color:var(--green)}} .error-row{{padding:12px;border:1px solid rgba(255,127,133,.25);border-radius:10px;margin-top:8px}} .error-reason{{color:#ffa5aa;margin-top:5px}}
.nav-pills{{display:flex;gap:7px;overflow:auto;padding-bottom:2px;margin:0 0 16px}} .nav-pills a{{text-decoration:none;color:#aebdc6;padding:7px 10px;border:1px solid var(--line);border-radius:999px;white-space:nowrap;font-size:11px}} .nav-pills a:hover{{color:white;border-color:#3a6470}} .research-note{{font-size:11px;color:var(--muted);margin-top:10px}}
@media(max-width:1100px){{.layout{{grid-template-columns:1fr}}.rail{{position:static;grid-template-columns:repeat(2,minmax(0,1fr))}}.metric-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}
@media(max-width:760px){{.shell{{padding:14px 12px 54px}}.topbar{{margin:-14px -12px 16px;padding:12px}}.brand small{{display:none}}.hero{{grid-template-columns:1fr}}.hero h2{{font-size:24px}}.metric-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.fixture-grid,.signal-grid,.watchdog-grid,.maturity-grid,.rail{{grid-template-columns:1fr}}.phase-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.teams{{grid-template-columns:minmax(0,1fr) 50px minmax(0,1fr)}}.team{{display:grid;justify-items:start;gap:5px}}.team.away{{justify-items:end}}.team strong{{font-size:11px;max-width:115px}}.signal-stats{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="shell" data-dashboard-meta="{meta_json}" data-generated-at="{_esc(generated)}">
<header class="topbar">
  <div class="brand"><span class="brand-mark">◈</span><div><h1>Soccer Edge</h1><small>Control Tower · evidence-first</small></div></div>
  <div class="status-cluster"><span class="live-dot"></span><span class="pill {_status_class(status)}">{_esc(status)}</span><span id="refreshState" class="refresh-state">Snapshot {_esc(generated)}</span></div>
</header>

<section class="hero">
  <div class="hero-main">
    <div class="eyebrow">DASHBOARD CONTROL TOWER V1</div>
    <h2>Live slate, real pipeline KPIs, zero invented state.</h2>
    <p>Read-only operational view. The page renders the latest persisted tick and keeps it visible if refresh checks fail. Logos, scores and partial-match states are shown only from grounded team IDs or stored fixture data.</p>
    <div class="hero-tags">
      <span class="pill ok">Research-first dashboard</span>
      <span class="pill neutral">Read-only operational view</span>
      <span class="pill warn">Production-valid markets: {_num(tower.get('production_valid_market_count'), '0')}</span>
    </div>
  </div>
  <div class="hero-side">
    <div class="eyebrow">LATEST PERSISTED TICK</div>
    <div class="hero-state">{_esc(status)}</div>
    <div class="hero-meta mono">{_esc(pipeline_version)}</div>
    <div class="hero-meta">{_esc(generated)}</div>
    <div class="hero-meta">Provider requests added by dashboard: <strong>0</strong></div>
  </div>
</section>

<div class="metric-grid">{slate_metrics}</div>
<nav class="nav-pills">
  <a href="#todays_slate">Full slate</a><a href="#strong">Strong signals</a><a href="#control">Control Tower</a>
  <a href="#maturity">Model Maturity</a><a href="#watchdogs">Maturation Watchdogs</a><a href="#phases">Phases 14–24</a>
</nav>

<div class="layout">
<main class="main">
  {_slate_section(slate)}
  <section class="panel" id="strong">
    <div class="panel-head"><div><div class="eyebrow">VERIFIED SIGNALS</div><h2>Strong Sport Signals</h2></div><span class="count">{_num(strong.get('total'), '0')} total</span></div>
    {_signal_grid(strong, empty="No STRONG / VERY_STRONG rows in this tick.")}
  </section>
  <section class="panel">
    <div class="panel-head"><div><div class="eyebrow">VALUE / EXECUTION</div><h2>Market Watch</h2></div></div>
    <div class="metric-grid">{''.join((_metric('Value plays', values.get('total')), _metric('Waiting price', waiting_price.get('total')), _metric('Waiting XI', waiting_xi.get('total'))))}</div>
  </section>
  {feed_sections}
  <section class="panel" id="control">
    <div class="panel-head"><div><div class="eyebrow">OPERATIONS</div><h2>Control Tower</h2></div><span class="count">{_esc(health.get('last_tick'))}</span></div>
    <div class="metric-grid">{operator_metrics}</div>
    <div class="metric-grid">{scheduler_cards}</div>
  </section>
  <section class="panel" id="maturity">
    <div class="panel-head"><div><div class="eyebrow">VALIDATION</div><h2>Model Maturity</h2><p class="panel-copy">{state_note}</p></div></div>
    <div class="maturity-grid">{maturation_rows}</div>
    <div class="gates" style="margin-top:16px">{gate_rows or "<div class='empty'>No validation gate telemetry.</div>"}</div>
  </section>
  <section class="panel" id="watchdogs">
    <div class="panel-head"><div><div class="eyebrow">EVIDENCE MONITORING</div><h2>Maturation Watchdogs</h2><p class="panel-copy">{watchdog_note}</p></div></div>
    <div class="watchdog-grid">{watchdog_rows}</div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><div class="eyebrow">FAULT SURFACE</div><h2>Pipeline Errors</h2></div><span class="count">{_num(errors.get('count'), '0')} current</span></div>
    {_errors_html(errors)}
  </section>
  <section class="panel" id="phases">
    <div class="panel-head"><div><div class="eyebrow">ROADMAP</div><h2>Phases 14–24</h2></div></div>
    <div class="phase-grid">{phase_rows or "<div class='empty'>No phase telemetry.</div>"}</div>
  </section>
</main>
<aside class="rail">
  <section class="panel">
    <div class="panel-head"><div><div class="eyebrow">SYSTEM</div><h2>System Health</h2></div></div>
    <div class="health-grid">{health_cards}</div>
    <div class="research-note">Research-first dashboard · no thresholds, gates, model weights or canonical BET logic are changed by this view.</div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><div class="eyebrow">REFRESH</div><h2>Snapshot Fallback</h2></div></div>
    <p class="panel-copy">The browser checks the persisted product view every 60 seconds. A newer tick reloads the dashboard. If the check fails, this snapshot stays on screen.</p>
    <div id="fallbackState" class="pill ok">Snapshot available</div>
  </section>
</aside>
</div>
</div>
<script>
(() => {{
  const generated = {json.dumps(str(generated or ""))};
  const state = document.getElementById("refreshState");
  const fallback = document.getElementById("fallbackState");
  let timer = null;
  function mark(text, bad=false) {{
    if (state) state.textContent = text;
    if (fallback) {{
      fallback.textContent = bad ? "Snapshot mode · refresh unavailable" : "Live refresh check OK";
      fallback.className = "pill " + (bad ? "warn" : "ok");
    }}
  }}
  async function check() {{
    if (document.hidden) return;
    try {{
      const res = await fetch("/product/views?limit=25", {{cache:"no-store", headers:{{"Accept":"application/json"}}}});
      if (!res.ok) throw new Error("HTTP " + res.status);
      const payload = await res.json();
      const next = String(payload.generated_at_utc || "");
      mark("Refresh OK · " + new Date().toLocaleTimeString([], {{hour:"2-digit",minute:"2-digit"}}));
      if (next && generated && next !== generated) window.location.reload();
    }} catch (err) {{
      mark("Refresh unavailable · showing saved snapshot", true);
    }}
  }}
  function schedule() {{
    clearInterval(timer);
    timer = setInterval(check, 60000);
  }}
  window.addEventListener("online", () => {{ check(); schedule(); }});
  window.addEventListener("offline", () => mark("Offline · showing saved snapshot", true));
  document.addEventListener("visibilitychange", () => {{ if (!document.hidden) check(); }});
  schedule();
}})();
</script>
</body>
</html>"""
