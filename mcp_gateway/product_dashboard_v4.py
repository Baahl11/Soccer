from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.5.0"
MODEL_VERSION = "SOCCER_PRODUCT_DASHBOARD_V4_1.5.0"

DISPLAY_VIEWS = (
    ("todays_slate", "Today's Slate"),
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


def _status_class(value: Any) -> str:
    text = str(value or "").upper()
    if text == "OK" or any(token in text for token in ("HEALTHY", "LIVE", "READY", "PASS", "TICK_OBSERVED")):
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
        "RESEARCH_ONLY": "Research",
        "DEFER_COOLDOWN": "Cooldown",
        "NOT_VERIFIED": "Not verified",
    }
    return mapping.get(text, text.replace("_", " ").title())


def _edge_value(row: dict[str, Any]) -> float | None:
    value = row.get("calibrated_edge_pp")
    if value is None:
        value = row.get("prob_edge_pp")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _signal_card(row: dict[str, Any], *, accent: str = "cyan") -> str:
    fixture = f"{_esc(row.get('home'))} <span>vs</span> {_esc(row.get('away'))}"
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
        f"<h3>{fixture}</h3>"
        f"<div class='market-name'>{_esc(row.get('market'))}</div>"
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
    fixture = f"{_esc(row.get('home'))} vs {_esc(row.get('away'))}"
    edge = _edge_value(row)
    edge_html = "N/V" if edge is None else f"{edge:+.1f} pp"
    return (
        "<tr>"
        f"<td><strong>{fixture}</strong><div class='muted'>{_esc(row.get('league'))}</div></td>"
        f"<td>{_esc(row.get('market'))}<div class='muted'>{_esc(row.get('selection'))} {_esc(row.get('line'))}</div></td>"
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
    strong = views.get("strong_sport_signals") if isinstance(views.get("strong_sport_signals"), dict) else {}
    values = views.get("value_plays") if isinstance(views.get("value_plays"), dict) else {}
    waiting_price = views.get("waiting_for_price") if isinstance(views.get("waiting_for_price"), dict) else {}
    waiting_xi = views.get("waiting_for_xi") if isinstance(views.get("waiting_for_xi"), dict) else {}

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
        _metric("Fixtures scanned", pipeline.get("fixtures_scanned")),
        _metric("Scheduler mode", pipeline.get("scheduler_mode")),
        _metric("Unseen processed", pipeline.get("scheduler_unseen_processed")),
        _metric("Starvation", pipeline.get("scheduler_starvation_count")),
        _metric("TT close candidates", pipeline.get("team_totals_maturation_candidates")),
        _metric("Primary close candidates", pipeline.get("primary_clv_maturation_candidates")),
        _metric("API calls", pipeline.get("api_calls"), f"cap {_num(pipeline.get('api_call_cap'))}"),
        _metric("API remaining", health.get("api_football_remaining")),
    ))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#07111d">
<meta name="description" content="Soccer Edge Intelligence — evidence-first football market analysis and model signals.">
<title>Soccer Edge · Intelligence</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--bg:#050912;--surface:#08111c;--surface2:#0b1724;--border:#17283a;--text:#f3f7fb;--muted:#72859a;--cyan:#47c9ff;--blue:#5588ff;--green:#2ed7a3;--amber:#f4bc55;--red:#ff6879;--violet:#9a7cff}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 80% -10%,rgba(71,201,255,.14),transparent 28%),radial-gradient(circle at 10% 15%,rgba(154,124,255,.08),transparent 26%),linear-gradient(180deg,#040811,#07111c 45%,#050a12);color:var(--text)}}a{{color:inherit;text-decoration:none}}.shell{{max-width:1540px;margin:auto;padding:22px 26px 64px}}.topnav{{height:58px;display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:1px solid rgba(255,255,255,.06)}}.brand{{display:flex;align-items:center;gap:11px;font-weight:900;letter-spacing:.02em}}.mark{{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(135deg,#0b2840,#11203b);border:1px solid #1d4e72;color:var(--cyan);box-shadow:0 0 28px rgba(71,201,255,.12)}}.navlinks{{display:flex;gap:4px;align-items:center}}.navlinks a{{font-size:12px;color:#899bad;padding:8px 10px;border-radius:8px}}.navlinks a:hover{{color:#fff;background:#0c1d2d}}.beta{{font-size:10px;color:#c9d7e3;border:1px solid #294158;background:#0a1723;padding:5px 8px;border-radius:999px}}.hero{{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,.85fr);gap:28px;padding:48px 0 28px;align-items:end}}.eyebrow{{font-size:10px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:var(--cyan)}}h1{{font-size:clamp(36px,6vw,70px);line-height:.98;letter-spacing:-.06em;margin:10px 0 16px;max-width:850px}}.hero p{{font-size:15px;line-height:1.7;color:#8fa3b7;max-width:720px;margin:0}}.hero-side{{border:1px solid var(--border);background:linear-gradient(180deg,rgba(12,25,39,.95),rgba(7,16,26,.95));border-radius:18px;padding:18px;box-shadow:0 18px 50px rgba(0,0,0,.2)}}.live-row{{display:flex;align-items:center;justify-content:space-between;gap:18px}}.live{{display:flex;align-items:center;gap:8px;font-weight:850;font-size:12px}}.dot{{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 16px rgba(46,215,163,.8)}}.stamp{{font-size:10px;color:#62798f;text-align:right}}.trust-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px}}.trust{{padding:12px;border-radius:10px;border:1px solid #142c40;background:#071620}}.trust strong{{display:block;font-size:12px}}.trust span{{display:block;color:#5e768a;font-size:9px;margin-top:4px;line-height:1.35}}.overview{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin:18px 0 30px}}.overview-card{{border:1px solid var(--border);border-radius:13px;padding:17px;background:linear-gradient(180deg,#0a1724,#08121d)}}.overview-card .label{{font-size:10px;color:#71859a;text-transform:uppercase;letter-spacing:.08em}}.overview-card .value{{font-size:30px;font-weight:950;margin-top:4px}}.overview-card .sub{{font-size:10px;color:#5d7488;margin-top:3px}}.positive{{color:var(--green)}}.section{{margin-top:26px}}.section-head{{display:flex;justify-content:space-between;align-items:end;gap:18px;margin-bottom:12px}}.section-head h2{{font-size:22px;letter-spacing:-.03em;margin:3px 0 0}}.count{{font-size:11px;color:#62788c}}.signal-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px}}.signal-card{{position:relative;overflow:hidden;border:1px solid var(--border);border-radius:14px;background:linear-gradient(180deg,#0a1724,#07111c);padding:15px;min-height:216px}}.signal-card:before{{content:"";position:absolute;inset:0 auto 0 0;width:2px;background:var(--cyan)}}.signal-card.green:before{{background:var(--green)}}.signal-card-top{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.league{{font-size:9px;color:#6c8397;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.signal-card h3{{font-size:17px;letter-spacing:-.03em;margin:18px 0 12px}}.signal-card h3 span{{color:#51677b;font-weight:500;font-size:12px}}.market-name{{font-size:10px;color:#6d8396;text-transform:uppercase;letter-spacing:.06em}}.selection{{font-size:14px;font-weight:800;margin-top:4px}}.selection strong{{color:var(--cyan)}}.signal-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:18px}}.signal-stats div{{min-width:0}}.signal-stats span{{display:block;color:#536b80;font-size:8px;text-transform:uppercase;letter-spacing:.06em}}.signal-stats strong{{display:block;font-size:11px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.radar{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.radar-card{{border:1px solid var(--border);border-radius:14px;background:#08131f;overflow:hidden}}.radar-head{{display:flex;justify-content:space-between;align-items:center;padding:14px 15px;border-bottom:1px solid var(--border)}}.radar-head h3{{font-size:14px;margin:0}}.radar-body{{padding:13px}}.panel{{background:linear-gradient(180deg,rgba(10,22,35,.98),rgba(7,16,27,.98));border:1px solid var(--border);border-radius:14px;overflow:hidden}}.panel-head{{padding:14px 16px;display:flex;justify-content:space-between;gap:18px;align-items:center;border-bottom:1px solid var(--border)}}.panel-head h2{{font-size:16px;margin:3px 0}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:880px}}th,td{{text-align:left;padding:12px 15px;border-bottom:1px solid #122230;vertical-align:middle}}th{{color:#5e7489;font-size:8px;text-transform:uppercase;letter-spacing:.1em}}td{{font-size:11px;color:#c5d2dd}}.muted{{color:var(--muted)}}.signal{{font-weight:850}}.edge{{color:var(--green);font-weight:900}}.pill{{display:inline-flex;align-items:center;border-radius:999px;font-size:8px;font-weight:900;letter-spacing:.04em;padding:4px 7px;border:1px solid transparent}}.pill.ok{{color:#6ff0c7;background:#0b382c;border-color:#155944}}.pill.warn{{color:#f5ca78;background:#332710;border-color:#58431f}}.pill.bad{{color:#ff8795;background:#3b1620;border-color:#642431}}.pill.neutral{{color:#9fb1c0;background:#172431;border-color:#27394a}}.lab{{display:grid;grid-template-columns:1.1fr .9fr;gap:12px}}.maturity-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;padding:13px}}.maturity-card{{background:#081722;border:1px solid #152b3d;border-radius:10px;padding:12px}}.maturity-top{{display:flex;justify-content:space-between;gap:8px;font-size:11px}}.maturity-kind{{color:#4e6c83;font-size:8px;margin-top:3px}}.maturity-value{{font-size:18px;font-weight:900;margin-top:11px}}.maturity-meta{{color:#617c91;font-size:9px;margin-top:8px;line-height:1.35}}.track{{height:6px;background:#101e2b;border-radius:999px;margin-top:7px;overflow:hidden;border:1px solid #172a3b}}.fill{{height:100%;background:linear-gradient(90deg,#228ed1,#2ed7a3);border-radius:999px}}.fill.complete{{background:linear-gradient(90deg,#1ebf8c,#67efc5)}}.fill.unknown{{width:0!important}}.gate-list{{padding:13px 16px}}.gate{{margin:10px 0 14px}}.gate-top{{display:flex;justify-content:space-between;gap:12px;font-size:11px}}.gate-top span{{color:#9db0c1}}.operator{{margin-top:34px;border-top:1px solid rgba(255,255,255,.06);padding-top:24px}}.operator summary{{cursor:pointer;list-style:none;display:flex;justify-content:space-between;align-items:center;padding:14px 0;font-weight:900}}.operator summary::-webkit-details-marker{{display:none}}.operator-kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-bottom:12px}}.metric-card{{padding:13px 14px;border:1px solid var(--border);border-radius:10px;background:#07131e}}.metric-label{{color:#71869b;font-size:9px;text-transform:uppercase;letter-spacing:.07em}}.metric-value{{font-size:20px;font-weight:900;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.metric-sub{{color:#536c82;font-size:9px;margin-top:2px}}.watchdog-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;padding:13px}}.watchdog-card{{background:#081722;border:1px solid #152b3d;border-radius:9px;padding:12px}}.watchdog-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;font-size:11px}}.watchdog-reason{{color:#91a6b8;font-size:9px;margin-top:8px;line-height:1.35;word-break:break-word}}.watchdog-source{{color:#405d72;font-size:8px;margin-top:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.phase-grid{{display:grid;grid-template-columns:repeat(11,minmax(105px,1fr));gap:7px;padding:13px;overflow-x:auto}}.phase-card{{padding:10px;border:1px solid #152b3d;background:#081722;border-radius:9px;min-width:108px}}.phase-num{{color:#4e6c83;font-size:8px;font-weight:800;letter-spacing:.08em}}.phase-name{{font-size:11px;font-weight:800;margin:5px 0 8px;min-height:26px}}.empty{{padding:18px;color:#63798d;font-size:11px}}.empty-success{{display:flex;gap:12px;align-items:center;padding:16px;border:1px dashed #174335;border-radius:10px;background:rgba(25,100,75,.08);margin:13px}}.check{{width:28px;height:28px;display:grid;place-items:center;border-radius:50%;background:#0c4c38;color:#6df2c2;font-weight:900}}.error-row{{display:flex;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border)}}.error-reason{{color:var(--red);font-size:10px;max-width:48%;text-align:right}}.mono{{font-variant-numeric:tabular-nums;font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace}}.footer{{margin-top:26px;color:#526b80;font-size:10px;line-height:1.55;padding:14px 0}}
@media(max-width:1180px){{.signal-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.overview{{grid-template-columns:repeat(2,minmax(0,1fr))}}.maturity-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.watchdog-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.hero{{grid-template-columns:1fr}}}}
@media(max-width:760px){{.shell{{padding:14px 14px 44px}}.navlinks{{display:none}}.hero{{padding-top:30px}}h1{{font-size:42px}}.overview,.signal-grid,.radar,.lab,.operator-kpis,.maturity-grid,.watchdog-grid{{grid-template-columns:1fr}}.trust-grid{{grid-template-columns:1fr}}.signal-stats{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body data-meta="{meta_json}">
<div class="shell">
<header class="topnav">
<a class="brand" href="#top"><span class="mark">SE</span><span>Soccer Edge</span></a>
<nav class="navlinks"><a href="#signals">Signals</a><a href="#radar">Radar</a><a href="#markets">Markets</a><a href="#model-lab">Model Lab</a></nav>
<span class="beta">SUBSCRIBER PREVIEW</span>
</header>

<main id="top">
<section class="hero">
<div><div class="eyebrow">EVIDENCE-FIRST FOOTBALL INTELLIGENCE</div><h1>Find the signal.<br>Ignore the noise.</h1><p>Soccer Edge turns model output, market price, lineup status and validation evidence into one decision surface. No forced picks, no hidden backfills, and no invented data.</p></div>
<aside class="hero-side"><div class="live-row"><div class="live"><span class="dot"></span>{_esc(status)}</div><div class="stamp">Last persisted tick<br><span class="mono">{_esc(generated)}</span></div></div><div class="trust-grid"><div class="trust"><strong>Price-aware</strong><span>Signals keep price and freshness state visible.</span></div><div class="trust"><strong>Evidence-gated</strong><span>Research markets remain research until evidence matures.</span></div><div class="trust"><strong>Read-only</strong><span>This surface cannot alter model or bet logic.</span></div></div></aside>
</section>

<section class="overview" aria-label="Today overview">
<div class="overview-card"><div class="label">Strong Sport Signals</div><div class="value positive mono">{_num(strong.get('total'),'0')}</div><div class="sub">verified in latest tick</div></div>
<div class="overview-card"><div class="label">Value Plays</div><div class="value mono">{_num(values.get('total'),'0')}</div><div class="sub">calibrated mismatch candidates</div></div>
<div class="overview-card"><div class="label">Waiting for Price</div><div class="value mono">{_num(waiting_price.get('total'),'0')}</div><div class="sub">signal exists, price not ready</div></div>
<div class="overview-card"><div class="label">Waiting for XI</div><div class="value mono">{_num(waiting_xi.get('total'),'0')}</div><div class="sub">lineup confirmation pending</div></div>
</section>

<section class="section" id="signals"><div class="section-head"><div><div class="eyebrow">DECISION DESK</div><h2>Strong Sport Signals</h2></div><span class="count">Research-first dashboard · {_num(strong.get('total'),'0')} total</span></div>{_signal_grid(strong, accent='green', empty='No strong verified signal in the latest persisted tick.')}</section>
<section class="section"><div class="section-head"><div><div class="eyebrow">MARKET MISPRICING</div><h2>Value Plays</h2></div><span class="count">Calibrated probability required · {_num(values.get('total'),'0')} total</span></div>{_signal_grid(values, accent='cyan', empty='No calibrated value play in the latest persisted tick.')}</section>

<section class="section" id="radar"><div class="section-head"><div><div class="eyebrow">MARKET RADAR</div><h2>What is blocking execution?</h2></div><span class="count">No hidden assumptions</span></div><div class="radar"><div class="radar-card"><div class="radar-head"><h3>Waiting for Price</h3><span class="pill warn">{_num(waiting_price.get('total'),'0')}</span></div><div class="radar-body">{_signal_grid(waiting_price, accent='cyan', empty='No current price-watch signals.')}</div></div><div class="radar-card"><div class="radar-head"><h3>Waiting for XI</h3><span class="pill warn">{_num(waiting_xi.get('total'),'0')}</span></div><div class="radar-body">{_signal_grid(waiting_xi, accent='green', empty='No current lineup-watch signals.')}</div></div></div></section>

<section class="section" id="model-lab"><div class="section-head"><div><div class="eyebrow">MODEL TRANSPARENCY</div><h2>Model Maturity</h2></div><span class="count">{state_note}</span></div><div class="lab"><div class="panel"><div class="panel-head"><div><div class="eyebrow">MATURATION CONTROL TOWER</div><h2>Market evidence</h2></div><span class="count">Explicit gates only</span></div><div class="maturity-grid">{maturation_rows}</div></div><div class="panel"><div class="panel-head"><div><div class="eyebrow">VALIDATION GATES</div><h2>Evidence thresholds</h2></div></div><div class="gate-list">{gate_rows}</div></div></div></section>

<section class="section" id="markets"><div class="section-head"><div><div class="eyebrow">FULL MARKET BOARD</div><h2>Today's Slate & market views</h2></div><span class="count">Latest persisted data only</span></div>{feed_sections}</section>

<details class="operator" id="control-tower">
<summary><span>Control Tower · operator diagnostics</span><span class="count">System Health · {watchdog_note}</span></summary>
<section class="panel"><div class="panel-head"><div><div class="eyebrow">SYSTEM HEALTH</div><h2>System Health</h2></div><span class="count">Pipeline {_esc(pipeline_version)}</span></div><div class="operator-kpis">{operator_metrics}</div></section>
<section class="panel" style="margin-top:12px"><div class="panel-head"><div><div class="eyebrow">PIPELINE SAFETY</div><h2>Pipeline Errors</h2></div><span class="count">{_num(errors.get('count'),'0')} current</span></div>{_errors_html(errors)}</section>
<section class="panel" style="margin-top:12px"><div class="panel-head"><div><div class="eyebrow">VALIDATION · READ ONLY</div><h2>Maturation Watchdogs</h2></div><span class="count">{watchdog_note}</span></div><div class="watchdog-grid">{watchdog_rows}</div></section>
<section class="panel" style="margin-top:12px"><div class="panel-head"><div><div class="eyebrow">ROADMAP</div><h2>Phases 14–24</h2></div><span class="count">Production-valid markets: {_num(tower.get('production_valid_market_count'),'0')}</span></div><div class="phase-grid">{phase_rows}</div></section>
</details>

<div class="footer">Read-only operational view. Soccer Edge presents persisted model and market evidence; it does not guarantee outcomes, manufacture missing data, promote research markets, or alter betting logic. N/V means the current field is not verified in the latest payload.</div>
</main>
</div>
</body>
</html>"""
