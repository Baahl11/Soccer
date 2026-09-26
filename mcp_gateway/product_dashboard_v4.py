from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.2.0"
MODEL_VERSION = "SOCCER_PRODUCT_DASHBOARD_V4_1.2.0"

DISPLAY_VIEWS = (
    ("strong_sport_signals", "Strong Sport Signals"),
    ("value_plays", "Value Plays"),
    ("waiting_for_price", "Waiting for Price"),
    ("waiting_for_xi", "Waiting for XI"),
    ("todays_slate", "Today's Slate"),
    ("team_totals", "Team Totals"),
    ("first_half", "1H"),
    ("second_half", "2H"),
    ("corners", "Corners"),
    ("player_props", "Player Props"),
)


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
    if any(token in text for token in ("HEALTHY", "LIVE", "READY", "PASS", "TICK_OBSERVED")):
        return "ok"
    if any(token in text for token in ("WAIT", "HOLD", "LOCKED", "RESEARCH", "VALIDATION", "PENDING", "N/V", "NOT_VERIFIED")):
        return "warn"
    if any(token in text for token in ("ERROR", "FAILED", "DEGRADED", "OFFLINE")):
        return "bad"
    return "neutral"


def _row_html(row: dict[str, Any]) -> str:
    fixture = f"{_esc(row.get('home'))} vs {_esc(row.get('away'))}"
    market = _esc(row.get("market"))
    selection = _esc(row.get("selection"))
    line = _esc(row.get("line"))
    price = _esc(row.get("price"))
    signal = _esc(row.get("model_signal"))
    execution = _esc(row.get("execution_status"))
    league = _esc(row.get("league"))
    edge = row.get("calibrated_edge_pp")
    if edge is None:
        edge = row.get("prob_edge_pp")
    edge_html = "N/V" if edge is None else f"{float(edge):+.1f} pp"
    return (
        "<tr>"
        f"<td><strong>{fixture}</strong><div class='muted'>{league}</div></td>"
        f"<td>{market}<div class='muted'>{selection} {line}</div></td>"
        f"<td class='mono'>{price}</td>"
        f"<td><span class='signal'>{signal}</span></td>"
        f"<td class='edge mono'>{_esc(edge_html)}</td>"
        f"<td><span class='pill {_status_class(execution)}'>{execution}</span></td>"
        "</tr>"
    )


def _view_section(key: str, title: str, view: dict[str, Any]) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    total = int(view.get("total") or 0) if isinstance(view, dict) else 0
    if not rows:
        body = "<div class='empty'>No verified rows in the latest persisted tick.</div>"
    else:
        body = (
            "<div class='table-wrap'><table>"
            "<thead><tr><th>Fixture</th><th>Market</th><th>Price</th><th>Signal</th><th>Edge</th><th>Execution</th></tr></thead>"
            "<tbody>" + "".join(_row_html(row) for row in rows) + "</tbody></table></div>"
        )
    return (
        f"<section class='panel feed-panel' id='{_esc(key)}'>"
        f"<div class='panel-head'><div><div class='eyebrow'>LIVE VIEW</div><h2>{_esc(title)}</h2></div>"
        f"<span class='count'>{total} total</span></div>{body}</section>"
    )


def _health_card(label: str, value: Any, detail: Any = None) -> str:
    cls = _status_class(value)
    detail_html = f"<div class='health-detail'>{_esc(detail)}</div>" if detail is not None else ""
    return (
        "<div class='health-card'>"
        f"<div class='health-icon {cls}'></div>"
        f"<div><div class='health-label'>{_esc(label)}</div>"
        f"<div class='health-value {cls}'>{_esc(value)}</div>{detail_html}</div>"
        "</div>"
    )


def _metric(label: str, value: Any, sub: Any = None) -> str:
    sub_html = f"<div class='metric-sub'>{_esc(sub)}</div>" if sub is not None else ""
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{_esc(label)}</div>"
        f"<div class='metric-value mono'>{_num(value)}</div>{sub_html}"
        "</div>"
    )


def _gate_html(gate: dict[str, Any]) -> str:
    current = gate.get("current")
    target = gate.get("target")
    label = _esc(gate.get("label"))
    unit = _esc(gate.get("unit"))
    if current is None or target in (None, 0):
        value = f"N/V / {_num(target)}"
        width = 0
        cls = "unknown"
    else:
        width = max(0, min(100, int(float(current) / float(target) * 100)))
        value = f"{_num(current)} / {_num(target)}"
        cls = "complete" if float(current) >= float(target) else "progress"
    return (
        "<div class='gate'>"
        f"<div class='gate-top'><span>{label}</span><strong class='mono'>{_esc(value)} {_esc(unit)}</strong></div>"
        f"<div class='track'><div class='fill {cls}' style='width:{width}%'></div></div>"
        "</div>"
    )


def _phase_html(phase: dict[str, Any]) -> str:
    status = phase.get("status") or "NOT_VERIFIED"
    cls = _status_class(status)
    return (
        "<div class='phase-card'>"
        f"<div class='phase-num'>PHASE {_esc(phase.get('phase'))}</div>"
        f"<div class='phase-name'>{_esc(phase.get('name'))}</div>"
        f"<span class='pill {cls}'>{_esc(status)}</span>"
        "</div>"
    )


def _errors_html(errors: dict[str, Any]) -> str:
    count = int(errors.get("count") or 0) if isinstance(errors, dict) else 0
    rows = errors.get("rows") if isinstance(errors, dict) and isinstance(errors.get("rows"), list) else []
    if count == 0:
        return (
            "<div class='empty-success'><span class='check'>✓</span>"
            "<div><strong>No current pipeline errors</strong>"
            "<div class='muted'>Latest persisted tick contains no PIPELINE_ERROR rows.</div></div></div>"
        )
    cards = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixture = f"{_esc(row.get('home'))} vs {_esc(row.get('away'))}"
        cards.append(
            "<div class='error-row'>"
            f"<div><strong>{fixture}</strong><div class='muted'>{_esc(row.get('league'))} · {_esc(row.get('stage'))}</div></div>"
            f"<div class='error-reason'>{_esc(row.get('reason'))}</div></div>"
        )
    return "".join(cards) or "<div class='empty'>Pipeline errors detected; detail unavailable.</div>"


def render_dashboard(product_payload: dict[str, Any]) -> str:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    tower = views.get("control_tower") if isinstance(views.get("control_tower"), dict) else {}
    health = tower.get("system_health") if isinstance(tower.get("system_health"), dict) else {}
    pipeline = tower.get("pipeline") if isinstance(tower.get("pipeline"), dict) else {}
    errors = tower.get("errors") if isinstance(tower.get("errors"), dict) else {}
    gates = tower.get("validation_gates") if isinstance(tower.get("validation_gates"), list) else []
    phases = tower.get("phases") if isinstance(tower.get("phases"), list) else []

    generated = product_payload.get("generated_at_utc")
    pipeline_version = product_payload.get("pipeline_version")
    status = tower.get("status") or product_payload.get("status")
    strong_total = (views.get("strong_sport_signals") or {}).get("total") if isinstance(views.get("strong_sport_signals"), dict) else 0
    value_total = (views.get("value_plays") or {}).get("total") if isinstance(views.get("value_plays"), dict) else 0
    wait_xi_total = (views.get("waiting_for_xi") or {}).get("total") if isinstance(views.get("waiting_for_xi"), dict) else 0

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": pipeline_version,
        "generated_at_utc": generated,
        "status": status,
        "production_promotion_allowed": False,
    }
    meta_json = html.escape(json.dumps(meta, ensure_ascii=False), quote=True)

    health_cards = "".join((
        _health_card("Runtime", health.get("runtime")),
        _health_card("Postgres", health.get("postgres")),
        _health_card("Scheduler", health.get("scheduler")),
        _health_card("API-Football", health.get("api_football"), f"{_num(health.get('api_football_remaining'))} left"),
        _health_card("Galaxy", health.get("galaxy")),
        _health_card("Budget", health.get("daily_budget_mode")),
    ))

    scan_dates = pipeline.get("scan_dates") if isinstance(pipeline.get("scan_dates"), list) else []
    date_label = " · ".join(str(value) for value in scan_dates[:3]) if scan_dates else "N/V"
    metrics = "".join((
        _metric("Fixtures scanned", pipeline.get("fixtures_scanned")),
        _metric(
            "Leagues scanned",
            pipeline.get("unique_leagues_scanned"),
            f"{_num(pipeline.get('unique_countries_scanned'))} countries",
        ),
        _metric("Dates scanned", len(scan_dates) if scan_dates else None, date_label),
        _metric(
            "Market handoff",
            pipeline.get("market_capture_handoff_fixture_count"),
            "A/B/C future fixtures",
        ),
        _metric("Due", pipeline.get("due")),
        _metric("Deep dives", pipeline.get("deep_dives")),
        _metric("Events", pipeline.get("events")),
        _metric("Research visible", pipeline.get("research_visible")),
        _metric("API calls", pipeline.get("api_calls"), f"cap {_num(pipeline.get('api_call_cap'))}"),
        _metric("Scheduler mode", pipeline.get("scheduler_mode"), f"schema {_esc(pipeline.get('scheduler_schema_version'))}"),
        _metric(
            "Unseen processed",
            pipeline.get("scheduler_unseen_processed"),
            f"actionable {_num(pipeline.get('scheduler_actionable_processed'))}",
        ),
        _metric(
            "Starvation",
            pipeline.get("scheduler_starvation_count"),
            f"{_esc(pipeline.get('scheduler_due_analyzed_pct'))}% analyzed",
        ),
        _metric(
            "Planned leagues",
            pipeline.get("scheduler_planned_unique_leagues"),
            f"urgent {_num(pipeline.get('scheduler_urgent_actionable_count'))}",
        ),
        _metric(
            "TT close candidates",
            pipeline.get("team_totals_maturation_candidates"),
            f"matured {_num(pipeline.get('team_totals_later_quote_refreshes'))}",
        ),
        _metric(
            "Primary close candidates",
            pipeline.get("primary_clv_maturation_candidates"),
            f"refreshed {_num(pipeline.get('primary_clv_maturation_refreshed'))}",
        ),
    ))

    gate_rows = "".join(_gate_html(g) for g in gates if isinstance(g, dict))
    phase_rows = "".join(_phase_html(p) for p in phases if isinstance(p, dict))
    feed_sections = "".join(
        _view_section(key, title, views.get(key) if isinstance(views.get(key), dict) else {})
        for key, title in DISPLAY_VIEWS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soccer Edge · Control Tower</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --bg:#050b12; --surface:#09121d; --surface2:#0c1724; --border:#17283a;
  --text:#edf5fb; --muted:#71849a; --cyan:#34b7ff; --green:#20d69a;
  --amber:#f4b74a; --red:#ff5f72; --violet:#9b7cff;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:
  radial-gradient(circle at 82% -10%, rgba(36,128,192,.13), transparent 30%),
  linear-gradient(180deg,#050b12 0%,#07101a 100%); color:var(--text); }}
a {{ color:inherit; }}
.shell {{ min-height:100vh; display:grid; grid-template-columns:220px minmax(0,1fr); }}
.sidebar {{ position:sticky; top:0; height:100vh; border-right:1px solid var(--border); background:rgba(5,12,20,.94); padding:24px 16px; }}
.brand {{ display:flex; gap:10px; align-items:center; font-weight:900; letter-spacing:.04em; margin:2px 8px 28px; }}
.brand-mark {{ width:30px; height:30px; border-radius:9px; display:grid; place-items:center; color:var(--cyan); border:1px solid #17496b; background:#071c2d; }}
.nav-title {{ color:#50677d; font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.14em; margin:24px 10px 8px; }}
.nav a {{ display:flex; align-items:center; gap:9px; text-decoration:none; color:#8fa2b6; padding:10px 11px; border-radius:8px; margin:3px 0; font-size:13px; }}
.nav a:hover,.nav a.active {{ color:#dff4ff; background:#0d2d47; box-shadow:inset 2px 0 0 var(--cyan); }}
.main {{ min-width:0; padding:28px 30px 60px; max-width:1760px; width:100%; margin:auto; }}
.topbar {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:22px; }}
.eyebrow {{ color:var(--cyan); font-size:10px; letter-spacing:.13em; font-weight:800; text-transform:uppercase; }}
h1 {{ font-size:32px; letter-spacing:-.04em; margin:4px 0 5px; }}
h2 {{ font-size:17px; margin:3px 0; letter-spacing:-.02em; }}
.subtitle,.muted {{ color:var(--muted); }}
.subtitle {{ font-size:13px; }}
.livebox {{ text-align:right; font-size:12px; color:var(--muted); }}
.live-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); box-shadow:0 0 10px rgba(32,214,154,.75); margin-right:6px; }}
.panel {{ background:linear-gradient(180deg,rgba(11,23,36,.96),rgba(8,17,28,.96)); border:1px solid var(--border); border-radius:12px; overflow:hidden; box-shadow:0 14px 36px rgba(0,0,0,.16); }}
.panel-head {{ padding:15px 17px; display:flex; justify-content:space-between; gap:18px; align-items:center; border-bottom:1px solid var(--border); }}
.count {{ color:#6f8498; font-size:12px; }}
.health-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:9px; padding:13px; }}
.health-card {{ min-width:0; display:flex; align-items:center; gap:10px; background:#091a27; border:1px solid #123047; border-radius:9px; padding:12px; }}
.health-icon {{ width:9px; height:9px; border-radius:50%; flex:0 0 auto; }}
.health-icon.ok {{ background:var(--green); box-shadow:0 0 9px rgba(32,214,154,.7); }}
.health-icon.warn {{ background:var(--amber); }}
.health-icon.bad {{ background:var(--red); }}
.health-icon.neutral {{ background:#64788d; }}
.health-label {{ font-size:10px; color:#8194a8; text-transform:uppercase; letter-spacing:.06em; }}
.health-value {{ font-weight:800; font-size:12px; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.health-value.ok {{ color:var(--green); }} .health-value.warn {{ color:var(--amber); }} .health-value.bad {{ color:var(--red); }}
.health-detail {{ color:#5e7488; font-size:10px; margin-top:2px; }}
.kpi-strip {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin:14px 0; }}
.metric-card {{ padding:15px 16px; border:1px solid var(--border); border-radius:11px; background:#08131f; }}
.metric-label {{ color:#71869b; font-size:10px; text-transform:uppercase; letter-spacing:.07em; }}
.metric-value {{ font-size:24px; font-weight:900; margin-top:5px; }}
.metric-sub {{ color:#536c82; font-size:10px; margin-top:2px; }}
.grid-main {{ display:grid; grid-template-columns:1.05fr .95fr; gap:14px; margin-bottom:14px; }}
.gate-list {{ padding:13px 16px 17px; }}
.gate {{ margin:11px 0 16px; }}
.gate-top {{ display:flex; justify-content:space-between; gap:12px; font-size:12px; }}
.gate-top span {{ color:#a8b9c8; }} .gate-top strong {{ font-size:11px; }}
.track {{ height:7px; background:#101e2b; border-radius:999px; margin-top:7px; overflow:hidden; border:1px solid #172a3b; }}
.fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg,#1689d2,#25d5a1); }}
.fill.unknown {{ width:0!important; }}
.fill.complete {{ background:linear-gradient(90deg,#17b982,#54e9b7); }}
.error-body {{ padding:15px; }}
.empty-success {{ display:flex; gap:12px; align-items:center; min-height:108px; padding:18px; border:1px dashed #174335; border-radius:10px; background:rgba(25,100,75,.08); }}
.check {{ width:30px; height:30px; display:grid; place-items:center; border-radius:50%; background:#0c4c38; color:#6df2c2; font-weight:900; }}
.error-row {{ display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid var(--border); padding:11px 2px; }}
.error-reason {{ color:var(--red); font-size:11px; max-width:48%; text-align:right; }}
.phase-grid {{ display:grid; grid-template-columns:repeat(11,minmax(110px,1fr)); gap:8px; padding:13px; overflow-x:auto; }}
.phase-card {{ padding:11px; border:1px solid #152b3d; background:#081722; border-radius:9px; min-width:112px; }}
.phase-num {{ color:#4e6c83; font-size:9px; font-weight:800; letter-spacing:.08em; }}
.phase-name {{ font-size:12px; font-weight:800; margin:5px 0 9px; min-height:30px; }}
.pill {{ display:inline-flex; align-items:center; border-radius:999px; font-size:9px; font-weight:800; letter-spacing:.04em; padding:4px 7px; border:1px solid transparent; }}
.pill.ok {{ color:#63ecc1; background:#0c382d; border-color:#155c48; }}
.pill.warn {{ color:#f6ca73; background:#362914; border-color:#5a431f; }}
.pill.bad {{ color:#ff8794; background:#3d1720; border-color:#652330; }}
.pill.neutral {{ color:#9fb0c0; background:#182532; border-color:#27394a; }}
.section-title {{ margin:28px 0 12px; display:flex; justify-content:space-between; align-items:end; }}
.section-title h2 {{ font-size:21px; }}
.quick-cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-bottom:14px; }}
.quick {{ background:#081722; border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.quick .qv {{ font-size:25px; font-weight:900; margin-top:3px; }}
.quick.green .qv {{ color:var(--green); }} .quick.amber .qv {{ color:var(--amber); }} .quick.cyan .qv {{ color:var(--cyan); }}
.feed-panel {{ margin-bottom:14px; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:900px; }}
th,td {{ text-align:left; padding:12px 16px; border-bottom:1px solid #122331; vertical-align:middle; }}
th {{ color:#60768b; font-size:9px; text-transform:uppercase; letter-spacing:.09em; }}
td {{ font-size:12px; color:#c5d2dd; }}
.signal {{ color:#d8e8f4; font-weight:800; }} .edge {{ color:var(--green); font-weight:900; }}
.mono {{ font-variant-numeric:tabular-nums; font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; }}
.empty {{ padding:20px 17px; color:#63798d; font-size:12px; }}
.footer {{ color:#526b80; font-size:11px; padding:22px 2px; }}
@media (max-width:1180px) {{
  .shell {{ grid-template-columns:76px minmax(0,1fr); }}
  .brand span,.nav a span,.nav-title {{ display:none; }}
  .brand {{ justify-content:center; margin-left:0; margin-right:0; }}
  .nav a {{ justify-content:center; font-size:16px; }}
  .health-grid,.kpi-strip {{ grid-template-columns:repeat(3,1fr); }}
}}
@media (max-width:760px) {{
  .shell {{ display:block; }}
  .sidebar {{ display:none; }}
  .main {{ padding:18px 13px 40px; }}
  .topbar {{ display:block; }} .livebox {{ text-align:left; margin-top:10px; }}
  .health-grid,.kpi-strip,.quick-cards {{ grid-template-columns:repeat(2,1fr); }}
  .grid-main {{ grid-template-columns:1fr; }}
  h1 {{ font-size:27px; }}
}}
</style>
</head>
<body data-meta="{meta_json}">
<div class="shell">
<aside class="sidebar">
  <div class="brand"><div class="brand-mark">SE</div><span>SOCCER<br>EDGE</span></div>
  <div class="nav">
    <a href="#today"><b>⌂</b><span>Today</span></a>
    <a href="#strong_sport_signals"><b>↗</b><span>Edge Feed</span></a>
    <a href="#today"><b>◫</b><span>Matches</span></a>
    <a href="#market-lab"><b>◇</b><span>Markets</span></a>
    <a href="#performance"><b>⌁</b><span>Performance</span></a>
    <a href="#today"><b>☆</b><span>My Edge</span></a>
    <div class="nav-title">LAB (Admin)</div>
    <a class="active" href="#control-tower"><b>◉</b><span>Control Tower</span></a>
    <a href="#market-lab"><b>⌬</b><span>Research Lab</span></a>
  </div>
</aside>
<main class="main" id="control-tower">
  <div class="topbar">
    <div><div class="eyebrow">LAB · ADMIN</div><h1>Control Tower</h1>
      <div class="subtitle">System health, data pipeline and model validation status</div></div>
    <div class="livebox"><div><span class="live-dot"></span>{_esc(status)}</div>
      <div>Last tick {_esc(health.get("last_tick"))}</div>
      <div>Pipeline {_esc(pipeline_version)}</div></div>
  </div>

  <section class="panel">
    <div class="panel-head"><div><div class="eyebrow">INFRASTRUCTURE</div><h2>System Health</h2></div>
      <span class="count">Read-only · no provider calls</span></div>
    <div class="health-grid">{health_cards}</div>
  </section>

  <div class="kpi-strip">{metrics}</div>

  <div class="grid-main">
    <section class="panel">
      <div class="panel-head"><div><div class="eyebrow">VALIDATION</div><h2>Model Maturity</h2></div>
        <span class="count">Explicit gates only</span></div>
      <div class="gate-list">{gate_rows}</div>
    </section>
    <section class="panel">
      <div class="panel-head"><div><div class="eyebrow">RUNTIME</div><h2>Pipeline Errors</h2></div>
        <span class="pill {_status_class('HEALTHY' if int(errors.get('count') or 0) == 0 else 'DEGRADED')}">{_num(errors.get('count'), '0')}</span></div>
      <div class="error-body">{_errors_html(errors)}</div>
    </section>
  </div>

  <section class="panel" id="market-lab">
    <div class="panel-head"><div><div class="eyebrow">ROADMAP</div><h2>Phases 14–24</h2></div>
      <span class="count">Production-valid markets: {_num(tower.get('production_valid_market_count'), '0')}</span></div>
    <div class="phase-grid">{phase_rows}</div>
  </section>

  <div class="section-title" id="today"><div><div class="eyebrow">FOOTBALL INTELLIGENCE</div><h2>Today's Pulse</h2></div>
    <span class="count">Research-first dashboard</span></div>
  <div class="quick-cards">
    <div class="quick green"><div class="metric-label">Strong signals</div><div class="qv mono">{_num(strong_total,'0')}</div></div>
    <div class="quick cyan"><div class="metric-label">Value plays</div><div class="qv mono">{_num(value_total,'0')}</div></div>
    <div class="quick amber"><div class="metric-label">Waiting XI</div><div class="qv mono">{_num(wait_xi_total,'0')}</div></div>
  </div>

  <div id="performance">{feed_sections}</div>
  <div class="footer">Read-only operational view. Research/validation status remains authoritative; this dashboard does not promote markets or alter betting logic. N/V means the current counter is not present in the latest runtime payload and is intentionally not inferred.</div>
</main>
</div>
</body>
</html>"""
