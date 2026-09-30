from __future__ import annotations

import html
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_MATCH_DETAIL_V4_1.0.0"
MAX_DETAIL_ROWS = 8

DETAIL_VIEW_PRIORITY = (
    "strong_sport_signals",
    "value_plays",
    "waiting_for_price",
    "waiting_for_xi",
    "todays_slate",
)


def _esc(value: Any) -> str:
    if value is None or value == "":
        return "N/V"
    return html.escape(str(value), quote=True)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _prob(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/V"
    if number < 0.0 or number > 1.0:
        return "N/V"
    return f"{number * 100:.1f}%"


def _edge(value: Any) -> str:
    try:
        return f"{float(value):+.2f} pp" if value is not None else "N/V"
    except (TypeError, ValueError):
        return "N/V"


def _score(value: Any) -> str:
    try:
        return f"{float(value):.1f}" if value is not None else "N/V"
    except (TypeError, ValueError):
        return "N/V"


def _blockers(row: dict[str, Any]) -> str:
    values = row.get("blockers")
    if not isinstance(values, list) or not values:
        return "N/V"
    return " · ".join(_esc(value) for value in values if value is not None)


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    fixture_id = row.get("fixture_id")
    if fixture_id is not None:
        return (
            fixture_id,
            row.get("market_family"),
            row.get("market"),
            row.get("selection"),
            row.get("line"),
        )
    return (
        row.get("home"),
        row.get("away"),
        row.get("kickoff"),
        row.get("market"),
        row.get("selection"),
        row.get("line"),
    )


def build_detail_rows(product_payload: dict[str, Any], *, limit: int = MAX_DETAIL_ROWS) -> list[dict[str, Any]]:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for view_name in DETAIL_VIEW_PRIORITY:
        view = views.get(view_name) if isinstance(views.get(view_name), dict) else {}
        rows = view.get("rows") if isinstance(view.get("rows"), list) else []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            identity = _identity(raw)
            if identity in seen:
                continue
            seen.add(identity)
            output.append({**raw, "_detail_source_view": view_name})
            if len(output) >= max(0, int(limit)):
                return output
    return output


def _kv(label: str, value: Any, *, mono: bool = False) -> str:
    cls = " mono" if mono else ""
    return f"<div class='v217-kv'><span>{_esc(label)}</span><strong class='{cls.strip()}'>{_esc(value)}</strong></div>"


def _detail(row: dict[str, Any]) -> str:
    fixture = f"{_esc(row.get('home'))} vs {_esc(row.get('away'))}"
    market = _esc(row.get("market"))
    selection = _esc(row.get("selection"))
    line = _esc(row.get("line"))
    status = _esc(row.get("execution_status"))
    signal = _esc(row.get("model_signal"))
    signal_score = _score(row.get("model_signal_score"))
    calibrated = _prob(_first(row, "p_model_calibrated", "p_calibrated", "calibrated_probability", "model_probability_calibrated"))
    raw = _prob(_first(row, "p_raw", "model_probability"))
    market_fair = _prob(_first(row, "p_market_fair", "p_market_devig"))
    edge = _edge(_first(row, "calibrated_edge_pp", "prob_edge_pp"))
    price = _esc(row.get("price"))
    bookmaker = _esc(row.get("bookmaker"))
    price_status = _esc(row.get("price_resolution_status"))
    price_source = _esc(row.get("price_resolution_source"))
    provider_update = _esc(row.get("price_resolution_provider_update"))
    price_policy = _esc(row.get("price_resolution_reference_policy"))
    calibration = _dict(row.get("phase16_binary_calibration_diagnostics"))
    calibrator_status = _first(row, "phase16_calibration_status") or calibration.get("calibrator_status")
    reason = _esc(row.get("reason"))
    blockers = _blockers(row)
    signal_basis = _esc(row.get("model_signal_basis"))
    execution_basis = _esc(row.get("execution_status_basis"))

    return f"""
<details class="v217-card">
  <summary>
    <div class="v217-summary-main"><span class="v217-league">{_esc(row.get('league'))} · {_esc(row.get('stage'))}</span><strong>{fixture}</strong><small>{market} · {selection} {line}</small></div>
    <div class="v217-summary-state"><span>{signal} · {signal_score}</span><b>{status}</b></div>
  </summary>
  <div class="v217-body">
    <div class="v217-section"><div class="v217-label">MARKET & PROBABILITY</div><div class="v217-grid">
      {_kv('Price', price, mono=True)}{_kv('Bookmaker', bookmaker)}{_kv('Raw model', raw, mono=True)}{_kv('Calibrated', calibrated, mono=True)}{_kv('Market fair', market_fair, mono=True)}{_kv('Edge', edge, mono=True)}
    </div></div>
    <div class="v217-section"><div class="v217-label">TRACEABILITY</div><div class="v217-grid">
      {_kv('Fixture ID', row.get('fixture_id'), mono=True)}{_kv('Kickoff', row.get('kickoff'), mono=True)}{_kv('Data tier', row.get('data_tier'))}{_kv('Price status', price_status)}{_kv('Price source', price_source)}{_kv('Provider update', provider_update, mono=True)}
    </div></div>
    <div class="v217-section"><div class="v217-label">DECISION CONTEXT</div><div class="v217-grid wide">
      {_kv('Reason', reason)}{_kv('Blockers', blockers)}{_kv('Signal basis', signal_basis)}{_kv('Execution basis', execution_basis)}{_kv('Calibration', calibrator_status)}{_kv('Reference policy', price_policy)}
    </div></div>
  </div>
</details>
"""


def render_fragment(product_payload: dict[str, Any]) -> str:
    rows = build_detail_rows(product_payload)
    cards = "".join(_detail(row) for row in rows)
    if not cards:
        cards = "<div class='v217-empty'>No verified match-detail rows in the latest persisted product payload.</div>"
    return f"""
<style>
.v217{{margin:14px 0;border:1px solid #1b3448;background:linear-gradient(155deg,#091620,#080f16);border-radius:14px;overflow:hidden}}
.v217-head{{padding:16px 18px;display:flex;justify-content:space-between;gap:16px;align-items:flex-start;border-bottom:1px solid #172a3b}}
.v217-head p{{margin:4px 0 0;color:#7890a5;font-size:11px;line-height:1.5;max-width:780px}}
.v217-preview{{padding:5px 8px;border:1px solid #36536c;background:#0c2130;color:#b8eaff;border-radius:999px;font-size:9px;font-weight:900;white-space:nowrap}}
.v217-list{{padding:13px;display:grid;gap:9px}}
.v217-card{{border:1px solid #153047;background:#08141e;border-radius:11px;overflow:hidden}}
.v217-card summary{{list-style:none;cursor:pointer;display:flex;justify-content:space-between;gap:16px;align-items:center;padding:13px 14px}}
.v217-card summary::-webkit-details-marker{{display:none}}
.v217-summary-main{{min-width:0;display:grid;gap:3px}}.v217-summary-main strong{{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.v217-summary-main small{{color:#718ba0;font-size:10px}}
.v217-league{{color:#526f86;font-size:9px;text-transform:uppercase;letter-spacing:.05em}}
.v217-summary-state{{text-align:right;display:grid;gap:4px;flex:0 0 auto}}.v217-summary-state span{{color:#85a1b7;font-size:9px}}.v217-summary-state b{{font-size:9px;color:#f4c66e}}
.v217-body{{border-top:1px solid #142737;padding:13px}}
.v217-section+ .v217-section{{margin-top:14px}}.v217-label{{color:#45bff7;font-size:9px;font-weight:900;letter-spacing:.09em;margin-bottom:8px}}
.v217-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}}.v217-grid.wide{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.v217-kv{{min-width:0;background:#091925;border:1px solid #143044;border-radius:9px;padding:9px}}.v217-kv span{{display:block;color:#607b91;font-size:8px;text-transform:uppercase;letter-spacing:.04em}}.v217-kv strong{{display:block;margin-top:4px;font-size:10px;color:#c8d8e4;word-break:break-word}}.v217-kv strong.mono{{font-family:"SFMono-Regular",Consolas,monospace;font-variant-numeric:tabular-nums}}
.v217-empty{{margin:13px;border:1px dashed #294053;border-radius:10px;padding:16px;color:#71889b;font-size:11px}}
@media(max-width:1000px){{.v217-grid{{grid-template-columns:repeat(3,1fr)}}.v217-grid.wide{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:620px){{.v217-head,.v217-card summary{{display:block}}.v217-preview{{display:inline-block;margin-top:9px}}.v217-summary-state{{text-align:left;margin-top:9px}}.v217-grid,.v217-grid.wide{{grid-template-columns:1fr 1fr}}}}@media(max-width:420px){{.v217-grid,.v217-grid.wide{{grid-template-columns:1fr}}}}
</style>
<section class="v217" id="match-detail">
<div class="v217-head"><div><div class="eyebrow">MATCH DETAIL · V217</div><h2>Premium Signal Detail</h2><p>Expandable traceability for the latest persisted rows. Preview only: access control is not enforced until Auth + Entitlements. Missing probabilities, prices or provenance remain N/V.</p></div><span class="v217-preview">PREMIUM PREVIEW · {len(rows)} ROWS</span></div>
<div class="v217-list">{cards}</div>
</section>
"""
