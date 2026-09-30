from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_COMMERCIAL_APP_V4_1.0.0"
PRODUCT_MODE = "PREVIEW_MODE"

FREE_FEATURES = (
    "Today's verified slate",
    "Market readiness radar",
    "Model maturity overview",
    "Read-only public intelligence",
)

PRO_FEATURES = (
    "Strong signal desk",
    "Calibrated value plays",
    "Advanced market families",
    "Verified performance history",
    "Favorites and alerts",
)


def _esc(value: Any) -> str:
    if value is None:
        return "N/V"
    return html.escape(str(value), quote=True)


def _num(value: Any, fallback: str = "0") -> str:
    if value is None:
        return fallback
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _esc(value)


def _edge(row: dict[str, Any]) -> str:
    value = row.get("calibrated_edge_pp")
    if value is None:
        value = row.get("prob_edge_pp")
    try:
        return f"{float(value):+.1f} pp" if value is not None else "N/V"
    except (TypeError, ValueError):
        return "N/V"


def _friendly_status(value: Any) -> str:
    text = str(value or "NOT_VERIFIED").upper()
    mapping = {
        "BET": "Ready",
        "READY": "Ready",
        "WAIT_PRICE": "Price watch",
        "WAIT_FRESH_QUOTE": "Fresh price",
        "WAIT_XI": "Lineup watch",
        "RESEARCH_ONLY": "Research",
        "DEFER_COOLDOWN": "Cooldown",
        "NOT_VERIFIED": "Not verified",
    }
    return mapping.get(text, text.replace("_", " ").title())


def _signal_card(row: dict[str, Any]) -> str:
    home = _esc(row.get("home"))
    away = _esc(row.get("away"))
    return (
        "<article class='pick-card'>"
        "<div class='pick-top'>"
        f"<span class='league'>{_esc(row.get('league'))}</span>"
        f"<span class='status'>{_esc(_friendly_status(row.get('execution_status')))}</span>"
        "</div>"
        f"<h3>{home} <span>vs</span> {away}</h3>"
        f"<div class='market'>{_esc(row.get('market'))}</div>"
        f"<div class='selection'>{_esc(row.get('selection'))} <strong>{_esc(row.get('line'))}</strong></div>"
        "<div class='pick-stats'>"
        f"<div><small>Price</small><strong>{_esc(row.get('price'))}</strong></div>"
        f"<div><small>Edge</small><strong>{_edge(row)}</strong></div>"
        f"<div><small>Signal</small><strong>{_esc(row.get('model_signal'))}</strong></div>"
        "</div>"
        "</article>"
    )


def _cards(view: dict[str, Any], empty: str, limit: int = 6) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not rows:
        return f"<div class='empty'>{_esc(empty)}</div>"
    return "<div class='pick-grid'>" + "".join(_signal_card(row) for row in rows[:limit]) + "</div>"


def _feature_list(items: tuple[str, ...]) -> str:
    return "".join(f"<li><span>✓</span>{_esc(item)}</li>" for item in items)


def _maturity_cards(product_payload: dict[str, Any]) -> str:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    tower = views.get("control_tower") if isinstance(views.get("control_tower"), dict) else {}
    snapshot = tower.get("maturity_snapshot") if isinstance(tower.get("maturity_snapshot"), dict) else {}
    mt = snapshot.get("maturation_control_tower") if isinstance(snapshot.get("maturation_control_tower"), dict) else {}
    families = mt.get("families") if isinstance(mt.get("families"), list) else []
    cards: list[str] = []
    for node in families:
        if not isinstance(node, dict):
            continue
        current = node.get("current")
        target = node.get("target")
        try:
            pct = max(0, min(100, int(float(current) / float(target) * 100))) if current is not None and target not in (None, 0) else 0
        except (TypeError, ValueError, ZeroDivisionError):
            pct = 0
        ratio = f"{_num(current, 'N/V')} / {_num(target, 'N/V')}" if target is not None else _num(current, "N/V")
        cards.append(
            "<div class='maturity'>"
            f"<div class='maturity-head'><strong>{_esc(node.get('label'))}</strong><span>{_esc(node.get('status'))}</span></div>"
            f"<div class='ratio'>{_esc(ratio)}</div>"
            f"<div class='track'><i style='width:{pct}%'></i></div>"
            f"<small>{_esc(node.get('blocker') or 'No explicit blocker')}</small>"
            "</div>"
        )
    return "".join(cards) or "<div class='empty'>Maturation telemetry is not present in the current snapshot.</div>"


def render_app(product_payload: dict[str, Any]) -> str:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    strong = views.get("strong_sport_signals") if isinstance(views.get("strong_sport_signals"), dict) else {}
    value = views.get("value_plays") if isinstance(views.get("value_plays"), dict) else {}
    waiting_price = views.get("waiting_for_price") if isinstance(views.get("waiting_for_price"), dict) else {}
    waiting_xi = views.get("waiting_for_xi") if isinstance(views.get("waiting_for_xi"), dict) else {}
    today = views.get("todays_slate") if isinstance(views.get("todays_slate"), dict) else {}
    performance = views.get("performance") if isinstance(views.get("performance"), dict) else {}

    generated = product_payload.get("generated_at_utc")
    pipeline_version = product_payload.get("pipeline_version")
    meta = html.escape(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "product_mode": PRODUCT_MODE,
        "pipeline_version": pipeline_version,
        "generated_at_utc": generated,
        "auth_enabled": False,
        "billing_enabled": False,
        "entitlements_enforced": False,
        "provider_requests_added": 0,
    }, ensure_ascii=False), quote=True)

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Soccer Edge · App Preview</title><style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--bg:#05080d;--surface:#0b1119;--line:#1b2b3b;--text:#f4f8fb;--muted:#7f93a8;--cyan:#41c7ff;--green:#38dda3;--amber:#ffc55b}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 80% -10%,rgba(65,199,255,.12),transparent 30%),var(--bg);color:var(--text)}}a{{color:inherit;text-decoration:none}}.wrap{{max-width:1380px;margin:auto;padding:0 24px 64px}}.top{{height:72px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.06);position:sticky;top:0;background:rgba(5,8,13,.88);backdrop-filter:blur(14px);z-index:20}}.brand{{font-weight:950;font-size:20px}}.brand b{{color:var(--cyan)}}.nav{{display:flex;gap:20px;color:#a9b8c7;font-size:13px}}.account{{display:flex;gap:9px;align-items:center}}.badge{{font-size:10px;font-weight:900;padding:6px 9px;border-radius:999px;border:1px solid #31506b;color:#bfeeff;background:#0b2132}}.btn{{border:1px solid #274057;background:#0c1722;color:#dce9f4;padding:9px 13px;border-radius:9px;font-size:12px;font-weight:800}}.hero{{padding:56px 0 28px;display:grid;grid-template-columns:1.25fr .75fr;gap:28px;align-items:end}}.eyebrow{{font-size:10px;letter-spacing:.14em;color:var(--cyan);font-weight:900;text-transform:uppercase}}h1{{font-size:48px;line-height:1.02;letter-spacing:-.055em;margin:10px 0 14px;max-width:820px}}.lead{{color:#9eb0c1;max-width:720px;line-height:1.6;font-size:15px}}.hero-card{{border:1px solid var(--line);background:linear-gradient(155deg,#101925,#0a1018);border-radius:18px;padding:20px}}.hero-card strong{{font-size:34px;display:block;margin:4px 0}}.hero-card small{{color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0 34px}}.stat{{border:1px solid var(--line);background:var(--surface);border-radius:13px;padding:14px}}.stat small{{display:block;color:var(--muted);text-transform:uppercase;font-size:9px}}.stat strong{{display:block;font-size:25px;margin-top:5px}}section{{margin:34px 0}}.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:14px}}h2{{font-size:23px;margin:3px 0}}.section-head p{{color:var(--muted);font-size:12px;margin:0}}.pick-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.pick-card{{border:1px solid var(--line);background:linear-gradient(180deg,#0d1721,#090f16);border-radius:14px;padding:16px;min-width:0}}.pick-top{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.league{{font-size:10px;color:#768ea4;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.status{{font-size:9px;font-weight:900;color:#79e6bf;background:#0b3428;border:1px solid #17523f;padding:4px 7px;border-radius:999px}}.pick-card h3{{font-size:16px;margin:14px 0 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.pick-card h3 span{{color:#526578;font-size:11px;font-weight:500}}.market{{font-size:11px;color:#8198ad}}.selection{{font-size:19px;font-weight:900;margin-top:4px}}.pick-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:15px;border-top:1px solid #162432;padding-top:12px}}.pick-stats small{{display:block;color:#587084;font-size:9px;text-transform:uppercase}}.pick-stats strong{{font-size:12px}}.radar{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.radar-box,.maturity,.plan{{border:1px solid var(--line);background:var(--surface);border-radius:14px;padding:18px}}.radar-num{{font-size:38px;font-weight:950}}.radar-box p{{font-size:12px;color:var(--muted)}}.maturity-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}.maturity{{padding:13px}}.maturity-head{{display:flex;justify-content:space-between;gap:8px;font-size:11px}}.maturity-head span{{color:var(--amber);font-size:9px}}.ratio{{font-size:20px;font-weight:900;margin:11px 0 8px}}.track{{height:6px;background:#13202c;border-radius:999px;overflow:hidden}}.track i{{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}}.maturity small{{color:#60788c;display:block;margin-top:8px;font-size:9px}}.history{{border:1px solid var(--line);background:#0b1119;border-radius:16px;padding:20px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}.history span{{color:var(--muted);font-size:10px;text-transform:uppercase}}.history strong{{display:block;margin-top:8px}}.plans{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.plan.pro{{border-color:#31536c;background:linear-gradient(160deg,#0c1d2c,#101421)}}.plan h3{{font-size:24px;margin:4px 0}}.plan ul{{list-style:none;padding:0}}.plan li{{font-size:12px;color:#a8b9c8;padding:7px 0}}.plan li span{{color:var(--green);margin-right:8px}}.locked,.empty{{border:1px dashed #38485a;border-radius:10px;padding:14px;color:#8295a8;font-size:11px}}.footer{{margin-top:42px;border-top:1px solid #152230;padding-top:20px;color:#5f7487;font-size:11px}}@media(max-width:900px){{.hero{{grid-template-columns:1fr}}h1{{font-size:37px}}.stats,.maturity-grid{{grid-template-columns:repeat(2,1fr)}}.pick-grid{{grid-template-columns:1fr 1fr}}.plans{{grid-template-columns:1fr}}}}@media(max-width:620px){{.wrap{{padding:0 14px 40px}}.nav{{display:none}}h1{{font-size:32px}}.stats,.pick-grid,.radar,.maturity-grid,.history{{grid-template-columns:1fr}}}}
</style></head><body data-product='{meta}'><div class='wrap'><header class='top'><div class='brand'>SOCCER <b>EDGE</b></div><nav class='nav'><a href='#today'>Today</a><a href='#radar'>Radar</a><a href='#maturity'>Maturity</a><a href='#history'>History</a><a href='#membership'>Membership</a></nav><div class='account'><span class='badge'>PREVIEW</span><span class='btn'>Sign in · soon</span></div></header><main><section class='hero'><div><div class='eyebrow'>Verified football intelligence</div><h1>See the signal. See the evidence. Skip the noise.</h1><p class='lead'>Soccer Edge surfaces persisted model signals, market readiness and validation status. Missing evidence stays missing; the app does not invent prices, edges or results.</p></div><div class='hero-card'><small>Product status</small><strong>Preview</strong><small>Auth, billing and entitlements are intentionally disabled until the account layer is implemented.</small></div></section><div class='stats'><div class='stat'><small>Strong signals</small><strong>{_num(strong.get('total'))}</strong></div><div class='stat'><small>Value plays</small><strong>{_num(value.get('total'))}</strong></div><div class='stat'><small>Price watch</small><strong>{_num(waiting_price.get('total'))}</strong></div><div class='stat'><small>Lineup watch</small><strong>{_num(waiting_xi.get('total'))}</strong></div></div><section id='today'><div class='section-head'><div><div class='eyebrow'>Decision desk</div><h2>Strong signals</h2></div><p>Latest persisted tick · {_esc(generated)}</p></div>{_cards(strong, 'No verified strong signals in the latest persisted tick.')}</section><section><div class='section-head'><div><div class='eyebrow'>Market mismatch</div><h2>Value plays</h2></div><p>Requires calibrated probability evidence.</p></div>{_cards(value, 'No verified value plays in the latest persisted tick.')}</section><section id='radar'><div class='section-head'><div><div class='eyebrow'>Readiness</div><h2>Market radar</h2></div><p>What is actionable vs what is still waiting.</p></div><div class='radar'><div class='radar-box'><div class='radar-num'>{_num(waiting_price.get('total'))}</div><strong>Waiting for price</strong><p>Model context exists, but a usable verified quote is still missing or stale.</p></div><div class='radar-box'><div class='radar-num'>{_num(waiting_xi.get('total'))}</div><strong>Waiting for XI</strong><p>Signal is held until lineup evidence satisfies the current product state.</p></div></div></section><section id='maturity'><div class='section-head'><div><div class='eyebrow'>Evidence</div><h2>Model maturity</h2></div><p>Read-only validation counters.</p></div><div class='maturity-grid'>{_maturity_cards(product_payload)}</div></section><section id='history'><div class='section-head'><div><div class='eyebrow'>Track record</div><h2>Verified performance history</h2></div><p>V216 will expose the settled ledger; nothing is backfilled here.</p></div><div class='history'><div><span>OOS framework</span><strong>{_esc(performance.get('oos_status'))}</strong></div><div><span>Promotion shadow</span><strong>{_esc(performance.get('promotion_status'))}</strong></div><div><span>Risk framework</span><strong>{_esc(performance.get('risk_status'))}</strong></div></div></section><section id='membership'><div class='section-head'><div><div class='eyebrow'>Membership</div><h2>Free vs Pro</h2></div><p>Entitlements are design-only in V215.</p></div><div class='plans'><article class='plan'><div class='eyebrow'>Free</div><h3>Explorer</h3><p class='lead'>Transparent public intelligence and product discovery.</p><ul>{_feature_list(FREE_FEATURES)}</ul><div class='locked'>Current preview behaves as a public read-only surface. No account is created.</div></article><article class='plan pro'><div class='eyebrow'>Pro</div><h3>Edge Pro</h3><p class='lead'>The future subscriber tier for deeper signal access and workflow tools.</p><ul>{_feature_list(PRO_FEATURES)}</ul><div class='locked'>Locked until V218 Auth + V219 Entitlements. Billing is not connected and no purchase can occur.</div></article></div></section><section><div class='section-head'><div><div class='eyebrow'>Public slate</div><h2>Today's verified rows</h2></div><p>{_num(today.get('total'))} total in current view.</p></div>{_cards(today, 'No verified active rows in the latest persisted tick.', limit=9)}</section><div class='footer'>Soccer Edge App V215 · PREVIEW_MODE · read-only. This surface adds zero provider requests and does not alter models, thresholds, gates, strict-close, budget or canonical BET logic. Authentication, billing and entitlements are explicitly disabled.</div></main></div></body></html>"""
