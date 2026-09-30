from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_COMMERCIAL_SHELL_V4_1.1.0"
PRODUCT_MODE = "ENTITLEMENT_READY"

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


def _features(items: tuple[str, ...]) -> str:
    return "".join(f"<li><span>✓</span>{_esc(item)}</li>" for item in items)


def render_membership_fragment(product_payload: dict[str, Any]) -> str:
    from mcp_gateway import subscription_entitlements_v4, supabase_auth_v4

    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    strong = views.get("strong_sport_signals") if isinstance(views.get("strong_sport_signals"), dict) else {}
    values = views.get("value_plays") if isinstance(views.get("value_plays"), dict) else {}
    waiting_price = views.get("waiting_for_price") if isinstance(views.get("waiting_for_price"), dict) else {}
    waiting_xi = views.get("waiting_for_xi") if isinstance(views.get("waiting_for_xi"), dict) else {}
    performance = views.get("performance") if isinstance(views.get("performance"), dict) else {}
    auth_enabled = bool(supabase_auth_v4.auth_config().get("configured"))
    contract = subscription_entitlements_v4.plan_contract()

    meta = html.escape(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "product_mode": PRODUCT_MODE,
        "auth_enabled": auth_enabled,
        "billing_enabled": False,
        "entitlements_enforced": bool(contract.get("entitlements_enforced")),
        "authorization_source": contract.get("authorization_source"),
        "provider_requests_added": 0,
    }, ensure_ascii=False), quote=True)

    return f"""
<style>
.v215-shell{{margin:28px 0 14px;border:1px solid #1b3448;background:linear-gradient(155deg,#0b1924,#0a1119);border-radius:14px;overflow:hidden}}
.v215-head{{padding:18px 18px 14px;display:flex;justify-content:space-between;gap:20px;align-items:flex-start;border-bottom:1px solid #172a3b}}
.v215-head h2{{font-size:22px;margin:4px 0 5px}}
.v215-preview{{display:inline-flex;align-items:center;border:1px solid #27516d;background:#0a2131;color:#bdeaff;padding:6px 9px;border-radius:999px;font-size:9px;font-weight:900;letter-spacing:.08em}}
.v215-copy{{color:#7890a5;font-size:12px;line-height:1.55;max-width:760px}}
.v215-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;padding:13px}}
.v215-metric{{background:#08141e;border:1px solid #153047;border-radius:10px;padding:13px}}
.v215-metric span{{display:block;color:#667f94;font-size:9px;text-transform:uppercase;letter-spacing:.08em}}
.v215-metric strong{{display:block;font-size:24px;margin-top:4px}}
.v215-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 13px 13px}}
.v215-plan{{background:#091722;border:1px solid #173247;border-radius:11px;padding:16px}}
.v215-plan.pro{{background:linear-gradient(160deg,#0b2030,#101523);border-color:#28506a}}
.v215-plan h3{{font-size:21px;margin:4px 0}}
.v215-plan p{{color:#7a91a5;font-size:11px;line-height:1.5}}
.v215-plan ul{{list-style:none;margin:13px 0;padding:0}}
.v215-plan li{{color:#9db1c2;font-size:11px;padding:6px 0}}
.v215-plan li span{{color:#44dca7;margin-right:7px}}
.v215-lock{{border:1px dashed #355069;border-radius:9px;padding:10px;color:#7690a4;font-size:10px;line-height:1.45}}
.v215-history{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;padding:0 13px 13px}}
.v215-history div{{background:#08141e;border:1px solid #153047;border-radius:10px;padding:12px}}
.v215-history span{{display:block;color:#607a90;font-size:9px;text-transform:uppercase}}
.v215-history strong{{display:block;margin-top:5px;font-size:11px}}
@media(max-width:760px){{.v215-metrics,.v215-history{{grid-template-columns:repeat(2,1fr)}}.v215-grid{{grid-template-columns:1fr}}}}
@media(max-width:520px){{.v215-metrics,.v215-history{{grid-template-columns:1fr}}.v215-head{{display:block}}.v215-preview{{margin-top:10px}}}}
</style>
<section class="v215-shell" id="membership" data-product="{meta}">
  <div class="v215-head">
    <div><div class="eyebrow">COMMERCIAL SHELL · V219</div><h2>Soccer Edge Membership</h2><div class="v215-copy">Supabase Auth is connected and the Free/Pro entitlement contract is enforced by server-side resolution plus database RLS. Billing is intentionally still disabled until V220. The persisted Soccer Edge payload remains authoritative and missing evidence is never inferred.</div></div>
    <span class="v215-preview">ENTITLEMENTS READY</span>
  </div>
  <div class="v215-metrics">
    <div class="v215-metric"><span>Strong signals</span><strong class="mono">{_num(strong.get('total'))}</strong></div>
    <div class="v215-metric"><span>Value plays</span><strong class="mono">{_num(values.get('total'))}</strong></div>
    <div class="v215-metric"><span>Waiting price</span><strong class="mono">{_num(waiting_price.get('total'))}</strong></div>
    <div class="v215-metric"><span>Waiting XI</span><strong class="mono">{_num(waiting_xi.get('total'))}</strong></div>
  </div>
  <div class="v215-grid">
    <article class="v215-plan"><div class="eyebrow">FREE</div><h3>Explorer</h3><p>Public read-only intelligence and transparent product discovery.</p><ul>{_features(FREE_FEATURES)}</ul><div class="v215-lock">Default account plan. Missing entitlement row, expired Pro or inactive billing state resolves to Free.</div></article>
    <article class="v215-plan pro"><div class="eyebrow">PRO</div><h3>Edge Pro</h3><p>Subscriber access for deeper signals and workflow tools.</p><ul>{_features(PRO_FEATURES)}</ul><div class="v215-lock">Requires a persisted ACTIVE or TRIALING Pro entitlement. Browser clients cannot grant or modify Pro. Checkout is added in V220.</div></article>
  </div>
  <div class="v215-history">
    <div><span>OOS framework</span><strong>{_esc(performance.get('oos_status'))}</strong></div>
    <div><span>Promotion shadow</span><strong>{_esc(performance.get('promotion_status'))}</strong></div>
    <div><span>Risk framework</span><strong>{_esc(performance.get('risk_status'))}</strong></div>
  </div>
</section>
"""
