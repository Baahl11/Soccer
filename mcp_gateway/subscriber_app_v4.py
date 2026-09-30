from __future__ import annotations

import asyncio
import html
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4, subscription_entitlements_v4, supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_APP_V4_1.0.0"

_FREE_ROW_KEYS = (
    "fixture_id",
    "kickoff",
    "league",
    "country",
    "home_team",
    "away_team",
    "market_family",
    "market",
    "period",
    "stage",
    "status",
    "execution_status",
    "reason",
    "blocker",
)
_ADVANCED_VIEW_KEYS = (
    "strong_sport_signals",
    "value_plays",
    "team_totals",
    "first_half",
    "second_half",
    "corners",
    "player_props",
    "performance",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(node: Any) -> list[dict[str, Any]]:
    value = _dict(node).get("rows")
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _free_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in _FREE_ROW_KEYS if key in row}


def _public_view(node: Any, limit: int = 20) -> dict[str, Any]:
    data = _dict(node)
    rows = [_free_row(row) for row in _rows(data)[:limit]]
    return {
        "total": data.get("total", len(rows)),
        "rows": rows,
        "redacted_for_free": True,
    }


def _maturity_overview(views: dict[str, Any]) -> list[dict[str, Any]]:
    tower = _dict(views.get("control_tower"))
    snapshot = _dict(tower.get("maturity_snapshot"))
    control = _dict(snapshot.get("maturation_control_tower"))
    families = control.get("families")
    if not isinstance(families, list):
        return []
    result: list[dict[str, Any]] = []
    for family in families:
        if not isinstance(family, dict):
            continue
        result.append({
            "id": family.get("id"),
            "label": family.get("label"),
            "status": family.get("status"),
            "current": family.get("current"),
            "target": family.get("target"),
        })
    return result


def anonymous_entitlement() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ANONYMOUS_FREE",
        "authenticated": False,
        "effective_plan": subscription_entitlements_v4.FREE_PLAN,
        "effective_plan_reason": "PUBLIC_DEFAULT_FREE",
        "feature_access": subscription_entitlements_v4.feature_access(subscription_entitlements_v4.FREE_PLAN),
        "billing_enabled": False,
        "entitlements_enforced": True,
        "provider_requests_added": 0,
    }


def build_subscriber_payload(product: dict[str, Any], entitlement: dict[str, Any]) -> dict[str, Any]:
    views = _dict(product.get("views"))
    plan = str(entitlement.get("effective_plan") or subscription_entitlements_v4.FREE_PLAN).upper()
    is_pro = plan == subscription_entitlements_v4.PRO_PLAN

    public = {
        "verified_slate": _public_view(views.get("todays_slate")),
        "waiting_for_price": _public_view(views.get("waiting_for_price")),
        "waiting_for_xi": _public_view(views.get("waiting_for_xi")),
        "maturity_overview": _maturity_overview(views),
    }
    locked_counts = {
        key: _dict(views.get(key)).get("total", len(_rows(views.get(key))))
        for key in _ADVANCED_VIEW_KEYS
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "SUBSCRIBER_APP_READY",
        "authenticated": bool(entitlement.get("authenticated")),
        "effective_plan": plan,
        "effective_plan_reason": entitlement.get("effective_plan_reason"),
        "feature_access": entitlement.get("feature_access") or subscription_entitlements_v4.feature_access(plan),
        "generated_at_utc": product.get("generated_at_utc"),
        "pipeline_version": product.get("pipeline_version"),
        "public": public,
        "locked_counts": locked_counts,
        "provider_requests_added": 0,
    }
    if entitlement.get("user"):
        result["user"] = entitlement.get("user")

    if is_pro:
        result["pro"] = {
            key: views.get(key)
            for key in _ADVANCED_VIEW_KEYS
            if key in views
        }
        result["pro"]["todays_slate"] = views.get("todays_slate")
        result["premium_unlocked"] = True
    else:
        result["pro"] = None
        result["premium_unlocked"] = False
    return result


async def _load_product() -> dict[str, Any] | None:
    payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    if not isinstance(payload, dict):
        return None
    payload.setdefault("status", "ok")
    payload["database_persisted"] = True
    payload["database_error"] = None
    result = product_views_v4.build_views(payload, limit=product_views_v4.MAX_ROWS_PER_VIEW)
    result["generated_at_utc"] = payload.get("generated_at_utc")
    result["pipeline_version"] = payload.get("version")
    result["source"] = "POSTGRES_LATEST_PIPELINE_RUN"
    return result


async def app_data(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if token:
        entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
        if not entitlement.get("ok") and not entitlement.get("authenticated"):
            return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)
    else:
        entitlement = anonymous_entitlement()

    try:
        product = await _load_product()
    except Exception as exc:
        return JSONResponse({"error": "SUBSCRIBER_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(product, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)
    return JSONResponse(build_subscriber_payload(product, entitlement))


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _app_html() -> str:
    auth = supabase_auth_v4.public_auth_config()
    config = {
        "supabase_url": auth.get("project_url"),
        "publishable_key": auth.get("publishable_key"),
        "auth_status": auth.get("status"),
        "checkout_endpoint": f"{auth.get('project_url')}/functions/v1/create-checkout-session" if auth.get("project_url") else None,
        "portal_endpoint": f"{auth.get('project_url')}/functions/v1/create-customer-portal" if auth.get("project_url") else None,
    }
    cfg = _safe_json(config)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Soccer Edge</title>
<style>
:root{{--bg:#061019;--panel:#0a1722;--panel2:#0d1e2b;--line:#18364b;--muted:#7590a4;--text:#eef7fb;--green:#5de3b0;--blue:#78c8ff;--amber:#f2c66d}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0,#10293a 0,#061019 34%,#050b11 100%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh}}button,input{{font:inherit}}a{{color:inherit}}
.wrap{{max-width:1180px;margin:auto;padding:22px}}.top{{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:10px 0 22px}}.brand{{font-size:24px;font-weight:900;letter-spacing:-.03em}}.brand span{{color:var(--green)}}.badge{{padding:7px 10px;border:1px solid var(--line);border-radius:999px;font-size:11px;font-weight:800;background:#08141e;color:var(--blue)}}
.hero{{border:1px solid var(--line);border-radius:18px;background:linear-gradient(150deg,#0c2130,#09131d);padding:26px;margin-bottom:14px}}.hero h1{{font-size:42px;line-height:1;margin:8px 0 12px;letter-spacing:-.05em}}.hero p{{color:#8aa3b5;max-width:760px;line-height:1.55}}.eyebrow{{font-size:10px;letter-spacing:.13em;font-weight:900;color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}}.card{{grid-column:span 6;border:1px solid var(--line);border-radius:14px;background:rgba(8,20,30,.88);padding:16px}}.card.full{{grid-column:1/-1}}.card.third{{grid-column:span 4}}.card h2{{font-size:16px;margin:5px 0 12px}}.muted{{color:var(--muted);font-size:12px;line-height:1.55}}.actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.btn{{border:1px solid #24506b;background:#0d2535;color:#dff5ff;padding:9px 12px;border-radius:9px;font-size:11px;font-weight:800;cursor:pointer}}.btn.primary{{background:#0f5b43;border-color:#1d8a67;color:#effff8}}.btn:disabled{{opacity:.45;cursor:not-allowed}}
.form{{display:grid;grid-template-columns:1fr 1fr auto auto;gap:8px}}input{{background:#07131d;border:1px solid #1b3a50;color:#eaf6fb;padding:10px 11px;border-radius:9px;min-width:0}}.status{{margin-top:9px;color:var(--muted);font-size:11px;min-height:16px}}.metric{{font-size:28px;font-weight:900}}.rows{{display:grid;gap:8px}}.row{{border:1px solid #143044;background:#07131c;border-radius:10px;padding:11px}}.rowtop{{display:flex;justify-content:space-between;gap:10px}}.teams{{font-weight:800;font-size:12px}}.meta{{color:#6f8a9e;font-size:10px;margin-top:5px;line-height:1.45}}.lock{{border:1px dashed #3f5260;color:#8aa0b1;border-radius:10px;padding:11px;font-size:11px}}.maturity{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.mat{{border:1px solid #153348;border-radius:9px;padding:10px;background:#07141d}}.mat b{{display:block;font-size:11px}}.mat span{{font-size:10px;color:#7892a5}}.notice{{padding:10px 12px;border:1px solid #5d4a22;background:#2c2413;color:#f3d587;border-radius:9px;font-size:11px;margin-bottom:12px;display:none}}
@media(max-width:800px){{.card,.card.third{{grid-column:1/-1}}.form{{grid-template-columns:1fr}}.hero h1{{font-size:34px}}.maturity{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.wrap{{padding:14px}}.top{{align-items:flex-start}}.maturity{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<div class="top"><div class="brand">Soccer <span>Edge</span></div><div class="badge" id="planBadge">FREE</div></div>
<div id="checkoutNotice" class="notice"></div>
<section class="hero"><div class="eyebrow">SUBSCRIBER APP · V220</div><h1>Signals without the noise.</h1><p>Verified slate, market readiness and model maturity are available on Explorer. Edge Pro unlocks strong signals, calibrated value, advanced markets, premium match detail and verified performance.</p><div class="actions"><button class="btn primary" id="upgradeBtn">Upgrade to Edge Pro</button><button class="btn" id="portalBtn">Manage subscription</button><button class="btn" id="logoutBtn">Sign out</button></div></section>
<div class="grid">
<section class="card full"><div class="eyebrow">ACCOUNT</div><h2>Sign in or create an account</h2><div class="form"><input id="email" type="email" placeholder="Email"><input id="password" type="password" placeholder="Password"><button class="btn" id="signinBtn">Sign in</button><button class="btn" id="signupBtn">Create account</button></div><div class="status" id="authStatus"></div></section>
<section class="card"><div class="eyebrow">TODAY</div><h2>Verified slate</h2><div id="slate" class="rows"></div></section>
<section class="card"><div class="eyebrow">RADAR</div><h2>Market readiness</h2><div id="radar" class="rows"></div></section>
<section class="card full"><div class="eyebrow">MATURITY</div><h2>Model maturity</h2><div id="maturity" class="maturity"></div></section>
<section class="card full"><div class="eyebrow">EDGE PRO</div><h2>Premium intelligence</h2><div id="pro" class="rows"></div></section>
</div></div>
<script id="appConfig" type="application/json">{cfg}</script>
<script>
const cfg=JSON.parse(document.getElementById('appConfig').textContent);const $=id=>document.getElementById(id);const tokenKey='soccer_edge_access_token';
const esc=v=>String(v??'N/V').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function token(){{return sessionStorage.getItem(tokenKey)||''}}function setStatus(t){{$('authStatus').textContent=t||''}}
function rowHtml(r){{const teams=[r.home_team,r.away_team].filter(Boolean).join(' vs ')||r.fixture_id||'Market';const market=r.market_family||r.market||r.period||'';const st=r.execution_status||r.status||r.stage||'';return `<div class="row"><div class="rowtop"><div class="teams">${{esc(teams)}}</div><div class="badge">${{esc(st)}}</div></div><div class="meta">${{esc(market)}} · ${{esc(r.kickoff||r.league||'')}}</div></div>`}}
function renderRows(id,node,empty='No verified rows right now.'){{const rows=node?.rows||[];$(id).innerHTML=rows.length?rows.slice(0,12).map(rowHtml).join(''):`<div class="lock">${{esc(empty)}}</div>`}}
function render(data){{$('planBadge').textContent=data.effective_plan||'FREE';renderRows('slate',data.public?.verified_slate);const wp=data.public?.waiting_for_price?.total??0,wx=data.public?.waiting_for_xi?.total??0;$('radar').innerHTML=`<div class="row"><div class="metric">${{esc(wp)}}</div><div class="meta">Waiting for price</div></div><div class="row"><div class="metric">${{esc(wx)}}</div><div class="meta">Waiting for XI</div></div>`;const fam=data.public?.maturity_overview||[];$('maturity').innerHTML=fam.length?fam.map(f=>`<div class="mat"><b>${{esc(f.label||f.id)}}</b><span>${{esc(f.status)}} · ${{esc(f.current)}} / ${{esc(f.target)}}</span></div>`).join(''):'<div class="lock">Maturity snapshot unavailable.</div>';if(data.premium_unlocked&&data.pro){{const chunks=[];for(const [k,v] of Object.entries(data.pro)){{if(v&&typeof v==='object'&&'total'in v)chunks.push(`<div class="row"><div class="rowtop"><div class="teams">${{esc(k.replaceAll('_',' '))}}</div><div class="badge">${{esc(v.total)}}</div></div><div class="meta">Unlocked for Edge Pro</div></div>`)}}$('pro').innerHTML=chunks.join('')||'<div class="lock">Pro is active. No premium rows are available right now.</div>'}}else{{const counts=data.locked_counts||{};$('pro').innerHTML=`<div class="lock">Edge Pro is locked. Strong signals: ${{esc(counts.strong_sport_signals??0)}} · Value plays: ${{esc(counts.value_plays??0)}}. Sign in and activate Pro to unlock row-level details.</div>`}}$('logoutBtn').style.display=data.authenticated?'inline-block':'none';$('portalBtn').disabled=!data.authenticated;$('upgradeBtn').disabled=!data.authenticated;}}
async function loadData(){{const h={{}};if(token())h.Authorization=`Bearer ${{token()}}`;const r=await fetch('/app/data',{{headers:h}});if(r.status===401){{sessionStorage.removeItem(tokenKey);setStatus('Session expired. Sign in again.');return loadData()}}const d=await r.json();render(d);if(d.authenticated&&d.user?.email)setStatus(`Signed in as ${{d.user.email}} · ${{d.effective_plan}}`)}}
async function auth(path){{if(!cfg.supabase_url||!cfg.publishable_key)throw new Error('AUTH_NOT_CONFIGURED');const email=$('email').value.trim(),password=$('password').value;if(!email||!password)throw new Error('Email and password are required');const r=await fetch(`${{cfg.supabase_url}}${{path}}`,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify({{email,password}})}});const d=await r.json();if(!r.ok)throw new Error(d.msg||d.error_description||d.message||'Authentication failed');if(d.access_token){{sessionStorage.setItem(tokenKey,d.access_token);await loadData();return 'Signed in.'}}return 'Account created. Check your email if confirmation is required.'}}
$('signinBtn').onclick=async()=>{{try{{setStatus(await auth('/auth/v1/token?grant_type=password'))}}catch(e){{setStatus(e.message)}}}};$('signupBtn').onclick=async()=>{{try{{setStatus(await auth('/auth/v1/signup'))}}catch(e){{setStatus(e.message)}}}};$('logoutBtn').onclick=async()=>{{sessionStorage.removeItem(tokenKey);setStatus('Signed out.');await loadData()}};
async function edge(endpoint){{if(!token())throw new Error('Sign in first.');const r=await fetch(endpoint,{{method:'POST',headers:{{apikey:cfg.publishable_key,Authorization:`Bearer ${{token()}}`,'Content-Type':'application/json'}},body:'{{}}'}});const d=await r.json();if(!r.ok)throw new Error(d.error==='BILLING_NOT_CONFIGURED'?'Billing is staged but Stripe credentials are not connected yet.':d.error||d.detail||'Billing request failed');if(d.url)location.href=d.url;}}
$('upgradeBtn').onclick=async()=>{{try{{await edge(cfg.checkout_endpoint)}}catch(e){{setStatus(e.message)}}}};$('portalBtn').onclick=async()=>{{try{{await edge(cfg.portal_endpoint)}}catch(e){{setStatus(e.message)}}}};
const qp=new URLSearchParams(location.search);if(qp.get('checkout')){{$('checkoutNotice').style.display='block';$('checkoutNotice').textContent=qp.get('checkout')==='success'?'Checkout completed. Your Pro entitlement will appear after the signed webhook is processed.':'Checkout canceled. No plan change was made.'}}loadData().catch(e=>setStatus(e.message));
</script></body></html>"""


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_app_html())
