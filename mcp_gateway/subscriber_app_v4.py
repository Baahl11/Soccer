from __future__ import annotations

import asyncio
import html
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4, subscription_entitlements_v4, supabase_auth_v4

SCHEMA_VERSION = "1.0.1"
MODEL_VERSION = "SOCCER_SUBSCRIBER_APP_V4_1.0.1"

_FREE_ROW_KEYS = ("fixture_id","kickoff","league","country","home_team","away_team","market_family","market","period","stage","status","execution_status","reason","blocker")
_ADVANCED_VIEW_KEYS = ("strong_sport_signals","value_plays","team_totals","first_half","second_half","corners","player_props","performance")


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
    return {"total": data.get("total", len(rows)), "rows": rows, "redacted_for_free": True}


def _maturity_overview(views: dict[str, Any]) -> list[dict[str, Any]]:
    families = _dict(_dict(_dict(views.get("control_tower")).get("maturity_snapshot")).get("maturation_control_tower")).get("families")
    if not isinstance(families, list):
        return []
    return [{k: family.get(k) for k in ("id","label","status","current","target")} for family in families if isinstance(family, dict)]


def anonymous_entitlement() -> dict[str, Any]:
    return {"ok":True,"status":"ANONYMOUS_FREE","authenticated":False,"effective_plan":subscription_entitlements_v4.FREE_PLAN,"effective_plan_reason":"PUBLIC_DEFAULT_FREE","feature_access":subscription_entitlements_v4.feature_access(subscription_entitlements_v4.FREE_PLAN),"billing_enabled":False,"entitlements_enforced":True,"provider_requests_added":0}


def build_subscriber_payload(product: dict[str, Any], entitlement: dict[str, Any]) -> dict[str, Any]:
    views = _dict(product.get("views"))
    plan = str(entitlement.get("effective_plan") or subscription_entitlements_v4.FREE_PLAN).upper()
    is_pro = plan == subscription_entitlements_v4.PRO_PLAN
    is_owner = bool(entitlement.get("owner")) or _dict(entitlement.get("user")).get("role") == "OWNER"
    is_admin = bool(entitlement.get("admin")) or is_owner
    public = {
        "verified_slate": _public_view(views.get("todays_slate")),
        "waiting_for_price": _public_view(views.get("waiting_for_price")),
        "waiting_for_xi": _public_view(views.get("waiting_for_xi")),
        "maturity_overview": _maturity_overview(views),
    }
    locked_counts = {key: _dict(views.get(key)).get("total", len(_rows(views.get(key)))) for key in _ADVANCED_VIEW_KEYS}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "SUBSCRIBER_APP_READY",
        "authenticated": bool(entitlement.get("authenticated")),
        "effective_plan": plan,
        "display_role": "OWNER" if is_owner else ("ADMIN" if is_admin else plan),
        "owner": is_owner,
        "admin": is_admin,
        "subscription_required": bool(entitlement.get("subscription_required", not is_owner)),
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
        result["pro"] = {key: views.get(key) for key in _ADVANCED_VIEW_KEYS if key in views}
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
        return JSONResponse({"error":"SUBSCRIBER_DATA_UNAVAILABLE","detail":str(exc)[:200]}, status_code=503)
    if not isinstance(product, dict):
        return JSONResponse({"error":"NO_PERSISTED_PIPELINE_RUN"}, status_code=503)
    return JSONResponse(build_subscriber_payload(product, entitlement))


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _app_html() -> str:
    auth = supabase_auth_v4.public_auth_config()
    cfg = _safe_json({
        "supabase_url": auth.get("project_url"),
        "publishable_key": auth.get("publishable_key"),
        "auth_status": auth.get("status"),
        "checkout_endpoint": f"{auth.get('project_url')}/functions/v1/create-checkout-session" if auth.get("project_url") else None,
        "portal_endpoint": f"{auth.get('project_url')}/functions/v1/create-customer-portal" if auth.get("project_url") else None,
    })
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Soccer Edge</title>
<style>:root{{--bg:#061019;--panel:#0a1722;--line:#18364b;--muted:#8298aa;--text:#eef7fb;--green:#5de3b0;--blue:#78c8ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0,#10293a 0,#061019 35%,#050b11 100%);color:var(--text);font-family:Inter,system-ui,sans-serif}}button,input{{font:inherit}}.wrap{{max-width:1180px;margin:auto;padding:22px}}.top{{display:flex;justify-content:space-between;align-items:center;padding:10px 0 22px}}.brand{{font-size:25px;font-weight:900}}.brand span{{color:var(--green)}}.badge{{padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#08141e;color:var(--blue);font-size:12px;font-weight:900}}.hero,.card{{border:1px solid var(--line);border-radius:18px;background:rgba(8,20,30,.9);padding:22px;margin-bottom:14px}}.hero h1{{font-size:40px;margin:8px 0}}.hero p,.muted,.status{{color:var(--muted)}}.eyebrow{{font-size:11px;letter-spacing:.12em;font-weight:900;color:var(--green)}}.actions,.form{{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}}.btn{{border:1px solid #24506b;background:#0d2535;color:#e8f7ff;padding:10px 13px;border-radius:10px;font-weight:800;cursor:pointer}}.btn.primary{{background:#0f5b43;border-color:#1d8a67}}input{{flex:1;min-width:180px;background:#07131d;border:1px solid #1b3a50;color:#fff;padding:11px;border-radius:10px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.full{{grid-column:1/-1}}.rows,.maturity{{display:grid;gap:8px}}.maturity{{grid-template-columns:repeat(3,1fr)}}.row,.mat,.lock{{border:1px solid #153348;border-radius:11px;padding:12px;background:#07141d}}.rowtop{{display:flex;justify-content:space-between;gap:8px}}.teams{{font-weight:800}}.meta{{font-size:11px;color:var(--muted);margin-top:5px}}.ownerbox{{display:none;border:1px solid #1d8a67;background:#0b2c24;border-radius:12px;padding:12px;margin-top:12px;color:#bdf9df}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}.full{{grid-column:auto}}.maturity{{grid-template-columns:1fr}}.hero h1{{font-size:34px}}}}</style></head>
<body><div class="wrap"><div class="top"><div class="brand">Soccer <span>Edge</span></div><div class="badge" id="planBadge">FREE</div></div>
<section class="hero"><div class="eyebrow">APP DE SUSCRIPTOR · V223</div><h1>Señales sin ruido.</h1><p>Cartelera verificada, estado del mercado, maduración y acceso premium según tu cuenta.</p><div id="ownerBox" class="ownerbox"><b>OWNER · ADMIN</b><div class="meta">Acceso interno completo · Edge Pro desbloqueado · No requiere suscripción.</div></div><div class="actions"><button class="btn primary" id="upgradeBtn">Mejorar a Edge Pro</button><button class="btn" id="portalBtn">Administrar suscripción</button><button class="btn" id="logoutBtn">Cerrar sesión</button></div></section>
<div class="grid"><section class="card full" id="accountCard"><div class="eyebrow">CUENTA</div><h2>Inicia sesión o crea una cuenta</h2><div class="form"><input id="email" type="email" placeholder="Correo"><input id="password" type="password" placeholder="Contraseña"><button class="btn" id="signinBtn">Iniciar sesión</button><button class="btn" id="signupBtn">Crear cuenta</button></div><div class="status" id="authStatus"></div></section>
<section class="card"><div class="eyebrow">HOY</div><h2>Cartelera verificada</h2><div id="slate" class="rows"></div></section><section class="card"><div class="eyebrow">RADAR</div><h2>Estado del mercado</h2><div id="radar" class="rows"></div></section><section class="card full"><div class="eyebrow">MADURACIÓN</div><h2>Estado de modelos</h2><div id="maturity" class="maturity"></div></section><section class="card full"><div class="eyebrow">EDGE PRO</div><h2>Inteligencia premium</h2><div id="pro" class="rows"></div></section></div></div>
<script id="appConfig" type="application/json">{cfg}</script><script>
const cfg=JSON.parse(document.getElementById('appConfig').textContent),$=id=>document.getElementById(id),tokenKey='soccer_edge_access_token';
const esc=v=>String(v??'N/V').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const token=()=>sessionStorage.getItem(tokenKey)||'';const setStatus=t=>$('authStatus').textContent=t||'';
function rowHtml(r){{const teams=[r.home_team,r.away_team].filter(Boolean).join(' vs ')||r.fixture_id||'Mercado';return `<div class="row"><div class="rowtop"><div class="teams">${{esc(teams)}}</div><div class="badge">${{esc(r.execution_status||r.status||r.stage||'')}}</div></div><div class="meta">${{esc(r.market_family||r.market||r.period||'')}} · ${{esc(r.kickoff||r.league||'')}}</div></div>`}}
function renderRows(id,node,empty='Sin filas verificadas ahora.'){{const rows=node?.rows||[];$(id).innerHTML=rows.length?rows.slice(0,12).map(rowHtml).join(''):`<div class="lock">${{esc(empty)}}</div>`}}
function render(data){{const owner=!!data.owner||data.display_role==='OWNER';$('planBadge').textContent=owner?'OWNER · ADMIN':(data.display_role||data.effective_plan||'FREE');$('ownerBox').style.display=owner?'block':'none';$('upgradeBtn').style.display=owner||data.effective_plan==='PRO'?'none':'inline-block';$('portalBtn').style.display=owner?'none':'inline-block';$('accountCard').style.display=data.authenticated?'none':'block';$('logoutBtn').style.display=data.authenticated?'inline-block':'none';renderRows('slate',data.public?.verified_slate);const wp=data.public?.waiting_for_price?.total??0,wx=data.public?.waiting_for_xi?.total??0;$('radar').innerHTML=`<div class="row"><b>${{esc(wp)}}</b><div class="meta">Esperando precio</div></div><div class="row"><b>${{esc(wx)}}</b><div class="meta">Esperando XI</div></div>`;const fam=data.public?.maturity_overview||[];$('maturity').innerHTML=fam.length?fam.map(f=>`<div class="mat"><b>${{esc(f.label||f.id)}}</b><div class="meta">${{esc(f.status)}} · ${{esc(f.current)}} / ${{esc(f.target)}}</div></div>`).join(''):'<div class="lock">Snapshot no disponible.</div>';if(data.premium_unlocked&&data.pro){{const chunks=[];for(const [k,v] of Object.entries(data.pro))if(v&&typeof v==='object'&&'total'in v)chunks.push(`<div class="row"><div class="rowtop"><b>${{esc(k.replaceAll('_',' '))}}</b><span class="badge">${{esc(v.total)}}</span></div><div class="meta">Desbloqueado</div></div>`);$('pro').innerHTML=chunks.join('')||'<div class="lock">Acceso premium activo; no hay filas premium ahora.</div>'}}else $('pro').innerHTML='<div class="lock">Edge Pro bloqueado.</div>';if(data.authenticated&&data.user?.email)setStatus(`Sesión: ${{data.user.email}} · ${{owner?'OWNER/ADMIN':data.effective_plan}}`)}}
async function loadData(){{const h={{}};if(token())h.Authorization=`Bearer ${{token()}}`;const r=await fetch('/app/data',{{headers:h}});if(r.status===401){{sessionStorage.removeItem(tokenKey);return loadData()}}const d=await r.json();if(!r.ok)throw new Error(d.error||'No se pudo cargar');render(d)}}
async function auth(path){{const email=$('email').value.trim(),password=$('password').value;if(!email||!password)throw new Error('Correo y contraseña requeridos');const r=await fetch(`${{cfg.supabase_url}}${{path}}`,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify({{email,password}})}}),d=await r.json();if(!r.ok)throw new Error(d.msg||d.error_description||d.message||'Error de autenticación');if(d.access_token){{sessionStorage.setItem(tokenKey,d.access_token);await loadData();return}}setStatus('Cuenta creada. Confirma tu correo si es necesario.')}}
$('signinBtn').onclick=async()=>{{try{{await auth('/auth/v1/token?grant_type=password')}}catch(e){{setStatus(e.message)}}}};$('signupBtn').onclick=async()=>{{try{{await auth('/auth/v1/signup')}}catch(e){{setStatus(e.message)}}}};$('logoutBtn').onclick=async()=>{{sessionStorage.removeItem(tokenKey);await loadData()}};
async function edge(endpoint){{if(!token())throw new Error('Inicia sesión primero');const r=await fetch(endpoint,{{method:'POST',headers:{{apikey:cfg.publishable_key,Authorization:`Bearer ${{token()}}`,'Content-Type':'application/json'}},body:'{{}}'}}),d=await r.json();if(!r.ok)throw new Error(d.error||d.detail||'Error de facturación');if(d.url)location.href=d.url}}
$('upgradeBtn').onclick=async()=>{{try{{await edge(cfg.checkout_endpoint)}}catch(e){{setStatus(e.message)}}}};$('portalBtn').onclick=async()=>{{try{{await edge(cfg.portal_endpoint)}}catch(e){{setStatus(e.message)}}}};loadData().catch(e=>setStatus(e.message));
</script></body></html>'''


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_app_html())
