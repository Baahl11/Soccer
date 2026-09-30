from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4, subscription_entitlements_v4, supabase_auth_v4

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_APP_V4_1.1.0"

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
    keys = ("key","id","label","status","current","target","unique_fixtures","evidence_kind","blocker","source")
    return [{k: family.get(k) for k in keys if k in family} for family in families if isinstance(family, dict)]


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
        "schema_version": SCHEMA_VERSION,"model_version": MODEL_VERSION,"status": "SUBSCRIBER_APP_READY",
        "authenticated": bool(entitlement.get("authenticated")),"effective_plan": plan,
        "display_role": "OWNER" if is_owner else ("ADMIN" if is_admin else plan),"owner": is_owner,"admin": is_admin,
        "subscription_required": bool(entitlement.get("subscription_required", not is_owner)),
        "effective_plan_reason": entitlement.get("effective_plan_reason"),
        "feature_access": entitlement.get("feature_access") or subscription_entitlements_v4.feature_access(plan),
        "generated_at_utc": product.get("generated_at_utc"),"pipeline_version": product.get("pipeline_version"),
        "public": public,"locked_counts": locked_counts,"provider_requests_added": 0,
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
        "supabase_url": auth.get("project_url"),"publishable_key": auth.get("publishable_key"),"auth_status": auth.get("status"),
        "checkout_endpoint": f"{auth.get('project_url')}/functions/v1/create-checkout-session" if auth.get("project_url") else None,
        "portal_endpoint": f"{auth.get('project_url')}/functions/v1/create-customer-portal" if auth.get("project_url") else None,
    })
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Soccer Edge Control Tower</title>
<style>:root{{--bg:#050a0f;--panel:#0a131c;--panel2:#0d1923;--line:#193143;--muted:#8294a3;--text:#f2f7fa;--green:#56e0ad;--blue:#6fc7ff;--amber:#f1c46b}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#08131d,#050a0f 45%);color:var(--text);font-family:Inter,system-ui,sans-serif}}button,input{{font:inherit}}.wrap{{max-width:1160px;margin:auto;padding:18px}}.top{{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;align-items:center;padding:14px 0;background:rgba(5,10,15,.92);backdrop-filter:blur(10px)}}.brand{{font-size:25px;font-weight:950}}.brand span,.eyebrow{{color:var(--green)}}.badge{{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:#08131c;color:var(--blue);font-size:12px;font-weight:900}}.hero,.card{{border:1px solid var(--line);border-radius:18px;background:rgba(10,19,28,.94);padding:20px;margin-bottom:14px}}.hero{{display:grid;grid-template-columns:1fr auto;gap:18px;align-items:center}}.hero h1{{font-size:32px;margin:5px 0}}.hero p,.muted,.status,.meta{{color:var(--muted)}}.eyebrow{{font-size:11px;letter-spacing:.12em;font-weight:900}}.actions,.form,.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.btn,.tab{{border:1px solid #25465c;background:#0d2230;color:#eaf7ff;padding:9px 12px;border-radius:10px;font-weight:800;cursor:pointer}}.tab.active{{border-color:#2c8d6c;color:#bdf9df;background:#0b2c24}}input{{flex:1;min-width:180px;background:#07131d;border:1px solid #1b3a50;color:#fff;padding:11px;border-radius:10px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.full{{grid-column:1/-1}}.rows,.maturity{{display:grid;gap:9px}}.maturity{{grid-template-columns:repeat(3,1fr)}}.row,.mat,.empty{{border:1px solid #173348;border-radius:12px;padding:13px;background:#07141d}}.rowtop{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}}.teams{{font-weight:850}}.meta{{font-size:12px;margin-top:6px;line-height:1.45}}.mat .progress{{height:6px;background:#102431;border-radius:99px;margin:10px 0;overflow:hidden}}.mat .progress i{{display:block;height:100%;background:var(--green)}}.blocker{{font-size:11px;color:var(--amber);margin-top:7px;word-break:break-word}}.ownerbox{{display:none;border:1px solid #1d8a67;background:#0b2c24;border-radius:12px;padding:11px;color:#bdf9df}}.detailgrid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:9px}}.kv{{background:#0b1b26;border-radius:8px;padding:7px 9px;font-size:11px;overflow-wrap:anywhere}}.kv b{{color:#9db1bf}}h2{{margin:4px 0 14px}}@media(max-width:760px){{.grid,.hero{{grid-template-columns:1fr}}.full{{grid-column:auto}}.maturity{{grid-template-columns:1fr}}.hero h1{{font-size:27px}}.detailgrid{{grid-template-columns:1fr}}}}</style></head>
<body><div class="wrap"><div class="top"><div class="brand">Soccer <span>Edge</span></div><div class="badge" id="planBadge">FREE</div></div>
<section class="hero"><div><div class="eyebrow">CONTROL TOWER · V224</div><h1>Lo importante, primero.</h1><p>Partidos, señales, valor y maduración con evidencia verificable.</p></div><div><div id="ownerBox" class="ownerbox"><b>OWNER · ADMIN</b><div class="meta">Acceso completo · sin suscripción.</div></div><div class="actions"><button class="btn" id="upgradeBtn">Edge Pro</button><button class="btn" id="portalBtn">Suscripción</button><button class="btn" id="logoutBtn">Cerrar sesión</button></div></div></section>
<section class="card" id="accountCard"><div class="eyebrow">CUENTA</div><h2>Acceso</h2><div class="form"><input id="email" type="email" placeholder="Correo"><input id="password" type="password" placeholder="Contraseña"><button class="btn" id="signinBtn">Iniciar sesión</button><button class="btn" id="signupBtn">Crear cuenta</button></div><div class="status" id="authStatus"></div></section>
<div class="grid"><section class="card"><div class="eyebrow">HOY</div><h2>Cartelera</h2><div id="slate" class="rows"></div></section><section class="card"><div class="eyebrow">RADAR</div><h2>Mercado</h2><div id="radar" class="rows"></div></section><section class="card full"><div class="eyebrow">MADURACIÓN</div><h2>Roadmap de evidencia</h2><div id="maturity" class="maturity"></div></section><section class="card full"><div class="eyebrow">EDGE PRO</div><h2>Inteligencia premium</h2><div id="proTabs" class="tabs"></div><div id="pro" class="rows"></div></section></div></div>
<script id="appConfig" type="application/json">{cfg}</script><script>
const cfg=JSON.parse(document.getElementById('appConfig').textContent),$=id=>document.getElementById(id),AK='soccer_edge_access_token',RK='soccer_edge_refresh_token';
const esc=v=>String(v??'N/V').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const token=()=>localStorage.getItem(AK)||'',refreshToken=()=>localStorage.getItem(RK)||'';const setStatus=t=>$('authStatus').textContent=t||'';
function saveSession(d){{if(d.access_token)localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token)}}function clearSession(){{localStorage.removeItem(AK);localStorage.removeItem(RK)}}
function usefulEntries(r){{const skip=new Set(['home_team','away_team','fixture_id']);return Object.entries(r||{{}}).filter(([k,v])=>!skip.has(k)&&v!==null&&v!==''&&typeof v!=='object').slice(0,10)}}
function rowHtml(r){{const teams=[r.home_team||r.home,r.away_team||r.away].filter(Boolean).join(' vs ')||r.fixture_id||r.market||r.selection||'Señal';const status=r.execution_status||r.status||r.stage||r.classification||'';const kv=usefulEntries(r).map(([k,v])=>`<div class="kv"><b>${{esc(k.replaceAll('_',' '))}}</b><br>${{esc(v)}}</div>`).join('');return `<div class="row"><div class="rowtop"><div class="teams">${{esc(teams)}}</div>${{status?`<div class="badge">${{esc(status)}}</div>`:''}}</div>${{kv?`<div class="detailgrid">${{kv}}</div>`:''}}</div>`}}
function renderRows(id,node,empty='Sin evidencia disponible ahora.'){{const rows=node?.rows||[];$(id).innerHTML=rows.length?rows.slice(0,20).map(rowHtml).join(''):`<div class="empty">${{esc(empty)}}</div>`}}
let currentData=null,currentProKey='strong_sport_signals';function renderProKey(k){{currentProKey=k;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.k===k));const n=currentData?.pro?.[k];renderRows('pro',n,`No hay filas verificadas en ${{k.replaceAll('_',' ')}} en este snapshot.`)}}
function render(data){{currentData=data;const owner=!!data.owner||data.display_role==='OWNER';$('planBadge').textContent=owner?'OWNER · ADMIN':(data.display_role||data.effective_plan||'FREE');$('ownerBox').style.display=owner?'block':'none';$('upgradeBtn').style.display=owner||data.effective_plan==='PRO'?'none':'inline-block';$('portalBtn').style.display=owner?'none':'inline-block';$('accountCard').style.display=data.authenticated?'none':'block';$('logoutBtn').style.display=data.authenticated?'inline-block':'none';renderRows('slate',data.public?.verified_slate);const wp=data.public?.waiting_for_price?.total??0,wx=data.public?.waiting_for_xi?.total??0;$('radar').innerHTML=`<div class="row"><b>${{esc(wp)}}</b><div class="meta">Esperando precio</div></div><div class="row"><b>${{esc(wx)}}</b><div class="meta">Esperando XI</div></div>`;const fam=data.public?.maturity_overview||[];$('maturity').innerHTML=fam.length?fam.map(f=>{{const pct=f.target?Math.min(100,Math.round((Number(f.current||0)/Number(f.target))*100)):0;return `<div class="mat"><div class="rowtop"><b>${{esc(f.label||f.id||f.key)}}</b><span class="badge">${{esc(f.status)}}</span></div><div class="progress"><i style="width:${{pct}}%"></i></div><div class="meta">${{esc(f.current)}} / ${{esc(f.target)}} · ${{pct}}%${{f.unique_fixtures!=null?` · ${{esc(f.unique_fixtures)}} fixtures`:''}}</div>${{f.blocker?`<div class="blocker">Bloqueo: ${{esc(f.blocker)}}</div>`:''}}</div>`}}).join(''):'<div class="empty">Snapshot no disponible.</div>';if(data.premium_unlocked&&data.pro){{const keys=Object.keys(data.pro).filter(k=>data.pro[k]&&typeof data.pro[k]==='object'&&('rows'in data.pro[k]||'total'in data.pro[k]));$('proTabs').innerHTML=keys.map(k=>`<button class="tab" data-k="${{esc(k)}}">${{esc(k.replaceAll('_',' '))}} · ${{esc(data.pro[k]?.total??data.pro[k]?.rows?.length??0)}}</button>`).join('');$('proTabs').querySelectorAll('.tab').forEach(b=>b.onclick=()=>renderProKey(b.dataset.k));if(!keys.includes(currentProKey))currentProKey=keys[0];if(currentProKey)renderProKey(currentProKey)}}else{{$('proTabs').innerHTML='';$('pro').innerHTML='<div class="empty">Edge Pro bloqueado.</div>'}}}}
async function refreshSession(){{if(!refreshToken())return false;const r=await fetch(`${{cfg.supabase_url}}/auth/v1/token?grant_type=refresh_token`,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify({{refresh_token:refreshToken()}})}}),d=await r.json();if(!r.ok||!d.access_token){{clearSession();return false}}saveSession(d);return true}}
async function loadData(retry=true){{const h={{}};if(token())h.Authorization=`Bearer ${{token()}}`;const r=await fetch('/app/data',{{headers:h,cache:'no-store'}});if(r.status===401&&retry&&await refreshSession())return loadData(false);if(r.status===401)clearSession();const d=await r.json();if(!r.ok&&r.status!==401)throw new Error(d.error||'No se pudo cargar');if(r.status===401)return loadData(false);render(d)}}
async function auth(path){{const email=$('email').value.trim(),password=$('password').value;if(!email||!password)throw new Error('Correo y contraseña requeridos');const r=await fetch(`${{cfg.supabase_url}}${{path}}`,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify({{email,password}})}}),d=await r.json();if(!r.ok)throw new Error(d.msg||d.error_description||d.message||'Error de autenticación');if(d.access_token){{saveSession(d);await loadData();return}}setStatus('Cuenta creada. Confirma tu correo si es necesario.')}}
$('signinBtn').onclick=async()=>{{try{{await auth('/auth/v1/token?grant_type=password')}}catch(e){{setStatus(e.message)}}}};$('signupBtn').onclick=async()=>{{try{{await auth('/auth/v1/signup')}}catch(e){{setStatus(e.message)}}}};$('logoutBtn').onclick=async()=>{{clearSession();await loadData()}};
async function edge(endpoint){{if(!token())throw new Error('Inicia sesión primero');const r=await fetch(endpoint,{{method:'POST',headers:{{apikey:cfg.publishable_key,Authorization:`Bearer ${{token()}}`,'Content-Type':'application/json'}},body:'{{}}'}}),d=await r.json();if(!r.ok)throw new Error(d.error||d.detail||'Error de facturación');if(d.url)location.href=d.url}}$('upgradeBtn').onclick=async()=>{{try{{await edge(cfg.checkout_endpoint)}}catch(e){{setStatus(e.message)}}}};$('portalBtn').onclick=async()=>{{try{{await edge(cfg.portal_endpoint)}}catch(e){{setStatus(e.message)}}}};loadData().catch(e=>setStatus(e.message));
</script></body></html>'''


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_app_html())
