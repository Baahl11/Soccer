from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import subscriber_app_v4
from mcp_gateway import subscriber_product_v234
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PRODUCT_V235_1.1.0"

_PUBLIC_PRESENTATION_KEYS = (
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


def _owner_view_entitlement(entitlement: dict[str, Any]) -> dict[str, Any]:
    out = dict(entitlement)
    user = out.get("user") if isinstance(out.get("user"), dict) else {}
    owner = bool(out.get("owner")) or str(user.get("role") or "").upper() == "OWNER"
    admin = bool(out.get("admin")) or owner or str(user.get("role") or "").upper() == "ADMIN"
    if owner or admin:
        out["owner"] = owner
        out["admin"] = admin
        out["effective_plan"] = subscription_entitlements_v4.PRO_PLAN
        out["effective_plan_reason"] = "OWNER_ADMIN_PRESENTATION_ACCESS"
        out["feature_access"] = subscription_entitlements_v4.feature_access(subscription_entitlements_v4.PRO_PLAN)
    return out


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(node: Any) -> list[dict[str, Any]]:
    rows = _dict(node).get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _team_identity(row: dict[str, Any]) -> dict[str, Any]:
    teams = _dict(row.get("teams"))
    home_meta = _dict(teams.get("home"))
    away_meta = _dict(teams.get("away"))

    home_id = _positive_int(_first(row, "home_team_id", "home_id")) or _positive_int(home_meta.get("id"))
    away_id = _positive_int(_first(row, "away_team_id", "away_id")) or _positive_int(away_meta.get("id"))

    home_logo = _first(row, "home_team_logo", "home_logo", "home_team_logo_url") or home_meta.get("logo")
    away_logo = _first(row, "away_team_logo", "away_logo", "away_team_logo_url") or away_meta.get("logo")

    if not (isinstance(home_logo, str) and home_logo.startswith(("https://", "http://"))) and home_id:
        home_logo = f"https://media.api-sports.io/football/teams/{home_id}.png"
    if not (isinstance(away_logo, str) and away_logo.startswith(("https://", "http://"))) and away_id:
        away_logo = f"https://media.api-sports.io/football/teams/{away_id}.png"

    out: dict[str, Any] = {}
    if home_id:
        out["home_team_id"] = home_id
    if away_id:
        out["away_team_id"] = away_id
    if isinstance(home_logo, str) and home_logo.startswith(("https://", "http://")):
        out["home_team_logo"] = home_logo
    if isinstance(away_logo, str) and away_logo.startswith(("https://", "http://")):
        out["away_team_logo"] = away_logo
    return out


def _public_fixture_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {key: row.get(key) for key in _PUBLIC_PRESENTATION_KEYS if key in row}
    public.update(_team_identity(row))
    return public


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("fixture_id") or ""),
        str(row.get("kickoff") or ""),
        str(row.get("home_team") or row.get("home") or ""),
        str(row.get("away_team") or row.get("away") or ""),
    )


def _enrich_rows(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_row_key(row): row for row in raw_rows}
    by_fixture = {
        str(row.get("fixture_id")): row
        for row in raw_rows
        if row.get("fixture_id") not in (None, "")
    }
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        source = by_fixture.get(str(row.get("fixture_id"))) if row.get("fixture_id") not in (None, "") else None
        source = source or by_key.get(_row_key(row)) or row
        out.update(_team_identity(source))
        enriched.append(out)
    return enriched


def _build_payload(product: dict[str, Any], entitlement: dict[str, Any]) -> dict[str, Any]:
    payload = subscriber_app_v4.build_subscriber_payload(product, entitlement)
    views = _dict(product.get("views"))
    raw_slate = _dict(views.get("todays_slate"))
    raw_rows = _rows(raw_slate)

    public = _dict(payload.get("public"))
    verified = _dict(public.get("verified_slate"))
    # V235.1: the public slate is a browse surface, not a premium signal surface.
    # Expose every persisted fixture row while keeping model/price/edge fields redacted.
    verified["rows"] = [_public_fixture_row(row) for row in raw_rows]
    verified["total"] = raw_slate.get("total", len(raw_rows))
    verified["redacted_for_free"] = True
    verified["full_slate"] = True
    public["verified_slate"] = verified
    payload["public"] = public

    pro = payload.get("pro")
    if isinstance(pro, dict):
        todays = _dict(pro.get("todays_slate"))
        todays["rows"] = _enrich_rows(_rows(todays), raw_rows)
        todays["total"] = todays.get("total", raw_slate.get("total", len(todays["rows"])))
        pro["todays_slate"] = todays

    payload["frontend_version"] = "V235_FULL_SLATE_LOGOS_1.1.0"
    payload["provider_requests_added"] = 0
    return payload


def _visual_fragment() -> str:
    return r'''
<style id="SOCCER_V235_VISUAL_MOBILE_STYLE">
:root{--v235-safe-bottom:max(12px,env(safe-area-inset-bottom));--v235-safe-top:max(10px,env(safe-area-inset-top))}
#v233AccountDock{top:max(14px,var(--v235-safe-top));right:18px}.header{padding-right:118px}.panel,.hero-main,.market-card,.score-card,.filter-card{backdrop-filter:saturate(110%)}
button,.tab,.toolbtn,.v233-account-btn,.v233-primary,.v233-secondary{-webkit-tap-highlight-color:transparent;transition:border-color .14s ease,background .14s ease,color .14s ease,transform .14s ease}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #58bbff;outline-offset:2px}.toolbtn:hover,.tab:hover{border-color:#347394;color:#dcecf4}.hero-main[data-v233-clickable="1"]:active,#feedbody tr:active{transform:translateY(1px)}
#feed .panel.scroll{overflow:auto;scrollbar-width:thin;scrollbar-color:#1d4b66 transparent}.table th:first-child,.table td:first-child{position:sticky;left:0;z-index:2;background:#081a27}.table th:first-child{z-index:3;background:#091d2b}.table tbody tr:hover td{background-color:#0a2030}.table tbody tr:hover td:first-child{background:#0a2030}
.tabs{overflow-x:auto;flex-wrap:nowrap;padding-bottom:3px;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}.tab{flex:0 0 auto}.match-head{gap:14px}.teams-head{min-width:0}.teams-head>div:not(.crest){min-width:0}.match-name{overflow-wrap:anywhere}.quality{min-width:110px}
#myedge .signal{align-items:center}.v231-action{min-width:32px;min-height:32px}.placeholder,.v231-empty,.v234-empty,.v233-free-lock{background:linear-gradient(180deg,#081923,#07131c)}
.mobile-nav{gap:4px;justify-content:flex-start;overflow-x:auto;overscroll-behavior-x:contain;scroll-snap-type:x proximity;padding:7px 8px calc(7px + env(safe-area-inset-bottom));scrollbar-width:none}.mobile-nav::-webkit-scrollbar{display:none}.mobile-nav button{flex:0 0 auto;min-width:68px;min-height:42px;padding:6px 9px;border-radius:8px;scroll-snap-align:start}.mobile-nav button.active{background:#0b2b3d;color:var(--green)}
.v235-mobile-lab{border-left:1px solid #214052!important}.v235-mobile-account{color:#58bbff!important}
.v235-slate-panel{padding:0!important;overflow:hidden}.v235-slate-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 15px;border-bottom:1px solid #173448}.v235-slate-head h3{margin:0;font-size:13px}.v235-slate-count{display:inline-flex;align-items:center;gap:6px;border:1px solid #20506a;background:#092233;color:#69c8ff;border-radius:999px;padding:5px 8px;font-size:8px;font-weight:900;white-space:nowrap}.v235-slate-list{display:grid}.v235-fixture{display:grid;grid-template-columns:68px minmax(0,1fr) auto;align-items:center;gap:11px;padding:11px 14px;border-bottom:1px solid #102b3b;min-height:66px}.v235-fixture:last-child{border-bottom:0}.v235-kickoff{color:#7f98a9;font-size:9px;line-height:1.35}.v235-kickoff b{display:block;color:#c9dbe5;font-size:11px}.v235-teams{display:grid;gap:6px;min-width:0}.v235-team{display:flex;align-items:center;gap:7px;min-width:0;font-weight:800;font-size:10px}.v235-team-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.v235-logo-wrap{width:25px;height:25px;flex:0 0 25px;border-radius:50%;background:#0c2635;border:1px solid #1b4054;display:grid;place-items:center;overflow:hidden}.v235-logo-wrap img{width:21px;height:21px;object-fit:contain}.v235-logo-fallback{font-size:8px;font-weight:950;color:#8bb5ca}.v235-fixture-meta{text-align:right;display:grid;gap:5px;justify-items:end}.v235-league{max-width:130px;color:#6f899b;font-size:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v235-state{border:1px solid #24516a;border-radius:999px;padding:4px 7px;color:#70cfff;background:#092132;font-size:7px;font-weight:900;text-transform:uppercase;white-space:nowrap}.v235-state.ready{border-color:#21624e;color:#68e3b6;background:#09281f}.v235-empty{padding:24px 15px;color:#7f97a7;text-align:center;font-size:10px}.v235-slate-note{padding:8px 14px;border-top:1px solid #102b3b;color:#5f7c8e;font-size:8px;line-height:1.45}
@media(max-width:1050px){body{padding-top:0}.main{padding:calc(56px + env(safe-area-inset-top)) 12px calc(78px + env(safe-area-inset-bottom))}.header{padding-right:0;align-items:flex-start}.header>div:first-child{min-width:0}.header .live,.header .preview-chip{max-width:45%;text-align:right;justify-content:flex-end}.header h1{font-size:23px;line-height:1.08}.subtitle{font-size:10px;line-height:1.35}.metrics{gap:8px}.metric{padding:12px}.metric .n{font-size:20px}.hero-main{padding:14px}.market-name{font-size:18px}.triple{gap:5px}.stat{padding:8px}.stat b{font-size:16px}.panel{padding:12px}.match-head{display:grid;grid-template-columns:1fr}.teams-head{gap:9px;display:grid;grid-template-columns:36px minmax(0,1fr) 36px;align-items:center}.crest{width:36px;height:36px}.quality{text-align:left;display:flex;gap:8px;flex-wrap:wrap}.filters{gap:9px}.filter-card{position:static}.toolbar{justify-content:flex-start;overflow-x:auto;flex-wrap:nowrap;padding-bottom:5px}.toolbtn{flex:0 0 auto;min-height:34px}.table{min-width:760px}.market-grid{gap:8px}.health-card,.fresh{min-height:64px}.chain,.wire{grid-template-columns:repeat(2,minmax(0,1fr))}.barrow{grid-template-columns:78px minmax(70px,1fr) 38px}.v233-auth-card{max-height:calc(100vh - 24px);overflow:auto;padding:16px}#v233AccountDock{position:fixed;top:calc(8px + env(safe-area-inset-top));right:10px}.v233-plan{display:none}.v233-account-btn{min-height:38px;padding:8px 12px}.v234-market-table{min-width:620px}#v234MatchTabPanel{overflow:auto}.v235-fixture{grid-template-columns:58px minmax(0,1fr) auto;padding:10px 11px}.v235-league{max-width:92px}}
@media(max-width:560px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric:last-child{grid-column:1/-1}.triple{grid-template-columns:1fr}.stat{display:flex;justify-content:space-between;align-items:center}.stat small,.stat b{display:inline;margin:0}.bottom-grid{grid-template-columns:1fr}.health,.fresh-grid{grid-template-columns:1fr 1fr}.chain,.wire{grid-template-columns:1fr 1fr}.ph{align-items:flex-start}.ph h3{line-height:1.3}.header .live{font-size:8px}.market-name{font-size:17px}.bigprob{font-size:20px}.v235-fixture{grid-template-columns:50px minmax(0,1fr)}.v235-fixture-meta{grid-column:2;display:flex;justify-content:space-between;align-items:center;width:100%;text-align:left}.v235-league{max-width:52vw}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important}}
</style>
<script id="SOCCER_V235_VISUAL_MOBILE_SCRIPT">
(()=>{
 const AK='soccer_edge_access_token';
 const labels=[['today','Today'],['feed','Feed'],['matches','Match'],['markets','Markets'],['performance','Performance'],['myedge','My Edge'],['tower','Tower'],['research','Lab']];
 const nav=document.querySelector('.mobile-nav');
 const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function activate(id){document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===id));document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===id));window.scrollTo({top:0,behavior:'smooth'})}
 if(nav){nav.innerHTML=labels.map(([id,label])=>`<button data-page="${id}" class="${id==='today'?'active':''} ${id==='tower'||id==='research'?'v235-mobile-lab':''}">${label}</button>`).join('')+`<button type="button" id="v235MobileAccount" class="v235-mobile-account">Account</button>`;nav.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>activate(b.dataset.page));document.getElementById('v235MobileAccount')?.addEventListener('click',()=>document.getElementById('v233AccountBtn')?.click())}
 document.querySelectorAll('.sidebar [data-page]').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.page)));
 const app=document.querySelector('.app');if(app)app.dataset.frontend='V235_FULL_SLATE_LOGOS';

 const initial=name=>String(name||'?').trim().slice(0,2).toUpperCase();
 const logo=(url,name)=>`<span class="v235-logo-wrap">${url?`<img loading="lazy" decoding="async" src="${esc(url)}" alt="${esc(name)} crest"><span class="v235-logo-fallback" style="display:none">${esc(initial(name))}</span>`:`<span class="v235-logo-fallback">${esc(initial(name))}</span>`}</span>`;
 function kickoffParts(raw){
   if(!raw)return ['TBD','Kickoff'];const d=new Date(raw);if(Number.isNaN(d.getTime()))return [String(raw).slice(0,5),'Kickoff'];
   const time=new Intl.DateTimeFormat(undefined,{hour:'2-digit',minute:'2-digit'}).format(d);
   const day=new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric'}).format(d);return [time,day];
 }
 function rowStatus(r){return String(r.execution_status||r.status||r.stage||r.blocker||'Fixture Only').replaceAll('_',' ')}
 function uniqueRows(rows){
   const seen=new Set();return (rows||[]).filter(r=>{const k=String(r.fixture_id||`${r.kickoff||''}|${r.home_team||''}|${r.away_team||''}`);if(seen.has(k))return false;seen.add(k);return true}).sort((a,b)=>{const x=Date.parse(a.kickoff||''),y=Date.parse(b.kickoff||'');return (Number.isFinite(x)?x:9e15)-(Number.isFinite(y)?y:9e15)});
 }
 function findSlatePanel(){
   const direct=document.getElementById('upcoming');if(direct)return {root:direct,panel:direct.closest('.panel')};
   const panels=[...document.querySelectorAll('#today .bottom-grid > .panel,#today .panel')];
   const panel=panels.find(p=>/Upcoming Matches|Verified Slate/i.test(p.textContent||''));
   if(!panel)return null;panel.classList.add('v235-slate-panel');panel.innerHTML='<div class="v235-slate-head"><h3>Upcoming Matches</h3><span id="v235SlateCount" class="v235-slate-count">Full Slate</span></div><div id="v235FullSlate" class="v235-slate-list"></div><div class="v235-slate-note">Fixtures render independently from deep analysis. Model/market intelligence appears only when persisted and entitled.</div>';return {root:document.getElementById('v235FullSlate'),panel};
 }
 function renderFullSlate(rows,total){
   const target=findSlatePanel();if(!target)return;const list=uniqueRows(rows);const panel=target.panel;if(panel){panel.classList.add('v235-slate-panel');if(target.root.id==='upcoming'){panel.innerHTML='<div class="v235-slate-head"><h3>Upcoming Matches</h3><span id="v235SlateCount" class="v235-slate-count"></span></div><div id="v235FullSlate" class="v235-slate-list"></div><div class="v235-slate-note">Fixtures render independently from deep analysis. Model/market intelligence appears only when persisted and entitled.</div>'}}
   const root=document.getElementById('v235FullSlate')||target.root,count=document.getElementById('v235SlateCount');if(count)count.textContent=`Full Slate (${total??list.length})`;
   if(!root)return;root.innerHTML=list.length?list.map(r=>{const [time,day]=kickoffParts(r.kickoff),home=r.home_team||r.home||'Home',away=r.away_team||r.away||'Away',state=rowStatus(r),ready=/READY|LIVE|STRONG/i.test(state);return `<div class="v235-fixture" data-fixture-id="${esc(r.fixture_id||'')}"><div class="v235-kickoff"><b>${esc(time)}</b>${esc(day)}</div><div class="v235-teams"><div class="v235-team">${logo(r.home_team_logo,home)}<span class="v235-team-name">${esc(home)}</span></div><div class="v235-team">${logo(r.away_team_logo,away)}<span class="v235-team-name">${esc(away)}</span></div></div><div class="v235-fixture-meta"><span class="v235-league">${esc(r.league||r.country||'Competition')}</span><span class="v235-state ${ready?'ready':''}">${esc(state)}</span></div></div>`}).join(''):'<div class="v235-empty">No upcoming persisted fixtures in the latest slate.</div>';
   root.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{img.style.display='none';const fb=img.nextElementSibling;if(fb)fb.style.display='inline'}));
 }
 async function refreshFullSlate(){
   try{const token=localStorage.getItem(AK)||'',headers=token?{Authorization:`Bearer ${token}`}:{},res=await fetch('/app/data',{headers,cache:'no-store'}),data=await res.json();if(!res.ok)return;const pro=data?.pro?.todays_slate,free=data?.public?.verified_slate,slate=pro?.rows?.length?pro:free;renderFullSlate(slate?.rows||[],slate?.total)}catch(_){/* existing UI remains truthful */}
 }
 setTimeout(refreshFullSlate,220);setTimeout(refreshFullSlate,1200);
})();
</script>
'''


def product_html() -> str:
    html = subscriber_product_v234.product_html()
    marker = "</body>"
    fragment = _visual_fragment()
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(product_html(), headers={"Cache-Control": "no-store"})


async def app_data(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if token:
        entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
        if not entitlement.get("ok") or not entitlement.get("authenticated"):
            return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)
        entitlement = _owner_view_entitlement(entitlement)
    else:
        entitlement = subscriber_app_v4.anonymous_entitlement()
    try:
        product = await subscriber_app_v4._load_product()
    except Exception as exc:
        return JSONResponse({"error": "SUBSCRIBER_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(product, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)
    return JSONResponse(_build_payload(product, entitlement), headers={"Cache-Control": "no-store"})


async def match_data(request: Request) -> JSONResponse:
    return await subscriber_product_v234.match_data(request)


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "mobile_pages": ["Today", "Feed", "Match", "Markets", "Performance", "My Edge", "Tower", "Lab", "Account"],
        "desktop_mockup_shell_preserved": True,
        "mobile_horizontal_navigation": True,
        "safe_area_support": True,
        "full_upcoming_slate": True,
        "team_logos_from_persisted_identity": True,
        "anonymous_fixture_browse": True,
        "owner_admin_presentation_access": "PRO_VIEW_WITHOUT_BILLING_MUTATION",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
