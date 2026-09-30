from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import subscriber_app_v4
from mcp_gateway import subscriber_product_v234
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PRODUCT_V235_1.0.0"


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
@media(max-width:1050px){body{padding-top:0}.main{padding:calc(56px + env(safe-area-inset-top)) 12px calc(78px + env(safe-area-inset-bottom))}.header{padding-right:0;align-items:flex-start}.header>div:first-child{min-width:0}.header .live,.header .preview-chip{max-width:45%;text-align:right;justify-content:flex-end}.header h1{font-size:23px;line-height:1.08}.subtitle{font-size:10px;line-height:1.35}.metrics{gap:8px}.metric{padding:12px}.metric .n{font-size:20px}.hero-main{padding:14px}.market-name{font-size:18px}.triple{gap:5px}.stat{padding:8px}.stat b{font-size:16px}.panel{padding:12px}.match-head{display:grid;grid-template-columns:1fr}.teams-head{gap:9px;display:grid;grid-template-columns:36px minmax(0,1fr) 36px;align-items:center}.crest{width:36px;height:36px}.quality{text-align:left;display:flex;gap:8px;flex-wrap:wrap}.filters{gap:9px}.filter-card{position:static}.toolbar{justify-content:flex-start;overflow-x:auto;flex-wrap:nowrap;padding-bottom:5px}.toolbtn{flex:0 0 auto;min-height:34px}.table{min-width:760px}.market-grid{gap:8px}.health-card,.fresh{min-height:64px}.chain,.wire{grid-template-columns:repeat(2,minmax(0,1fr))}.barrow{grid-template-columns:78px minmax(70px,1fr) 38px}.v233-auth-card{max-height:calc(100vh - 24px);overflow:auto;padding:16px}#v233AccountDock{position:fixed;top:calc(8px + env(safe-area-inset-top));right:10px}.v233-plan{display:none}.v233-account-btn{min-height:38px;padding:8px 12px}.v234-market-table{min-width:620px}#v234MatchTabPanel{overflow:auto}}
@media(max-width:560px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric:last-child{grid-column:1/-1}.triple{grid-template-columns:1fr}.stat{display:flex;justify-content:space-between;align-items:center}.stat small,.stat b{display:inline;margin:0}.bottom-grid{grid-template-columns:1fr}.health,.fresh-grid{grid-template-columns:1fr 1fr}.chain,.wire{grid-template-columns:1fr 1fr}.ph{align-items:flex-start}.ph h3{line-height:1.3}.header .live{font-size:8px}.market-name{font-size:17px}.bigprob{font-size:20px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important}}
</style>
<script id="SOCCER_V235_VISUAL_MOBILE_SCRIPT">
(()=>{
 const labels=[['today','Today'],['feed','Feed'],['matches','Match'],['markets','Markets'],['performance','Performance'],['myedge','My Edge'],['tower','Tower'],['research','Lab']];
 const nav=document.querySelector('.mobile-nav');
 function activate(id){document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===id));document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===id));window.scrollTo({top:0,behavior:'smooth'})}
 if(nav){nav.innerHTML=labels.map(([id,label])=>`<button data-page="${id}" class="${id==='today'?'active':''} ${id==='tower'||id==='research'?'v235-mobile-lab':''}">${label}</button>`).join('')+`<button type="button" id="v235MobileAccount" class="v235-mobile-account">Account</button>`;nav.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>activate(b.dataset.page));document.getElementById('v235MobileAccount')?.addEventListener('click',()=>document.getElementById('v233AccountBtn')?.click())}
 document.querySelectorAll('.sidebar [data-page]').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.page)));
 const app=document.querySelector('.app');if(app)app.dataset.frontend='V235_VISUAL_MOBILE_PARITY';
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
    if not token:
        return await subscriber_app_v4.app_data(request)
    entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not entitlement.get("ok") or not entitlement.get("authenticated"):
        return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)
    entitlement = _owner_view_entitlement(entitlement)
    try:
        product = await subscriber_app_v4._load_product()
    except Exception as exc:
        return JSONResponse({"error": "SUBSCRIBER_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(product, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)
    return JSONResponse(subscriber_app_v4.build_subscriber_payload(product, entitlement))


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
        "owner_admin_presentation_access": "PRO_VIEW_WITHOUT_BILLING_MUTATION",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
