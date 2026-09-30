from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import subscriber_preview_data_v231
from mcp_gateway import subscriber_product_v233
from mcp_gateway import subscriber_ui_contract_v231
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PRODUCT_V234_1.0.0"

_GOALS = {"1X2", "BTTS", "FT_TOTALS", "HOME_TT", "AWAY_TT", "1H", "2H"}
_CORNERS = {"FT_CORNERS", "TEAM_CORNERS"}
_CARDS = {"CARDS", "TEAM_CARDS"}
_PLAYERS = {"SHOTS", "SOT", "GOALSCORER", "GOALSCORER_ANYTIME", "ASSISTS", "PLAYER_CARDS", "GK_SAVES"}


def _family(row: dict[str, Any]) -> str:
    market = row.get("market") if isinstance(row.get("market"), dict) else {}
    return str(market.get("family") or "").upper()


def _group_fixture_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"Goals": [], "Corners": [], "Cards": [], "Players": [], "Market": [], "Model": []}
    for row in rows:
        family = _family(row)
        if family in _GOALS:
            groups["Goals"].append(row)
        if family in _CORNERS:
            groups["Corners"].append(row)
        if family in _CARDS:
            groups["Cards"].append(row)
        if family in _PLAYERS:
            groups["Players"].append(row)
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        model = row.get("model") if isinstance(row.get("model"), dict) else {}
        if any(pricing.get(key) is not None for key in ("price", "market_probability", "fair_price", "edge_pp")):
            groups["Market"].append(row)
        if any(model.get(key) is not None for key in ("probability", "raw_probability", "version")):
            groups["Model"].append(row)
    return groups


def _enhancement_fragment() -> str:
    return r'''
<style id="SOCCER_V234_FUNCTIONAL_STYLE">
#feed .filter-card select,#feed .filter-card input{width:100%;background:#07141e;border:1px solid #183a50;border-radius:6px;color:#c9d8e2;padding:8px;font:inherit;font-size:9px}
#feed .filter-card input[type=range]{padding:0;height:6px;accent-color:#4de1ad}.v234-range-value{float:right;color:#65e7b5;font-weight:900}
#v234MatchTabPanel{display:none;margin-top:10px}.v234-market-table{width:100%;border-collapse:collapse;font-size:9px}.v234-market-table th{color:#69889b;text-transform:uppercase;text-align:left;padding:7px;border-bottom:1px solid #19384b;font-size:8px}.v234-market-table td{padding:8px 7px;border-bottom:1px solid #102b3c}.v234-market-table td.edge{color:#4de1ad;font-weight:900}.v234-empty{padding:24px;text-align:center;color:#7892a4;border:1px dashed #29475b;border-radius:9px}.v234-count{color:#64bfff;font-size:9px}.v234-filter-summary{margin:7px 0 0;color:#6f899b;font-size:8px}
</style>
<script id="SOCCER_V234_FUNCTIONAL_SCRIPT">
(()=>{
 const AK='soccer_edge_access_token';let DATA=null,CURRENT=null;
 const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=v=>v===null||v===undefined||v===''?null:Number(v);
 const pct=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${(Math.abs(n)<=1?n*100:n).toFixed(1)}%`};
 const pp=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${n>=0?'+':''}${n.toFixed(1)} pp`};
 const price=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':n.toFixed(2)};
 const conf=r=>{let n=num(r?.raw?.confidence_score??r?.raw?.model_confidence??r?.raw?.model_signal_score);if(n===null)return null;if(Math.abs(n)<=1)n*=100;return n};
 const status=r=>String(r?.state?.status||'WATCH').toUpperCase();
 const match=r=>r?.match?.label||[r?.match?.home,r?.match?.away].filter(Boolean).join(' vs ')||'Fixture';
 const market=r=>r?.market?.selection||r?.market?.family||'Market';
 const key=r=>[r?.match?.fixture_id,r?.market?.family,r?.market?.selection].join('|');
 const token=()=>localStorage.getItem(AK)||'';
 const getJson=async path=>{const r=await fetch(path,{headers:token()?{Authorization:`Bearer ${token()}`}:{},cache:'no-store'});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d};

 function setupFilters(){
   const card=document.querySelector('#feed .filter-card');if(!card||!DATA)return;
   card.innerHTML=`<div class="eyebrow">Filters</div>
    <label>League</label><select id="v234League"><option value="">All leagues</option></select>
    <label>Market</label><select id="v234Market"><option value="">All markets</option></select>
    <label>Kickoff</label><select id="v234Kickoff"><option value="">All times</option><option value="1">Next 1 hour</option><option value="3">Next 3 hours</option><option value="12">Next 12 hours</option></select>
    <label>Edge (pp) <span class="v234-range-value" id="v234EdgeVal">0+</span></label><input id="v234Edge" type="range" min="0" max="20" step="1" value="0">
    <label>Confidence <span class="v234-range-value" id="v234ConfVal">0+</span></label><input id="v234Conf" type="range" min="0" max="100" step="5" value="0">
    <label>Status</label><select id="v234Status"><option value="">All statuses</option></select><div id="v234FilterSummary" class="v234-filter-summary"></div>`;
   const rows=DATA.edge_feed?.rows||[];
   const uniq=fn=>[...new Set(rows.map(fn).filter(Boolean))].sort();
   const fill=(id,vals)=>{const s=document.getElementById(id);if(s)vals.forEach(v=>s.insertAdjacentHTML('beforeend',`<option value="${esc(v)}">${esc(v)}</option>`))};
   fill('v234League',uniq(r=>r.match?.league||r.match?.country));fill('v234Market',uniq(r=>r.market?.family));fill('v234Status',uniq(status));
   ['v234League','v234Market','v234Kickoff','v234Edge','v234Conf','v234Status'].forEach(id=>document.getElementById(id)?.addEventListener('input',()=>applyFilters('All')));
   renderFeed(rows);applyFilters('All');
 }
 function renderFeed(rows){
   const body=document.getElementById('feedbody');if(!body)return;
   body.innerHTML=rows.length?rows.map(r=>`<tr data-v234-key="${esc(key(r))}" data-v234-fixture="${esc(r.match?.fixture_id)}"><td class="team-cell"><span class="team-dot">⚽</span>${esc(match(r))}</td><td>${esc(market(r))}</td><td>${esc(pct(r.model?.probability))}</td><td>${esc(pct(r.pricing?.market_probability))}</td><td class="edge">${esc(pp(r.pricing?.edge_pp))}</td><td>${esc(price(r.pricing?.price))}</td><td>${conf(r)==null?'—':Math.round(conf(r))}</td><td><span class="status">${esc(status(r))}</span></td></tr>`).join(''):'<tr><td colspan="8">No persisted Edge Feed rows.</td></tr>';
   body.querySelectorAll('tr[data-v234-fixture]').forEach(tr=>tr.onclick=()=>openFixture(tr.dataset.v234Fixture));
 }
 function kickoffHours(r){const raw=r?.match?.kickoff;if(!raw)return null;const t=Date.parse(raw);if(!Number.isFinite(t))return null;return (t-Date.now())/3600000}
 function xiConfirmed(r){const v=String(r?.raw?.lineup_status??r?.raw?.xi_status??r?.raw?.confirmed_xi??r?.raw?.lineups_confirmed??'').toUpperCase();return ['CONFIRMED','TRUE','1','YES'].includes(v)}
 function applyFilters(mode){
   if(!DATA)return;const rows=DATA.edge_feed?.rows||[],strong=new Set((DATA.today?.strong_signals||[]).map(key));
   const league=document.getElementById('v234League')?.value||'',fam=document.getElementById('v234Market')?.value||'',kh=Number(document.getElementById('v234Kickoff')?.value||0),edgeMin=Number(document.getElementById('v234Edge')?.value||0),confMin=Number(document.getElementById('v234Conf')?.value||0),st=document.getElementById('v234Status')?.value||'';
   document.getElementById('v234EdgeVal').textContent=`${edgeMin}+`;document.getElementById('v234ConfVal').textContent=`${confMin}+`;
   let shown=0;document.querySelectorAll('#feedbody tr[data-v234-key]').forEach((tr,i)=>{const r=rows[i];let ok=!!r;const h=kickoffHours(r),e=num(r?.pricing?.edge_pp),c=conf(r);ok=ok&&(!league||(r.match?.league||r.match?.country)===league)&&(!fam||r.market?.family===fam)&&(!st||status(r)===st)&&(e===null?edgeMin===0:e>=edgeMin)&&(c===null?confMin===0:c>=confMin)&&(!kh||(h!==null&&h>=0&&h<=kh));if(mode==='Strong Only')ok=ok&&strong.has(key(r));if(mode==='Ready Only')ok=ok&&status(r)==='READY';if(mode==='Next 3 Hours')ok=ok&&(h!==null&&h>=0&&h<=3);if(mode==='Confirmed XI')ok=ok&&xiConfirmed(r);tr.style.display=ok?'':'none';if(ok)shown++});
   const s=document.getElementById('v234FilterSummary');if(s)s.textContent=`${shown} of ${rows.length} persisted rows`;
 }
 function setupToolbar(){document.querySelectorAll('#feed .toolbtn').forEach(btn=>{btn.onclick=()=>{document.querySelectorAll('#feed .toolbtn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');applyFilters(btn.textContent.trim())}})}

 async function openFixture(fid){if(!fid)return;try{CURRENT=await getJson(`/app/match?fixture_id=${encodeURIComponent(fid)}`);showPage('matches');renderFamilyTab('Overview')}catch(e){const p=ensureTabPanel();p.style.display='block';p.innerHTML=`<div class="v234-empty">Match detail unavailable · ${esc(e.message)}</div>`}}
 function showPage(id){document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===id));window.scrollTo({top:0,behavior:'smooth'})}
 function ensureTabPanel(){let p=document.getElementById('v234MatchTabPanel');if(!p){p=document.createElement('div');p.id='v234MatchTabPanel';p.className='panel';document.querySelector('#matches .match-shell')?.appendChild(p)}return p}
 function rowTable(rows,label){if(!rows?.length)return `<div class="v234-empty">No persisted ${esc(label)} rows for this fixture snapshot. Missing market data is not synthesized.</div>`;return `<div class="ph"><h3>${esc(label)} · persisted markets</h3><span class="v234-count">${rows.length} rows</span></div><div style="overflow:auto"><table class="v234-market-table"><thead><tr><th>Market</th><th>Model</th><th>Market</th><th>Edge</th><th>Price</th><th>Status</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(market(r))}</td><td>${esc(pct(r.model?.probability))}</td><td>${esc(pct(r.pricing?.market_probability))}</td><td class="edge">${esc(pp(r.pricing?.edge_pp))}</td><td>${esc(price(r.pricing?.price))}</td><td>${esc(status(r))}</td></tr>`).join('')}</tbody></table></div>`}
 function renderFamilyTab(label){
   const panel=ensureTabPanel(),cards=[...document.querySelectorAll('#matches .match-grid .score-card')];
   if(label==='Overview'){panel.style.display='none';cards.forEach(c=>c.style.display='');return}
   cards.forEach(c=>c.style.display='none');panel.style.display='block';const rows=CURRENT?.markets_by_tab?.[label]||[];panel.innerHTML=rowTable(rows,label);
 }
 function setupTabs(){document.querySelectorAll('#matches .tabs .tab').forEach(tab=>{tab.onclick=()=>{document.querySelectorAll('#matches .tabs .tab').forEach(x=>x.classList.remove('active'));tab.classList.add('active');renderFamilyTab(tab.textContent.trim())}})}
 function bindFixtureClicks(){
   const hero=document.querySelector('#today .hero-main');const top=DATA?.today?.top_edge;if(hero&&top?.match?.fixture_id)hero.onclick=()=>openFixture(top.match.fixture_id);
   document.querySelectorAll('#feedbody tr[data-v234-fixture]').forEach(tr=>tr.onclick=()=>openFixture(tr.dataset.v234Fixture));
 }
 async function init(){if(!token())return;try{DATA=await getJson('/app-preview/data');setupFilters();setupToolbar();setupTabs();bindFixtureClicks();setTimeout(()=>{renderFeed(DATA.edge_feed?.rows||[]);applyFilters('All');setupToolbar();setupTabs();bindFixtureClicks()},900)}catch(_){}}
 setTimeout(init,180);
})();
</script>
'''


def product_html() -> str:
    html = subscriber_product_v233.product_html()
    marker = "</body>"
    fragment = _enhancement_fragment()
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(product_html(), headers={"Cache-Control": "no-store"})


async def match_data(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return JSONResponse({"error": "AUTH_REQUIRED"}, status_code=401)
    entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not entitlement.get("ok") or not entitlement.get("authenticated"):
        return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)
    is_owner = bool(entitlement.get("owner")) or (entitlement.get("user") or {}).get("role") == "OWNER"
    is_pro = str(entitlement.get("effective_plan") or "").upper() == subscription_entitlements_v4.PRO_PLAN
    if not (is_owner or is_pro):
        return JSONResponse({"error": "MATCH_DETAIL_REQUIRES_PRO"}, status_code=403)
    fixture_text = str(request.query_params.get("fixture_id") or "").strip()
    if not fixture_text:
        return JSONResponse({"error": "FIXTURE_ID_REQUIRED"}, status_code=400)
    fixture_value: Any = int(fixture_text) if fixture_text.isdigit() else fixture_text
    try:
        payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    except Exception as exc:
        return JSONResponse({"error": "MATCH_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)

    preview = subscriber_preview_data_v231.build_preview_payload(payload)
    feed = (preview.get("edge_feed") or {}).get("rows") or []
    selected = next((row for row in feed if str(((row.get("match") or {}).get("fixture_id"))) == fixture_text), None)
    raw_rows = subscriber_preview_data_v231._fixture_rows(payload, fixture_value)
    adapted = subscriber_ui_contract_v231.adapt_rows(raw_rows)
    if not isinstance(selected, dict):
        selected = next((row for row in adapted if str(((row.get("match") or {}).get("fixture_id"))) == fixture_text), None)
    if not isinstance(selected, dict):
        return JSONResponse({"error": "FIXTURE_NOT_IN_PERSISTED_SNAPSHOT"}, status_code=404)
    detail = subscriber_preview_data_v231._match_detail(payload, selected)
    groups = _group_fixture_rows(adapted)
    return JSONResponse({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "fixture_id": fixture_text,
        "selected": selected,
        "detail": detail,
        "markets_by_tab": groups,
        "fixture_market_rows": len(adapted),
        "source": "POSTGRES_LATEST_PIPELINE_RUN",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    })


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "edge_feed_filters": ["league", "market", "kickoff", "edge_pp", "confidence", "status"],
        "edge_feed_modes": ["All", "Strong Only", "Ready Only", "Next 3 Hours", "Confirmed XI"],
        "match_tabs_backed_by_persisted_rows": ["Goals", "Corners", "Cards", "Players", "Market", "Model"],
        "missing_market_rows_are_synthesized": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
