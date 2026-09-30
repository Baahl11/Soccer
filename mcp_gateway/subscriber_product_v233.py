from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import subscriber_preview_data_v231
from mcp_gateway import subscriber_preview_maturity_live_v232
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PRODUCT_V233_1.0.0"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _account_fragment() -> str:
    cfg = supabase_auth_v4.public_auth_config()
    public_cfg = {
        "supabase_url": cfg.get("project_url"),
        "publishable_key": cfg.get("publishable_key"),
        "auth_configured": bool(cfg.get("configured")),
    }
    return f'''
<script id="cfg" type="application/json">{_safe_json(public_cfg)}</script>
<style id="SOCCER_V233_PRODUCT_STYLE">
#v233AccountDock{{position:fixed;right:18px;top:16px;z-index:45;display:flex;gap:7px;align-items:center}}
.v233-account-btn,.v233-primary,.v233-secondary{{border:1px solid #21475f;background:#0a1d2a;color:#dcebf3;border-radius:8px;padding:8px 11px;font:inherit;font-size:9px;font-weight:850;cursor:pointer}}
.v233-account-btn:hover,.v233-secondary:hover{{border-color:#2f6685}}.v233-primary{{background:#0b3a2f;border-color:#246b56;color:#7df0c8}}
.v233-plan{{border:1px solid #1f6a52;background:#0b3027;color:#61e5b5;border-radius:999px;padding:5px 8px;font-size:8px;font-weight:900}}
#v233Auth{{position:fixed;inset:0;background:#02070bbd;backdrop-filter:blur(8px);z-index:70;display:none;place-items:center;padding:18px}}
#v233Auth.open{{display:grid}}.v233-auth-card{{width:min(440px,100%);background:linear-gradient(180deg,#0b1f2c,#07131c);border:1px solid #21475d;border-radius:15px;padding:20px;box-shadow:0 30px 90px #000c}}
.v233-auth-card h2{{margin:0 0 4px;font-size:22px}}.v233-auth-card p{{margin:0 0 14px;color:#7e96a6;font-size:10px;line-height:1.5}}
.v233-auth-card input,.v233-auth-card select{{width:100%;margin:0 0 8px;padding:10px;background:#071722;color:#eef7fb;border:1px solid #193d52;border-radius:8px;font:inherit}}
.v233-auth-actions{{display:flex;gap:7px;flex-wrap:wrap}}#v233AuthStatus{{display:block;min-height:16px;margin-top:8px;color:#e9ba62;font-size:9px}}
.v233-billing{{margin-top:12px;border-top:1px solid #163448;padding-top:12px}}.v233-billing-grid{{display:grid;grid-template-columns:1fr auto;gap:7px;align-items:center}}
.v233-billing select{{width:100%;padding:8px;background:#071722;color:#eaf5fa;border:1px solid #193d52;border-radius:7px;font-size:9px}}
.v233-free-lock{{border:1px dashed #2b4a5d;background:#081720;border-radius:10px;padding:16px;color:#8198a7;text-align:center;font-size:10px;line-height:1.5}}
.v233-tab-empty{{display:none;border:1px dashed #29485d;background:#081720;border-radius:9px;padding:24px;text-align:center;color:#7d95a6;font-size:10px;grid-column:1/-1}}
#feedbody tr{{cursor:pointer}}#feedbody tr:hover{{background:#0a2030}}.hero-main[data-v233-clickable="1"]{{cursor:pointer}}.hero-main[data-v233-clickable="1"]:hover{{border-color:#2a6687}}
.v233-hidden{{display:none!important}}.v233-selected-row{{background:#0a2636!important}}
@media(max-width:1050px){{#v233AccountDock{{top:8px;right:8px}}}}
</style>
<div id="v233AccountDock"><span id="v233Plan" class="v233-plan">EXPLORER</span><button id="v233AccountBtn" class="v233-account-btn" type="button">Sign in</button></div>
<div id="v233Auth" aria-hidden="true"><div class="v233-auth-card">
  <h2>Soccer Edge</h2><p id="v233AuthCopy">Sign in to unlock your account. Explorer remains available without fabricated premium data.</p>
  <input id="v233Email" type="email" autocomplete="email" placeholder="Email">
  <input id="v233Password" type="password" autocomplete="current-password" placeholder="Password">
  <div class="v233-auth-actions"><button id="v233SignIn" class="v233-primary" type="button">Sign in</button><button id="v233SignUp" class="v233-secondary" type="button">Create account</button><button id="v233SignOut" class="v233-secondary" type="button">Sign out</button><button id="v233CloseAuth" class="v233-secondary" type="button">Close</button></div>
  <span id="v233AuthStatus"></span>
  <div class="v233-billing"><b style="font-size:10px">Edge Pro · Founding Beta</b><p style="margin:5px 0 8px">Choose billing market explicitly. Language and IP never select a price.</p><div class="v233-billing-grid"><select id="v233BillingMarket"><option value="">Billing market</option><option value="US">United States — US$14.99 / month</option><option value="MX_LATAM">Mexico / LATAM — MX$249 / month</option></select><button id="v233Upgrade" class="v233-primary" type="button">Upgrade</button></div><button id="v233Portal" class="v233-secondary" style="margin-top:7px" type="button">Manage subscription</button><span id="v233BillingStatus" style="display:block;min-height:14px;margin-top:6px;color:#7f97a8;font-size:9px"></span></div>
</div></div>
'''


def _interaction_script() -> str:
    return r'''
<script id="SOCCER_V233_PRODUCT_SCRIPT">
(()=>{
  const AK='soccer_edge_access_token', RK='soccer_edge_refresh_token', MK='soccer_edge_billing_market';
  const cfg=(()=>{try{return JSON.parse(document.getElementById('cfg')?.textContent||'{}')}catch(_){return{}}})();
  let APP=null, PREVIEW=null, selectedFixture=null;
  const $=id=>document.getElementById(id), esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const token=()=>localStorage.getItem(AK)||'';
  const openAuth=()=>{$('v233Auth')?.classList.add('open');$('v233Auth')?.setAttribute('aria-hidden','false')};
  const closeAuth=()=>{$('v233Auth')?.classList.remove('open');$('v233Auth')?.setAttribute('aria-hidden','true')};
  const gotoPage=id=>{document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===id));window.scrollTo({top:0,behavior:'smooth'})};
  const authFetch=async(path,body)=>{const r=await fetch(`${cfg.supabase_url}${path}`,{method:'POST',headers:{apikey:cfg.publishable_key,'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error_description||d.msg||d.error||`HTTP ${r.status}`);return d};
  const api=async(path)=>{const h={};if(token())h.Authorization=`Bearer ${token()}`;const r=await fetch(path,{headers:h,cache:'no-store'});const d=await r.json().catch(()=>({}));if(!r.ok)throw Object.assign(new Error(d.error||`HTTP ${r.status}`),{status:r.status,data:d});return d};

  function syncAccount(){
    const logged=!!token(), plan=APP?.effective_plan||PREVIEW?.effective_plan||(logged?'FREE':'EXPLORER');
    if($('v233Plan'))$('v233Plan').textContent=String(plan).toUpperCase();
    if($('v233AccountBtn'))$('v233AccountBtn').textContent=logged?'Account':'Sign in';
    if($('v233SignOut'))$('v233SignOut').style.display=logged?'inline-block':'none';
  }

  async function loadApp(){
    try{APP=await api('/app/data')}catch(_){APP=null}
    syncAccount();
    if(!APP)return;
    if(!token()||String(APP.effective_plan||'').toUpperCase()!=='PRO')renderExplorer(APP);
  }

  function flatMatch(r){return [r.home_team||r.home,r.away_team||r.away].filter(Boolean).join(' vs ')||String(r.fixture_id||'Fixture')}
  function renderExplorer(d){
    const pub=d.public||{}, slate=pub.verified_slate||{}, wp=pub.waiting_for_price||{}, wx=pub.waiting_for_xi||{};
    const metrics=document.querySelectorAll('#today .metrics .metric .n');
    const vals=[slate.total??(slate.rows||[]).length,'—','—',wx.total??(wx.rows||[]).length,'—'];metrics.forEach((x,i)=>{if(x)x.textContent=vals[i]});
    const hero=document.querySelector('#today .hero-main');if(hero){const r=(slate.rows||[])[0];hero.innerHTML=r?`<div class="eyebrow">EXPLORER · VERIFIED SLATE</div><div class="match-name">${esc(flatMatch(r))}</div><div class="market-name">${esc(r.market||r.market_family||'Persisted market')}</div><div class="badges"><span class="badge">${esc(r.status||r.execution_status||r.stage||'WATCH')}</span></div><p class="note">Model probability, market probability, edge and premium match detail unlock only when entitlement permits.</p>`:'<div class="v233-free-lock">No verified slate rows in the latest persisted snapshot.</div>'}
    const strong=document.querySelector('#today .hero-edge > .panel:nth-child(2) .signal-list');if(strong)strong.innerHTML='<div class="v233-free-lock">Strong Signals are Edge Pro. Explorer shows the verified slate and wait states without inventing premium values.</div>';
    const panels=document.querySelectorAll('#today .bottom-grid > .panel');
    const compact=rows=>(rows||[]).slice(0,5).map(r=>`<div class="compact-row"><b>${esc(flatMatch(r))}</b><span>${esc(r.market||r.market_family||'Market')}</span><span>${esc(r.status||r.execution_status||'WATCH')}</span></div>`).join('')||'<div class="placeholder">No rows.</div>';
    if(panels[0])panels[0].innerHTML='<div class="ph"><h3>Waiting for Price</h3></div>'+compact(wp.rows);
    if(panels[1])panels[1].innerHTML='<div class="ph"><h3>Waiting for XI</h3></div>'+compact(wx.rows);
    if(panels[2])panels[2].innerHTML='<div class="ph"><h3>Verified Slate</h3></div>'+compact(slate.rows);
    ['feed','matches','performance','research'].forEach(id=>{const page=$(id);if(page&&!page.dataset.v233locked){page.dataset.v233locked='1';page.querySelectorAll('.panel,.market-grid,.match-shell').forEach((n,i)=>{if(i===0)n.innerHTML='<div class="v233-free-lock">Edge Pro unlocks this evidence surface. No premium values are reconstructed for Explorer.</div>'})}})
  }

  async function loadPreview(){
    if(!token())return;
    try{PREVIEW=await api('/app-preview/data');syncAccount();setTimeout(()=>{wireRows();wireHero();wireFilters();wireTabs()},300)}catch(e){if(e.status!==403)console.warn('preview load',e.message)}
  }

  function wireHero(){
    const top=PREVIEW?.today?.top_edge,hero=document.querySelector('#today .hero-main');if(!top||!hero)return;const fid=top.match?.fixture_id;if(fid==null)return;hero.dataset.v233Clickable='1';hero.onclick=()=>selectFixture(fid);
  }
  function wireRows(){
    const rows=PREVIEW?.edge_feed?.rows||[], trs=[...document.querySelectorAll('#feedbody tr')];
    trs.forEach((tr,i)=>{const r=rows[i];if(!r)return;const fid=r.match?.fixture_id;tr.dataset.fixtureId=fid??'';tr.onclick=()=>{if(fid!=null)selectFixture(fid)}})
  }

  async function selectFixture(fid){
    selectedFixture=String(fid);gotoPage('matches');
    try{const d=await api(`/app/match?fixture_id=${encodeURIComponent(fid)}`);renderMatch(d);document.querySelectorAll('#feedbody tr').forEach(tr=>tr.classList.toggle('v233-selected-row',tr.dataset.fixtureId===String(fid)))}catch(e){const h=document.querySelector('#matches .match-head');if(h)h.innerHTML=`<div class="v233-free-lock">Match detail unavailable · ${esc(e.message)}</div>`}
  }

  function pct(v){const n=Number(v);return Number.isFinite(n)?`${(Math.abs(n)<=1?n*100:n).toFixed(1)}%`:'—'}
  function renderMatch(d){
    const r=d.selected||{}, detail=d.detail||{}, m=r.match||{}, market=r.market||{}, pricing=r.pricing||{}, model=r.model||{};
    const h=document.querySelector('#matches .match-head');if(h)h.innerHTML=`<div class="teams-head"><div class="crest">${esc((m.home||'?').slice(0,3).toUpperCase())}</div><div><small class="subtitle">${esc(m.league||m.country||'Competition')} · ${esc(m.kickoff||'Kickoff N/V')}</small><div class="match-name">${esc(m.home||'Home')} <span class="vs">vs</span> ${esc(m.away||'Away')}</div></div><div class="crest">${esc((m.away||'?').slice(0,3).toUpperCase())}</div></div><div class="quality"><span>Data <b class="green">${esc(detail.model_context?.data_quality||'N/V')}</b></span><span>Lineup <b class="green">${esc(detail.model_context?.lineup||'N/V')}</b></span><span>Status <b class="green">${esc(r.state?.status||'WATCH')}</b></span></div>`;
    const cards=[...document.querySelectorAll('#matches .match-grid .score-card')],p=detail.outcome_probabilities,x=detail.expected_goals;
    if(cards[0])cards[0].innerHTML=p?`<h4>Match Result Probability</h4><div class="probline"><div class="probbox"><small>HOME</small><b class="green">${pct(p.home)}</b></div><div class="probbox"><small>DRAW</small><b>${pct(p.draw)}</b></div><div class="probbox"><small>AWAY</small><b>${pct(p.away)}</b></div></div>`:`<h4>${esc(market.selection||market.family||'Selected market')}</h4><div class="bigprob">${pct(model.probability)}</div><div class="subtitle">Persisted model probability</div>`;
    if(cards[1])cards[1].innerHTML=x?`<h4>Expected Goals (λ)</h4><div class="bigprob">${x.home==null?'—':Number(x.home).toFixed(2)} <span class="vs">–</span> ${x.away==null?'—':Number(x.away).toFixed(2)}</div>`:'<h4>Expected Goals (λ)</h4><div class="v231-empty">N/V for this persisted fixture snapshot.</div>';
    if(cards[2])cards[2].innerHTML=`<h4>Edge Gap · ${esc(market.selection||market.family||'Market')}</h4><div class="bigprob green">${pricing.edge_pp==null?'—':`${Number(pricing.edge_pp)>=0?'+':''}${Number(pricing.edge_pp).toFixed(1)} pp`}</div><div class="subtitle">Model ${pct(model.probability)} · Market ${pct(pricing.market_probability)}</div>`;
    if(cards[4]){const sm=detail.score_matrix||[];cards[4].innerHTML=`<h4>Score Matrix · top persisted scorelines</h4><div class="v231-score-grid">${sm.length?sm.map(x=>`<div class="v231-score"><b>${esc(x.score)}</b><small>${pct(x.probability)}</small></div>`).join(''):'<div class="v231-empty">N/V</div>'}</div>`}
    if(cards[3]){const sp=detail.sport_profile||[];cards[3].innerHTML=`<h4>Sport Profile</h4><div class="bars">${sp.length?sp.map(x=>`<div class="barrow"><span>${esc(x.label)}</span><div class="bar"><div class="fill" style="width:${Math.max(0,Math.min(100,Number(x.score)||0))}%"></div></div><b>${Math.round(Number(x.score)||0)}</b></div>`).join(''):'<div class="v231-empty">N/V</div>'}</div>`}
    wireTabs();
  }

  function wireTabs(){
    const tabs=[...document.querySelectorAll('#matches .tabs .tab')],cards=[...document.querySelectorAll('#matches .match-grid .score-card')];if(!tabs.length)return;
    let empty=document.querySelector('#matches .v233-tab-empty');if(!empty){empty=document.createElement('div');empty.className='v233-tab-empty';document.querySelector('#matches .match-grid')?.appendChild(empty)}
    const groups={Overview:[0,1,2,3,4,5],Goals:[1,4,5],Market:[0,2],Model:[0,1,3,4],Corners:[],Cards:[],Players:[]};
    tabs.forEach(tab=>{tab.onclick=()=>{tabs.forEach(x=>x.classList.remove('active'));tab.classList.add('active');const label=tab.textContent.trim(),show=groups[label]??[];cards.forEach((c,i)=>c.classList.toggle('v233-hidden',!show.includes(i)));if(show.length){empty.style.display='none'}else{empty.style.display='block';empty.innerHTML=`No persisted ${esc(label)} detail is available for this selected fixture snapshot. Soccer Edge does not synthesize missing family data.`}}})
  }

  function wireFilters(){
    const rows=PREVIEW?.edge_feed?.rows||[],filterCard=document.querySelector('#feed .filter-card');if(!filterCard||!rows.length)return;
    const sels=[...filterCard.querySelectorAll('.select')];
    const unique=fn=>[...new Set(rows.map(fn).filter(Boolean))].sort();
    if(sels[0])sels[0].innerHTML='<option value="">All</option>'+unique(r=>r.match?.league||r.match?.country).map(x=>`<option>${esc(x)}</option>`).join('');
    if(sels[1])sels[1].innerHTML='<option value="">All</option>'+unique(r=>r.market?.family).map(x=>`<option>${esc(x)}</option>`).join('');
    const apply=()=>{const league=sels[0]?.value||'',fam=sels[1]?.value||'';const trs=[...document.querySelectorAll('#feedbody tr')];trs.forEach((tr,i)=>{const r=rows[i],ok=!!r&&(!league||(r.match?.league||r.match?.country)===league)&&(!fam||r.market?.family===fam);tr.style.display=ok?'':'none'})};sels.slice(0,2).forEach(s=>s&&s.addEventListener('change',apply));
    const strongKeys=new Set((PREVIEW?.today?.strong_signals||[]).map(r=>[r.match?.fixture_id,r.market?.family,r.market?.selection].join('|')));
    document.querySelectorAll('#feed .toolbtn').forEach(btn=>{btn.onclick=()=>{document.querySelectorAll('#feed .toolbtn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');const mode=btn.textContent.trim();const trs=[...document.querySelectorAll('#feedbody tr')];trs.forEach((tr,i)=>{const r=rows[i];let ok=!!r;if(mode==='Strong Only')ok=strongKeys.has([r?.match?.fixture_id,r?.market?.family,r?.market?.selection].join('|'));else if(mode==='Ready Only')ok=String(r?.state?.status||'').toUpperCase()==='READY';else if(mode==='Confirmed XI')ok=['CONFIRMED','TRUE'].includes(String(r?.raw?.lineup_status??r?.raw?.xi_status??r?.raw?.confirmed_xi??'').toUpperCase());tr.style.display=ok?'':'none'})}})
  }

  async function checkout(){const market=$('v233BillingMarket')?.value||'';if(!['US','MX_LATAM'].includes(market)){textBill('Choose US or Mexico/LATAM first.');return}if(!token()){textBill('Sign in before Checkout.');openAuth();return}localStorage.setItem(MK,market);try{const r=await fetch(`${cfg.supabase_url}/functions/v1/create-checkout-session`,{method:'POST',headers:{Authorization:`Bearer ${token()}`,apikey:cfg.publishable_key,'Content-Type':'application/json'},body:JSON.stringify({billing_market:market})});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error||'CHECKOUT_FAILED');location.href=d.url}catch(e){textBill(e.message)}}
  async function portal(){if(!token()){openAuth();return}try{const r=await fetch(`${cfg.supabase_url}/functions/v1/create-customer-portal`,{method:'POST',headers:{Authorization:`Bearer ${token()}`,apikey:cfg.publishable_key,'Content-Type':'application/json'},body:'{}'});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error||'PORTAL_FAILED');location.href=d.url}catch(e){textBill(e.message)}}
  const textBill=v=>{if($('v233BillingStatus'))$('v233BillingStatus').textContent=v};

  $('v233AccountBtn')?.addEventListener('click',openAuth);$('v233CloseAuth')?.addEventListener('click',closeAuth);$('v233Upgrade')?.addEventListener('click',checkout);$('v233Portal')?.addEventListener('click',portal);
  $('v233SignOut')?.addEventListener('click',()=>{localStorage.removeItem(AK);localStorage.removeItem(RK);location.reload()});
  $('v233SignIn')?.addEventListener('click',async()=>{try{const d=await authFetch('/auth/v1/token?grant_type=password',{email:$('v233Email').value,password:$('v233Password').value});localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);location.reload()}catch(e){$('v233AuthStatus').textContent=e.message}});
  $('v233SignUp')?.addEventListener('click',async()=>{try{const d=await authFetch('/auth/v1/signup',{email:$('v233Email').value,password:$('v233Password').value});if(d.access_token)localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);$('v233AuthStatus').textContent=d.access_token?'Account created. Reloading…':'Account created. Check your email if confirmation is required.';if(d.access_token)setTimeout(()=>location.reload(),400)}catch(e){$('v233AuthStatus').textContent=e.message}});
  const prior=localStorage.getItem(MK)||'';if($('v233BillingMarket')&&['US','MX_LATAM'].includes(prior))$('v233BillingMarket').value=prior;
  document.title='Soccer Edge';document.querySelectorAll('.preview-chip').forEach(x=>{if(x.textContent.includes('PREVIEW'))x.textContent=x.textContent.replace('PREVIEW','LIVE')});
  syncAccount();loadApp();loadPreview();setTimeout(()=>{wireRows();wireHero();wireTabs()},1000);
})();
</script>
'''


def product_html() -> str:
    html = subscriber_preview_maturity_live_v232._html()
    html = html.replace("Soccer Edge · Product Preview", "Soccer Edge", 1)
    html = html.replace("Mockup Preview", "Soccer Edge", 1)
    marker = "</body>"
    fragment = _account_fragment() + _interaction_script()
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
    fixture_id = str(request.query_params.get("fixture_id") or "").strip()
    if not fixture_id:
        return JSONResponse({"error": "FIXTURE_ID_REQUIRED"}, status_code=400)
    try:
        payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    except Exception as exc:
        return JSONResponse({"error": "MATCH_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)
    product = subscriber_preview_data_v231.build_preview_payload(payload)
    rows = (product.get("edge_feed") or {}).get("rows") or []
    candidate = next((row for row in rows if str(((row.get("match") or {}).get("fixture_id"))) == fixture_id), None)
    if not isinstance(candidate, dict):
        return JSONResponse({"error": "FIXTURE_NOT_IN_PERSISTED_FEED"}, status_code=404)
    detail = subscriber_preview_data_v231._match_detail(payload, candidate)
    return JSONResponse({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "fixture_id": fixture_id,
        "selected": candidate,
        "detail": detail,
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
        "source_of_truth": "PANEL_DE_ANALISIS_SOCCER_EDGE_MOCKUP",
        "pages": ["Today", "Edge Feed", "Matches", "Markets", "Performance", "My Edge", "Control Tower", "Research Lab"],
        "match_tabs": ["Overview", "Goals", "Corners", "Cards", "Players", "Market", "Model"],
        "dynamic_match_selection": True,
        "explorer_missing_premium_values_are_redacted": True,
        "browser_sends_price_id": False,
        "billing_market_selection": "EXPLICIT_USER_CHOICE",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
