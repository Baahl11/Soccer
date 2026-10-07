from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_TODAY_V236_1.0.0"


def _positive_ids(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for piece in str(raw or "").split(","):
        try:
            value = int(piece.strip())
        except (TypeError, ValueError):
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= 100:
            break
    return values


def _logo(team_id: Any) -> str | None:
    try:
        value = int(team_id)
    except (TypeError, ValueError):
        return None
    return f"https://media.api-sports.io/football/teams/{value}.png" if value > 0 else None


def _load_fixture_identities(fixture_ids: list[int]) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []
    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, kickoff, status, league, country,
                       home_team_id, home_team, away_team_id, away_team
                FROM soccer_fixtures
                WHERE fixture_id = ANY(%s)
                """,
                (fixture_ids,),
            )
            rows = cur.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        fixture_id, kickoff, status, league, country, home_id, home, away_id, away = row
        result.append(
            {
                "fixture_id": fixture_id,
                "kickoff": kickoff.isoformat() if hasattr(kickoff, "isoformat") else kickoff,
                "status": status,
                "league": league,
                "country": country,
                "home_team_id": home_id,
                "home_team": home,
                "home_team_logo": _logo(home_id),
                "away_team_id": away_id,
                "away_team": away,
                "away_team_logo": _logo(away_id),
            }
        )
    return result


async def fixture_identities(request: Request) -> JSONResponse:
    """Public presentation metadata for already-persisted fixtures.

    This endpoint never calls API-Football. It only joins fixture/team identity from
    the local Postgres fixture registry so the UI can render names and crests.
    """
    fixture_ids = _positive_ids(request.query_params.get("ids", ""))
    try:
        rows = await asyncio.to_thread(_load_fixture_identities, fixture_ids)
        status = "OK"
        detail = None
    except Exception as exc:
        rows = []
        status = "IDENTITY_UNAVAILABLE"
        detail = str(exc)[:160]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "rows": rows,
        "requested": len(fixture_ids),
        "returned": len(rows),
        "source": "POSTGRES_SOCCER_FIXTURES",
        "provider_requests_added": 0,
    }
    if detail:
        payload["detail"] = detail
    return JSONResponse(payload, headers={"Cache-Control": "public, max-age=60"})


_STYLE = r'''
<style id="SOCCER_V236_TODAY_STYLE">
#today .metrics{gap:11px;margin-bottom:14px}
#today .metric{padding:16px 15px;min-height:76px}
#today .metric .n{font-size:27px;line-height:1.05}
#today .metric .label{font-size:9px;margin-top:6px}
#today .panel{border-color:#1a4057}
#today .ph h3{font-size:13px}
#today .hero-main{padding:18px}
#today .hero-main .match-name{font-size:20px;line-height:1.25}
#today .hero-main .market-name{font-size:22px}
#today .signal b{font-size:11px}
#today .signal small{font-size:9.5px}
#today .bottom-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;gap:12px!important}
#today .bottom-grid>.panel{min-width:0}
#today .bottom-grid .compact-row b{font-size:10px}
#today .bottom-grid .compact-row span{font-size:9px}
.v236-upcoming{padding:0!important;margin-top:2px!important;overflow:hidden;box-shadow:0 18px 45px rgba(0,0,0,.18)!important}
.v236-slate-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 18px;border-bottom:1px solid #17384c;background:linear-gradient(180deg,#0b2231,#091a27)}
.v236-slate-head h3{margin:0;font-size:15px!important}.v236-slate-sub{display:block;margin-top:3px;color:#6f899b;font-size:9px;font-weight:500}
.v236-slate-count{display:inline-flex;align-items:center;border:1px solid #24607e;background:#0a2a3c;color:#74ceff;border-radius:999px;padding:6px 10px;font-size:9px;font-weight:900;white-space:nowrap}
.v236-slate-list{display:grid;background:linear-gradient(180deg,#081923,#07151e)}
.v236-fixture{display:grid;grid-template-columns:92px minmax(320px,1fr) 190px;align-items:center;gap:18px;padding:14px 18px;border-bottom:1px solid #102d3e;min-height:82px;transition:background .14s ease,border-color .14s ease}
.v236-fixture:hover{background:#0a2030;border-bottom-color:#19465f}.v236-fixture:last-child{border-bottom:0}
.v236-kickoff{color:#708da0;font-size:9px;line-height:1.4}.v236-kickoff b{display:block;color:#e1edf3;font-size:13px;margin-bottom:2px}.v236-kickoff .v236-date{color:#668496}
.v236-teams{display:grid;gap:8px;min-width:0}.v236-team{display:flex;align-items:center;gap:10px;min-width:0}.v236-team strong{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v236-team.home strong{color:#f0f7fa}.v236-team.away strong{color:#d5e2e9}
.v236-crest{width:32px;height:32px;flex:0 0 32px;border-radius:50%;display:grid;place-items:center;background:#0d2a3b;border:1px solid #23516b;overflow:hidden;position:relative}.v236-crest img{width:27px;height:27px;object-fit:contain;position:relative;z-index:2}.v236-crest span{font-size:8px;font-weight:950;color:#8db5c8;position:absolute;inset:0;display:grid;place-items:center}
.v236-meta{display:grid;justify-items:end;gap:7px;text-align:right;min-width:0}.v236-league{max-width:190px;color:#7796a9;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v236-state{border:1px solid #285873;background:#0a2535;color:#78cfff;border-radius:999px;padding:5px 9px;font-size:8px;font-weight:900;text-transform:uppercase;white-space:nowrap}.v236-state.ready{border-color:#267258;background:#0b3025;color:#65e7b6}.v236-state.wait{border-color:#6a5425;background:#2a220f;color:#e8be66}.v236-state.research{border-color:#2b5f88;background:#112b46;color:#73bdff}.v236-state.pass{border-color:#344b5b;background:#15222c;color:#a8bdc9}
.v236-slate-foot{padding:10px 18px;border-top:1px solid #102c3c;color:#628092;font-size:8.5px;line-height:1.5;background:#07141d}
.v236-hero-identity{display:flex;align-items:center;gap:8px;margin-bottom:8px}.v236-hero-identity .v236-crest{width:36px;height:36px;flex-basis:36px}.v236-hero-identity .v236-crest img{width:30px;height:30px}.v236-hero-vs{font-size:9px;color:#668698;font-weight:800}.v236-signal-crests{display:inline-flex;gap:3px;margin-right:6px;vertical-align:middle}.v236-signal-crests .v236-crest{width:19px;height:19px;flex-basis:19px}.v236-signal-crests .v236-crest img{width:16px;height:16px}.v236-signal-crests .v236-crest span{font-size:6px}
@media(max-width:1180px){.v236-fixture{grid-template-columns:78px minmax(230px,1fr) 150px;gap:12px}.v236-meta{justify-items:end}.v236-league{max-width:150px}}
@media(max-width:760px){#today .metrics{grid-template-columns:repeat(2,minmax(0,1fr))}#today .metric:last-child{grid-column:1/-1}#today .bottom-grid{grid-template-columns:1fr!important}.v236-slate-head{padding:14px}.v236-fixture{grid-template-columns:62px minmax(0,1fr);gap:10px;padding:12px 14px}.v236-meta{grid-column:2;display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;text-align:left}.v236-league{max-width:55vw}.v236-team strong{font-size:11px}.v236-crest{width:29px;height:29px;flex-basis:29px}.v236-crest img{width:24px;height:24px}.v236-slate-count{font-size:8px}.v236-slate-head h3{font-size:14px!important}}
</style>
'''


_SCRIPT = r'''
<script id="SOCCER_V236_TODAY_SCRIPT">
(() => {
  const AK='soccer_edge_access_token';
  let cachedRows=[];
  let identityByFixture={};
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=v=>String(v??'').trim();
  const initials=name=>text(name).split(/\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase()||'FC';
  const token=()=>localStorage.getItem(AK)||'';

  async function fetchJson(path, withAuth=true){
    const headers={};if(withAuth&&token())headers.Authorization=`Bearer ${token()}`;
    const r=await fetch(path,{headers,cache:'no-store'});const d=await r.json().catch(()=>({}));
    if(!r.ok)throw Object.assign(new Error(d.error||`HTTP ${r.status}`),{status:r.status});return d;
  }
  async function loadAppData(){
    try{return await fetchJson('/app/data',true)}catch(e){if(e.status===401&&token())return await fetchJson('/app/data',false);throw e}
  }
  function fixtureId(r){return r?.fixture_id??r?.match?.fixture_id??null}
  function uniqueFixtures(rows){
    const seen=new Set(),out=[];
    for(const r of rows||[]){if(!r||typeof r!=='object')continue;const id=fixtureId(r);const key=id!=null?`id:${id}`:`${r.kickoff||''}|${r.home_team||r.home||''}|${r.away_team||r.away||''}`;if(seen.has(key))continue;seen.add(key);out.push(r)}
    out.sort((a,b)=>{const x=Date.parse(a.kickoff||a.match?.kickoff||''),y=Date.parse(b.kickoff||b.match?.kickoff||'');return (Number.isFinite(x)?x:9e15)-(Number.isFinite(y)?y:9e15)});return out;
  }
  async function loadIdentities(rows){
    const ids=[...new Set(rows.map(fixtureId).filter(v=>Number.isFinite(Number(v))).map(Number))];if(!ids.length)return{};
    try{const d=await fetchJson(`/app/fixture-identities?ids=${encodeURIComponent(ids.join(','))}`,false);return Object.fromEntries((d.rows||[]).map(r=>[String(r.fixture_id),r]))}catch(_){return{}}
  }
  function merged(r){
    const id=fixtureId(r),m=id!=null?(identityByFixture[String(id)]||{}):{};
    return {...r,...m,fixture_id:id??m.fixture_id,home_team:r.home_team||r.home||r.match?.home||m.home_team,away_team:r.away_team||r.away||r.match?.away||m.away_team,league:r.league||r.match?.league||m.league,country:r.country||r.match?.country||m.country,kickoff:r.kickoff||r.match?.kickoff||m.kickoff,status:r.execution_status||r.status||r.stage||r.state?.status||m.status,home_team_logo:r.home_team_logo||r.home_logo||m.home_team_logo,away_team_logo:r.away_team_logo||r.away_logo||m.away_team_logo};
  }
  function stateInfo(raw){
    const s=text(raw||'Fixture Only').toUpperCase();
    if(['READY','BET','LIVE'].some(x=>s.includes(x)))return['Ready','ready'];
    if(s.includes('WAIT_XI')||s.includes('WAIT GK')||s.includes('WAIT_GK')||s.includes('LINEUP'))return['XI Pending','wait'];
    if(s.includes('WAIT_PRICE')||s.includes('WAIT FRESH')||s.includes('WAIT_FRESH'))return['Odds Pending','wait'];
    if(s.includes('RESEARCH'))return['Research Only','research'];
    if(s.includes('PASS'))return['No Bet Yet','pass'];
    if(s.includes('DEEP'))return['Deep Pending','wait'];
    return [s.replaceAll('_',' ')||'Fixture Only','research'];
  }
  function kickoff(raw){
    if(!raw)return['TBD','Kickoff'];const d=new Date(raw);if(Number.isNaN(d.getTime()))return[text(raw).slice(0,8),'Kickoff'];
    const time=new Intl.DateTimeFormat(undefined,{hour:'2-digit',minute:'2-digit'}).format(d);const date=new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric'}).format(d);return[time,date];
  }
  function crest(url,name){return `<span class="v236-crest">${url?`<img loading="lazy" decoding="async" src="${esc(url)}" alt="${esc(name)} crest">`:''}<span>${esc(initials(name))}</span></span>`}
  function ensurePanel(){
    let panel=document.getElementById('v236UpcomingPanel');if(panel)return panel;
    const bottom=document.querySelector('#today .bottom-grid');if(!bottom)return null;
    panel=bottom.querySelector(':scope > .panel:nth-child(3)');if(!panel){panel=document.createElement('div');panel.className='panel'}
    panel.id='v236UpcomingPanel';panel.classList.add('v236-upcoming');
    const grid=document.querySelector('#today .grid2');if(grid)grid.insertAdjacentElement('afterend',panel);
    return panel;
  }
  function renderSlate(){
    const panel=ensurePanel();if(!panel)return;const rows=cachedRows.map(merged);const count=rows.length;
    panel.innerHTML=`<div class="v236-slate-head"><div><h3>Upcoming Matches</h3><span class="v236-slate-sub">Full fixture browser · intelligence enriches progressively</span></div><span class="v236-slate-count">Full Slate (${count})</span></div><div class="v236-slate-list">${rows.length?rows.map(r=>{const [time,date]=kickoff(r.kickoff),home=r.home_team||'Home',away=r.away_team||'Away',[label,tone]=stateInfo(r.status);return `<article class="v236-fixture" data-fixture-id="${esc(r.fixture_id||'')}"><div class="v236-kickoff"><b>${esc(time)}</b><span class="v236-date">${esc(date)}</span></div><div class="v236-teams"><div class="v236-team home">${crest(r.home_team_logo,home)}<strong>${esc(home)}</strong></div><div class="v236-team away">${crest(r.away_team_logo,away)}<strong>${esc(away)}</strong></div></div><div class="v236-meta"><span class="v236-league">${esc(r.league||r.country||'Competition')}</span><span class="v236-state ${tone}">${esc(label)}</span></div></article>`}).join(''):'<div class="v231-empty">No upcoming persisted fixtures in the latest slate.</div>'}</div><div class="v236-slate-foot">Every fixture remains visible even before deep analysis. Missing model, market or XI data stays missing instead of hiding the match.</div>`;
    panel.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{img.style.display='none'}));
  }
  function matchIdentityByLabel(label){
    const want=text(label).toLowerCase();if(!want)return null;
    for(const base of cachedRows){const r=merged(base),candidate=`${text(r.home_team)} vs ${text(r.away_team)}`.toLowerCase();if(candidate===want)return r}return null;
  }
  function decorateHero(){
    const hero=document.querySelector('#today .hero-main'),name=hero?.querySelector('.match-name');if(!hero||!name)return;hero.querySelector('.v236-hero-identity')?.remove();const r=matchIdentityByLabel(name.textContent);if(!r)return;
    const block=document.createElement('div');block.className='v236-hero-identity';block.innerHTML=`${crest(r.home_team_logo,r.home_team)}<span class="v236-hero-vs">VS</span>${crest(r.away_team_logo,r.away_team)}`;name.insertAdjacentElement('beforebegin',block);block.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{img.style.display='none'}));
  }
  function decorateSignals(){
    document.querySelectorAll('#today .signal-list .signal b').forEach(b=>{if(b.parentElement?.querySelector('.v236-signal-crests'))return;const r=matchIdentityByLabel(b.textContent);if(!r)return;const span=document.createElement('span');span.className='v236-signal-crests';span.innerHTML=`${crest(r.home_team_logo,r.home_team)}${crest(r.away_team_logo,r.away_team)}`;b.insertAdjacentElement('beforebegin',span);span.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{img.style.display='none'}))});
  }
  function waitFlag(flag,eventName,timeout=2500){
    if(window[flag])return Promise.resolve();
    return new Promise(resolve=>{let done=false,timer=null;const finish=()=>{if(done)return;done=true;if(timer)clearTimeout(timer);window.removeEventListener(eventName,finish);resolve()};window.addEventListener(eventName,finish,{once:true});timer=setTimeout(finish,timeout)});
  }
  async function refresh(dataOverride=null){
    try{const data=dataOverride||window.__SOCCER_EDGE_APP_DATA__||await loadAppData(),slate=data?.public?.verified_slate||data?.pro?.todays_slate||{};cachedRows=uniqueFixtures(slate.rows||[]);identityByFixture=await loadIdentities(cachedRows);renderSlate();decorateHero();decorateSignals();return true}catch(_){renderSlate();return false}
  }
  async function start(data){
    await Promise.all([waitFlag('__SOCCER_EDGE_V234_READY__','soccer-edge:v234-ready'),waitFlag('__SOCCER_EDGE_V235_READY__','soccer-edge:v235-ready')]);
    await refresh(data||window.__SOCCER_EDGE_APP_DATA__||null);
    window.__SOCCER_EDGE_TODAY_DATA__={rows:cachedRows,identities:identityByFixture};window.__SOCCER_EDGE_TODAY_READY__=true;
    window.dispatchEvent(new CustomEvent('soccer-edge:today-ready',{detail:window.__SOCCER_EDGE_TODAY_DATA__}));
  }
  if(window.__SOCCER_EDGE_APP_READY__)start(window.__SOCCER_EDGE_APP_DATA__);else window.addEventListener('soccer-edge:app-data-ready',e=>start(e?.detail||null),{once:true});
  window.addEventListener('focus',()=>refresh(window.__SOCCER_EDGE_APP_DATA__||null));
})();
</script>
'''


def inject(html: str) -> str:
    fragment = _STYLE + _SCRIPT
    marker = "</body>"
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment


def install(subscriber_product_module: Any) -> None:
    if hasattr(subscriber_product_module, "_v236_base_product_html"):
        return
    base = subscriber_product_module.product_html
    subscriber_product_module._v236_base_product_html = base

    def product_html_v236() -> str:
        return inject(base())

    subscriber_product_module.product_html = product_html_v236


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "today_layout": "MOCKUP_ALIGNED_FULL_WIDTH_SLATE",
        "fixture_identity_source": "POSTGRES_SOCCER_FIXTURES",
        "team_crest_source": "API_SPORTS_MEDIA_FROM_PERSISTED_TEAM_ID",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
