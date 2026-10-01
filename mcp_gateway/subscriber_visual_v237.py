from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_VISUAL_V237_1.0.0"

_STYLE = r'''
<style id="SOCCER_V237_VISUAL_STYLE">
/* Approved mockup parity: logos are transparent crests, not circular avatars. */
#today .v236-crest{background:transparent!important;border:0!important;border-radius:0!important;overflow:visible!important}
#today .v236-crest img{width:100%!important;height:100%!important;object-fit:contain!important;filter:drop-shadow(0 4px 8px rgba(0,0,0,.22))}
#today .v236-crest>span{display:none!important}

#today .hero-main{padding:0!important;overflow:hidden;background:linear-gradient(135deg,#082132,#071823)!important;border-color:#1d506d!important}
.v237-hero{display:grid;grid-template-columns:minmax(280px,.92fr) minmax(300px,1.08fr);min-height:245px}
.v237-hero-left{padding:22px 24px;display:grid;align-content:center;border-right:1px solid #163a4f;background:radial-gradient(circle at 34% 48%,rgba(15,77,108,.32),transparent 52%)}
.v237-hero-kicker{font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#55e5b4;margin-bottom:17px}
.v237-faceoff{display:grid;grid-template-columns:minmax(90px,1fr) 34px minmax(90px,1fr);align-items:center;gap:8px}
.v237-side{display:grid;justify-items:center;gap:10px;min-width:0}
.v237-side .v236-crest{width:82px!important;height:82px!important;flex-basis:82px!important}
.v237-side strong{font-size:18px;line-height:1.15;text-align:center;max-width:180px;overflow-wrap:anywhere}
.v237-vs{font-weight:900;color:#8aa7b8;font-size:14px;text-align:center}
.v237-kickoff{margin-top:15px;text-align:center;color:#86a3b4;font-size:10px}
.v237-hero-right{padding:24px;display:grid;align-content:center;gap:16px}
.v237-market{font-size:22px;font-weight:950;line-height:1.12;color:#f4f9fb}
.v237-hero-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));background:#0a2231;border:1px solid #19445d;border-radius:10px;overflow:hidden}
.v237-hero-stat{padding:12px 14px;border-right:1px solid #17394d}.v237-hero-stat:last-child{border-right:0}
.v237-hero-stat small{display:block;font-size:8px;color:#7691a2;text-transform:uppercase}.v237-hero-stat b{display:block;font-size:21px;margin-top:3px}.v237-hero-stat.edge b{color:#4de1ad}
.v237-badges{display:flex;gap:7px;flex-wrap:wrap}.v237-badges .badge{padding:6px 9px;font-size:9px}

#v236UpcomingPanel{margin-top:16px!important;border-radius:14px!important}
.v236-slate-head{padding:16px 18px!important}.v236-slate-head h3{font-size:17px!important}
.v236-fixture{grid-template-columns:110px minmax(0,1fr) 190px!important;min-height:92px!important;padding:12px 18px!important;gap:16px!important}
.v236-kickoff b{font-size:14px!important}.v236-kickoff{font-size:10px!important}
.v236-teams{display:grid!important;grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr)!important;gap:10px!important;align-items:center!important}
.v236-team{display:grid!important;grid-template-columns:44px minmax(0,1fr)!important;align-items:center!important;gap:10px!important}
.v236-team.away{grid-template-columns:44px minmax(0,1fr)!important}
.v236-team .v236-crest{width:44px!important;height:44px!important;flex-basis:44px!important}
.v236-team strong{font-size:13px!important;white-space:normal!important;line-height:1.2!important}
.v237-row-vs{font-size:10px;font-weight:900;color:#658397;text-align:center}
.v236-meta{gap:8px!important}.v236-state{font-size:8px!important;padding:6px 9px!important}.v236-league{font-size:9px!important}

#today .signal-list .v236-signal-crests .v236-crest{width:25px!important;height:25px!important;flex-basis:25px!important}

@media(max-width:860px){
 .v237-hero{grid-template-columns:1fr}.v237-hero-left{border-right:0;border-bottom:1px solid #163a4f;padding:18px}.v237-hero-right{padding:18px}.v237-side .v236-crest{width:70px!important;height:70px!important;flex-basis:70px!important}.v237-side strong{font-size:16px}.v237-market{font-size:19px}
 .v236-fixture{grid-template-columns:74px minmax(0,1fr)!important}.v236-meta{grid-column:2!important}.v236-teams{grid-template-columns:minmax(0,1fr) 18px minmax(0,1fr)!important}.v236-team{grid-template-columns:38px minmax(0,1fr)!important;gap:7px!important}.v236-team .v236-crest{width:38px!important;height:38px!important;flex-basis:38px!important}.v236-team strong{font-size:11px!important}.v237-row-vs{font-size:9px}
}
@media(max-width:520px){
 .v237-faceoff{grid-template-columns:minmax(0,1fr) 26px minmax(0,1fr)}.v237-side .v236-crest{width:62px!important;height:62px!important;flex-basis:62px!important}.v237-side strong{font-size:14px}.v237-hero-stats{grid-template-columns:1fr}.v237-hero-stat{border-right:0;border-bottom:1px solid #17394d;display:flex;align-items:center;justify-content:space-between}.v237-hero-stat:last-child{border-bottom:0}.v237-hero-stat b{font-size:18px}
 .v236-fixture{grid-template-columns:62px minmax(0,1fr)!important;padding:11px 12px!important}.v236-teams{grid-template-columns:1fr!important;gap:6px!important}.v237-row-vs{display:none}.v236-team{grid-template-columns:34px minmax(0,1fr)!important}.v236-team .v236-crest{width:34px!important;height:34px!important;flex-basis:34px!important}
}
</style>
'''

_SCRIPT = r'''
<script id="SOCCER_V237_VISUAL_SCRIPT">
(()=>{
 const AK='soccer_edge_access_token',ASSET_V='237';
 const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const token=()=>localStorage.getItem(AK)||'';
 const crest=(url,name,cls='')=>`<span class="v236-crest ${cls}">${url?`<img loading="eager" decoding="async" src="${esc(url)}${String(url).includes('?')?'&':'?'}ui=${ASSET_V}" alt="${esc(name)} crest">`:''}<span>${esc(String(name||'FC').slice(0,2).toUpperCase())}</span></span>`;
 async function getJson(path,auth=true){const h={};if(auth&&token())h.Authorization=`Bearer ${token()}`;const r=await fetch(path,{headers:h,cache:'no-store'});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d}
 function fid(r){return r?.fixture_id??r?.match?.fixture_id??null}
 function identityRow(r,map){const id=fid(r),m=id!=null?(map[String(id)]||{}):{};return {...r,...m,fixture_id:id??m.fixture_id,home_team:r.home_team||r.home||r.match?.home||m.home_team,away_team:r.away_team||r.away||r.match?.away||m.away_team,league:r.league||r.match?.league||m.league,kickoff:r.kickoff||r.match?.kickoff||m.kickoff,home_team_logo:r.home_team_logo||m.home_team_logo,away_team_logo:r.away_team_logo||m.away_team_logo}}
 function lookupByLabel(rows,map,label){const target=String(label||'').trim().toLowerCase();for(const base of rows){const r=identityRow(base,map),candidate=`${r.home_team||''} vs ${r.away_team||''}`.toLowerCase();if(candidate===target)return r}return null}
 function recomposeHero(rows,map){
   const hero=document.querySelector('#today .hero-main');if(!hero||hero.dataset.v237==='1')return;
   const matchName=hero.querySelector('.match-name')?.textContent?.trim();const r=lookupByLabel(rows,map,matchName);if(!r)return;
   const kicker=hero.querySelector('.eyebrow')?.textContent?.trim()||r.league||'Football Intelligence';
   const market=hero.querySelector('.market-name')?.textContent?.trim()||'Market';
   const vals=[...hero.querySelectorAll('.triple .stat b')].map(x=>x.textContent?.trim()||'—');
   const badges=hero.querySelector('.badges')?.innerHTML||'';
   const kickoff=r.kickoff?new Intl.DateTimeFormat(undefined,{weekday:'short',hour:'2-digit',minute:'2-digit'}).format(new Date(r.kickoff)):'Kickoff TBD';
   hero.dataset.v237='1';hero.innerHTML=`<div class="v237-hero"><div class="v237-hero-left"><div class="v237-hero-kicker">${esc(kicker)}</div><div class="v237-faceoff"><div class="v237-side">${crest(r.home_team_logo,r.home_team)}<strong>${esc(r.home_team)}</strong></div><div class="v237-vs">VS</div><div class="v237-side">${crest(r.away_team_logo,r.away_team)}<strong>${esc(r.away_team)}</strong></div></div><div class="v237-kickoff">${esc(kickoff)}</div></div><div class="v237-hero-right"><div class="v237-market">${esc(market)}</div><div class="v237-hero-stats"><div class="v237-hero-stat"><small>Model</small><b>${esc(vals[0]||'—')}</b></div><div class="v237-hero-stat"><small>Market</small><b>${esc(vals[1]||'—')}</b></div><div class="v237-hero-stat edge"><small>Edge</small><b>${esc(vals[2]||'—')}</b></div></div><div class="v237-badges">${badges}</div></div></div>`;
   hero.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{img.style.display='none'}));
 }
 function fixSlateRows(){
   document.querySelectorAll('#v236UpcomingPanel .v236-teams').forEach(teams=>{if(teams.querySelector('.v237-row-vs'))return;const home=teams.querySelector('.v236-team.home'),away=teams.querySelector('.v236-team.away');if(!home||!away)return;const vs=document.createElement('span');vs.className='v237-row-vs';vs.textContent='VS';home.insertAdjacentElement('afterend',vs)});
   document.querySelectorAll('#v236UpcomingPanel img').forEach(img=>{if(img.dataset.v237==='1')return;img.dataset.v237='1';const u=new URL(img.src,location.href);u.searchParams.set('ui',ASSET_V);img.src=u.toString()});
 }
 async function run(){
   try{let data;try{data=await getJson('/app/data',true)}catch(_){data=await getJson('/app/data',false)}const slate=data?.public?.verified_slate||data?.pro?.todays_slate||{},rows=slate.rows||[];const ids=[...new Set(rows.map(fid).filter(Boolean))];let map={};if(ids.length){const d=await getJson(`/app/fixture-identities?ids=${encodeURIComponent(ids.join(','))}`,false);map=Object.fromEntries((d.rows||[]).map(x=>[String(x.fixture_id),x]))}recomposeHero(rows,map);fixSlateRows()}catch(_){fixSlateRows()}
 }
 [350,1200,2600].forEach(ms=>setTimeout(run,ms));
 const obs=new MutationObserver(()=>{fixSlateRows();const h=document.querySelector('#today .hero-main');if(h&&!h.dataset.v237)setTimeout(run,30)});setTimeout(()=>{const today=document.getElementById('today');if(today)obs.observe(today,{childList:true,subtree:true})},200);
})();
</script>
'''


def inject(html: str) -> str:
    fragment = _STYLE + _SCRIPT
    marker = "</body>"
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment


def install(subscriber_product_module: Any) -> None:
    if hasattr(subscriber_product_module, "_v237_base_product_html"):
        return
    base = subscriber_product_module.product_html
    subscriber_product_module._v237_base_product_html = base
    subscriber_product_module.product_html = lambda: inject(base())


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "visual_target": "APPROVED_MOCKUP_TEAM_CREST_PARITY",
        "asset_cache_buster": "ui=237",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
