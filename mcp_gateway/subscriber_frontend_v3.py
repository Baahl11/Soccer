from __future__ import annotations

import gzip
from pathlib import Path

_PAYLOAD_PATH = Path(__file__).with_name("subscriber_frontend_v3.py.gz")
_SOURCE = gzip.decompress(_PAYLOAD_PATH.read_bytes()).decode("utf-8")
exec(compile(_SOURCE, str(Path(__file__).with_suffix(".source.py")), "exec"), globals(), globals())

# V3 presentation compatibility shim.
# The V2 Today contract wraps registry identity under row.fixture while the
# transitional V3 preview expects a flattened row. Normalize only the browser
# payload before React consumes it. No model, market, provider, or persistence
# logic is changed.
_v3_source_html = _html

def _html() -> str:
    rendered = _v3_source_html()
    shim = r"""
<script>
(() => {
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const response = await originalFetch(input, init);
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (response.ok && url.includes('/app/api/v2/today')) {
      try {
        const payload = await response.clone().json();
        if (payload && payload.slate && Array.isArray(payload.slate.rows)) {
          payload.slate.rows = payload.slate.rows.map(row =>
            row && row.fixture ? Object.assign({}, row, row.fixture) : row
          );
          return new Response(JSON.stringify(payload), {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
          });
        }
      } catch (_) {}
    }
    if (response.ok && url.includes('/app/api/v2/match/')) {
      response.clone().json().then(payload => {
        window.__V3_MATCH_DATA__ = payload;
        window.dispatchEvent(new CustomEvent('v3-match-data', {detail: payload}));
      }).catch(() => {});
    }
    return response;
  };
})();
</script>
"""
    visual_overrides = r"""
<style>
/* V3 premium composition pass — screenshot target 1000164422.jpg */
:root{
  --panel:#071822;
  --panel2:#091f2b;
  --line:#153a4d;
  --line2:#1e536d;
}
main{max-width:1180px}
.match-center{background:#05131c;border-color:#173d50;box-shadow:0 28px 70px #0009}
.mc-hero{padding:14px 18px 13px;background:
  radial-gradient(circle at 50% -20%,rgba(28,123,162,.22),transparent 47%),
  linear-gradient(180deg,#071d29 0%,#05141d 100%)}
.mc-top{min-height:32px}.mc-top .brand>span{box-shadow:0 0 26px rgba(11,117,184,.28)}
.mc-meta{margin-top:7px}
.faceoff{margin:11px auto 8px;max-width:650px}
.team-face .crest{box-shadow:0 10px 24px #0005}
.hero-status{margin-top:4px}
.status-strip>div{background:linear-gradient(180deg,#071923,#06131b)}
.tabs{background:#06151e}
.tabs button.active{box-shadow:inset 0 0 0 1px #2b6c8a}
.mc-body{padding:8px}
.feature-callout{padding:11px 13px;border-radius:10px;margin-bottom:7px;box-shadow:inset 0 1px 0 rgba(255,255,255,.015)}
.feature-callout h2{max-width:760px}
.dashboard-grid{gap:7px}
.panel{margin-bottom:7px;border-radius:10px;padding:10px;box-shadow:inset 0 1px 0 rgba(255,255,255,.012)}
.panel>header{margin-bottom:8px}
.panel h3{font-size:10px}
.rate-bars{gap:8px}
.rate-track{height:7px}
.coverage-panel{justify-content:flex-start}
.donut-wrap{width:82px;height:82px}
.donut-label strong{font-size:16px}
.form-chart{height:64px}
.takeaways{gap:5px}
.takeaways>div{padding:7px}
.missing-compact{min-height:56px;padding:7px}
.missing-compact span{font-size:16px}
.dashboard-grid.primary>.panel:has(.missing-compact){min-height:0}
.dashboard-grid.primary>.panel:has(.missing-compact) .missing-compact{min-height:58px}
.sport-grid .panel:has(.missing-compact){min-height:92px}
.metric-grid>div,.availability-grid>div{background:#061820}
.heat-cell{box-shadow:inset 0 0 0 1px rgba(255,255,255,.015)}
@media(max-width:820px){
  .mc-hero{padding:9px 10px 10px}
  .mc-top{min-height:28px}
  .mc-top .brand>span{width:29px;height:29px}
  .mc-meta{margin-top:5px}
  .faceoff{margin:8px auto 6px}
  .team-face .crest{width:50px;height:50px;flex-basis:50px;border-radius:13px}
  .team-face b{font-size:12px}
  .status-strip>div{padding:7px 8px}
  .status-strip b{font-size:8.5px}
  .tabs{padding:4px 6px}
  .mc-body{padding:6px}
  .feature-callout{padding:9px 10px}
  .feature-callout h2{font-size:10px;line-height:1.28}
  .feature-callout p{font-size:6px}
  .dashboard-grid.primary{grid-template-columns:1fr 1fr;gap:6px}
  .dashboard-grid.primary .wide{grid-column:1/-1;grid-row:auto}
  .dashboard-grid.primary>.panel:not(.wide){min-height:132px}
  .dashboard-grid.primary>.panel:has(.missing-compact){min-height:112px}
  .dashboard-grid.primary>.panel:has(.missing-compact) .missing-compact{min-height:64px}
  .coverage-panel{display:grid;grid-template-columns:1fr;justify-items:center;text-align:center;gap:4px}
  .donut-wrap{width:64px;height:64px}
  .donut-label strong{font-size:13px}
  .coverage-panel>div:last-child strong{font-size:7.5px}
  .coverage-panel>div:last-child span{font-size:5px}
  .form-chart{height:52px}
  .form-legends{gap:3px}
  .form-dots i{width:14px;height:14px}
  .takeaways{grid-template-columns:1fr 1fr}
  .sport-grid{grid-template-columns:1fr 1fr;gap:6px}
  .sport-grid .panel{min-height:132px}
  .sport-grid .panel:has(.missing-compact){min-height:98px}
  .xg b{font-size:18px}
  .dist-bars{height:74px}
}
@media(max-width:390px){
  .dashboard-grid.primary{grid-template-columns:1fr 1fr}
  .dashboard-grid.primary .wide{grid-column:1/-1}
  .sport-grid{grid-template-columns:1fr 1fr}
  .sport-grid .panel{min-height:124px}
  .takeaways{grid-template-columns:1fr}
  .team-face b{max-width:122px}
}
</style>
"""
    premium_pass = r"""
<style>
/* V3 premium pass 2 — closer to approved Match Center composition */
.match-center{position:relative}
.match-center::before{content:'';position:absolute;inset:0;pointer-events:none;opacity:.18;background:
linear-gradient(rgba(41,110,140,.08) 1px,transparent 1px),
linear-gradient(90deg,rgba(41,110,140,.06) 1px,transparent 1px);background-size:28px 28px}
.match-center>*{position:relative;z-index:1}
.mc-hero{min-height:0}
.mc-top{padding-bottom:1px}
.mc-meta b{font-weight:700;color:#829baa}
.faceoff{max-width:560px}
.team-face .crest{background:linear-gradient(180deg,#0d3142,#092331);border-color:#23546b}
.status-strip{box-shadow:inset 0 1px 0 rgba(255,255,255,.015)}
.tabs{backdrop-filter:blur(10px)}
.feature-callout{display:grid;grid-template-columns:minmax(0,1fr) 34px;align-items:center}
.feature-callout::after{content:'';position:absolute;inset:auto 14px 0 14px;height:1px;background:linear-gradient(90deg,transparent,#2b6b8655,transparent)}
.dashboard-grid.primary>.panel.wide{background:
radial-gradient(circle at 92% 0,rgba(65,174,226,.08),transparent 34%),
linear-gradient(180deg,#081c28,#06151e)}
.dashboard-grid.primary>.panel:not(.wide){background:linear-gradient(180deg,#071a25,#06151e)}
.panel.v3-goal{border-color:#1b465a}
.panel.v3-coverage{border-color:#1d4f53}
.panel.v3-form{border-color:#194458}
.panel.v3-takeaways{background:linear-gradient(180deg,#071923,#06141d)}
.v3-goal .rate-row b{font-size:9px}
.v3-goal .rate-track{box-shadow:inset 0 0 0 1px rgba(255,255,255,.015)}
.v3-coverage .coverage-panel{min-height:82px}
.v3-form .form-chart{border-left:1px solid #173847;border-bottom:1px solid #173847;background:
repeating-linear-gradient(to top,transparent 0,transparent 18px,#102c3a 19px)}
.v3-form .form-chart circle{filter:drop-shadow(0 0 4px currentColor)}
.v3-takeaways .takeaways>div{background:linear-gradient(180deg,#071b25,#071720)}
.sport-grid>.panel:nth-child(1){border-color:#245941}.sport-grid>.panel:nth-child(2){border-color:#4a4425}
.sport-grid>.panel:nth-child(3){border-color:#1d4960}.sport-grid>.panel:nth-child(4){border-color:#1d4e47}
.heat-grid{background:linear-gradient(180deg,#071a25,#06151d);padding:4px;border-radius:7px}
.market-warning{box-shadow:inset 0 1px 0 rgba(255,255,255,.02)}
@media(max-width:820px){
  .mc-hero{padding-bottom:8px}
  .faceoff{max-width:500px}
  .status-strip{position:relative}
  .feature-callout{grid-template-columns:minmax(0,1fr) 31px}
  .dashboard-grid.primary>.panel.wide{min-height:118px}
  .dashboard-grid.primary>.panel:not(.wide){min-height:118px}
  .v3-coverage .coverage-panel{min-height:74px}
  .v3-form .form-chart{height:48px}
  .v3-takeaways{margin-top:1px}
  .v3-takeaways .takeaways>div{min-height:58px}
  .sport-grid>.panel{min-height:118px}
}
</style>
<script>
(() => {
  const NS='http://www.w3.org/2000/svg';
  function decorate(){
    document.querySelectorAll('.panel').forEach(panel=>{
      const title=(panel.querySelector('h3')?.textContent||'').trim().toLowerCase();
      panel.classList.toggle('v3-goal',title==='goal-rate baseline');
      panel.classList.toggle('v3-coverage',title==='input coverage');
      panel.classList.toggle('v3-form',title==='form momentum');
      panel.classList.toggle('v3-takeaways',title==='key sporting takeaways');
    });

    document.querySelectorAll('.form-chart').forEach(svg=>{
      if(svg.dataset.v3points==='1')return;
      const panel=svg.closest('.panel');
      const groups=[...panel.querySelectorAll('.form-dots')];
      if(!groups.length)return;
      const classes=['home','away'];
      groups.slice(0,2).forEach((g,gi)=>{
        const seq=[...g.querySelectorAll('i')].map(x=>x.textContent.trim().toUpperCase()).filter(Boolean);
        seq.forEach((x,i)=>{
          const cx=8+(i*(84/Math.max(1,seq.length-1)));
          const cy=x==='W'?10:x==='D'?27:44;
          const circle=document.createElementNS(NS,'circle');
          circle.setAttribute('cx',String(cx));
          circle.setAttribute('cy',String(cy));
          circle.setAttribute('r','3.2');
          circle.setAttribute('fill',gi===0?'#42aee2':'#48d19f');
          circle.setAttribute('stroke','#06131b');
          circle.setAttribute('stroke-width','1.2');
          svg.appendChild(circle);
        });
      });
      svg.dataset.v3points='1';
    });
  }
  const obs=new MutationObserver(decorate);
  obs.observe(document.documentElement,{childList:true,subtree:true});
  setTimeout(decorate,0);
})();
</script>
"""
    premium_board = r"""
<style>
/* Mockup intelligence board: probability / xG / edge / profile / matrix / distribution */
.v3-board{margin:0 0 7px;border:1px solid #174054;border-radius:11px;background:
radial-gradient(circle at 10% 0,rgba(55,170,218,.08),transparent 30%),
linear-gradient(180deg,#071923,#05141d);padding:8px;box-shadow:inset 0 1px 0 rgba(255,255,255,.018)}
.v3-board-head{display:flex;justify-content:space-between;align-items:flex-end;gap:8px;margin-bottom:6px}
.v3-board-head span{display:block;color:#4fddb0;font-size:5.5px;font-weight:950;letter-spacing:.13em}
.v3-board-head h3{margin:2px 0 0;font-size:9px}
.v3-board-head small{color:#627f8f;font-size:5px;font-weight:950;letter-spacing:.1em}
.v3-top-grid,.v3-bottom-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.v3-bottom-grid{margin-top:5px}
.v3-bi{min-width:0;border:1px solid #15394b;border-radius:8px;background:linear-gradient(180deg,#081b26,#06151e);padding:7px}
.v3-bi>header{display:flex;justify-content:space-between;gap:5px;align-items:center;margin-bottom:6px}
.v3-bi>header b{font-size:6.5px}.v3-bi>header span{font-size:4.5px;color:#607d8d;font-weight:900;letter-spacing:.06em}
.v3-probs{display:grid;grid-template-columns:repeat(3,1fr);gap:3px}.v3-p{padding:6px 2px;border-radius:6px;background:#0a202c;text-align:center;border:1px solid #173c50}
.v3-p span{display:block;font-size:4.5px;color:#6f8a99;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v3-p b{display:block;margin-top:3px;font-size:10px}
.v3-p.h b{color:#59ddb0}.v3-p.d b{color:#ddb456}.v3-p.a b{color:#55b8e3}.v3-probbar{display:flex;height:5px;border-radius:99px;overflow:hidden;margin-top:5px;background:#102b3a}
.v3-probbar i{height:100%}.v3-probbar .h{background:#4bdfa9}.v3-probbar .d{background:#dfb754}.v3-probbar .a{background:#42afe3}
.v3-xg{display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:3px;text-align:center;padding-top:4px}.v3-xg span{display:block;color:#6c8797;font-size:4.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v3-xg b{display:block;margin-top:3px;font-size:16px}.v3-xg em{font-style:normal;color:#496777;padding-bottom:4px;font-size:7px}
.v3-edge{display:grid;gap:5px}.v3-edge-row{display:grid;grid-template-columns:34px minmax(0,1fr) 30px;gap:4px;align-items:center}.v3-edge-row span{font-size:4.5px;color:#6c8797}
.v3-edge-track{height:5px;border-radius:99px;background:#102b39;overflow:hidden}.v3-edge-track i{display:block;height:100%;border-radius:99px}.v3-edge-track .model{background:#4bdfa9}.v3-edge-track .market{background:#42afe3}
.v3-edge-row b{font-size:5.5px;text-align:right}.v3-edge-big{margin-top:2px;color:#5ddfb1;font-size:11px;font-weight:950}
.v3-profile{display:grid;gap:4px}.v3-profile-row{display:grid;grid-template-columns:50px minmax(0,1fr) 20px;gap:4px;align-items:center}.v3-profile-row span,.v3-profile-row b{font-size:4.5px}.v3-profile-row span{color:#6d8796;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v3-profile-row div{height:4px;border-radius:99px;background:#102a39;overflow:hidden}.v3-profile-row i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#247a60,#54d6a8)}.v3-profile-row b{text-align:right}
.v3-heat{display:grid;gap:2px}.v3-hcell{aspect-ratio:1;border:1px solid #1b4b64;border-radius:2px;display:grid;place-items:center}.v3-hcell b{font-size:4px;color:white}.v3-axis{font-size:4px;color:#637f8f;display:grid;place-items:center}
.v3-dist{height:64px;display:grid;grid-template-columns:repeat(5,1fr);gap:3px;align-items:end;border-bottom:1px solid #173849}.v3-dcol{height:100%;display:grid;grid-template-rows:1fr auto;gap:2px;text-align:center}
.v3-dpair{height:100%;display:flex;justify-content:center;align-items:flex-end;gap:1px}.v3-dpair i{display:block;width:38%;min-height:2px;border-radius:2px 2px 0 0}.v3-dpair .h{background:#42afe3}.v3-dpair .a{background:#4bdfa9}.v3-dcol b{font-size:4.5px;color:#687f8d}
.v3-mini-missing{min-height:52px;display:grid;place-content:center;text-align:center;color:#607d8d;border:1px dashed #234657;border-radius:6px;font-size:5px;line-height:1.4;padding:5px}
.v3-fallback{display:grid;gap:5px}.v3-fallback-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px;align-items:center}.v3-fallback-row span{font-size:4.5px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v3-fallback-row b{font-size:6px}.v3-fallback-track{height:5px;border-radius:99px;background:#102a39;overflow:hidden}.v3-fallback-track i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2a83af,#53bee7)}
.v3-form-mini{display:grid;gap:5px}.v3-form-team{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px;align-items:center}.v3-form-team>span{font-size:4.5px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.v3-form-dots{display:flex;gap:2px}.v3-form-dots i{font-style:normal;width:14px;height:14px;border-radius:4px;display:grid;place-items:center;font-size:4.5px;font-weight:950;border:1px solid #25485a}.v3-form-dots .w{background:#0b2e24;color:#61dfb0;border-color:#24634d}.v3-form-dots .d{background:#2b230f;color:#e1ba62;border-color:#685327}.v3-form-dots .l{background:#2a151a;color:#ef8792;border-color:#66353c}
.v3-coverage-mini{display:grid;grid-template-columns:48px minmax(0,1fr);gap:6px;align-items:center}.v3-coverage-ring{--p:0%;width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#4bdfa9 var(--p),#123243 0);position:relative}.v3-coverage-ring::after{content:'';position:absolute;inset:6px;border-radius:50%;background:#071923}.v3-coverage-ring b{position:relative;z-index:1;font-size:9px}.v3-coverage-copy b{display:block;font-size:6px}.v3-coverage-copy span{display:block;margin-top:2px;font-size:4.5px;color:#6f8998;line-height:1.35}
.v3-gate{display:grid;gap:5px}.v3-gate b{font-size:7px}.v3-gate span{font-size:4.5px;color:#6f8998;line-height:1.35}.v3-gate-pill{display:inline-flex;width:max-content;max-width:100%;padding:4px 6px;border:1px solid #625326;border-radius:6px;background:#231d0c;color:#dfb754;font-size:4.5px;font-weight:900}
@media(max-width:820px){
 .v3-board{padding:6px;margin-bottom:6px}.v3-board-head{margin-bottom:5px}
 .v3-top-grid,.v3-bottom-grid{gap:4px}.v3-bi{padding:6px}
 .v3-p{padding:5px 1px}.v3-p b{font-size:9px}.v3-xg b{font-size:14px}.v3-edge-big{font-size:10px}
 .v3-profile-row{grid-template-columns:42px minmax(0,1fr) 18px}.v3-dist{height:56px}
}
@media(max-width:365px){
 .v3-top-grid{grid-template-columns:1.15fr .85fr}.v3-top-grid .v3-edge-card{grid-column:1/-1}
 .v3-bottom-grid{grid-template-columns:1fr 1fr}.v3-bottom-grid .v3-profile-card{grid-column:1/-1}
}
</style>
<script>
(() => {
  let matchData = window.__V3_MATCH_DATA__ || null;
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=v=>{if(v===null||v===undefined||v==='')return null;const x=Number(v);return Number.isFinite(x)?x:null};
  const p=v=>{const x=n(v);return x===null?null:(Math.abs(x)<=1?x*100:x)};
  const miss=t=>'<div class="v3-mini-missing">'+esc(t)+'<br>NOT VERIFIED</div>';
  const score=r=>{
    if(Number.isFinite(Number(r?.home_goals))&&Number.isFinite(Number(r?.away_goals)))return [Number(r.home_goals),Number(r.away_goals)];
    const m=String(r?.score||'').match(/^(\d+)\s*[-:]\s*(\d+)$/);return m?[Number(m[1]),Number(m[2])]:null;
  };

  function probs(d,f){
    const q=d?.sport_context?.outcome_probabilities||{},raw=[p(q.home),p(q.draw),p(q.away)];
    if(raw.some(x=>x===null))return miss('1X2 probabilities');
    const sum=raw.reduce((a,b)=>a+b,0)||100,vals=raw.map(x=>x/sum*100),labs=[f.home_team||'Home','Draw',f.away_team||'Away'],cls=['h','d','a'];
    return '<div class="v3-probs">'+vals.map((v,i)=>'<div class="v3-p '+cls[i]+'"><span>'+esc(labs[i])+'</span><b>'+v.toFixed(1)+'%</b></div>').join('')+'</div>'+
      '<div class="v3-probbar">'+vals.map((v,i)=>'<i class="'+cls[i]+'" style="width:'+v.toFixed(2)+'%"></i>').join('')+'</div>';
  }
  function xg(d,f){
    const q=d?.sport_context?.expected_goals||{},h=n(q.home),a=n(q.away);
    if(h===null||a===null)return miss('Expected goals');
    return '<div class="v3-xg"><div><span>'+esc(f.home_team||'Home')+'</span><b>'+h.toFixed(2)+'</b></div><em>—</em><div><span>'+esc(f.away_team||'Away')+'</span><b>'+a.toFixed(2)+'</b></div></div>';
  }
  function edge(d){
    const q=d?.projection_ladder||{},model=p(q.raw_sport_probability),market=p(q.fair_market_probability),edge=n(q.probability_edge_pp);
    if(model===null||market===null)return miss('Model vs market');
    const mx=Math.max(model,market,1);
    return '<div class="v3-edge">'+
      '<div class="v3-edge-row"><span>Model</span><div class="v3-edge-track"><i class="model" style="width:'+(model/mx*100).toFixed(1)+'%"></i></div><b>'+model.toFixed(1)+'%</b></div>'+
      '<div class="v3-edge-row"><span>Market</span><div class="v3-edge-track"><i class="market" style="width:'+(market/mx*100).toFixed(1)+'%"></i></div><b>'+market.toFixed(1)+'%</b></div>'+
      '<div class="v3-edge-big">'+(edge===null?'EDGE N/V':((edge>=0?'+':'')+edge.toFixed(1)+' pp'))+'</div></div>';
  }
  function profile(d){
    const rows=d?.sport_context?.sport_profile||[];
    if(!rows.length)return miss('Sport profile');
    return '<div class="v3-profile">'+rows.slice(0,6).map(r=>{
      const v=Math.max(0,Math.min(100,n(r.score)||0));
      return '<div class="v3-profile-row"><span>'+esc(r.label||r.key||'Metric')+'</span><div><i style="width:'+v+'%"></i></div><b>'+v.toFixed(0)+'</b></div>';
    }).join('')+'</div>';
  }
  function heat(d){
    const rows=(d?.sport_context?.score_matrix||[]).map(r=>({s:score(r),v:p(r.probability)})).filter(x=>x.s&&x.v!==null);
    if(!rows.length)return miss('Score matrix');
    const maxGoal=Math.min(3,Math.max(2,...rows.flatMap(x=>x.s))),mx=Math.max(1,...rows.map(x=>x.v)),map=new Map(rows.map(x=>[x.s[0]+'-'+x.s[1],x.v]));
    let out='<div class="v3-heat" style="grid-template-columns:12px repeat('+(maxGoal+1)+',1fr)"><span></span>';
    for(let a=0;a<=maxGoal;a++)out+='<span class="v3-axis">'+a+'</span>';
    for(let h=0;h<=maxGoal;h++){out+='<span class="v3-axis">'+h+'</span>';for(let a=0;a<=maxGoal;a++){const q=map.get(h+'-'+a)||0;out+='<div class="v3-hcell" style="background:rgba(39,151,205,'+Math.max(.05,q/mx).toFixed(2)+')"><b>'+(q?q.toFixed(0)+'%':'')+'</b></div>';}}
    return out+'</div>';
  }
  function dist(d){
    const rows=d?.sport_context?.score_matrix||[],b=['0','1','2','3','4+'],h=Object.fromEntries(b.map(k=>[k,0])),a=Object.fromEntries(b.map(k=>[k,0]));let found=false;
    rows.forEach(r=>{const s=score(r),q=p(r.probability);if(!s||q===null)return;found=true;h[s[0]>=4?'4+':String(s[0])]+=q;a[s[1]>=4?'4+':String(s[1])]+=q;});
    if(!found)return miss('Goal distribution');
    const mx=Math.max(1,...Object.values(h),...Object.values(a));
    return '<div class="v3-dist">'+b.map(k=>'<div class="v3-dcol"><div class="v3-dpair"><i class="h" style="height:'+(h[k]/mx*100).toFixed(1)+'%"></i><i class="a" style="height:'+(a[k]/mx*100).toFixed(1)+'%"></i></div><b>'+k+'</b></div>').join('')+'</div>';
  }
  function featureMap(d){return d?.relational_evidence?.feature_snapshots?.[0]?.payload?.features||{};}
  function fval(fs,key){const r=fs?.[key];return r&&r.value!==null&&r.value!==undefined?r.value:null;}
  function formSeq(v){return String(v||'').toUpperCase().replace(/[^WDL]/g,'').split('').slice(-5);}
  function formDots(seq){return '<div class="v3-form-dots">'+seq.map(x=>'<i class="'+x.toLowerCase()+'">'+x+'</i>').join('')+'</div>';}
  function rateFallback(d,f){
    const fs=featureMap(d),h=n(fval(fs,'team_performance.home_goal_rate_blend')),a=n(fval(fs,'team_performance.away_goal_rate_blend')),t=n(fval(fs,'team_performance.total_goal_rate_blend'));
    if(h===null&&a===null)return miss('Goal-rate baseline');
    const mx=Math.max(1,...[h,a,t].filter(x=>x!==null));
    const row=(name,v)=>'<div class="v3-fallback-row"><span>'+esc(name)+'</span><b>'+(v===null?'N/V':v.toFixed(2))+'</b></div><div class="v3-fallback-track"><i style="width:'+(v===null?0:Math.max(3,v/mx*100).toFixed(1))+'%"></i></div>';
    return '<div class="v3-fallback">'+row(f.home_team||'Home',h)+row(f.away_team||'Away',a)+(t!==null?row('Combined',t):'')+'</div>';
  }
  function formFallback(d,f){
    const fs=featureMap(d),h=formSeq(fval(fs,'team_performance.home_form')),a=formSeq(fval(fs,'team_performance.away_form'));
    if(!h.length&&!a.length)return miss('Recent form');
    return '<div class="v3-form-mini"><div class="v3-form-team"><span>'+esc(f.home_team||'Home')+'</span>'+formDots(h)+'</div><div class="v3-form-team"><span>'+esc(f.away_team||'Away')+'</span>'+formDots(a)+'</div></div>';
  }
  function coverageFallback(d){
    const fs=featureMap(d),keys=['team_performance.home_form','team_performance.away_form','team_performance.home_goal_rate_blend','team_performance.away_goal_rate_blend','team_performance.home_goals_for_avg','team_performance.away_goals_for_avg','availability.home_formation','availability.away_formation','availability.both_xi_confirmed','availability.both_goalkeepers_confirmed','context.venue','context.city'];
    const good=keys.filter(k=>fval(fs,k)!==null).length,pct=Math.round(good/keys.length*100);
    return '<div class="v3-coverage-mini"><div class="v3-coverage-ring" style="--p:'+pct+'%"><b>'+pct+'%</b></div><div class="v3-coverage-copy"><b>'+good+' / '+keys.length+'</b><span>verified sport/context inputs</span></div></div>';
  }
  function availabilityFallback(d){
    const fs=featureMap(d),home=fval(fs,'availability.home_formation'),away=fval(fs,'availability.away_formation'),xi=fval(fs,'availability.both_xi_confirmed'),gk=fval(fs,'availability.both_goalkeepers_confirmed');
    if(home===null&&away===null&&xi===null&&gk===null)return miss('Availability');
    return '<div class="v3-fallback">'+
      '<div class="v3-fallback-row"><span>Home formation</span><b>'+esc(home??'N/V')+'</b></div>'+
      '<div class="v3-fallback-row"><span>Away formation</span><b>'+esc(away??'N/V')+'</b></div>'+
      '<div class="v3-fallback-row"><span>Both XI</span><b>'+(xi===null?'N/V':xi?'YES':'NO')+'</b></div>'+
      '<div class="v3-fallback-row"><span>Goalkeepers</span><b>'+(gk===null?'N/V':gk?'YES':'NO')+'</b></div></div>';
  }
  function scoringFallback(d,f){
    const fs=featureMap(d),hgf=n(fval(fs,'team_performance.home_goals_for_avg')),hga=n(fval(fs,'team_performance.home_goals_against_avg')),agf=n(fval(fs,'team_performance.away_goals_for_avg')),aga=n(fval(fs,'team_performance.away_goals_against_avg'));
    if([hgf,hga,agf,aga].every(x=>x===null))return rateFallback(d,f);
    return '<div class="v3-fallback">'+
      '<div class="v3-fallback-row"><span>'+esc(f.home_team||'Home')+' GF/GA</span><b>'+(hgf===null?'N/V':hgf.toFixed(2))+' / '+(hga===null?'N/V':hga.toFixed(2))+'</b></div>'+
      '<div class="v3-fallback-row"><span>'+esc(f.away_team||'Away')+' GF/GA</span><b>'+(agf===null?'N/V':agf.toFixed(2))+' / '+(aga===null?'N/V':aga.toFixed(2))+'</b></div></div>';
  }
  function marketGateFallback(d){
    const cls=String(d?.decision_summary?.classification||d?.analyst_review?.status||'RESEARCH').replaceAll('_',' ');
    return '<div class="v3-gate"><div class="v3-gate-pill">'+esc(cls)+'</div><b>No verified model-vs-market edge</b><span>Market remains secondary until sporting projection and fair-price inputs are both verified.</span></div>';
  }
  function board(d){
    const f=d?.fixture||{},ctx=d?.sport_context||{},scoreRows=ctx.score_matrix||[];
    const q=ctx.outcome_probabilities||{},hasProb=[p(q.home),p(q.draw),p(q.away)].every(x=>x!==null);
    const x=ctx.expected_goals||{},hasXg=n(x.home)!==null&&n(x.away)!==null;
    const hasProfile=(ctx.sport_profile||[]).length>0;
    const hasScore=scoreRows.some(r=>score(r)&&p(r.probability)!==null);
    const ladder=d?.projection_ladder||{},hasEdge=p(ladder.raw_sport_probability)!==null&&p(ladder.fair_market_probability)!==null;
    return '<section id="v3-premium-board" class="v3-board">'+
      '<div class="v3-board-head"><div><span>MATCH INTELLIGENCE</span><h3>Sport model snapshot</h3></div><small>SPORT FIRST</small></div>'+
      '<div class="v3-top-grid">'+
        '<article class="v3-bi"><header><b>'+(hasProb?'Result Probability':'Goal-rate Split')+'</b><span>'+(hasProb?'1X2':'VERIFIED')+'</span></header>'+(hasProb?probs(d,f):rateFallback(d,f))+'</article>'+
        '<article class="v3-bi"><header><b>'+(hasXg?'Expected Goals':'Recent Form')+'</b><span>'+(hasXg?'xG':'W/D/L')+'</span></header>'+(hasXg?xg(d,f):formFallback(d,f))+'</article>'+
        '<article class="v3-bi v3-edge-card"><header><b>'+(hasEdge?'Edge Gap':'Market Gate')+'</b><span>AFTER SPORT</span></header>'+(hasEdge?edge(d):marketGateFallback(d))+'</article>'+
      '</div>'+
      '<div class="v3-bottom-grid">'+
        '<article class="v3-bi v3-profile-card"><header><b>'+(hasProfile?'Sport Profile':'Data Coverage')+'</b><span>'+(hasProfile?'MODEL':'EVIDENCE')+'</span></header>'+(hasProfile?profile(d):coverageFallback(d))+'</article>'+
        '<article class="v3-bi"><header><b>'+(hasScore?'Score Matrix':'Availability')+'</b><span>'+(hasScore?'FT':'SPORT')+'</span></header>'+(hasScore?heat(d):availabilityFallback(d))+'</article>'+
        '<article class="v3-bi"><header><b>'+(hasScore?'Goal Distribution':'Scoring Context')+'</b><span>'+(hasScore?'MODEL':'SPORT')+'</span></header>'+(hasScore?dist(d):scoringFallback(d,f))+'</article>'+
      '</div></section>';
  }
  function overviewActive(){
    const active=document.querySelector('.tabs button.active');
    return !active || String(active.textContent||'').trim().toLowerCase()==='overview';
  }
  function sync(){
    const body=document.querySelector('.mc-body');if(!body)return;
    const current=document.getElementById('v3-premium-board');
    if(!overviewActive()){if(current)current.remove();return;}
    if(!matchData||current)return;
    body.insertAdjacentHTML('afterbegin',board(matchData));
  }
  window.addEventListener('v3-match-data',e=>{matchData=e.detail;sync();});
  document.addEventListener('click',e=>{if(e.target.closest('.tabs button'))setTimeout(sync,0);});
  const obs=new MutationObserver(()=>sync());obs.observe(document.documentElement,{childList:true,subtree:true});
  setTimeout(sync,0);
})();
</script>
"""
    marker = "</head>"
    if marker in rendered:
        rendered = rendered.replace(marker, shim + visual_overrides + premium_pass + premium_board + marker, 1)
    else:
        rendered = shim + visual_overrides + premium_pass + rendered
    return rendered
