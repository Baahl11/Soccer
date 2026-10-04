from __future__ import annotations

import re

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import subscriber_preview_v230


_TOWER_PLACEHOLDER = r'''
<section class="page" id="tower">
  <div class="header"><div><h1>Control Tower</h1><div class="subtitle">System health and evidence freshness are separate concepts</div></div><span class="preview-chip amber">LOADING OBSERVABILITY</span></div>
  <div class="health">
    <div class="health-card"><b>● Render</b><small>—</small></div><div class="health-card"><b>● Postgres</b><small>—</small></div><div class="health-card"><b>● Scheduler</b><small>—</small></div><div class="health-card"><b>● API-Football</b><small>—</small></div><div class="health-card"><b>● Galaxy</b><small>—</small></div><div class="health-card"><b>● Last Tick</b><small>—</small></div>
  </div>
  <div class="section-title">Data freshness</div>
  <div class="fresh-grid"><div class="fresh"><b>Runtime telemetry</b><small>Loading persisted observability…</small></div><div class="fresh"><b>Postgres pipeline</b><small>Loading persisted observability…</small></div><div class="fresh"><b>Maturation artifact</b><small>Loading persisted observability…</small></div><div class="fresh"><b>Evidence age</b><small>Loading persisted observability…</small></div></div>
  <div class="pipeline-grid" style="margin-top:10px">
    <div class="panel"><div class="ph"><h3>Current Pipeline</h3></div><div class="mini-kpis"><div class="mini"><b>—</b><small>fixtures scanned</small></div><div class="mini"><b>—</b><small>due</small></div><div class="mini"><b>—</b><small>deep dives</small></div><div class="mini"><b>—</b><small>events</small></div><div class="mini"><b>—</b><small>research visible</small></div><div class="mini"><b>—</b><small>API calls</small></div></div></div>
    <div class="panel"><div class="ph"><h3>Pipeline Errors</h3><span class="status research">—</span></div><div class="v231-empty">Loading persisted runtime errors…</div></div>
    <div class="panel"><div class="ph"><h3>Model Maturity</h3><span class="status research">LOADING</span></div><div class="maturity"><div class="v231-empty">Loading persisted validation reports…</div></div></div>
  </div>
  <div class="panel"><div class="ph"><h3>Persisted maturity evidence</h3><span class="status research">LOADING</span></div><div class="chain"><div><b>—</b><small>Waiting for public Control Tower snapshot</small></div></div><p class="note">No design-time counters or synthetic pipeline errors are shown on the live product surface.</p></div>
</section>
'''

_LIVE_STYLE = r'''
<style id="v231-live-style">
.v231-action{border:1px solid #21475f;background:#0b1c28;color:#8fb1c8;border-radius:7px;padding:4px 7px;font:inherit;cursor:pointer}.v231-action.on{border-color:#18b88a;color:#76ecc8;background:#0c3029}.v231-muted{color:#71899c}.v231-stack{display:grid;gap:7px;margin-top:9px}.v231-kv{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid #112b3b;font-size:10px}.v231-kv:last-child{border-bottom:0}.v231-kv span{color:#71899c}.v231-score-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:10px}.v231-score{border:1px solid #17384c;background:#091923;border-radius:8px;padding:8px;text-align:center}.v231-score b{display:block;font-size:13px}.v231-score small{color:#6e879a}.v231-alert{display:flex;gap:8px;padding:9px 0;border-bottom:1px solid #112b3b}.v231-alert:last-child{border-bottom:0}.v231-alert b{font-size:10px}.v231-alert small{display:block;color:#728a9d;margin-top:2px}.v231-trackline{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid #112b3b}.v231-progress{height:5px;background:#102b3a;border-radius:999px;overflow:hidden;margin-top:5px}.v231-progress i{display:block;height:100%;background:#25b98a}.v231-panel-note{color:#71899c;font-size:9px;line-height:1.5;margin-top:8px}.v231-empty{padding:18px 4px;color:#71899c;font-size:10px}.perf-table td,.perf-table th{white-space:nowrap}
</style>
'''

_LIVE_SCRIPT = r'''
<script id="v231-live-preview">
(() => {
  const AK='soccer_edge_access_token';
  const SAVED='soccer_edge_saved_signals_v1';
  const TRACKED='soccer_edge_tracked_markets_v1';
  let LIVE=null;
  let PERF=null;

  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>v===null||v===undefined||v===''?null:Number(v);
  const pct=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${(n*100).toFixed(1)}%`};
  const pp=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${n>=0?'+':''}${n.toFixed(2)} pp`};
  const edge=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${n>=0?'+':''}${n.toFixed(1)} pp`};
  const price=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':n.toFixed(2)};
  const metric=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':n.toFixed(3)};
  const conf=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':String(Math.round(n<=1?n*100:n))};
  const text=(el,v)=>{if(el)el.textContent=v??'—'};
  const market=r=>r?.market?.selection||r?.market?.family||'Market';
  const match=r=>r?.match?.label||'Fixture';
  const league=r=>r?.match?.league||r?.match?.country||'';
  const status=r=>String(r?.state?.status||'WATCH').replaceAll('_',' ');
  const statusClass=s=>{s=String(s||'').toUpperCase();if(s==='READY'||s==='BET'||s==='HEALTHY'||s==='OK'||s==='LIVE')return'ready';if(s.includes('WAIT')||s==='WATCH'||s.includes('STALE'))return'wait';if(s.includes('RESEARCH')||s.includes('NOT VERIFIED')||s.includes('HOLD'))return'research';return'pass'};
  const compact=(r,last='')=>`<div class="compact-row"><b>${esc(match(r))}</b><span>${esc(market(r))}</span><span>${esc(last||edge(r?.pricing?.edge_pp))}</span></div>`;
  const jsonGet=(key)=>{try{const v=JSON.parse(localStorage.getItem(key)||'[]');return Array.isArray(v)?v:[]}catch(_){return[]}};
  const jsonSet=(key,v)=>localStorage.setItem(key,JSON.stringify(v));
  const rowKey=r=>[r?.match?.fixture_id,r?.market?.family,r?.market?.selection].map(v=>String(v??'')).join('|');
  const familyLabel=f=>({HOME_TT:'Team Totals',AWAY_TT:'Team Totals',FT_TOTALS:'FT Totals',FT_CORNERS:'Corners',TEAM_CORNERS:'Corners',CARDS:'Cards',PLAYER_CARDS:'Player Props',SHOTS:'Player Props',SOT:'Player Props',GOALSCORER:'Player Props',ASSISTS:'Player Props',GK_SAVES:'Player Props'}[String(f||'').toUpperCase()]||String(f||'').replaceAll('_',' '));

  function neutralizeMocks(){
    document.querySelectorAll('#today .metrics .metric .n,#performance .metrics .metric .n').forEach(x=>x.textContent='—');
    const feed=document.getElementById('feedbody');if(feed)feed.innerHTML='<tr><td colspan="8">Loading persisted data…</td></tr>';
    const mg=document.getElementById('marketgrid');if(mg)mg.innerHTML='<div class="v231-empty">Loading market catalog…</div>';
    const perfBody=document.querySelector('#performance .perf-table tbody');if(perfBody)perfBody.innerHTML='<tr><td colspan="7">Loading persisted validation evidence…</td></tr>';
    const myPanels=document.querySelectorAll('#myedge .grid2 > .panel');myPanels.forEach(p=>p.innerHTML='<div class="v231-empty">Loading personal preview…</div>');
    const research=document.querySelector('#research .market-grid');if(research)research.innerHTML='<div class="v231-empty">Loading persisted research context…</div>';
    const towerChip=document.querySelector('#tower .preview-chip');if(towerChip){towerChip.textContent='LOADING OBSERVABILITY';towerChip.classList.add('amber')}
  }

  function renderToday(data){
    const t=data.today||{}, m=t.metrics||{};
    const metrics=document.querySelectorAll('#today .metrics .metric .n');
    const vals=[m.matches_scanned,m.deep_analyzed,m.strong_edges,m.waiting_xi,m.pass];
    metrics.forEach((el,i)=>text(el,vals[i]??'—'));
    const live=document.querySelector('#today .header .live');
    if(live)live.innerHTML=`<span class="dot"></span> LIVE DATA · ${esc(data.generated_at_local||data.generated_at_utc||'persisted')}`;
    const chip=document.querySelector('#today .preview-chip');if(chip)chip.textContent='LIVE DATA';

    const top=t.top_edge;
    const hero=document.querySelector('#today .hero-main');
    if(hero&&top){
      text(hero.querySelector('.match-name'),match(top));
      text(hero.querySelector('.market-name'),market(top));
      const stats=hero.querySelectorAll('.triple .stat b');
      if(stats[0])stats[0].textContent=pct(top.model?.probability);
      if(stats[1])stats[1].textContent=pct(top.pricing?.market_probability);
      if(stats[2])stats[2].textContent=edge(top.pricing?.edge_pp);
      const badges=hero.querySelector('.badges');
      if(badges)badges.innerHTML=[
        top.pricing?.price!=null?`<span class="badge">Price ${price(top.pricing.price)}</span>`:'',
        top.pricing?.fair_price!=null?`<span class="badge">Fair ${price(top.pricing.fair_price)}</span>`:'',
        top.pricing?.bookmaker?`<span class="badge">${esc(top.pricing.bookmaker)}</span>`:'',
        `<span class="badge">${esc(status(top))}</span>`
      ].filter(Boolean).join('');
    } else if(hero){
      hero.innerHTML='<div class="eyebrow">No comparable priced edge</div><div class="match-name">Waiting for model + market pair</div><div class="market-name">Missing data stays missing — never converted to zero.</div>';
    }

    const strongList=document.querySelector('#today .hero-edge > .panel:nth-child(2) .signal-list');
    if(strongList){const rows=t.strong_signals||[];strongList.innerHTML=rows.length?rows.map(r=>`<div class="signal"><div><b>${esc(match(r))}</b><small>${esc(market(r))}${league(r)?` · ${esc(league(r))}`:''}</small></div><span class="pp">${esc(edge(r.pricing?.edge_pp))}</span></div>`).join(''):'<div class="placeholder">No strong signals in this persisted snapshot.</div>'}
    const panels=document.querySelectorAll('#today .bottom-grid > .panel');
    if(panels[0]){const rows=t.price_opportunities||[];panels[0].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[0].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No price waits now.</div>')}
    if(panels[1]){const rows=t.waiting_xi||[];panels[1].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[1].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No XI waits now.</div>')}
    if(panels[2]){const rows=t.upcoming||[];panels[2].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[2].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r,r.match?.kickoff||'')).join(''):'<div class="placeholder">No upcoming rows in snapshot.</div>')}
  }

  function toggleSaved(key){const rows=jsonGet(SAVED);const i=rows.indexOf(key);if(i>=0)rows.splice(i,1);else rows.push(key);jsonSet(SAVED,rows);renderFeed(LIVE);renderMyEdge(LIVE)}
  function toggleTracked(label){const rows=jsonGet(TRACKED);const i=rows.indexOf(label);if(i>=0)rows.splice(i,1);else rows.push(label);jsonSet(TRACKED,rows);renderMarkets(LIVE);renderMyEdge(LIVE)}

  function renderFeed(data){
    const body=document.getElementById('feedbody');if(!body)return;
    const rows=data.edge_feed?.rows||[], saved=new Set(jsonGet(SAVED));
    body.innerHTML=rows.length?rows.map(r=>{const key=rowKey(r),on=saved.has(key);return `<tr>
      <td class="team-cell"><button class="v231-action ${on?'on':''}" data-save="${esc(key)}" title="Save to My Edge">${on?'★':'☆'}</button> <span class="team-dot">⚽</span>${esc(match(r))}</td>
      <td>${esc(market(r))}</td><td>${esc(pct(r.model?.probability))}</td><td>${esc(pct(r.pricing?.market_probability))}</td><td class="edge">${esc(edge(r.pricing?.edge_pp))}</td><td>${esc(price(r.pricing?.price))}</td><td>${esc(conf(r.raw?.confidence_score??r.raw?.model_confidence??r.raw?.model_signal_score))}</td><td><span class="status ${statusClass(status(r))}">${esc(status(r))}</span></td></tr>`}).join(''):'<tr><td colspan="8">No persisted rows for Edge Feed.</td></tr>';
    body.querySelectorAll('[data-save]').forEach(b=>b.onclick=e=>{e.stopPropagation();toggleSaved(b.dataset.save)});
    const live=document.querySelector('#feed .header .live');if(live)live.innerHTML=`<span class="dot"></span> Persisted · ${esc(data.generated_at_local||data.generated_at_utc||'latest')}`;
  }

  function renderMatch(data){
    const mc=data.match_center||{}, r=mc.selected, d=mc.detail||{};
    const head=document.querySelector('#matches .match-head');const cards=document.querySelectorAll('#matches .match-grid .score-card');
    if(!r){if(head)head.innerHTML='<div class="v231-empty">No persisted match row available for Match Center.</div>';cards.forEach(c=>c.innerHTML='<div class="v231-empty">No live data.</div>');return}
    const ctx=d.model_context||{}, probs=d.outcome_probabilities, xg=d.expected_goals, matrix=d.score_matrix||[], profile=d.sport_profile||[];
    if(head){
      const initials=x=>String(x||'?').split(/\s+/).filter(Boolean).slice(0,3).map(v=>v[0]).join('').toUpperCase().slice(0,3);
      head.innerHTML=`<div class="teams-head"><div class="crest">${esc(initials(r.match?.home))}</div><div><small class="subtitle">${esc(r.match?.league||r.match?.country||'Competition')} · ${esc(r.match?.kickoff||'Kickoff not persisted')}</small><div class="match-name">${esc(r.match?.home||'Home')} <span class="vs">vs</span> ${esc(r.match?.away||'Away')}</div></div><div class="crest">${esc(initials(r.match?.away))}</div></div><div class="quality"><span>Data <b class="green">${esc(ctx.data_quality||'—')}</b></span><span>Lineup <b class="green">${esc(ctx.lineup||'—')}</b></span><span>Status <b class="green">${esc(status(r))}</b></span></div>`;
    }
    if(cards[0])cards[0].innerHTML=probs?`<h4>Match Result Probability</h4><div class="probline"><div class="probbox"><small>HOME</small><b class="green">${pct(probs.home)}</b></div><div class="probbox"><small>DRAW</small><b>${pct(probs.draw)}</b></div><div class="probbox"><small>AWAY</small><b>${pct(probs.away)}</b></div></div>`:`<h4>Selected Market Probability</h4><div class="bigprob">${esc(market(r))}</div><div class="v231-stack"><div class="v231-kv"><span>Model</span><b>${pct(r.model?.probability)}</b></div><div class="v231-kv"><span>Market</span><b>${pct(r.pricing?.market_probability)}</b></div></div><p class="v231-panel-note">A complete persisted 1X2 vector was not present for this fixture snapshot.</p>`;
    if(cards[1])cards[1].innerHTML=xg?`<h4>Expected Goals (λ)</h4><div class="bigprob">${xg.home==null?'—':Number(xg.home).toFixed(2)} <span class="vs">–</span> ${xg.away==null?'—':Number(xg.away).toFixed(2)}</div><div class="subtitle">Total ${xg.total==null?'—':Number(xg.total).toFixed(2)}</div>`:`<h4>Expected Goals (λ)</h4><div class="v231-empty">Not persisted for the selected fixture snapshot.</div>`;
    if(cards[2])cards[2].innerHTML=`<h4>Edge Gap · ${esc(market(r))}</h4><div class="bigprob green">${esc(edge(r.pricing?.edge_pp))}</div><div class="subtitle">Model ${pct(r.model?.probability)} · Market ${pct(r.pricing?.market_probability)}</div>`;
    if(cards[3])cards[3].innerHTML=profile.length?`<h4>Sport Profile</h4><div class="bars">${profile.map(p=>`<div class="barrow"><span>${esc(p.label)}</span><div class="bar"><div class="fill" style="width:${Math.max(0,Math.min(100,num(p.score)||0))}%"></div></div><b>${Math.round(num(p.score)||0)}</b></div>`).join('')}</div>`:`<h4>Model Context</h4><div class="v231-stack"><div class="v231-kv"><span>Confidence</span><b>${esc(conf(ctx.confidence))}</b></div><div class="v231-kv"><span>Disagreement</span><b>${esc(ctx.model_disagreement||'—')}</b></div><div class="v231-kv"><span>Stage</span><b>${esc(ctx.stage||'—')}</b></div><div class="v231-kv"><span>Reason</span><b>${esc(ctx.reason||'—')}</b></div></div><p class="v231-panel-note">Sport-profile bars appear only when their persisted feature scores exist.</p>`;
    if(cards[4])cards[4].innerHTML=matrix.length?`<h4>Score Matrix · top persisted outcomes</h4><div class="v231-score-grid">${matrix.map(s=>`<div class="v231-score"><b>${esc(s.score)}</b><small>${pct(s.probability)}</small></div>`).join('')}</div>`:`<h4>Score Matrix (FT)</h4><div class="v231-empty">No score-matrix distribution is persisted on the selected fixture rows.</div>`;
    if(cards[5])cards[5].innerHTML=`<h4>Market Readiness</h4><div class="v231-stack"><div class="v231-kv"><span>Status</span><b>${esc(status(r))}</b></div><div class="v231-kv"><span>Price</span><b>${esc(price(r.pricing?.price))}</b></div><div class="v231-kv"><span>Fair price</span><b>${esc(price(r.pricing?.fair_price))}</b></div><div class="v231-kv"><span>Book</span><b>${esc(r.pricing?.bookmaker||ctx.bookmaker||'—')}</b></div><div class="v231-kv"><span>Provider update</span><b>${esc(ctx.provider_update||r.state?.provider_update||'—')}</b></div></div>`;
    const tabs=document.querySelectorAll('#matches .tabs .tab');tabs.forEach((tab,i)=>{tab.classList.toggle('active',i===0);if(i>0){tab.title='Family-specific tab will activate only when that persisted derivative is selected';tab.style.opacity='.55'}});
    const live=document.querySelector('#matches .header .preview-chip');if(live)live.textContent=`LIVE MATCH · ${d.fixture_row_count??1} ROW${d.fixture_row_count===1?'':'S'}`;
  }

  function renderMarkets(data){
    const grid=document.getElementById('marketgrid');if(!grid)return;
    const families=data.markets?.families||[], tracked=new Set(jsonGet(TRACKED));
    grid.innerHTML=families.length?families.map(m=>{const on=tracked.has(m.label);return `<div class="market-card"><div style="display:flex;justify-content:space-between;gap:8px"><h3>${esc(m.label)}</h3><button class="v231-action ${on?'on':''}" data-track="${esc(m.label)}">${on?'Tracked':'Track'}</button></div><div class="count">${esc(m.live_rows??0)}<span class="subtitle"> live rows</span></div><small>${esc(m.description)}</small><div class="mini-line"><span>Maturity</span><b>${esc(m.maturity_current??'—')}/${esc(m.maturity_target??'—')} · ${esc(String(m.maturity_status||'NOT VERIFIED').replaceAll('_',' '))}</b></div>${m.blocker?`<div class="mini-line"><span>Blocker</span><b style="color:#eab95c">${esc(String(m.blocker).replaceAll('_',' '))}</b></div>`:''}</div>`}).join(''):'<div class="v231-empty">No market catalog available.</div>';
    grid.querySelectorAll('[data-track]').forEach(b=>b.onclick=()=>toggleTracked(b.dataset.track));
    const chip=document.querySelector('#markets .preview-chip');if(chip)chip.textContent='LIVE CATALOG';
  }

  function renderPerformance(perf){
    PERF=perf||{};const rows=Array.isArray(PERF.rows)?PERF.rows:[];
    const one=rows.find(r=>r.label==='1X2')||{},btts=rows.find(r=>r.label==='BTTS')||{};
    const metrics=document.querySelectorAll('#performance .metrics .metric');
    const values=[
      [PERF.weighted_avg_clv_pp==null?'—':pp(PERF.weighted_avg_clv_pp),'weighted avg CLV'],
      [metric(one.brier),'1X2 Brier'],
      [metric(btts.brier),'BTTS Brier'],
      [PERF.families_with_clv??'—','families with CLV'],
      [one.true_clv_rows==null?'—':`${one.true_clv_rows}/${one.true_clv_target??'—'}`,'1X2 maturation']
    ];
    metrics.forEach((box,i)=>{const v=values[i];if(!v)return;text(box.querySelector('.n'),v[0]);text(box.querySelector('.label'),v[1])});
    const table=document.querySelector('#performance .perf-table');
    if(table){const head=table.querySelector('thead');if(head)head.innerHTML='<tr><th>Market</th><th>Sample</th><th>True CLV</th><th>Avg CLV</th><th>Brier</th><th>Log Loss</th><th>Status</th></tr>';const body=table.querySelector('tbody');if(body)body.innerHTML=rows.length?rows.map(r=>`<tr><td><b>${esc(r.label)}</b></td><td>${esc(r.sample_n??'—')}</td><td>${esc(r.true_clv_rows??'—')}/${esc(r.true_clv_target??'—')}${r.true_clv_fixtures!=null?` · ${esc(r.true_clv_fixtures)} fx`:''}</td><td class="edge">${esc(r.avg_clv_pp==null?'—':pp(r.avg_clv_pp))}</td><td>${esc(metric(r.brier))}</td><td>${esc(metric(r.log_loss))}</td><td><span class="status ${statusClass(r.status)}">${esc(String(r.status||'NOT VERIFIED').replaceAll('_',' '))}</span></td></tr>`).join(''):'<tr><td colspan="7">No persisted validation rows available.</td></tr>'}
    const panels=document.querySelectorAll('#performance .grid2 > .panel');
    if(panels[1]){const clv=rows.filter(r=>r.true_clv_rows!=null);panels[1].innerHTML=`<div class="ph"><h3>CLV Summary</h3><span class="status ${statusClass(PERF.status)}">${esc(PERF.status||'N/V')}</span></div><div class="bigprob green">${PERF.weighted_avg_clv_pp==null?'—':esc(pp(PERF.weighted_avg_clv_pp))}</div><div class="v231-stack">${clv.length?clv.map(r=>{const cur=num(r.true_clv_rows),tar=num(r.true_clv_target),w=cur!=null&&tar?Math.min(100,cur/tar*100):0;return `<div><div class="v231-kv"><span>${esc(r.label)}</span><b>${esc(cur??'—')}/${esc(tar??'—')} · ${esc(r.avg_clv_pp==null?'—':pp(r.avg_clv_pp))}</b></div><div class="v231-progress"><i style="width:${w}%"></i></div></div>`}).join(''):'<div class="v231-empty">No family-specific true-CLV evidence.</div>'}</div><p class="v231-panel-note">Weighted CLV uses only families with both an observed average and non-zero true-CLV rows. No missing family is treated as zero.</p>`}
    const live=document.querySelector('#performance .header .live');if(live)live.innerHTML=`<span class="dot"></span> ${esc(PERF.status||'N/V')} · persisted validation reports`;

    const chain=document.querySelector('#tower .pipeline-grid + .panel');
    if(chain){const visible=rows.filter(r=>r.sample_n!=null||r.true_clv_rows!=null);chain.innerHTML=`<div class="ph"><h3>Validation evidence by family</h3><span class="status research">PERSISTED</span></div><div class="chain">${visible.slice(0,9).map(r=>`<div><b>${esc(r.true_clv_rows??r.sample_n??'—')}</b><small>${esc(r.label)} · ${r.true_clv_rows!=null?'CLV rows':'sample'}</small></div>`).join('')}</div><p class="note">Calibration and CLV cohorts are displayed as persisted; family samples are not silently combined into a production gate.</p>`}
  }

  function renderResearch(data,perf){
    const lab=data.research_lab||{},d=lab.selected_fixture||{},r=d.selected,ctx=d.model_context||{},fw=lab.firewall||{},grid=document.querySelector('#research .market-grid');
    const perfRows=Array.isArray(perf?.rows)?perf.rows:[];const fLabel=familyLabel(lab.selected_family);const familyPerf=perfRows.find(x=>String(x.label).toUpperCase()===String(fLabel).toUpperCase());
    if(grid){
      if(!r){grid.innerHTML='<div class="v231-empty">No persisted selected fixture is available for research inspection.</div>'}
      else grid.innerHTML=[
        `<div class="market-card"><h3>Selected research fixture</h3><div class="count" style="font-size:16px">${esc(match(r))}</div><small>${esc(market(r))}</small><div class="mini-line"><span>Stage</span><b>${esc(ctx.stage||'—')}</b></div><div class="mini-line"><span>Data tier</span><b>${esc(ctx.data_quality||'—')}</b></div></div>`,
        `<div class="market-card"><h3>Model evidence</h3><div class="count">${esc(conf(ctx.confidence))}</div><small>confidence · persisted if available</small><div class="mini-line"><span>Model</span><b>${esc(ctx.model_version||lab.runtime_model_version||'—')}</b></div><div class="mini-line"><span>Disagreement</span><b>${esc(ctx.model_disagreement||'—')}</b></div></div>`,
        `<div class="market-card"><h3>Distribution evidence</h3><div class="count">${esc((d.score_matrix||[]).length)}</div><small>persisted score outcomes exposed</small><div class="mini-line"><span>xG</span><b>${d.expected_goals?`${esc(d.expected_goals.home??'—')} – ${esc(d.expected_goals.away??'—')}`:'—'}</b></div><div class="mini-line"><span>1X2 vector</span><b>${d.outcome_probabilities?'PRESENT':'NOT PERSISTED'}</b></div></div>`,
        `<div class="market-card"><h3>${esc(fLabel||'Family')} validation</h3><div class="count">${familyPerf?.true_clv_rows??'—'}<span class="subtitle">/${familyPerf?.true_clv_target??'—'}</span></div><small>true CLV · ${esc(String(familyPerf?.status||'NOT VERIFIED').replaceAll('_',' '))}</small><div class="mini-line"><span>Brier</span><b>${esc(metric(familyPerf?.brier))}</b></div><div class="mini-line"><span>Log loss</span><b>${esc(metric(familyPerf?.log_loss))}</b></div></div>`
      ].join('');
    }
    const wire=document.querySelectorAll('#research .wire > div');
    const firewall=[['Research rows','isolated'],['Decision weight',fw.decision_weight??0],['Promotion',fw.production_promotion_allowed?'allowed':'blocked'],['Canonical BET',fw.canonical_bet_logic_changed?'changed':'unchanged'],['Model weights',fw.model_weights_changed?'changed':'unchanged'],['Provider requests',fw.provider_requests_added??0],['Strict close',fw.strict_close_changed?'changed':'unchanged']];
    wire.forEach((el,i)=>{const x=firewall[i];if(x)el.innerHTML=`${esc(x[0])}<span>${esc(x[1])}</span>`});
    const ph=document.querySelector('#research > .header .preview-chip');if(ph)ph.textContent='LIVE RESEARCH · DECISION WEIGHT 0';
  }

  function renderMyEdge(data){
    const panels=document.querySelectorAll('#myedge .grid2 > .panel');if(panels.length<2)return;
    const feed=data.edge_feed?.rows||[], savedKeys=jsonGet(SAVED), saved=new Set(savedKeys), savedRows=feed.filter(r=>saved.has(rowKey(r)));
    panels[0].innerHTML=`<div class="ph"><h3>Saved Signals</h3><span class="link">${savedRows.length} visible</span></div><div class="signal-list">${savedRows.length?savedRows.map(r=>`<div class="signal"><div><b>${esc(match(r))}</b><small>${esc(market(r))} · ${esc(edge(r.pricing?.edge_pp))}</small></div><div style="display:flex;align-items:center;gap:6px"><span class="status ${statusClass(status(r))}">${esc(status(r))}</span><button class="v231-action on" data-remove-save="${esc(rowKey(r))}">★</button></div></div>`).join(''):'<div class="v231-empty">No saved signals yet. Use ☆ in Edge Feed to save a live row.</div>'}</div><p class="v231-panel-note">Preview storage is device-local; no fake default favorites are preloaded.</p>`;
    panels[0].querySelectorAll('[data-remove-save]').forEach(b=>b.onclick=()=>toggleSaved(b.dataset.removeSave));
    const alerts=data.my_edge?.system_alerts||[], tracked=jsonGet(TRACKED);
    panels[1].innerHTML=`<div class="ph"><h3>Live Alerts</h3><span class="status ${alerts.length?'wait':'ready'}">${alerts.length}</span></div>${alerts.length?alerts.slice(0,6).map(a=>`<div class="v231-alert"><span class="status ${statusClass(a.severity)}">${esc(a.type)}</span><div><b>${esc(a.match||a.market||'System')}</b><small>${esc(String(a.message||'').replaceAll('_',' '))}</small></div></div>`).join(''):'<div class="v231-empty">No live price/XI/maturation alerts in the current snapshot.</div>'}<div class="section-title" style="margin-top:12px">Tracked Markets · this device</div>${tracked.length?tracked.map(x=>`<div class="v231-trackline"><b>${esc(x)}</b><button class="v231-action on" data-untrack="${esc(x)}">Tracked</button></div>`).join(''):'<div class="v231-empty">Track families from Markets to build your personal slate.</div>'}`;
    panels[1].querySelectorAll('[data-untrack]').forEach(b=>b.onclick=()=>toggleTracked(b.dataset.untrack));
    const chip=document.querySelector('#myedge .preview-chip');if(chip)chip.textContent='PRO · DEVICE PREVIEW';
  }

  function renderTower(data){
    const ct=data.control_tower||{},health=ct.system_health||{},pipe=ct.pipeline||{},mat=ct.maturation||{},monitoring=mat.monitoring||{},freshness=monitoring.report_freshness||{},evidenceAge=monitoring.evidence_age||{};
    const header=document.querySelector('#tower .header .preview-chip');const age=evidenceAge?.evidence?.age_hours;const matState=String(evidenceAge.status||freshness.status||mat.status||'NOT_VERIFIED').replaceAll('_',' ');
    if(header){header.textContent=`RUNTIME ${ct.runtime_generated_at_local||ct.runtime_generated_at_utc||'—'} · MATURITY ${matState}${age!=null?` ${age}h`:''}`;header.classList.toggle('amber',String(matState).toUpperCase()!=='OK')}
    const cards=document.querySelectorAll('#tower .health-card');const healthRows=[['Render',ct.status||'LIVE'],['Postgres',health.postgres],['Scheduler',health.scheduler],['API-Football',health.api_football_remaining!=null?`${health.api_football_remaining} LEFT`:health.api_football],['Galaxy',health.galaxy],['Last Tick',health.last_tick]];
    cards.forEach((card,i)=>{const row=healthRows[i];if(!row)return;text(card.querySelector('b'),`● ${row[0]}`);text(card.querySelector('small'),row[1]??'—');card.querySelector('b')?.classList.toggle('green',!String(row[1]||'').toUpperCase().includes('DEGRADED'))});
    const fresh=document.querySelectorAll('#tower .fresh');const frows=[['Runtime telemetry',ct.runtime_generated_at_local||ct.runtime_generated_at_utc],['Postgres pipeline',health.last_tick],['Maturation artifact',freshness?.evidence?.report_updated_at||mat.snapshot_generated_at_utc],['Evidence age',age==null?'Not verified':`${age} hours · ${matState}`]];
    fresh.forEach((el,i)=>{const x=frows[i];if(!x)return;text(el.querySelector('b'),x[0]);text(el.querySelector('small'),x[1]??'—');el.classList.toggle('stale',i===3&&String(matState).toUpperCase()!=='OK')});
    const minis=document.querySelectorAll('#tower .pipeline-grid > .panel:first-child .mini');const pvals=[[pipe.fixtures_scanned,'fixtures scanned'],[pipe.due,'due'],[pipe.deep_dives,'deep dives'],[pipe.events,'events'],[pipe.research_visible,'research visible'],[pipe.api_calls,'API calls']];minis.forEach((el,i)=>{const x=pvals[i];if(!x)return;text(el.querySelector('b'),x[0]??'—');text(el.querySelector('small'),x[1])});
    const errPanel=document.querySelector('#tower .pipeline-grid > .panel:nth-child(2)');if(errPanel){errPanel.querySelectorAll('.error,.placeholder,.v231-empty').forEach(x=>x.remove());const rows=Array.isArray(ct.errors?.rows)?ct.errors.rows:[];const freshnessIssue=String(matState).toUpperCase()!=='OK'?{reason:`Maturation evidence ${matState}`,stage:age!=null?`${age}h evidence age`:'freshness not verified'}:null;const merged=[...(freshnessIssue?[freshnessIssue]:[]),...rows].slice(0,4);const badge=errPanel.querySelector('.status');if(badge){badge.textContent=String(merged.length);badge.className=`status ${merged.length?'wait':'ready'}`}errPanel.insertAdjacentHTML('beforeend',merged.length?merged.map(e=>`<div class="error"><span class="errdot">!</span><div><b>${esc(String(e.reason||'Pipeline issue').replaceAll('_',' '))}</b><small>${esc(e.stage||e.fixture_id||'observability')}</small></div></div>`).join(''):'<div class="v231-empty">No pipeline errors in latest persisted runtime.</div>')}
    const maturityBox=document.querySelector('#tower .pipeline-grid > .panel:nth-child(3) .maturity');if(maturityBox){const families=Array.isArray(mat.families)?mat.families:[];maturityBox.innerHTML=families.length?families.map(f=>{const cur=num(f.current),tar=num(f.target),w=cur!==null&&tar&&tar>0?Math.max(0,Math.min(100,(cur/tar)*100)):0;return `<div><div class="matrow"><b>${esc(f.label||f.key)}</b><div class="mbar"><div class="mfill" style="width:${w.toFixed(1)}%"></div></div><span>${esc(cur??'—')}/${esc(tar??'—')}</span></div>${f.blocker?`<small style="display:block;color:#eab95c;margin:2px 0 0 79px;font-size:7px">${esc(String(f.blocker).replaceAll('_',' '))}</small>`:''}</div>`}).join(''):'<div class="v231-empty">Maturation reports unavailable.</div>'}
  }

  function publicTowerEnvelope(product){
    const ct=product?.views?.control_tower||{},snap=ct?.maturity_snapshot||{},mat=snap?.maturation_control_tower||{};
    return {control_tower:{status:ct.status,runtime_generated_at_utc:ct.generated_at_utc||product?.generated_at_utc,runtime_generated_at_local:ct.generated_at_local,pipeline_version:ct.pipeline_version||product?.pipeline_version,model_version:ct.model_version,system_health:ct.system_health||{},pipeline:ct.pipeline||{},errors:ct.errors||{count:0,rows:[]},maturation:{status:mat.status||snap.status,families:Array.isArray(mat.families)?mat.families:[],monitoring:mat.monitoring||{},snapshot_generated_at_utc:snap.generated_at_utc,reports_loaded:snap.reports_loaded,reports_expected:snap.reports_expected,errors:snap.errors||{}}}};
  }

  function renderPublicEvidence(data){
    const mat=data?.control_tower?.maturation||{},rows=Array.isArray(mat.families)?mat.families:[];
    const panel=document.querySelector('#tower .pipeline-grid + .panel');if(!panel)return;
    const evidence=rows.filter(x=>x&&typeof x==='object');
    panel.innerHTML=`<div class="ph"><h3>Persisted maturity evidence</h3><span class="status ${evidence.length?'ready':'research'}">${evidence.length?'LIVE':'N/V'}</span></div><div class="chain">${evidence.length?evidence.slice(0,9).map(f=>`<div><b>${esc(f.current??'—')}/${esc(f.target??'—')}</b><small>${esc(f.label||f.key||'Family')}${f.unique_fixtures!=null?` · ${esc(f.unique_fixtures)} fx`:''}</small></div>`).join(''):'<div><b>—</b><small>No persisted maturation families available.</small></div>'}</div><p class="note">Public read-only observability from /product/views. No design-time counters, provider calls, or synthetic errors are used.</p>`;
  }

  async function loadPublicTower(){
    const chip=document.querySelector('#tower .preview-chip');
    try{
      const r=await fetch('/product/views?limit=25',{cache:'no-store',headers:{Accept:'application/json'}});
      const product=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(product.error||`HTTP ${r.status}`);
      const data=publicTowerEnvelope(product);
      renderTower(data);renderPublicEvidence(data);
      if(chip)chip.classList.toggle('amber',String(data?.control_tower?.status||'').toUpperCase()!=='LIVE');
    }catch(err){
      if(chip){chip.textContent=`OBSERVABILITY UNAVAILABLE · ${esc(err.message)}`;chip.classList.add('amber')}
      const errPanel=document.querySelector('#tower .pipeline-grid > .panel:nth-child(2)');
      if(errPanel){errPanel.querySelectorAll('.error,.placeholder,.v231-empty').forEach(x=>x.remove());errPanel.insertAdjacentHTML('beforeend',`<div class="v231-empty">Public Control Tower unavailable · ${esc(err.message)}</div>`)}
      const maturityBox=document.querySelector('#tower .pipeline-grid > .panel:nth-child(3) .maturity');if(maturityBox)maturityBox.innerHTML='<div class="v231-empty">Persisted maturation reports unavailable.</div>';
    }
  }

  async function fetchJson(url,token){const r=await fetch(url,{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});let d={};try{d=await r.json()}catch(_){}if(!r.ok)throw new Error(d.error||`${url} HTTP ${r.status}`);return d}

  async function loadLive(){
    neutralizeMocks();
    loadPublicTower();
    const token=localStorage.getItem(AK)||'';
    if(!token){const live=document.querySelector('#today .header .live');if(live)live.innerHTML='<span class="dot" style="background:#eab95c"></span> LOGIN REQUIRED FOR LIVE PREVIEW';return}
    try{
      const results=await Promise.allSettled([fetchJson('/app-preview/data',token),fetchJson('/app-preview/performance',token)]);
      if(results[0].status!=='fulfilled')throw results[0].reason;
      LIVE=results[0].value;PERF=results[1].status==='fulfilled'?results[1].value:{status:'UNAVAILABLE',rows:[],error:String(results[1].reason||'Performance unavailable')};
      renderToday(LIVE);renderFeed(LIVE);renderMatch(LIVE);renderMarkets(LIVE);renderTower(LIVE);renderPerformance(PERF);renderResearch(LIVE,PERF);renderMyEdge(LIVE);
    }catch(err){const live=document.querySelector('#today .header .live');if(live)live.innerHTML=`<span class="dot" style="background:#ff6679"></span> LIVE DATA ERROR · ${esc(err.message)}`}
  }

  loadLive();
})();
</script>
'''


def _html() -> str:
    html = subscriber_preview_v230._html()
    # The approved v230 shell contains design-only Control Tower examples. Strip
    # them server-side so a failed/forbidden premium fetch can never look live.
    html = re.sub(
        r'<section class="page" id="tower">.*?</section>',
        _TOWER_PLACEHOLDER,
        html,
        count=1,
        flags=re.S,
    )
    marker = "</body>"
    payload = _LIVE_STYLE + _LIVE_SCRIPT
    return html.replace(marker, payload + marker, 1) if marker in html else html + payload


async def preview_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
