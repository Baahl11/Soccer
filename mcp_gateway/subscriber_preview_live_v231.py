from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import subscriber_preview_v230


_LIVE_SCRIPT = r'''
<script id="v231-live-preview">
(() => {
  const AK='soccer_edge_access_token';
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>v===null||v===undefined||v===''?null:Number(v);
  const pct=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${(n*100).toFixed(1)}%`};
  const edge=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':`${n>=0?'+':''}${n.toFixed(1)} pp`};
  const price=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':n.toFixed(2)};
  const conf=v=>{const n=num(v);return n===null||!Number.isFinite(n)?'—':(n<=1?Math.round(n*100):Math.round(n))};
  const text=(el,v)=>{if(el)el.textContent=v??'—'};
  const market=r=>r?.market?.selection||r?.market?.family||'Market';
  const match=r=>r?.match?.label||'Fixture';
  const league=r=>r?.match?.league||r?.match?.country||'';
  const status=r=>String(r?.state?.status||'WATCH').replaceAll('_',' ');
  const statusClass=s=>{s=String(s||'').toUpperCase();if(s==='READY'||s==='BET'||s==='HEALTHY'||s==='OK')return'ready';if(s.includes('WAIT')||s==='WATCH'||s.includes('STALE'))return'wait';if(s.includes('RESEARCH')||s.includes('NOT VERIFIED'))return'research';return'pass'};
  const compact=(r,last='')=>`<div class="compact-row"><b>${esc(match(r))}</b><span>${esc(market(r))}</span><span>${esc(last||edge(r?.pricing?.edge_pp))}</span></div>`;

  function renderToday(data){
    const t=data.today||{}, m=t.metrics||{};
    const metrics=document.querySelectorAll('#today .metrics .metric .n');
    const vals=[m.matches_scanned,m.deep_analyzed,m.strong_edges,m.waiting_xi,m.pass];
    metrics.forEach((el,i)=>text(el,vals[i]??'—'));
    const live=document.querySelector('#today .header .live');
    if(live)live.innerHTML=`<span class="dot"></span> LIVE DATA · ${esc(data.generated_at_local||data.generated_at_utc||'persisted')}`;
    const chip=document.querySelector('#today .preview-chip');
    if(chip)chip.textContent='LIVE DATA PREVIEW';

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
        top.model?.confidence!=null?`<span class="badge">Confidence ${conf(top.model.confidence)}</span>`:'',
        top.state?.data_quality?`<span class="badge good">Data ${esc(top.state.data_quality)}</span>`:'',
        top.state?.lineup?`<span class="badge good">XI ${esc(top.state.lineup)}</span>`:'',
        `<span class="badge">${esc(status(top))}</span>`
      ].filter(Boolean).join('');
    } else if(hero){
      hero.innerHTML='<div class="eyebrow">No comparable priced edge</div><div class="match-name">Waiting for model + market pair</div><div class="market-name">No fake zeroes</div>';
    }

    const strongList=document.querySelector('#today .hero-edge > .panel:nth-child(2) .signal-list');
    if(strongList){
      const rows=t.strong_signals||[];
      strongList.innerHTML=rows.length?rows.map(r=>`<div class="signal"><div><b>${esc(match(r))}</b><small>${esc(market(r))}${league(r)?` · ${esc(league(r))}`:''}</small></div><span class="pp">${esc(edge(r.pricing?.edge_pp))}</span></div>`).join(''):'<div class="placeholder">No strong signals in this persisted snapshot.</div>';
    }

    const panels=document.querySelectorAll('#today .bottom-grid > .panel');
    if(panels[0]){const rows=t.price_opportunities||[];panels[0].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[0].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No price waits now.</div>')}
    if(panels[1]){const rows=t.waiting_xi||[];panels[1].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[1].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No XI waits now.</div>')}
    if(panels[2]){const rows=t.upcoming||[];panels[2].querySelectorAll('.compact-row,.placeholder').forEach(x=>x.remove());panels[2].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r,r.match?.kickoff||'')).join(''):'<div class="placeholder">No upcoming rows in snapshot.</div>')}
  }

  function renderFeed(data){
    const body=document.getElementById('feedbody');
    if(!body)return;
    const rows=data.edge_feed?.rows||[];
    body.innerHTML=rows.length?rows.map(r=>`<tr>
      <td class="team-cell"><span class="team-dot">⚽</span>${esc(match(r))}</td>
      <td>${esc(market(r))}</td>
      <td>${esc(pct(r.model?.probability))}</td>
      <td>${esc(pct(r.pricing?.market_probability))}</td>
      <td class="edge">${esc(edge(r.pricing?.edge_pp))}</td>
      <td>${esc(price(r.pricing?.price))}</td>
      <td>${esc(conf(r.model?.confidence))}</td>
      <td><span class="status ${statusClass(status(r))}">${esc(status(r))}</span></td>
    </tr>`).join(''):'<tr><td colspan="8">No persisted rows for Edge Feed.</td></tr>';
    const live=document.querySelector('#feed .header .live');
    if(live)live.innerHTML=`<span class="dot"></span> Persisted · ${esc(data.generated_at_local||data.generated_at_utc||'latest')}`;
  }

  function renderTower(data){
    const ct=data.control_tower||{}, health=ct.system_health||{}, pipe=ct.pipeline||{}, mat=ct.maturation||{};
    const monitoring=mat.monitoring||{};
    const freshness=monitoring.report_freshness||{};
    const evidenceAge=monitoring.evidence_age||{};
    const header=document.querySelector('#tower .header .live');
    const age=evidenceAge?.evidence?.age_hours;
    const matState=String(evidenceAge.status||freshness.status||mat.status||'NOT_VERIFIED').replaceAll('_',' ');
    if(header)header.innerHTML=`<span class="dot"></span> Runtime ${esc(ct.runtime_generated_at_local||ct.runtime_generated_at_utc||'—')} · Maturity <span class="status ${statusClass(matState)}">${esc(matState)}${age!=null?` · ${esc(age)}h evidence age`:''}</span>`;

    const cards=document.querySelectorAll('#tower .health-card');
    const healthRows=[
      ['Render',ct.status||'LIVE'],
      ['Postgres',health.postgres],
      ['Scheduler',health.scheduler],
      ['API-Football',health.api_football_remaining!=null?`${health.api_football_remaining} LEFT`:health.api_football],
      ['Galaxy',health.galaxy],
      ['Last Tick',health.last_tick]
    ];
    cards.forEach((card,i)=>{const row=healthRows[i];if(!row)return;const b=card.querySelector('b'),s=card.querySelector('small');text(b,`● ${row[0]}`);text(s,row[1]??'—');if(b)b.className=String(row[1]||'').toUpperCase().includes('DEGRADED')?'':'green'});

    const minis=document.querySelectorAll('#tower .pipeline-grid > .panel:first-child .mini');
    const pvals=[
      [pipe.fixtures_scanned,'fixtures scanned'],[pipe.due,'due'],[pipe.deep_dives,'deep dives'],
      [pipe.events,'events'],[pipe.research_visible,'research visible'],[pipe.api_calls,'API calls']
    ];
    minis.forEach((el,i)=>{const x=pvals[i];if(!x)return;text(el.querySelector('b'),x[0]??'—');text(el.querySelector('small'),x[1])});

    const errPanel=document.querySelector('#tower .pipeline-grid > .panel:nth-child(2)');
    if(errPanel){
      errPanel.querySelectorAll('.error').forEach(x=>x.remove());
      const rows=Array.isArray(ct.errors?.rows)?ct.errors.rows:[];
      const freshnessIssue=String(matState).toUpperCase()!=='OK'?{reason:`Maturation evidence ${matState}`,stage:age!=null?`${age}h old`:'freshness not verified'}:null;
      const merged=[...(freshnessIssue?[freshnessIssue]:[]),...rows].slice(0,4);
      const badge=errPanel.querySelector('.status');if(badge){badge.textContent=String(merged.length);badge.className=`status ${merged.length?'wait':'ready'}`}
      errPanel.insertAdjacentHTML('beforeend',merged.length?merged.map(e=>`<div class="error"><span class="errdot">!</span><div><b>${esc(e.reason||'Pipeline issue')}</b><small>${esc(e.stage||e.fixture_id||'observability')}</small></div></div>`).join(''):'<div class="placeholder">No pipeline errors in latest persisted runtime.</div>');
    }

    const maturityBox=document.querySelector('#tower .pipeline-grid > .panel:nth-child(3) .maturity');
    if(maturityBox){
      const families=Array.isArray(mat.families)?mat.families:[];
      maturityBox.innerHTML=families.length?families.map(f=>{const cur=num(f.current),tar=num(f.target),w=cur!==null&&tar&&tar>0?Math.max(0,Math.min(100,(cur/tar)*100)):0;return `<div><div class="matrow"><b>${esc(f.label||f.key)}</b><div class="mbar"><div class="mfill" style="width:${w.toFixed(1)}%"></div></div><span>${esc(cur??'—')}/${esc(tar??'—')}</span></div>${f.blocker?`<small style="display:block;color:#eab95c;margin:2px 0 0 79px;font-size:7px">${esc(String(f.blocker).replaceAll('_',' '))}</small>`:''}</div>`}).join(''):'<div class="placeholder">Maturation reports unavailable.</div>';
    }

    const lower=document.querySelectorAll('#tower > .grid2 > .panel');
    if(lower[0])lower[0].innerHTML='<div class="ph"><h3>OOS Performance</h3><span class="status research">NEXT WIRING</span></div><div class="placeholder">Visual shell complete. OOS metrics will be connected only from persisted validation evidence.</div>';
    if(lower[1])lower[1].innerHTML='<div class="ph"><h3>CLV Summary</h3><span class="status research">NEXT WIRING</span></div><div class="placeholder">No mock CLV is shown while the persisted CLV adapter is being connected.</div>';
  }

  async function loadLive(){
    const token=localStorage.getItem(AK)||'';
    if(!token){
      const live=document.querySelector('#today .header .live');
      if(live)live.innerHTML='<span class="dot" style="background:#eab95c"></span> LOGIN REQUIRED FOR LIVE PREVIEW';
      return;
    }
    try{
      const r=await fetch('/app-preview/data',{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});
      const d=await r.json();
      if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);
      renderToday(d);renderFeed(d);renderTower(d);
    }catch(err){
      const live=document.querySelector('#today .header .live');
      if(live)live.innerHTML=`<span class="dot" style="background:#ff6679"></span> LIVE DATA ERROR · ${esc(err.message)}`;
    }
  }
  loadLive();
})();
</script>
'''


def _html() -> str:
    html = subscriber_preview_v230._html()
    marker = "</body>"
    return html.replace(marker, _LIVE_SCRIPT + marker, 1) if marker in html else html + _LIVE_SCRIPT


async def preview_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
