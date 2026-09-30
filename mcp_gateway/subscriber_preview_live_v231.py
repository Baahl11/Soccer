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
  const statusClass=s=>{s=String(s||'').toUpperCase();if(s==='READY'||s==='BET')return'ready';if(s.includes('WAIT'))return'wait';if(s.includes('RESEARCH'))return'research';return'pass'};
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
    if(panels[0]){const rows=t.price_opportunities||[];panels[0].querySelectorAll('.compact-row').forEach(x=>x.remove());panels[0].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No price waits now.</div>')}
    if(panels[1]){const rows=t.waiting_xi||[];panels[1].querySelectorAll('.compact-row').forEach(x=>x.remove());panels[1].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r)).join(''):'<div class="placeholder">No XI waits now.</div>')}
    if(panels[2]){const rows=t.upcoming||[];panels[2].querySelectorAll('.compact-row').forEach(x=>x.remove());panels[2].insertAdjacentHTML('beforeend',rows.length?rows.map(r=>compact(r,r.match?.kickoff||'')).join(''):'<div class="placeholder">No upcoming rows in snapshot.</div>')}
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
      renderToday(d);renderFeed(d);
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
