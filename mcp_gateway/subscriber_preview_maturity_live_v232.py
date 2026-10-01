from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import subscriber_preview_performance_live_v231


_STYLE = r'''
<style id="v232-maturity-style">
.v232-evidence{display:grid;gap:0;margin-top:7px}.v232-evidence .mini-line{min-height:19px}.v232-stage{color:#69d7b4!important}.v232-gate{color:#eab95c!important}.v232-source{color:#607d91;font-size:7px;margin-top:6px;line-height:1.35}.v232-truth-note{margin:10px 0 0;padding:8px 10px;border:1px solid #17384c;background:#091923;border-radius:8px;color:#7895aa;font-size:8px;line-height:1.45}
</style>
'''

_SCRIPT = r'''
<script id="v232-maturity-live">
(() => {
  const AK='soccer_edge_access_token';
  const TRACKED='soccer_edge_tracked_markets_v1';
  let MAT=null, LIVE=null;
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>v===null||v===undefined||v===''?null:Number(v);
  const human=v=>String(v??'—').replaceAll('_',' ');
  const getTracked=()=>{try{const v=JSON.parse(localStorage.getItem(TRACKED)||'[]');return Array.isArray(v)?v:[]}catch(_){return[]}};
  const setTracked=v=>localStorage.setItem(TRACKED,JSON.stringify(v));
  const toggleTracked=label=>{const v=getTracked(),i=v.indexOf(label);if(i>=0)v.splice(i,1);else v.push(label);setTracked(v);renderMarkets();};
  const fetchJson=async(path,token)=>{const r=await fetch(path,{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d};
  const oosText=m=>{
    const e=m?.model_evidence||{};
    if(m?.label==='Player Props')return `OOS ${e.current??0}${e.structural_profiles_max!=null?` · structural profiles ${e.structural_profiles_max}`:''}`;
    if(e.current==null)return '—';
    return `${e.current}${e.target!=null?`/${e.target}`:''} ${e.unit||''}${e.ready?' · READY':''}`.trim();
  };
  const marketText=m=>{
    if(m?.priced_rows==null&&m?.mapped_rows==null)return '—';
    return `${m?.priced_rows??0} priced · ${m?.mapped_rows??0} mapped`;
  };
  const clvText=m=>`${m?.true_clv_rows??'—'}/${m?.true_clv_target??'—'}${m?.true_clv_fixtures!=null?` · ${m.true_clv_fixtures} fx`:''}`;
  const liveByLabel=()=>Object.fromEntries(((LIVE?.markets?.families)||[]).map(x=>[x.label,x]));

  function renderMarkets(){
    if(!MAT)return;
    const grid=document.getElementById('marketgrid');if(!grid)return;
    const live=liveByLabel(),tracked=new Set(getTracked()),families=Array.isArray(MAT.families)?MAT.families:[];
    grid.innerHTML=families.length?families.map(m=>{
      const l=live[m.label]||{},on=tracked.has(m.label),e=m.model_evidence||{};
      const extras=[];
      if(m.label==='Team Totals'&&m.modeled_signal_fixtures!=null)extras.push(`<div class="mini-line"><span>Modeled fixtures</span><b>${esc(m.modeled_signal_fixtures)}</b></div>`);
      if(m.label==='Corners'&&e.current!=null)extras.push(`<div class="mini-line"><span>Formation gate</span><b>${esc(e.current)}/${esc(e.target??'—')}</b></div>`);
      if(m.label==='Cards'&&e.secondary){extras.push(`<div class="mini-line"><span>Referee / red-card OOS</span><b>${esc(e.secondary.yellow_referee_adjusted??'—')}/${esc(e.secondary.yellow_referee_target??'—')} ref · ${esc(e.secondary.red_card_oos??'—')}/${esc(e.secondary.red_card_market_review_target??'—')} red</b></div>`)}
      return `<div class="market-card" data-v232-maturity="1"><div style="display:flex;justify-content:space-between;gap:8px"><h3>${esc(m.label)}</h3><button class="v231-action ${on?'on':''}" data-v232-track="${esc(m.label)}">${on?'Tracked':'Track'}</button></div><div class="count">${esc(l.live_rows??0)}<span class="subtitle"> live rows</span></div><small>${esc(l.description||'Persisted market family')}</small><div class="v232-evidence"><div class="mini-line"><span>Stage</span><b class="v232-stage">${esc(human(m.stage))}</b></div><div class="mini-line"><span>Model / OOS</span><b>${esc(oosText(m))}</b></div><div class="mini-line"><span>Market evidence</span><b>${esc(marketText(m))}</b></div><div class="mini-line"><span>Strict True CLV</span><b>${esc(clvText(m))}</b></div>${extras.join('')}${m.next_gate?`<div class="mini-line"><span>Next gate</span><b class="v232-gate">${esc(human(m.next_gate))}</b></div>`:''}</div><div class="v232-source">${esc(m.source||'persisted validation')} · ${esc(m.report_status||'NOT VERIFIED')}</div></div>`;
    }).join(''):'<div class="v231-empty">No persisted maturity evidence available.</div>';
    grid.querySelectorAll('[data-v232-track]').forEach(b=>b.onclick=()=>toggleTracked(b.dataset.v232Track));
    const sub=document.querySelector('#markets .header .subtitle');if(sub)sub.textContent='Maturity = model/OOS evidence + observed market collection + strict True CLV. Each gate is shown separately.';
    const chip=document.querySelector('#markets .preview-chip');if(chip)chip.textContent='LIVE MATURITY · MULTI-GATE';
    if(!document.getElementById('v232-truth-note')){const note=document.createElement('div');note.id='v232-truth-note';note.className='v232-truth-note';note.textContent=MAT.truth_note||'True CLV is one maturity gate, not the whole maturity state.';grid.insertAdjacentElement('afterend',note)}
  }

  function renderTower(){
    if(!MAT)return;
    const panel=document.querySelector('#tower .pipeline-grid > .panel:nth-child(3)');if(!panel)return;
    const h=panel.querySelector('h3');if(h)h.textContent='Maturity Evidence';
    const badge=panel.querySelector('.status');if(badge){badge.textContent='MULTI-GATE';badge.className='status research'}
    const box=panel.querySelector('.maturity');if(!box)return;
    const rows=Array.isArray(MAT.families)?MAT.families:[];
    box.innerHTML=rows.map(m=>{
      const evidence=m.model_evidence||{};
      const formationPrimary=m.label==='Corners'&&num(evidence.current)!=null&&num(evidence.target)!=null;
      const cur=formationPrimary?num(evidence.current):num(m.true_clv_rows);
      const tar=formationPrimary?num(evidence.target):num(m.true_clv_target);
      const w=cur!=null&&tar?Math.max(0,Math.min(100,cur/tar*100)):0;
      const ratio=formationPrimary?`${cur}/${tar}`:`CLV ${cur??'—'}/${tar??'—'}`;
      const secondaryClv=formationPrimary?` · Strict CLV ${clvText(m)}`:'';
      return `<div><div class="matrow"><b>${esc(m.label)}</b><div class="mbar"><div class="mfill" style="width:${w}%"></div></div><span>${esc(ratio)}</span></div><small style="display:block;color:#7895aa;margin:2px 0 4px 79px;font-size:7px">${esc(human(m.stage))}${m.priced_rows!=null?` · ${esc(m.priced_rows)} priced`:''}${esc(secondaryClv)}</small></div>`;
    }).join('');
  }

  function patchPerformanceLabel(){
    const boxes=document.querySelectorAll('#performance .metrics .metric');
    if(boxes[4]){const label=boxes[4].querySelector('.label');if(label)label.textContent='1X2 True CLV'}
  }

  async function loadMaturity(){
    const token=localStorage.getItem(AK)||'';if(!token)return;
    try{
      const [mat,live]=await Promise.all([fetchJson('/app-preview/maturity',token),fetchJson('/app-preview/data',token)]);
      MAT=mat;LIVE=live;renderMarkets();renderTower();patchPerformanceLabel();
      setTimeout(()=>{renderMarkets();renderTower();patchPerformanceLabel()},900);
    }catch(err){const chip=document.querySelector('#markets .preview-chip');if(chip)chip.textContent=`MATURITY UNAVAILABLE · ${err.message}`}
  }
  setTimeout(loadMaturity,120);
})();
</script>
'''


def _html() -> str:
    html = subscriber_preview_performance_live_v231._html()
    head = "</head>"
    body = "</body>"
    if head in html:
        html = html.replace(head, _STYLE + head, 1)
    else:
        html = _STYLE + html
    return html.replace(body, _SCRIPT + body, 1) if body in html else html + _SCRIPT


async def preview_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
