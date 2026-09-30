from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import subscriber_preview_live_v231


_SCRIPT = r'''
<script id="v231-performance-live">
(() => {
  const AK='soccer_edge_access_token';
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=v=>v===null||v===undefined||v===''?null:Number(v);
  const f=(v,d=3)=>{const x=n(v);return x===null||!Number.isFinite(x)?'—':x.toFixed(d)};
  const pp=v=>{const x=n(v);return x===null||!Number.isFinite(x)?'—':`${x>=0?'+':''}${x.toFixed(2)} pp`};
  function renderPerformance(d){
    const page=document.getElementById('performance');if(!page)return;
    const rows=Array.isArray(d.rows)?d.rows:[];
    const avg=d.weighted_avg_clv_pp;
    const one=rows.find(r=>r.label==='1X2')||{};
    const btts=rows.find(r=>r.label==='BTTS')||{};
    const ft=rows.find(r=>r.label==='FT Totals')||{};
    const metrics=page.querySelectorAll('.metrics .metric .n');
    const vals=[pp(avg),f(one.brier),f(btts.brier),rows.reduce((a,r)=>a+(n(r.sample_n)||0),0),`${one.true_clv_rows??'—'}/${one.true_clv_target??'—'}`];
    metrics.forEach((el,i)=>{if(el)el.textContent=vals[i]??'—'});
    const live=page.querySelector('.header .live');if(live)live.innerHTML=`<span class="dot"></span> ${esc(d.status||'NOT VERIFIED')} · persisted validation reports`;
    const panels=page.querySelectorAll('.grid2 > .panel');
    if(panels[0]){
      const body=rows.slice(0,9).map(r=>`<tr><td>${esc(r.label)}</td><td>${esc(r.sample_n??'—')}</td><td>${esc(r.true_clv_rows??'—')}/${esc(r.true_clv_target??'—')}</td><td>${esc(f(r.brier))}</td><td>${esc(f(r.log_loss))}</td><td>${esc(f(r.ece))}</td><td>${esc(r.status||'—')}</td></tr>`).join('');
      panels[0].innerHTML=`<div class="ph"><h3>Persisted Validation Performance</h3><span class="status research">${esc(d.status||'N/V')}</span></div><table class="table perf-table"><thead><tr><th>Market</th><th>Sample</th><th>True CLV</th><th>Brier</th><th>Log Loss</th><th>ECE</th><th>Status</th></tr></thead><tbody>${body||'<tr><td colspan="7">No persisted validation metrics.</td></tr>'}</tbody></table>`;
    }
    if(panels[1]){
      const clvRows=rows.filter(r=>r.avg_clv_pp!==null&&r.avg_clv_pp!==undefined);
      panels[1].innerHTML=`<div class="ph"><h3>True CLV by Family</h3></div><div class="bigprob green">${esc(pp(avg))}</div><div class="signal-list" style="margin-top:8px">${clvRows.map(r=>`<div class="signal"><div><b>${esc(r.label)}</b><small>${esc(r.true_clv_rows??'—')} / ${esc(r.true_clv_target??'—')} rows · ${esc(r.source||'persisted report')}</small></div><span class="pp">${esc(pp(r.avg_clv_pp))}</span></div>`).join('')||'<div class="placeholder">No family CLV evidence yet.</div>'}</div>`;
    }
  }
  async function loadPerformance(){
    const token=localStorage.getItem(AK)||'';if(!token)return;
    try{const r=await fetch('/app-preview/performance',{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);renderPerformance(d)}catch(err){const live=document.querySelector('#performance .header .live');if(live)live.textContent=`Performance unavailable · ${err.message}`}
  }
  loadPerformance();
})();
</script>
'''


def _html() -> str:
    html = subscriber_preview_live_v231._html()
    marker = "</body>"
    return html.replace(marker, _SCRIPT + marker, 1) if marker in html else html + _SCRIPT


async def preview_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
