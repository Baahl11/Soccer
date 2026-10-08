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
    marker = "</head>"
    if marker in rendered:
        rendered = rendered.replace(marker, shim + visual_overrides + premium_pass + marker, 1)
    else:
        rendered = shim + visual_overrides + premium_pass + rendered
    return rendered
