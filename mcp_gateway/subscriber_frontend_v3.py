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
    marker = "</head>"
    if marker in rendered:
        return rendered.replace(marker, shim + visual_overrides + marker, 1)
    return shim + visual_overrides + rendered
