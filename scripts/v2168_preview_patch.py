from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "mcp_gateway/subscriber_preview_live_v231.py",
    "    if(!token){const live=document.querySelector('#today .header .live');if(live)live.innerHTML='<span class=\"dot\" style=\"background:#eab95c\"></span> LOGIN REQUIRED FOR LIVE PREVIEW';return}",
    "    if(!token){const live=document.querySelector('#today .header .live');if(live)live.innerHTML='<span class=\"dot\" style=\"background:#eab95c\"></span> LOGIN REQUIRED FOR LIVE PREVIEW';document.documentElement.classList.remove('v232-booting');return null}",
    "live no-token",
)
replace_once(
    "mcp_gateway/subscriber_preview_live_v231.py",
    "      renderToday(LIVE);renderFeed(LIVE);renderMatch(LIVE);renderMarkets(LIVE);renderTower(LIVE);renderPerformance(PERF);renderResearch(LIVE,PERF);renderMyEdge(LIVE);",
    "      renderToday(LIVE);renderFeed(LIVE);renderMatch(LIVE);renderMarkets(LIVE);renderTower(LIVE);renderPerformance(PERF);renderResearch(LIVE,PERF);renderMyEdge(LIVE);\n      const ready={data:LIVE,performance:PERF};window.__soccerEdgeLiveReady=ready;window.dispatchEvent(new CustomEvent('soccer-edge:live-ready',{detail:ready}));document.documentElement.classList.remove('v232-booting');return ready;",
    "live render",
)
replace_once(
    "mcp_gateway/subscriber_preview_live_v231.py",
    "    }catch(err){const live=document.querySelector('#today .header .live');if(live)live.innerHTML=`<span class=\"dot\" style=\"background:#ff6679\"></span> LIVE DATA ERROR · ${esc(err.message)}`}",
    "    }catch(err){const live=document.querySelector('#today .header .live');if(live)live.innerHTML=`<span class=\"dot\" style=\"background:#ff6679\"></span> LIVE DATA ERROR · ${esc(err.message)}`;document.documentElement.classList.remove('v232-booting');return null}",
    "live catch",
)
replace_once(
    "mcp_gateway/subscriber_preview_live_v231.py",
    "  loadLive();",
    "  window.__soccerEdgeLivePromise=loadLive();",
    "live bootstrap call",
)

replace_once(
    "mcp_gateway/subscriber_preview_performance_live_v231.py",
    'def _html() -> str:\n    html = subscriber_preview_live_v231._html()\n    marker = "</body>"\n    return html.replace(marker, _SCRIPT + marker, 1) if marker in html else html + _SCRIPT\n',
    'def _html() -> str:\n    # The main live bootstrap already fetches and renders performance.\n    # Avoid mounting a second renderer over the same DOM.\n    return subscriber_preview_live_v231._html()\n',
    "performance wrapper",
)

replace_once(
    "mcp_gateway/subscriber_preview_maturity_live_v232.py",
    "_STYLE = r'''\n<style id=\"v232-maturity-style\">",
    "_STYLE = r'''\n<script id=\"v232-boot-state\">document.documentElement.classList.add('v232-booting');</script>\n<style id=\"v232-maturity-style\">\nhtml.v232-booting body>*{opacity:0!important;pointer-events:none!important}\nhtml.v232-booting body::before{content:'SOCCER EDGE · loading live snapshot…';position:fixed;inset:0;display:grid;place-items:center;background:#07141d;color:#7895aa;font:600 11px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.08em;z-index:2147483647;visibility:visible}",
    "maturity boot style",
)
replace_once(
    "mcp_gateway/subscriber_preview_maturity_live_v232.py",
    """  async function loadMaturity(){
    const token=localStorage.getItem(AK)||'';if(!token)return;
    try{
      const [mat,live]=await Promise.all([fetchJson('/app-preview/maturity',token),fetchJson('/app-preview/data',token)]);
      MAT=mat;LIVE=live;renderMarkets();renderTower();patchPerformanceLabel();
      setTimeout(()=>{renderMarkets();renderTower();patchPerformanceLabel()},900);
    }catch(err){const chip=document.querySelector('#markets .preview-chip');if(chip)chip.textContent=`MATURITY UNAVAILABLE · ${err.message}`}
  }
  setTimeout(loadMaturity,120);""",
    """  async function liveSnapshot(){
    if(window.__soccerEdgeLiveReady?.data)return window.__soccerEdgeLiveReady.data;
    if(window.__soccerEdgeLivePromise&&typeof window.__soccerEdgeLivePromise.then==='function'){
      const ready=await window.__soccerEdgeLivePromise;return ready?.data||null;
    }
    return await new Promise(resolve=>{
      const timer=setTimeout(()=>resolve(null),2500);
      window.addEventListener('soccer-edge:live-ready',event=>{clearTimeout(timer);resolve(event.detail?.data||null)},{once:true});
    });
  }
  async function loadMaturity(){
    const token=localStorage.getItem(AK)||'';if(!token){document.documentElement.classList.remove('v232-booting');return}
    try{
      const [mat,live]=await Promise.all([fetchJson('/app-preview/maturity',token),liveSnapshot()]);
      MAT=mat;LIVE=live||{};renderMarkets();renderTower();patchPerformanceLabel();
    }catch(err){const chip=document.querySelector('#markets .preview-chip');if(chip)chip.textContent=`MATURITY UNAVAILABLE · ${err.message}`}
    finally{document.documentElement.classList.remove('v232-booting')}
  }
  loadMaturity();""",
    "maturity load",
)

Path("tests/test_v2168_preview_single_bootstrap.py").write_text(
    '''from mcp_gateway.subscriber_preview_live_v231 import _html as live_html\nfrom mcp_gateway.subscriber_preview_performance_live_v231 import _html as performance_html\nfrom mcp_gateway.subscriber_preview_maturity_live_v232 import _html as maturity_html\n\n\ndef test_v2168_single_visible_bootstrap():\n    html = maturity_html()\n    assert "v232-boot-state" in html\n    assert "v232-booting" in html\n    assert "window.__soccerEdgeLivePromise=loadLive()" in html\n    assert "soccer-edge:live-ready" in html\n    assert "setTimeout(loadMaturity,120)" not in html\n    assert "setTimeout(()=>{renderMarkets();renderTower();patchPerformanceLabel()},900)" not in html\n    assert "fetchJson('/app-preview/data',token)" not in html\n\n\ndef test_v2168_performance_wrapper_does_not_mount_second_renderer():\n    base = live_html()\n    wrapped = performance_html()\n    assert wrapped == base\n    assert wrapped.count('id="v231-live-preview"') == 1\n    assert 'id="v231-performance-live"' not in wrapped\n''',
    encoding="utf-8",
)
