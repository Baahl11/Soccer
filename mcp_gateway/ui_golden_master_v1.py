from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse


def _html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#040b12">
<title>Soccer Edge — Match Center Golden Master</title>
<style>
:root{
  --bg:#02090f;
  --shell:#05111a;
  --panel:#071923;
  --panel-hi:#0a1f2b;
  --line:#14384a;
  --line-hi:#21556c;
  --text:#f3f8fb;
  --muted:#7a94a3;
  --muted2:#58717f;
  --blue:#45afe2;
  --green:#4bd8a3;
  --gold:#d9ae4f;
  --red:#e97884;
  --radius:10px;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--text);background:var(--bg);color-scheme:dark
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:#01070c;color:var(--text)}
body{min-width:320px}
button{font:inherit}
.lab{
  min-height:100vh;
  background:
    radial-gradient(circle at 66% -10%,rgba(37,119,155,.18),transparent 30%),
    linear-gradient(180deg,#06131c 0,#02090f 100%);
}
.shell{
  max-width:1080px;margin:0 auto;min-height:100vh;display:grid;grid-template-columns:154px minmax(0,1fr);
  border-left:1px solid #0d2938;border-right:1px solid #0d2938;background:#041019
}
.rail{
  position:sticky;top:0;height:100vh;padding:20px 11px;border-right:1px solid #113244;background:#040d15
}
.brand{display:flex;align-items:center;gap:8px}
.brand-mark{
  width:34px;height:34px;border-radius:9px;display:grid;place-items:center;font-weight:950;font-size:11px;
  background:linear-gradient(145deg,#1187c3,#0a5c93);box-shadow:0 0 24px rgba(34,133,184,.3)
}
.brand b{font-size:9px;line-height:.88}
.rail .brand{margin:0 7px 24px}
.rail nav{display:grid;gap:4px}
.rail nav a{padding:9px 10px;border-radius:7px;color:#6e8796;font-size:7px;font-weight:850;text-decoration:none}
.rail nav a.active{background:#0b2636;border:1px solid #18455b;color:#fff}
.rail-note{position:absolute;left:17px;bottom:18px;color:#6f8795;font-size:6px;line-height:1.5}
.rail-note b{color:#92a8b3}
.workspace{min-width:0}
.topbar{
  height:48px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);
  background:rgba(5,17,26,.92);backdrop-filter:blur(10px)
}
.back{font-size:7px;color:#87a0ad;font-weight:800}
.live-wrap{display:flex;align-items:center;gap:9px;color:#718a98;font-size:6px}
.live{display:flex;align-items:center;gap:4px;color:#57dcae;font-weight:900}
.live i{width:6px;height:6px;border-radius:50%;background:#50d6a6;box-shadow:0 0 9px #50d6a6}
.hero{
  display:grid;grid-template-columns:minmax(0,1fr) 145px;gap:12px;padding:13px 14px 11px;border-bottom:1px solid var(--line);
  background:
    radial-gradient(circle at 34% -12%,rgba(32,113,150,.13),transparent 38%),
    linear-gradient(180deg,#071923,#06131b)
}
.hero-left{min-width:0}
.hero-meta{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:9px;color:#7992a1;font-size:6px}
.hero-meta b{color:#a1b3bd}
.faceoff{display:grid;grid-template-columns:1fr 30px 1fr;gap:8px;align-items:center}
.team{display:grid;justify-items:center;gap:5px;text-align:center;min-width:0}
.crest{
  width:58px;height:58px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(180deg,#11364a,#0a202c);
  border:1px solid #28566d;box-shadow:0 10px 24px #0006;position:relative
}
.crest::after{content:'';position:absolute;inset:4px;border:1px solid #ffffff0d;border-radius:50%}
.shield{
  width:36px;height:44px;clip-path:polygon(50% 0,92% 12%,85% 74%,50% 100%,15% 74%,8% 12%);
  display:grid;place-items:center;font-size:9px;font-weight:950;border:1px solid #ffffff25;color:white
}
.ars{background:linear-gradient(180deg,#e13a4b,#9b1424)}
.bha{background:linear-gradient(180deg,#2e78d3,#174791)}
.team b{font-size:12px;line-height:1.05;max-width:145px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.team span{font-size:5.7px;color:#6a8391}
.vs{text-align:center;font-size:9px;color:#557383;font-weight:950}
.quality{
  border:1px solid #17394a;border-radius:9px;background:#071822;align-self:start;overflow:hidden
}
.qrow{display:flex;justify-content:space-between;gap:7px;align-items:center;padding:7px 8px;border-bottom:1px solid #133548}
.qrow:last-child{border-bottom:0}.qrow span{font-size:5.5px;color:#6e8796}.qrow b{font-size:6px}
.good{color:#56ddb0}.info{color:#5bb8e3}.warn{color:#dfb657}
.tabs{display:flex;gap:4px;padding:7px 10px;border-bottom:1px solid var(--line);background:#06141d;overflow:auto}
.tabs button{border:0;background:transparent;color:#758f9e;padding:6px 10px;border-radius:6px;font-size:6.5px;font-weight:900}
.tabs button.active{background:#0d3043;color:white;border:1px solid #245b76}
.content{padding:9px}
.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin:1px 1px 8px}
.section-title h1{margin:0;font-size:12px;letter-spacing:-.02em}.section-title p{margin:2px 0 0;color:#6c8594;font-size:5.5px}
.section-title span{font-size:5.2px;color:#5f7988;white-space:nowrap}
.grid{display:grid;gap:7px}
.grid.top{grid-template-columns:1.1fr .72fr .78fr}
.grid.mid{grid-template-columns:.9fr .92fr 1.18fr;margin-top:7px}
.panel{
  min-width:0;border:1px solid var(--line);border-radius:var(--radius);padding:8px;background:
  linear-gradient(180deg,#081a25 0,#06151d 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.012)
}
.panel-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px}
.panel-head h3{margin:0;font-size:7.2px}.panel-head span{font-size:4.7px;color:#617b8b;font-weight:900;letter-spacing:.08em}
.prob-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.prob{padding:7px 3px;text-align:center;border:1px solid #173b4e;border-radius:6px;background:#071a24}
.prob span{display:block;color:#6e8797;font-size:4.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prob b{display:block;margin-top:3px;font-size:12px}.prob.home b{color:#59deb0}.prob.draw b{color:#ddb455}.prob.away b{color:#58b8e3}
.probbar{height:5px;display:flex;margin-top:6px;border-radius:999px;overflow:hidden;background:#102a39}
.probbar i{height:100%}.probbar .home{background:var(--green)}.probbar .draw{background:var(--gold)}.probbar .away{background:var(--blue)}
.xg{display:grid;grid-template-columns:1fr auto 1fr;gap:5px;align-items:end;text-align:center;padding:9px 0 3px}
.xg span{display:block;color:#6c8594;font-size:4.7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xg b{display:block;margin-top:4px;font-size:21px;letter-spacing:-.04em}.xg em{font-style:normal;color:#4e6a79;padding-bottom:5px}
.edge-bars{display:grid;gap:5px}
.edge-row{display:grid;grid-template-columns:34px minmax(0,1fr) 29px;gap:4px;align-items:center}
.edge-row span{font-size:4.6px;color:#6c8594}.edge-row b{font-size:5.2px;text-align:right}
.edge-track{height:5px;border-radius:999px;background:#102b39;overflow:hidden}.edge-track i{display:block;height:100%;border-radius:999px}
.edge-track .model{background:var(--green)}.edge-track .market{background:#3c8fbd}
.edge-value{margin-top:5px;font-size:15px;font-weight:950;color:#59ddb0}
.edge-caption{margin-top:2px;font-size:4.6px;color:#607a89}
.profile{display:grid;gap:5px}
.profile-row{display:grid;grid-template-columns:66px minmax(0,1fr) 32px;gap:5px;align-items:center}
.profile-row span{font-size:4.7px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.profile-row div{height:5px;background:#102a39;border-radius:999px;overflow:hidden}.profile-row i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#247b61,#53d5a7)}
.profile-row b{font-size:4.9px;text-align:right}.profile-row b.good{color:#57dcae}.profile-row b.neutral{color:#d8ae52}
.matrix{display:grid;grid-template-columns:14px repeat(5,1fr);gap:2px;align-items:center}
.axis{font-size:3.9px;color:#617b8a;text-align:center}
.cell{aspect-ratio:1;border:1px solid #1a4961;border-radius:2px;display:grid;place-items:center;font-size:4px;font-weight:850;color:#edf8fc}
.c1{background:#0b222f}.c2{background:#103547}.c3{background:#15536a}.c4{background:#1c718f}.c5{background:#2b91b6}
.matrix-note{margin-top:5px;color:#637d8c;font-size:4.4px;text-align:center}.matrix-note b{color:#cfe4ee}
.market-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.market-metric{padding:6px 3px;text-align:center;border:1px solid #17394a;border-radius:6px;background:#071923}
.market-metric span{display:block;color:#6d8695;font-size:4.3px}.market-metric b{display:block;margin-top:3px;font-size:8.8px}.market-metric b.green{color:#56ddb0}
.distribution{height:72px;display:grid;grid-template-columns:repeat(6,1fr);gap:4px;align-items:end;margin-top:7px;padding:0 3px 4px;border-bottom:1px solid #173748}
.dcol{height:100%;display:grid;grid-template-rows:1fr auto;gap:2px;text-align:center}
.bars{height:100%;display:flex;align-items:flex-end;justify-content:center;gap:1px}.bars i{display:block;width:38%;min-height:2px;border-radius:2px 2px 0 0}
.bars .h{background:#42afe3}.bars .a{background:#4bd7a3}.dcol b{font-size:4.4px;color:#687f8e}
.legend{display:flex;gap:8px;margin-top:5px;font-size:4.4px;color:#687f8e}.legend span{display:flex;align-items:center;gap:3px}.legend i{width:5px;height:5px;border-radius:50%}.legend .h{background:#42afe3}.legend .a{background:#4bd7a3}
.read{
  margin-top:7px;display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:8px;align-items:center;padding:7px 8px;
  border:1px solid #1a495d;border-left:3px solid var(--green);border-radius:8px;background:#071922
}
.read-mark{width:30px;height:30px;border-radius:7px;background:#0b322a;border:1px solid #26644f;display:grid;place-items:center;color:#62deb4;font-size:7px;font-weight:950}
.read b{display:block;font-size:6.5px}.read p{margin:2px 0 0;color:#6f8796;font-size:4.9px;line-height:1.35}.read strong{font-size:7.5px;color:#59ddb0;white-space:nowrap}
.audit-line{display:flex;justify-content:space-between;gap:8px;margin-top:7px;padding:0 2px;color:#526d7b;font-size:4.4px}
.footer-note{padding:10px 12px 12px;text-align:center;color:#4f6a78;font-size:4.5px}
@media(max-width:740px){
  .shell{display:block;border:0;max-width:none}.rail{display:none}.topbar{height:44px;padding:0 10px}
  .hero{grid-template-columns:minmax(0,1fr) 106px;padding:10px}.crest{width:52px;height:52px}.shield{width:32px;height:39px}.team b{font-size:10.5px;max-width:118px}
  .quality{border-radius:7px}.qrow{padding:6px}.qrow span{font-size:4.8px}.qrow b{font-size:5.2px}
  .tabs{padding:6px 7px}.tabs button{padding:6px 8px;font-size:6px}
  .content{padding:7px}.grid.top{grid-template-columns:1.15fr .72fr .86fr}.grid.mid{grid-template-columns:.92fr .93fr 1.15fr}
  .panel{padding:7px}.panel-head h3{font-size:6.5px}.prob{padding:6px 2px}.prob b{font-size:10px}.xg b{font-size:17px}
  .profile-row{grid-template-columns:54px minmax(0,1fr) 29px}.profile-row span{font-size:4.25px}.profile-row b{font-size:4.3px}
  .distribution{height:63px}
}
@media(max-width:430px){
  .topbar .back{font-size:6px}.live-wrap{font-size:5.3px}.hero{grid-template-columns:minmax(0,1fr) 96px;gap:7px}
  .crest{width:48px;height:48px}.team b{font-size:9.5px;max-width:105px}.hero-meta{font-size:5px}.qrow{padding:5px}
  .content{padding:6px}.section-title h1{font-size:10px}.section-title p{font-size:4.8px}
  .grid{gap:5px}.grid.top{grid-template-columns:1.18fr .82fr}.grid.top .edge-card{grid-column:1/-1}
  .grid.mid{grid-template-columns:1fr 1fr}.grid.mid .goals-card{grid-column:1/-1}
  .edge-card{display:grid;grid-template-columns:1fr 80px;gap:7px}.edge-card .panel-head{grid-column:1/-1;margin-bottom:2px}.edge-bars{align-self:start}.edge-value{font-size:13px}
  .profile-row{grid-template-columns:53px minmax(0,1fr) 27px}
  .goals-card{display:grid;grid-template-columns:112px minmax(0,1fr);column-gap:8px}.goals-card .panel-head{grid-column:1/-1}.goals-card .distribution{margin-top:0}.legend{margin-top:2px}
  .read{grid-template-columns:28px minmax(0,1fr) auto}.read-mark{width:28px;height:28px}
}
</style>
</head>
<body>
<div class="lab">
  <div class="shell">
    <aside class="rail">
      <div class="brand"><div class="brand-mark">SE</div><b>SOCCER<br>EDGE</b></div>
      <nav>
        <a href="#">Today</a>
        <a href="#">Edge Feed</a>
        <a class="active" href="#">Matches</a>
        <a href="#">Markets</a>
        <a href="#">Performance</a>
        <a href="#">My Edge</a>
      </nav>
      <div class="rail-note"><b>Football intelligence</b><br>Sport first. Market second.</div>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <span class="back">← Back to matches</span>
        <div class="live-wrap"><span class="live"><i></i> LIVE</span><span>Last update 2m ago</span></div>
      </header>

      <section class="hero">
        <div class="hero-left">
          <div class="hero-meta"><b>Premier League · Today</b><span>20:00</span></div>
          <div class="faceoff">
            <div class="team"><div class="crest"><div class="shield ars">ARS</div></div><b>Arsenal</b><span>Home</span></div>
            <div class="vs">VS</div>
            <div class="team"><div class="crest"><div class="shield bha">BHA</div></div><b>Brighton</b><span>Away</span></div>
          </div>
        </div>
        <aside class="quality">
          <div class="qrow"><span>Data Quality</span><b class="good">A</b></div>
          <div class="qrow"><span>Lineup</span><b class="good">Confirmed</b></div>
          <div class="qrow"><span>Market</span><b class="info">Fresh</b></div>
        </aside>
      </section>

      <nav class="tabs">
        <button class="active">Overview</button><button>Goals</button><button>Corners</button><button>Cards</button><button>Players</button><button>Market</button><button>Model</button>
      </nav>

      <main class="content">
        <div class="section-title">
          <div><h1>Match Center</h1><p>Why does this matchup matter?</p></div>
          <span>Premier League · model snapshot</span>
        </div>

        <section class="grid top">
          <article class="panel">
            <div class="panel-head"><h3>Match Result Probability</h3><span>MODEL</span></div>
            <div class="prob-cards">
              <div class="prob home"><span>HOME</span><b>67.1%</b></div>
              <div class="prob draw"><span>DRAW</span><b>20.3%</b></div>
              <div class="prob away"><span>AWAY</span><b>12.6%</b></div>
            </div>
            <div class="probbar"><i class="home" style="width:67.1%"></i><i class="draw" style="width:20.3%"></i><i class="away" style="width:12.6%"></i></div>
          </article>

          <article class="panel">
            <div class="panel-head"><h3>Expected Goals (λ)</h3><span>SPORT</span></div>
            <div class="xg"><div><span>Arsenal</span><b>2.08</b></div><em>—</em><div><span>Brighton</span><b>0.91</b></div></div>
          </article>

          <article class="panel edge-card">
            <div class="panel-head"><h3>Edge Gap</h3><span>EXAMPLE MARKET</span></div>
            <div class="edge-bars">
              <div class="edge-row"><span>Model</span><div class="edge-track"><i class="model" style="width:100%"></i></div><b>71.8%</b></div>
              <div class="edge-row"><span>Market</span><div class="edge-track"><i class="market" style="width:84%"></i></div><b>60.1%</b></div>
            </div>
            <div><div class="edge-value">+11.7 pp</div><div class="edge-caption">model − market fair</div></div>
          </article>
        </section>

        <section class="grid mid">
          <article class="panel">
            <div class="panel-head"><h3>Sport Profile</h3><span>FOOTBALL ONLY</span></div>
            <div class="profile">
              <div class="profile-row"><span>Attack strength</span><div><i style="width:83%"></i></div><b class="good">GOOD</b></div>
              <div class="profile-row"><span>Defense strength</span><div><i style="width:58%"></i></div><b class="neutral">NEUTRAL</b></div>
              <div class="profile-row"><span>Territorial control</span><div><i style="width:79%"></i></div><b class="good">GOOD</b></div>
              <div class="profile-row"><span>Recent form</span><div><i style="width:86%"></i></div><b class="good">GOOD</b></div>
              <div class="profile-row"><span>Home advantage</span><div><i style="width:81%"></i></div><b class="good">GOOD</b></div>
              <div class="profile-row"><span>Opponent quality</span><div><i style="width:54%"></i></div><b class="neutral">NEUTRAL</b></div>
            </div>
          </article>

          <article class="panel">
            <div class="panel-head"><h3>Score Matrix (FT)</h3><span>PROBABILITY</span></div>
            <div class="matrix">
              <span></span><span class="axis">0</span><span class="axis">1</span><span class="axis">2</span><span class="axis">3</span><span class="axis">4+</span>
              <span class="axis">0</span><div class="cell c1">2</div><div class="cell c2">5</div><div class="cell c2">6</div><div class="cell c1">3</div><div class="cell c1">1</div>
              <span class="axis">1</span><div class="cell c2">5</div><div class="cell c4">9</div><div class="cell c5">11</div><div class="cell c3">7</div><div class="cell c1">2</div>
              <span class="axis">2</span><div class="cell c2">4</div><div class="cell c5">10</div><div class="cell c4">9</div><div class="cell c3">6</div><div class="cell c1">2</div>
              <span class="axis">3</span><div class="cell c1">2</div><div class="cell c3">5</div><div class="cell c3">6</div><div class="cell c2">4</div><div class="cell c1">1</div>
              <span class="axis">4+</span><div class="cell c1">1</div><div class="cell c1">2</div><div class="cell c2">3</div><div class="cell c1">2</div><div class="cell c1">1</div>
            </div>
            <div class="matrix-note">Most likely: <b>2–1 · 11.4%</b></div>
          </article>

          <article class="panel goals-card">
            <div class="panel-head"><h3>Over / Under 2.5 Goals</h3><span>MODEL VS MARKET</span></div>
            <div class="market-grid">
              <div class="market-metric"><span>MODEL</span><b>64.4%</b></div>
              <div class="market-metric"><span>MARKET</span><b>55.3%</b></div>
              <div class="market-metric"><span>EDGE</span><b class="green">+8.9 pp</b></div>
            </div>
            <div class="distribution">
              <div class="dcol"><div class="bars"><i class="h" style="height:31%"></i><i class="a" style="height:52%"></i></div><b>0</b></div>
              <div class="dcol"><div class="bars"><i class="h" style="height:72%"></i><i class="a" style="height:88%"></i></div><b>1</b></div>
              <div class="dcol"><div class="bars"><i class="h" style="height:100%"></i><i class="a" style="height:58%"></i></div><b>2</b></div>
              <div class="dcol"><div class="bars"><i class="h" style="height:76%"></i><i class="a" style="height:27%"></i></div><b>3</b></div>
              <div class="dcol"><div class="bars"><i class="h" style="height:48%"></i><i class="a" style="height:11%"></i></div><b>4</b></div>
              <div class="dcol"><div class="bars"><i class="h" style="height:26%"></i><i class="a" style="height:5%"></i></div><b>5+</b></div>
            </div>
            <div class="legend"><span><i class="h"></i>Arsenal</span><span><i class="a"></i>Brighton</span></div>
          </article>
        </section>

        <section class="read">
          <div class="read-mark">SE</div>
          <div><b>Primary read</b><p>Sporting projection is built first. Market value is assessed only after the football case is established.</p></div>
          <strong>+11.7 pp</strong>
        </section>

        <div class="audit-line"><span>Data A · XI confirmed · Market fresh</span><span>Illustrative design-lab values</span></div>
      </main>

      <div class="footer-note">DESIGN LAB ONLY · SAMPLE DATA · NOT A LIVE PREDICTION</div>
    </section>
  </div>
</div>
</body>
</html>"""


async def design_match_center(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
