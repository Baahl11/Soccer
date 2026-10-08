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
  --bg:#020a10;
  --shell:#06121b;
  --panel:#081923;
  --panel-2:#0a1e2a;
  --line:#15384a;
  --line-hi:#20566f;
  --text:#f2f7fa;
  --muted:#7f98a6;
  --muted2:#58717f;
  --blue:#3ea9df;
  --cyan:#66c5ea;
  --green:#46d7a1;
  --gold:#d6aa45;
  --red:#e36e7b;
  --radius:14px;
  color-scheme:dark;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:#01080e;color:var(--text)}
body{min-width:320px}
button{font:inherit}
.prototype{
  min-height:100vh;
  background:
    radial-gradient(circle at 50% -12%,rgba(35,121,158,.18),transparent 30%),
    linear-gradient(180deg,#06131c 0,#020a10 100%);
  padding:14px 12px 44px;
}
.frame{
  max-width:720px;margin:0 auto;border:1px solid #14394b;border-radius:18px;overflow:hidden;
  background:#05131b;box-shadow:0 28px 80px #000a;
}
.topbar{
  height:48px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;
  border-bottom:1px solid var(--line);background:#06141d;
}
.brand{display:flex;align-items:center;gap:8px}
.brand-mark{
  width:30px;height:30px;border-radius:8px;display:grid;place-items:center;font-size:10px;font-weight:950;
  background:linear-gradient(145deg,#1188c4,#0a5d94);box-shadow:0 0 24px rgba(33,133,186,.3)
}
.brand b{font-size:9px;line-height:.86;letter-spacing:.01em}
.proto-badge{
  padding:5px 8px;border:1px solid #625529;border-radius:999px;background:#241e0d;color:#dfb85d;
  font-size:6px;font-weight:900;letter-spacing:.08em
}
.hero{
  display:grid;grid-template-columns:1fr 110px;padding:13px 14px 12px;gap:10px;
  border-bottom:1px solid var(--line);
  background:
    radial-gradient(circle at 25% 0,rgba(30,107,142,.11),transparent 38%),
    linear-gradient(180deg,#071923,#06131b);
}
.hero-left{min-width:0}
.hero-meta{display:flex;justify-content:space-between;gap:8px;color:#7c95a3;font-size:6px;margin-bottom:10px}
.hero-meta b{color:#9aafba}
.faceoff{display:grid;grid-template-columns:1fr 26px 1fr;align-items:center;gap:8px}
.team{display:grid;justify-items:center;text-align:center;gap:6px;min-width:0}
.crest{
  width:64px;height:64px;border-radius:50%;display:grid;place-items:center;position:relative;
  background:linear-gradient(180deg,#113348,#0a1e2a);border:1px solid #27546a;box-shadow:0 10px 24px #0007
}
.shield{
  width:39px;height:48px;clip-path:polygon(50% 0,92% 12%,85% 74%,50% 100%,15% 74%,8% 12%);
  display:grid;place-items:center;font-weight:950;font-size:10px;color:white;border:1px solid #ffffff22
}
.ars{background:linear-gradient(180deg,#df3044,#9b1424)}
.bha{background:linear-gradient(180deg,#2a72c8,#14418a)}
.team b{font-size:13px;line-height:1.05;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:138px}
.team span{font-size:6px;color:#6e8795}
.vs{text-align:center;font-size:10px;color:#567484;font-weight:950}
.hero-status{display:grid;align-content:start;gap:6px}
.qrow{
  display:flex;justify-content:space-between;gap:7px;align-items:center;
  padding:7px 8px;border:1px solid #17394a;border-radius:8px;background:#071923;
}
.qrow span{font-size:5.5px;color:#6f8998}.qrow b{font-size:6.2px}
.good{color:#55deb0}.warn{color:#ddb658}.info{color:#5eb9e3}
.tabs{
  display:flex;gap:4px;padding:7px 10px;border-bottom:1px solid var(--line);background:#06141d;overflow:auto
}
.tabs button{
  border:0;background:transparent;color:#758f9e;padding:7px 10px;border-radius:7px;font-size:6.5px;font-weight:900
}
.tabs button.active{background:#0e3043;color:#fff;border:1px solid #245b76}
.content{padding:9px}
.section-title{
  display:flex;justify-content:space-between;align-items:end;gap:10px;margin:1px 2px 8px
}
.section-title h2{margin:0;font-size:11px;letter-spacing:-.01em}.section-title span{font-size:5.5px;color:#637f8f}
.grid{display:grid;gap:7px}
.grid.top{grid-template-columns:1.25fr .75fr}
.grid.mid{grid-template-columns:1.05fr .95fr;margin-top:7px}
.panel{
  border:1px solid var(--line);border-radius:11px;background:linear-gradient(180deg,#081b26,#06151d);
  min-width:0;padding:9px;
}
.panel-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px}
.panel-head h3{margin:0;font-size:7.3px}.panel-head span{font-size:4.8px;color:#617c8b;font-weight:900;letter-spacing:.08em}
.prob-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.prob{
  padding:8px 4px;border:1px solid #173b4e;border-radius:7px;background:#071a25;text-align:center
}
.prob span{display:block;font-size:4.6px;color:#708998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prob b{display:block;margin-top:4px;font-size:13px}
.prob.home b{color:#5fe0b0}.prob.draw b{color:#dfb658}.prob.away b{color:#58b9e4}
.probbar{display:flex;height:6px;border-radius:999px;overflow:hidden;margin-top:7px;background:#102b39}
.probbar i{height:100%}.probbar .home{background:var(--green)}.probbar .draw{background:var(--gold)}.probbar .away{background:var(--blue)}
.xg{
  display:grid;grid-template-columns:1fr auto 1fr;gap:5px;align-items:end;text-align:center;padding:5px 0 1px
}
.xg span{display:block;font-size:4.8px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xg b{display:block;margin-top:4px;font-size:22px;letter-spacing:-.04em}.xg em{font-style:normal;color:#4e6c7c;padding-bottom:6px}
.edge-wrap{margin-top:7px}
.edge-row{display:grid;grid-template-columns:42px minmax(0,1fr) 31px;gap:5px;align-items:center;margin-bottom:5px}
.edge-row span{font-size:4.8px;color:#6e8796}.edge-row b{font-size:5.6px;text-align:right}
.edge-track{height:5px;background:#102b39;border-radius:999px;overflow:hidden}.edge-track i{display:block;height:100%;border-radius:999px}
.edge-track .model{background:var(--green)}.edge-track .market{background:#3c8fbd}
.edge-big{font-size:15px;font-weight:950;color:#59ddb0;margin-top:5px}
.profile{display:grid;gap:5px}
.profile-row{display:grid;grid-template-columns:72px minmax(0,1fr) 33px;gap:5px;align-items:center}
.profile-row span{font-size:5px;color:#708a99;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.profile-row div{height:5px;border-radius:999px;background:#102a39;overflow:hidden}
.profile-row i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#267d63,#53d5a7)}
.profile-row b{font-size:5.3px;text-align:right}.profile-row b.neutral{color:#d6ad54}.profile-row b.good{color:#57dcae}
.matrix{
  display:grid;grid-template-columns:16px repeat(5,1fr);gap:2px;align-items:center
}
.axis{font-size:4px;color:#607b8b;text-align:center}
.cell{
  aspect-ratio:1;border:1px solid #1b4960;border-radius:3px;display:grid;place-items:center;font-size:4.2px;font-weight:800;
  color:#eaf5fa
}
.c1{background:#0b2330}.c2{background:#0f3445}.c3{background:#14536b}.c4{background:#1c7290}.c5{background:#2b91b6}
.matrix-note{margin-top:5px;font-size:4.6px;color:#607b8b;text-align:center}
.feature{
  margin-top:7px;border:1px solid #214b61;border-radius:12px;background:
  radial-gradient(circle at 8% 0,rgba(63,168,221,.08),transparent 25%),
  linear-gradient(180deg,#081a24,#06141d);padding:10px
}
.feature-top{display:grid;grid-template-columns:1fr 116px;gap:9px;align-items:start}
.feature h3{margin:0;font-size:8px}.market-score{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.market-score div{padding:7px 4px;border:1px solid #173a4d;border-radius:7px;background:#071923;text-align:center}
.market-score span{display:block;font-size:4.4px;color:#6f8997}.market-score b{display:block;margin-top:3px;font-size:9px}
.market-score b.green{color:#56ddb0}
.distribution{margin-top:9px;display:grid;grid-template-columns:repeat(6,1fr);gap:6px;height:92px;align-items:end;border-bottom:1px solid #18384a;padding:0 5px 5px}
.dcol{height:100%;display:grid;grid-template-rows:1fr auto;gap:3px;text-align:center}
.bars{height:100%;display:flex;justify-content:center;align-items:flex-end;gap:2px}.bars i{display:block;width:37%;min-height:3px;border-radius:3px 3px 0 0}
.bars .h{background:#42afe3}.bars .a{background:#4bd7a3}.dcol b{font-size:4.8px;color:#687f8e}
.legend{display:flex;gap:10px;margin-top:6px;font-size:4.7px;color:#6d8796}
.legend span{display:flex;align-items:center;gap:3px}.legend i{width:5px;height:5px;border-radius:50%}.legend .h{background:#42afe3}.legend .a{background:#4bd7a3}
.insight{
  margin-top:7px;display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:8px;align-items:center;
  padding:8px 9px;border:1px solid #1b4b5f;border-left:3px solid var(--green);border-radius:9px;background:#071922
}
.insight-mark{width:34px;height:34px;border-radius:8px;background:#0b322a;border:1px solid #26654f;display:grid;place-items:center;font-size:8px;font-weight:950;color:#62deb4}
.insight b{display:block;font-size:7px}.insight p{margin:2px 0 0;font-size:5.1px;color:#708998;line-height:1.35}
.insight strong{font-size:8px;color:#59ddb0;white-space:nowrap}
.footer-note{padding:9px 12px 12px;color:#536f7e;font-size:4.8px;text-align:center}
@media(max-width:520px){
  .prototype{padding:0 0 34px}
  .frame{border-radius:0;border-left:0;border-right:0;max-width:none}
  .hero{grid-template-columns:1fr 93px;padding:10px}
  .crest{width:55px;height:55px}.shield{width:34px;height:42px}.team b{font-size:11px;max-width:118px}
  .hero-status{gap:5px}.qrow{padding:6px}.qrow span{font-size:4.8px}.qrow b{font-size:5.4px}
  .content{padding:7px}
  .grid.top{grid-template-columns:1.28fr .72fr}.grid.mid{grid-template-columns:1fr 1fr}
  .panel{padding:7px}.prob{padding:6px 2px}.prob b{font-size:11px}.xg b{font-size:18px}
  .profile-row{grid-template-columns:57px minmax(0,1fr) 29px}.profile-row span{font-size:4.6px}
  .feature-top{grid-template-columns:1fr 100px}.distribution{height:78px;gap:4px}
}
</style>
</head>
<body>
<div class="prototype">
  <div class="frame">
    <header class="topbar">
      <div class="brand"><div class="brand-mark">SE</div><b>SOCCER<br>EDGE</b></div>
      <div class="proto-badge">VISUAL GOLDEN MASTER · SAMPLE DATA</div>
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
      <aside class="hero-status">
        <div class="qrow"><span>Data Quality</span><b class="good">A</b></div>
        <div class="qrow"><span>Lineup</span><b class="good">Confirmed</b></div>
        <div class="qrow"><span>Market</span><b class="info">Fresh</b></div>
      </aside>
    </section>

    <nav class="tabs">
      <button class="active">Overview</button><button>Goals</button><button>Corners</button><button>Cards</button><button>Players</button><button>Market</button><button>Model</button>
    </nav>

    <main class="content">
      <div class="section-title"><h2>Match Center</h2><span>Why does this matchup matter?</span></div>

      <section class="grid top">
        <article class="panel">
          <div class="panel-head"><h3>Match Result Probability</h3><span>MODEL</span></div>
          <div class="prob-cards">
            <div class="prob home"><span>HOME</span><b>67.1%</b></div>
            <div class="prob draw"><span>DRAW</span><b>20.3%</b></div>
            <div class="prob away"><span>AWAY</span><b>12.6%</b></div>
          </div>
          <div class="probbar"><i class="home" style="width:67.1%"></i><i class="draw" style="width:20.3%"></i><i class="away" style="width:12.6%"></i></div>
          <div class="edge-wrap">
            <div class="edge-row"><span>Model</span><div class="edge-track"><i class="model" style="width:100%"></i></div><b>71.8%</b></div>
            <div class="edge-row"><span>Market</span><div class="edge-track"><i class="market" style="width:84%"></i></div><b>60.1%</b></div>
            <div class="edge-big">+11.7 pp</div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-head"><h3>Expected Goals (λ)</h3><span>SPORT</span></div>
          <div class="xg"><div><span>Arsenal</span><b>2.08</b></div><em>—</em><div><span>Brighton</span><b>0.91</b></div></div>
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
          <div class="matrix-note">Most likely exact score: <b>2–1 · 11.4%</b></div>
        </article>
      </section>

      <section class="feature">
        <div class="feature-top">
          <div>
            <div class="panel-head"><h3>Over / Under 2.5 Goals</h3><span>MODEL VS MARKET</span></div>
            <div class="market-score">
              <div><span>MODEL</span><b>64.4%</b></div>
              <div><span>MARKET</span><b>55.3%</b></div>
              <div><span>EDGE</span><b class="green">+8.9 pp</b></div>
            </div>
          </div>
          <div>
            <div class="panel-head"><h3>Goal Distribution</h3><span>MODEL</span></div>
            <div class="legend"><span><i class="h"></i>Arsenal</span><span><i class="a"></i>Brighton</span></div>
          </div>
        </div>

        <div class="distribution">
          <div class="dcol"><div class="bars"><i class="h" style="height:31%"></i><i class="a" style="height:52%"></i></div><b>0</b></div>
          <div class="dcol"><div class="bars"><i class="h" style="height:72%"></i><i class="a" style="height:88%"></i></div><b>1</b></div>
          <div class="dcol"><div class="bars"><i class="h" style="height:100%"></i><i class="a" style="height:58%"></i></div><b>2</b></div>
          <div class="dcol"><div class="bars"><i class="h" style="height:76%"></i><i class="a" style="height:27%"></i></div><b>3</b></div>
          <div class="dcol"><div class="bars"><i class="h" style="height:48%"></i><i class="a" style="height:11%"></i></div><b>4</b></div>
          <div class="dcol"><div class="bars"><i class="h" style="height:26%"></i><i class="a" style="height:5%"></i></div><b>5+</b></div>
        </div>

        <div class="insight">
          <div class="insight-mark">SE</div>
          <div><b>Primary read</b><p>Arsenal’s sporting projection leads. Market value is evaluated only after the raw football case is established.</p></div>
          <strong>+11.7 pp</strong>
        </div>
      </section>
    </main>

    <div class="footer-note">DESIGN LAB ONLY · illustrative values from the approved visual reference · not live match data</div>
  </div>
</div>
</body>
</html>"""


async def design_match_center(request: Request) -> HTMLResponse:
    return HTMLResponse(_html(), headers={"Cache-Control": "no-store"})
