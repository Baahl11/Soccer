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
  font-family:"Arial Narrow","Roboto Condensed",Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
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
  height:48px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid rgba(45,93,117,.34);
  background:rgba(4,15,23,.76);backdrop-filter:blur(16px) saturate(125%);-webkit-backdrop-filter:blur(16px) saturate(125%);
  box-shadow:inset 0 -1px 0 rgba(255,255,255,.015)
}
.back{font-size:7px;color:#87a0ad;font-weight:800}
.live-wrap{display:flex;align-items:center;gap:9px;color:#718a98;font-size:6px}
.live{display:flex;align-items:center;gap:4px;color:#57dcae;font-weight:900}
.live i{width:6px;height:6px;border-radius:50%;background:#50d6a6;box-shadow:0 0 9px #50d6a6}
.hero{
  display:grid;grid-template-columns:minmax(0,1fr) 145px;gap:12px;padding:13px 14px 11px;border-bottom:1px solid rgba(45,93,117,.34);
  background:
    radial-gradient(circle at 22% 0,rgba(57,156,198,.12),transparent 30%),
    radial-gradient(circle at 79% 18%,rgba(53,201,162,.055),transparent 20%),
    linear-gradient(180deg,rgba(8,28,39,.92),rgba(5,18,26,.96));
  position:relative;overflow:hidden
}
.hero::after{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(120deg,transparent 0%,rgba(255,255,255,.018) 36%,transparent 52%)
}
.hero-left{min-width:0}
.hero-meta{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;color:#7992a1;font-size:6px;letter-spacing:.01em}
.hero-meta b{color:#a1b3bd}
.hero-kicker{display:flex;align-items:center;gap:6px;margin:-2px 0 7px}
.hero-kicker span{padding:3px 5px;border:1px solid #265a72;border-radius:999px;background:#0a2635;color:#72c9ee;font-size:4.5px;font-weight:900;letter-spacing:.08em}
.hero-kicker b{font-size:5.2px;color:#6d8796;font-weight:800}
.faceoff{display:grid;grid-template-columns:1fr 30px 1fr;gap:8px;align-items:center}
.team{display:grid;justify-items:center;gap:5px;text-align:center;min-width:0}
.crest{
  width:60px;height:60px;border-radius:50%;display:grid;place-items:center;
  background:
    radial-gradient(circle at 35% 25%,rgba(255,255,255,.09),transparent 24%),
    linear-gradient(180deg,#12384d,#091e2a);
  border:1px solid rgba(62,127,158,.66);
  box-shadow:0 12px 26px rgba(0,0,0,.46),0 0 0 5px rgba(38,104,134,.08);
  position:relative
}
.crest::after{content:'';position:absolute;inset:4px;border:1px solid #ffffff0d;border-radius:50%}
.shield{
  width:37px;height:45px;clip-path:polygon(50% 0,92% 12%,85% 74%,50% 100%,15% 74%,8% 12%);
  display:grid;place-items:center;font-size:9px;font-weight:950;border:1px solid #ffffff2a;color:white;
  text-shadow:0 1px 2px #0008
}
.ars{background:linear-gradient(180deg,#e13a4b,#9b1424)}
.bha{background:linear-gradient(180deg,#2e78d3,#174791)}
.team b{font-size:13px;line-height:1.02;max-width:145px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.025em;font-weight:900}
.team span{font-size:5.7px;color:#6a8391}
.vs{text-align:center;font-size:9px;color:#557383;font-weight:950}
.quality{
  border:1px solid rgba(49,103,130,.42);border-radius:9px;
  background:linear-gradient(180deg,rgba(9,31,42,.72),rgba(6,22,30,.78));
  backdrop-filter:blur(12px) saturate(118%);-webkit-backdrop-filter:blur(12px) saturate(118%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 10px 24px rgba(0,0,0,.14);
  align-self:start;overflow:hidden
}
.qrow{display:flex;justify-content:space-between;gap:7px;align-items:center;padding:7px 8px;border-bottom:1px solid #133548}
.qrow:last-child{border-bottom:0}.qrow span{font-size:5.5px;color:#6e8796}.qrow b{font-size:6px}
.good{color:#56ddb0}.info{color:#5bb8e3}.warn{color:#dfb657}
.tabs{display:flex;gap:4px;padding:7px 10px;border-bottom:1px solid var(--line);
background:rgba(6,20,29,.8);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);overflow:auto}
.tabs button{border:0;background:transparent;color:#758f9e;padding:6px 10px;border-radius:6px;font-size:6.5px;font-weight:900}
.tabs button.active{background:#0d3043;color:white;border:1px solid #245b76}
.content{padding:9px;position:relative}
.content::before{content:'';position:absolute;left:0;right:0;top:0;height:48%;pointer-events:none;
background:radial-gradient(circle at 50% 0,rgba(52,150,190,.035),transparent 52%)}
.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin:1px 1px 8px}
.section-title h1{margin:0;font-size:13px;letter-spacing:-.035em;font-weight:900}.section-title p{margin:2px 0 0;color:#6c8594;font-size:5.5px}
.section-title span{font-size:5.2px;color:#5f7988;white-space:nowrap}
.grid{display:grid;gap:7px}
.grid.top{grid-template-columns:1.1fr .72fr .78fr}
.grid.mid{grid-template-columns:.9fr .92fr 1.18fr;margin-top:7px}
.intelligence-deck{
  display:grid;border:1px solid rgba(55,112,140,.38);border-radius:12px;overflow:hidden;
  background:
    radial-gradient(circle at 13% 0,rgba(67,170,216,.055),transparent 28%),
    linear-gradient(180deg,rgba(9,28,39,.84),rgba(5,19,27,.9));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.028),0 14px 34px rgba(0,0,0,.18)
}
.top-deck{grid-template-columns:1.14fr .72fr .82fr}
.mid-deck{grid-template-columns:.92fr .94fr 1.14fr;margin-top:7px}
.module{
  min-width:0;padding:9px;position:relative;background:transparent
}
.module+.module{border-left:1px solid rgba(38,82,104,.5)}
.module::before{
  content:'';position:absolute;left:9px;right:9px;top:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(125,204,234,.14),transparent)
}
.top-deck .probability-card{background:linear-gradient(180deg,rgba(13,39,52,.34),transparent)}
.top-deck .edge-card{background:linear-gradient(180deg,rgba(9,42,35,.16),transparent)}
.mid-deck .matrix-card{background:linear-gradient(180deg,rgba(11,37,50,.24),transparent)}
.chart-svg{width:100%;display:block;overflow:visible}
.edge-svg{height:58px;margin-top:2px}
.edge-axis{stroke:#173748;stroke-width:1}
.edge-model{stroke:#4bd8a3;stroke-width:5;stroke-linecap:round}
.edge-market{stroke:#3d98c6;stroke-width:5;stroke-linecap:round}
.edge-marker.model{fill:#4bd8a3}.edge-marker.market{fill:#3d98c6}
.edge-label{fill:#6f8998;font-size:5px;font-weight:800}
.edge-number{fill:#e9f4f8;font-size:5px;font-weight:900;text-anchor:end}
.dist-svg{height:92px;margin-top:5px}
.dist-grid{stroke:#173748;stroke-width:.8}
.dist-home{fill:url(#homeBar)}.dist-away{fill:url(#awayBar)}
.dist-label{fill:#687f8e;font-size:5px;text-anchor:middle}
.panel{
  min-width:0;border:1px solid rgba(55,112,140,.34);border-radius:var(--radius);padding:8px;
  background:
    linear-gradient(180deg,rgba(12,34,46,.78) 0%,rgba(6,21,29,.84) 100%);
  backdrop-filter:blur(12px) saturate(118%);
  -webkit-backdrop-filter:blur(12px) saturate(118%);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.035),
    inset 0 -1px 0 rgba(0,0,0,.18),
    0 10px 26px rgba(0,0,0,.18);
  position:relative;overflow:hidden
}
.panel::before{
  content:'';position:absolute;left:10px;right:10px;top:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(121,198,230,.18),transparent);pointer-events:none
}
.panel-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px}
.panel-head h3{margin:0;font-size:7.5px;letter-spacing:-.015em;font-weight:900}.panel-head span{font-size:4.7px;color:#617b8b;font-weight:900;letter-spacing:.08em}
.prob-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.prob{padding:7px 3px;text-align:center;border:1px solid rgba(45,94,118,.42);border-radius:6px;
background:linear-gradient(180deg,rgba(10,31,42,.66),rgba(7,24,33,.78));
box-shadow:inset 0 1px 0 rgba(255,255,255,.02)}
.prob span{display:block;color:#6e8797;font-size:4.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prob b{display:block;margin-top:3px;font-size:13px;letter-spacing:-.03em}.prob.home b{color:#59deb0}.prob.draw b{color:#ddb455}.prob.away b{color:#58b8e3}
.probbar{height:6px;display:flex;margin-top:7px;border-radius:999px;overflow:hidden;background:#102a39;box-shadow:inset 0 1px 2px #0007}
.probbar i{height:100%}.probbar .home{background:var(--green)}.probbar .draw{background:var(--gold)}.probbar .away{background:var(--blue)}
.xg{display:grid;grid-template-columns:1fr auto 1fr;gap:5px;align-items:end;text-align:center;padding:9px 0 3px}
.xg span{display:block;color:#6c8594;font-size:4.7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xg b{display:block;margin-top:4px;font-size:21px;letter-spacing:-.04em}.xg em{font-style:normal;color:#4e6a79;padding-bottom:5px}
.edge-bars{display:grid;gap:5px}
.edge-row{display:grid;grid-template-columns:34px minmax(0,1fr) 29px;gap:4px;align-items:center}
.edge-row span{font-size:4.6px;color:#6c8594}.edge-row b{font-size:5.2px;text-align:right}
.edge-track{height:5px;border-radius:999px;background:#102b39;overflow:hidden}.edge-track i{display:block;height:100%;border-radius:999px}
.edge-track .model{background:var(--green)}.edge-track .market{background:#3c8fbd}
.edge-value{margin-top:4px;font-size:17px;font-weight:950;color:#59ddb0;letter-spacing:-.035em;text-shadow:0 0 14px rgba(89,221,176,.12)}
.edge-caption{margin-top:2px;font-size:4.6px;color:#607a89}
.profile{display:grid;gap:5px}
.profile-row{display:grid;grid-template-columns:66px minmax(0,1fr) 32px;gap:5px;align-items:center}
.profile-row span{font-size:4.7px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.profile-row div{height:5px;background:#102a39;border-radius:999px;overflow:hidden}.profile-row i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#247b61,#53d5a7)}
.profile-row b{font-size:4.9px;text-align:right}.profile-row b.good{color:#57dcae}.profile-row b.neutral{color:#d8ae52}
.matrix{display:grid;grid-template-columns:14px repeat(5,1fr);gap:3px;align-items:center}
.axis{font-size:3.9px;color:#617b8a;text-align:center}
.cell{aspect-ratio:1;border:1px solid rgba(44,105,133,.55);border-radius:3px;display:grid;place-items:center;font-size:4px;font-weight:850;color:#edf8fc;
box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 1px 3px rgba(0,0,0,.18)}
.c1{background:linear-gradient(180deg,#0b2634,#091f2b)}
.c2{background:linear-gradient(180deg,#123e52,#0f3446)}
.c3{background:linear-gradient(180deg,#185f79,#145068)}
.c4{background:linear-gradient(180deg,#2380a0,#1b708d)}
.c5{background:linear-gradient(180deg,#39a7cf,#2a90b5);box-shadow:0 0 14px rgba(57,167,207,.16)}
.matrix-note{margin-top:5px;color:#637d8c;font-size:4.4px;text-align:center}.matrix-note b{color:#cfe4ee}
.market-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.market-metric{padding:6px 3px;text-align:center;border:1px solid rgba(45,94,118,.4);border-radius:6px;
background:linear-gradient(180deg,rgba(9,29,40,.68),rgba(7,23,32,.8));
box-shadow:inset 0 1px 0 rgba(255,255,255,.02)}
.market-metric span{display:block;color:#6d8695;font-size:4.3px}.market-metric b{display:block;margin-top:3px;font-size:8.8px}.market-metric b.green{color:#56ddb0}
.distribution{height:72px;display:grid;grid-template-columns:repeat(6,1fr);gap:4px;align-items:end;margin-top:7px;padding:0 3px 4px;border-bottom:1px solid #173748}
.dcol{height:100%;display:grid;grid-template-rows:1fr auto;gap:2px;text-align:center}
.bars{height:100%;display:flex;align-items:flex-end;justify-content:center;gap:1px}.bars i{display:block;width:38%;min-height:2px;border-radius:2px 2px 0 0}
.bars .h{background:linear-gradient(180deg,#58b8e8,#2e96c7);box-shadow:0 0 8px rgba(66,175,227,.16)}
.bars .a{background:linear-gradient(180deg,#59dda9,#2dbb86);box-shadow:0 0 8px rgba(75,215,163,.14)}
.dcol b{font-size:4.4px;color:#687f8e}
.legend{display:flex;gap:8px;margin-top:5px;font-size:4.4px;color:#687f8e}.legend span{display:flex;align-items:center;gap:3px}.legend i{width:5px;height:5px;border-radius:50%}.legend .h{background:#42afe3}.legend .a{background:#4bd7a3}
.read{
  margin-top:7px;display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:8px;align-items:center;padding:7px 8px;
  border:1px solid rgba(48,111,137,.42);border-left:3px solid var(--green);border-radius:8px;
  background:linear-gradient(180deg,rgba(8,30,40,.72),rgba(6,23,31,.82));
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025)
}
.read-mark{width:30px;height:30px;border-radius:7px;background:#0b322a;border:1px solid #26644f;display:grid;place-items:center;color:#62deb4;font-size:7px;font-weight:950}
.read b{display:block;font-size:6.5px}.read p{margin:2px 0 0;color:#6f8796;font-size:4.9px;line-height:1.35}.read strong{font-size:7.5px;color:#59ddb0;white-space:nowrap}
.audit-line{display:flex;justify-content:space-between;gap:8px;margin-top:7px;padding:0 2px;color:#526d7b;font-size:4.4px}
.footer-note{padding:10px 12px 12px;text-align:center;color:#4f6a78;font-size:4.5px}

.visual-strip{
  display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin:0 0 7px
}
.visual-stat{
  border:1px solid rgba(46,96,121,.34);border-radius:7px;padding:6px 7px;
  background:linear-gradient(180deg,rgba(9,28,38,.72),rgba(6,21,29,.82));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.018)
}
.visual-stat span{display:block;color:#617b8a;font-size:4.4px;font-weight:850;letter-spacing:.08em}
.visual-stat b{display:block;margin-top:3px;font-size:8px}
.spark{
  width:100%;height:34px;margin-top:5px;overflow:visible
}
.spark-grid{stroke:#183747;stroke-width:.7}
.spark-line{fill:none;stroke:#4bd8a3;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px rgba(75,216,163,.18))}
.spark-dot{fill:#4bd8a3;stroke:#06131b;stroke-width:1.1}
.panel-trend{display:grid;grid-template-columns:1fr 78px;gap:7px;align-items:end}
.microcopy{font-size:4.5px;color:#5f7988;line-height:1.35}
@media(max-width:740px){
  .shell{display:block;border:0;max-width:none}.rail{display:none}.topbar{height:44px;padding:0 10px}
  .hero{grid-template-columns:minmax(0,1fr) 106px;padding:10px}.crest{width:52px;height:52px}.shield{width:32px;height:39px}.team b{font-size:10.5px;max-width:118px}
  .quality{border-radius:7px}.qrow{padding:6px}.qrow span{font-size:4.8px}.qrow b{font-size:5.2px}
  .tabs{padding:6px 7px}.tabs button{padding:6px 8px;font-size:6px}
  .content{padding:7px}.grid.top{grid-template-columns:1.15fr .72fr .86fr}.grid.mid{grid-template-columns:.92fr .93fr 1.15fr}
  .visual-strip{grid-template-columns:repeat(4,1fr)}.visual-stat{padding:5px}.visual-stat b{font-size:7px}
  .panel{padding:7px}.panel-head h3{font-size:6.5px}.prob{padding:6px 2px}.prob b{font-size:10px}.xg b{font-size:17px}
  .profile-row{grid-template-columns:54px minmax(0,1fr) 29px}.profile-row span{font-size:4.25px}.profile-row b{font-size:4.3px}
  .distribution{height:63px}
}
@media(max-width:430px){
  .topbar .back{font-size:6px}.live-wrap{font-size:5.3px}.hero{grid-template-columns:minmax(0,1fr) 96px;gap:7px;padding:9px 8px 8px}
  .crest{width:48px;height:48px}.team b{font-size:9.5px;max-width:105px}.hero-meta{font-size:5px}.qrow{padding:5px}
  .content{padding:6px}.section-title h1{font-size:10.5px}.section-title p{font-size:4.8px}
  .visual-strip{gap:4px}.visual-stat span{font-size:4px}.visual-stat b{font-size:6.5px}
  .grid{gap:5px}.grid.top{grid-template-columns:1.18fr .82fr}.grid.top .edge-card{grid-column:1/-1}
  .grid.mid{grid-template-columns:1fr 1fr}.grid.mid .goals-card{grid-column:1/-1}
  .top-deck{grid-template-columns:1.12fr .88fr}.top-deck .edge-card{grid-column:1/-1;border-left:0;border-top:1px solid rgba(38,82,104,.5)}
  .mid-deck{grid-template-columns:1fr 1fr}.mid-deck .goals-card{grid-column:1/-1;border-left:0;border-top:1px solid rgba(38,82,104,.5)}
  .module{padding:7px}.module+.module{border-left:1px solid rgba(38,82,104,.5)}
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
          <div class="hero-kicker"><span>PRE-MATCH</span><b>Match Intelligence</b></div>
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

        <div class="visual-strip">
          <div class="visual-stat"><span>DATA QUALITY</span><b class="good">A</b></div>
          <div class="visual-stat"><span>CONFIDENCE</span><b>84</b></div>
          <div class="visual-stat"><span>XI</span><b class="good">CONFIRMED</b></div>
          <div class="visual-stat"><span>MARKET</span><b class="info">FRESH</b></div>
        </div>

        <section class="intelligence-deck top-deck">
          <article class="module probability-card">
            <div class="panel-head"><h3>Match Result Probability</h3><span>MODEL</span></div>
            <div class="prob-cards">
              <div class="prob home"><span>HOME</span><b>67.1%</b></div>
              <div class="prob draw"><span>DRAW</span><b>20.3%</b></div>
              <div class="prob away"><span>AWAY</span><b>12.6%</b></div>
            </div>
            <div class="probbar"><i class="home" style="width:67.1%"></i><i class="draw" style="width:20.3%"></i><i class="away" style="width:12.6%"></i></div>
          </article>

          <article class="module xg-card">
            <div class="panel-head"><h3>Expected Goals (λ)</h3><span>SPORT</span></div>
            <div class="panel-trend">
              <div class="xg"><div><span>Arsenal</span><b>2.08</b></div><em>—</em><div><span>Brighton</span><b>0.91</b></div></div>
              <svg class="spark" viewBox="0 0 78 34" aria-label="Recent xG trend">
                <line x1="2" y1="28" x2="76" y2="28" class="spark-grid"/>
                <line x1="2" y1="17" x2="76" y2="17" class="spark-grid"/>
                <polyline points="4,25 18,20 31,22 45,13 60,15 74,7" class="spark-line"/>
                <circle cx="74" cy="7" r="2.4" class="spark-dot"/>
              </svg>
            </div>
            <div class="microcopy">Recent attacking trend · visual reference</div>
          </article>

          <article class="module edge-card">
            <div class="panel-head"><h3>Edge Gap</h3><span>EXAMPLE MARKET</span></div>
            <svg class="chart-svg edge-svg" viewBox="0 0 230 58" aria-label="Model versus market">
              <line x1="54" y1="18" x2="202" y2="18" class="edge-axis"/>
              <line x1="54" y1="40" x2="202" y2="40" class="edge-axis"/>
              <text x="2" y="20" class="edge-label">MODEL</text>
              <text x="2" y="42" class="edge-label">MARKET</text>
              <line x1="54" y1="18" x2="190" y2="18" class="edge-model"/>
              <line x1="54" y1="40" x2="167" y2="40" class="edge-market"/>
              <circle cx="190" cy="18" r="4" class="edge-marker model"/>
              <circle cx="167" cy="40" r="4" class="edge-marker market"/>
              <text x="226" y="20" class="edge-number">71.8%</text>
              <text x="226" y="42" class="edge-number">60.1%</text>
            </svg>
            <div class="edge-value">+11.7 pp</div><div class="edge-caption">model − market fair</div>
          </article>
        </section>

        <section class="intelligence-deck mid-deck">
          <article class="module profile-card">
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

          <article class="module matrix-card">
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

          <article class="module goals-card">
            <div class="panel-head"><h3>Over / Under 2.5 Goals</h3><span>MODEL VS MARKET</span></div>
            <div class="market-grid">
              <div class="market-metric"><span>MODEL</span><b>64.4%</b></div>
              <div class="market-metric"><span>MARKET</span><b>55.3%</b></div>
              <div class="market-metric"><span>EDGE</span><b class="green">+8.9 pp</b></div>
            </div>
            <svg class="chart-svg dist-svg" viewBox="0 0 300 92" aria-label="Goal distribution">
              <defs>
                <linearGradient id="homeBar" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#58b8e8"/><stop offset="100%" stop-color="#2e96c7"/></linearGradient>
                <linearGradient id="awayBar" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#59dda9"/><stop offset="100%" stop-color="#2dbb86"/></linearGradient>
              </defs>
              <line x1="18" y1="72" x2="292" y2="72" class="dist-grid"/>
              <g transform="translate(28,0)"><rect x="0" y="51" width="10" height="21" rx="2" class="dist-home"/><rect x="12" y="38" width="10" height="34" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">0</text></g>
              <g transform="translate(70,0)"><rect x="0" y="28" width="10" height="44" rx="2" class="dist-home"/><rect x="12" y="15" width="10" height="57" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">1</text></g>
              <g transform="translate(112,0)"><rect x="0" y="8" width="10" height="64" rx="2" class="dist-home"/><rect x="12" y="35" width="10" height="37" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">2</text></g>
              <g transform="translate(154,0)"><rect x="0" y="24" width="10" height="48" rx="2" class="dist-home"/><rect x="12" y="54" width="10" height="18" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">3</text></g>
              <g transform="translate(196,0)"><rect x="0" y="42" width="10" height="30" rx="2" class="dist-home"/><rect x="12" y="65" width="10" height="7" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">4</text></g>
              <g transform="translate(238,0)"><rect x="0" y="56" width="10" height="16" rx="2" class="dist-home"/><rect x="12" y="69" width="10" height="3" rx="2" class="dist-away"/><text x="11" y="86" class="dist-label">5+</text></g>
            </svg>
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
