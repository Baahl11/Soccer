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
  --blue:#4a9fbe;
  --green:#63c4ad;
  --gold:#c8a45a;
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
  display:grid;grid-template-columns:minmax(0,1fr) 146px;gap:12px;padding:12px 14px 10px;border-bottom:1px solid rgba(45,93,117,.28);
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
.hero-meta{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;color:#849aa6;font-size:6.1px;letter-spacing:.015em}
.hero-meta b{color:#a1b3bd}
.hero-kicker{display:flex;align-items:center;gap:6px;margin:-1px 0 6px}
.hero-kicker span{padding:3px 5px;border:1px solid #265a72;border-radius:999px;background:#0a2635;color:#72c9ee;font-size:4.5px;font-weight:900;letter-spacing:.08em}
.hero-kicker b{font-size:5.2px;color:#6d8796;font-weight:800}
.faceoff{display:grid;grid-template-columns:1fr 42px 1fr;gap:10px;align-items:center}
.team{display:grid;justify-items:center;gap:5px;text-align:center;min-width:0}
.crest{
  width:56px;height:56px;border-radius:50%;display:grid;place-items:center;
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
.team b{font-size:12.5px;line-height:1.02;max-width:145px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.022em;font-weight:900}
.team span{font-size:5.4px;color:#6a8391;text-transform:uppercase;letter-spacing:.08em}
.vs{
  text-align:center;font-size:8px;color:#8aa1ad;font-weight:950;letter-spacing:.08em;position:relative
}
.vs::before,.vs::after{content:'';position:absolute;left:50%;transform:translateX(-50%);width:26px;height:1px;background:linear-gradient(90deg,transparent,#214a5e,transparent)}
.vs::before{top:-8px}.vs::after{bottom:-8px}
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
.top-deck{grid-template-columns:1.22fr .74fr .88fr}
.mid-deck{grid-template-columns:.88fr 1.02fr 1.10fr;margin-top:8px}
.module{
  min-width:0;padding:10px;position:relative;background:transparent
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
.edge-body{display:grid;grid-template-columns:minmax(0,1fr) 118px;gap:12px;align-items:stretch}
.edge-visual{min-width:0;display:grid;align-items:center}
.edge-svg{height:84px;margin-top:0}
.edge-base{stroke:#163747;stroke-width:1.4;stroke-linecap:round}
.edge-tick{stroke:#214a5c;stroke-width:1}
.edge-gap-glow{stroke:#e5bd5b;stroke-width:13;stroke-linecap:round;opacity:.18}
.edge-gap-focus{stroke:#f2c968;stroke-width:6.5;stroke-linecap:round}
.edge-gap-core{stroke:#fff0b2;stroke-width:2.2;stroke-linecap:round;opacity:.98}
.edge-market-dot{fill:#4a9fbe;stroke:#071923;stroke-width:2.2}
.edge-model-dot{fill:#63c4ad;stroke:#071923;stroke-width:2.2}
.edge-scale{fill:#536f7e;font-size:5px;font-weight:800;text-anchor:middle}
.edge-point-label{font-size:5px;font-weight:900;text-anchor:middle}
.edge-point-label.market{fill:#7db5c9}.edge-point-label.model{fill:#83c9b8}
.edge-gap-pill{fill:#2b2412;stroke:#c59b3f;stroke-width:1.1}
.edge-gap-pill-value{fill:#ffe08a;font-size:8.8px;font-weight:950;text-anchor:middle;letter-spacing:-.03em}
.edge-gap-pill-label{fill:#b99a55;font-size:3.7px;font-weight:900;text-anchor:middle;letter-spacing:.12em}
.edge-side-stats{
  display:grid;grid-template-rows:1fr 1fr;border-left:1px solid rgba(42,88,109,.42);
  background:linear-gradient(180deg,rgba(8,28,38,.32),rgba(5,20,28,.16))
}
.edge-side-stat{
  min-height:0;padding:8px 9px;display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:7px;align-content:center
}
.edge-side-stat+.edge-side-stat{border-top:1px solid rgba(42,88,109,.42)}
.edge-side-stat .k{font-size:4.3px;color:#607986;font-weight:900;letter-spacing:.09em}
.edge-side-stat .v{font-size:13px;color:#f1f7fa;font-weight:950;letter-spacing:-.035em;text-align:right;font-variant-numeric:tabular-nums}
.edge-side-stat .sub{grid-column:1/-1;margin-top:2px;font-size:4.2px;color:#617b89}
.edge-side-stat .sub b{color:#83a7b8;font-weight:900}
.edge-side-stat.conf .v{color:#61ddb1}
.edge-side-stat.conf .sub b{color:#61ddb1}
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
.prob-cards{
  display:grid;grid-template-columns:1.18fr .91fr .91fr;gap:0;border:1px solid rgba(45,94,118,.32);
  border-radius:7px;overflow:hidden;background:rgba(6,22,31,.46)
}
.prob{
  padding:8px 5px 7px;text-align:center;border:0;border-radius:0;
  background:linear-gradient(180deg,rgba(10,31,42,.30),rgba(7,24,33,.42));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.012)
}
.prob+.prob{border-left:1px solid rgba(45,94,118,.30)}
.prob.home{
  background:linear-gradient(180deg,rgba(60,145,126,.18),rgba(9,31,38,.54));
  box-shadow:inset 0 2px 0 rgba(99,196,173,.34),0 8px 18px rgba(0,0,0,.08)
}
.prob span{display:block;color:#6e8797;font-size:4.4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:850;letter-spacing:.05em}
.prob b{display:block;margin-top:3px;font-size:12.5px;letter-spacing:-.035em;font-variant-numeric:tabular-nums}
.prob.home b{font-size:15px;color:#73cbb8}.prob.draw b{color:#b9b29d}.prob.away b{color:#76afc5}
.prob-summary{
  display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:7px;
  padding-top:6px;border-top:1px solid rgba(42,86,108,.36);color:#657f8d
}
.prob-summary .lean{display:flex;align-items:center;gap:5px;font-size:4.7px;font-weight:850;letter-spacing:.05em}
.prob-summary .lean i{width:6px;height:6px;border-radius:50%;background:#73cbb8;box-shadow:0 0 7px rgba(115,203,184,.18)}
.prob-summary strong{font-size:5px;color:#a7bac4;letter-spacing:.03em}
.xg{display:grid;grid-template-columns:1fr auto 1fr;gap:5px;align-items:end;text-align:center;padding:9px 0 3px}
.xg span{display:block;color:#6c8594;font-size:4.7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xg b{display:block;margin-top:4px;font-size:23px;letter-spacing:-.055em;font-variant-numeric:tabular-nums}.xg em{font-style:normal;color:#4e6a79;padding-bottom:5px}
.edge-bars{display:grid;gap:5px}
.edge-row{display:grid;grid-template-columns:34px minmax(0,1fr) 29px;gap:4px;align-items:center}
.edge-row span{font-size:4.6px;color:#6c8594}.edge-row b{font-size:5.2px;text-align:right}
.edge-track{height:5px;border-radius:999px;background:#102b39;overflow:hidden}.edge-track i{display:block;height:100%;border-radius:999px}
.edge-track .model{background:var(--green)}.edge-track .market{background:#3c8fbd}
.edge-value{margin-top:4px;font-size:17px;font-weight:950;color:#59ddb0;letter-spacing:-.035em;text-shadow:0 0 14px rgba(89,221,176,.12)}
.edge-caption{margin-top:2px;font-size:4.6px;color:#607a89}
.profile{display:grid;gap:5px}
.profile-card{background:
  radial-gradient(circle at 15% 0,rgba(75,216,163,.035),transparent 35%),
  linear-gradient(180deg,rgba(7,25,34,.08),transparent)}
.profile-row{display:grid;grid-template-columns:66px minmax(0,1fr) 32px;gap:5px;align-items:center}
.profile-row span{font-size:4.7px;color:#6f8998;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.profile-row div{height:5px;background:#102633;border-radius:999px;overflow:hidden}.profile-row i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#356f7a,#77aeb2)}
.profile-row b{font-size:4.9px;text-align:right}.profile-row b.good{color:#77b9b1}.profile-row b.neutral{color:#b8a46f}
.matrix{display:grid;grid-template-columns:14px repeat(5,1fr);gap:3px;align-items:center}
.axis{font-size:4.5px;color:#6f8794;text-align:center;font-weight:850}
.cell{aspect-ratio:1;border:1px solid rgba(44,105,133,.55);border-radius:3px;display:grid;place-items:center;font-size:4.6px;font-weight:900;color:#edf8fc;
box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 1px 3px rgba(0,0,0,.18)}
.c1{background:linear-gradient(180deg,#0a202a,#081923)}
.c2{background:linear-gradient(180deg,#123746,#0e2c38)}
.c3{background:linear-gradient(180deg,#195367,#144556)}
.c4{background:linear-gradient(180deg,#24758b,#1c6175)}
.c5{background:linear-gradient(180deg,#3599ae,#2b8194);box-shadow:0 0 12px rgba(53,153,174,.16),inset 0 0 0 1px rgba(220,245,249,.09)}
.cell.hot{outline:1px solid rgba(207,173,92,.82);outline-offset:1px;box-shadow:0 0 14px rgba(200,164,90,.14),inset 0 0 0 1px rgba(255,255,255,.07)}
.matrix-note{margin-top:6px;color:#637d8c;font-size:4.6px;text-align:center}.matrix-note b{color:#cfe4ee}
.matrix-layout{display:block}
.scoring-profiles{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.team-profile{
  min-width:0;border:1px solid rgba(55,112,140,.30);border-radius:7px;padding:6px;
  background:linear-gradient(180deg,rgba(8,28,38,.48),rgba(5,19,27,.58))
}
.team-profile-head{display:flex;align-items:baseline;justify-content:space-between;gap:4px;margin-bottom:4px}
.team-profile-head b{font-size:5.4px;color:#e8f3f7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.team-profile-head span{font-size:3.8px;color:#607b89;font-weight:900;letter-spacing:.07em}
.mini-matrix{display:grid;grid-template-columns:9px repeat(5,minmax(0,1fr));gap:2px;align-items:center}
.mini-axis{font-size:3.3px;color:#68818e;text-align:center;font-weight:850}
.mini-cell{
  aspect-ratio:1;border:1px solid rgba(255,255,255,.055);border-radius:2px;display:grid;place-items:center;
  font-size:3.5px;font-weight:900;color:#eef8fb
}
.home-p1{background:#102929}.home-p2{background:#17443f}.home-p3{background:#1f6055}.home-p4{background:#2a7c69}.home-p5{background:#42a087;box-shadow:0 0 10px rgba(66,160,135,.12)}
.away-p1{background:#101e2b}.away-p2{background:#173249}.away-p3{background:#214a65}.away-p4{background:#2c6482}.away-p5{background:#3d83a1;box-shadow:0 0 10px rgba(61,131,161,.12)}
.profile-caption{margin-top:4px;text-align:center;font-size:3.5px;color:#587381;letter-spacing:.04em}
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
.legend{display:flex;gap:8px;margin-top:5px;font-size:4.4px;color:#687f8e}.legend span{display:flex;align-items:center;gap:3px}.legend i{width:5px;height:5px;border-radius:50%}.legend .h{background:#68b3cb}.legend .a{background:#82cbb8}
.read{
  margin-top:7px;display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:8px;align-items:center;padding:7px 8px;
  border:1px solid rgba(48,111,137,.42);border-left:3px solid var(--green);border-radius:8px;
  background:linear-gradient(180deg,rgba(8,30,40,.72),rgba(6,23,31,.82));
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025)
}
.read-mark{width:30px;height:30px;border-radius:7px;background:#0b322a;border:1px solid #26644f;display:grid;place-items:center;color:#62deb4;font-size:7px;font-weight:950}
.read b{display:block;font-size:6.5px}.read-tag{display:inline-block;margin-left:5px;padding:2px 4px;border-radius:999px;background:#0b3028;border:1px solid #205845;color:#59d7aa;font-size:4px;letter-spacing:.08em;vertical-align:1px}.read p{margin:2px 0 0;color:#6f8796;font-size:4.9px;line-height:1.35}.read strong{font-size:7.5px;color:#59ddb0;white-space:nowrap}
.audit-line{display:flex;justify-content:space-between;gap:8px;margin-top:7px;padding:0 2px;color:#526d7b;font-size:4.4px}
.footer-note{padding:10px 12px 12px;text-align:center;color:#4f6a78;font-size:4.5px}

.visual-strip{
  display:grid;grid-template-columns:repeat(4,1fr);margin:0 0 8px;
  border:1px solid rgba(44,92,116,.30);border-radius:8px;overflow:hidden;
  background:rgba(6,21,30,.58);box-shadow:inset 0 1px 0 rgba(255,255,255,.014)
}
.visual-stat{
  padding:6px 8px;background:transparent;min-width:0
}
.visual-stat+.visual-stat{border-left:1px solid rgba(39,82,103,.42)}
.visual-stat span{display:block;color:#607987;font-size:4.2px;font-weight:900;letter-spacing:.09em}
.visual-stat b{display:block;margin-top:2px;font-size:7.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.spark{
  width:100%;height:34px;margin-top:5px;overflow:visible
}
.spark-grid{stroke:#183747;stroke-width:.7}
.spark-line{fill:none;stroke:#4bd8a3;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px rgba(75,216,163,.18))}
.spark-dot{fill:#4bd8a3;stroke:#06131b;stroke-width:1.1}
.panel-trend{display:grid;grid-template-columns:1fr 78px;gap:7px;align-items:end}
.microcopy{font-size:4.5px;color:#5f7988;line-height:1.35}

.xg-card{background:
  radial-gradient(circle at 78% 12%,rgba(75,216,163,.055),transparent 34%),
  linear-gradient(180deg,rgba(8,27,37,.20),transparent)}
.xg-summary{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:end;margin-top:2px}
.xg-team{text-align:center;min-width:0}
.xg-team span{display:block;color:#6b8593;font-size:4.5px}
.xg-team b{display:block;margin-top:3px;font-size:22px;letter-spacing:-.055em;font-variant-numeric:tabular-nums}
.xg-sep{color:#466573;font-size:12px;padding-bottom:4px}
.xg-meta{display:grid;grid-template-columns:1fr 1fr;margin-top:5px;border-top:1px solid rgba(42,86,108,.34)}
.xg-meta div{padding:5px 4px;text-align:center}
.xg-meta div+div{border-left:1px solid rgba(42,86,108,.34)}
.xg-meta span{display:block;color:#5e7887;font-size:3.8px;font-weight:900;letter-spacing:.08em}
.xg-meta b{display:block;margin-top:2px;font-size:6.2px;color:#dcebf1}
.xg-meta b.good{color:#60deb2}
.xg-trend{height:32px;margin-top:3px}
.xg-trend .gridline{stroke:#173747;stroke-width:.7}
.xg-trend .area{fill:url(#xgArea)}
.xg-trend .line{fill:none;stroke:#63b9a7;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.xg-trend .dot{fill:#63b9a7;stroke:#06131b;stroke-width:1}
.xg-caption{margin-top:1px;color:#5e7785;font-size:3.8px;text-align:center}

.ou-read{display:grid;grid-template-columns:108px minmax(0,1fr);gap:10px;align-items:stretch}
.ou-hero{
  display:grid;align-content:center;border-right:1px solid rgba(42,86,108,.40);padding-right:10px
}
.ou-hero span{font-size:4px;color:#627c8a;font-weight:900;letter-spacing:.08em}
.ou-hero b{margin-top:3px;font-size:19px;line-height:1;color:#f3f8fb;letter-spacing:-.045em}
.ou-hero strong{margin-top:4px;font-size:6.3px;color:#58ddb0}
.ou-hero small{margin-top:2px;font-size:3.8px;color:#607a88}
.ou-gauge{position:relative;height:34px;margin:3px 0 1px}
.ou-gauge .base{position:absolute;left:8px;right:8px;top:17px;height:2px;border-radius:999px;background:#173849}
.ou-gauge .edge{position:absolute;left:55.3%;width:9.1%;top:15px;height:6px;border-radius:999px;background:#c8a45a;box-shadow:0 0 8px rgba(200,164,90,.16)}
.ou-gauge .mark{position:absolute;top:10px;width:12px;height:12px;border-radius:50%;transform:translateX(-50%);border:2px solid #071923}
.ou-gauge .market{left:55.3%;background:#4a9fbe}
.ou-gauge .model{left:64.4%;background:#63c4ad}
.ou-gauge .lab{position:absolute;top:0;transform:translateX(-50%);font-size:3.7px;font-weight:900;white-space:nowrap;background:none}
.ou-gauge .lab.market{left:55.3%;color:#7ab5ca}
.ou-gauge .lab.model{left:64.4%;color:#84cbb9}
.ou-scale{display:flex;justify-content:space-between;padding:0 8px;color:#506c7b;font-size:3.4px}
.dist-focus{fill:#0e2b37;opacity:.42}

/* Golden Master final polish */
.panel-head h3{font-size:7.7px}
.panel-head span{font-size:4.9px;color:#6b8390}
.section-title p{color:#748b98}
.axis{color:#78909d}
.mini-axis{color:#718894}
.profile-caption{color:#647d89}
.dist-label{fill:#708794}
.legend{color:#718794}
.microcopy,.xg-caption{color:#687f8b}

.matrix-layout{align-items:center}
.matrix-main{align-self:center}
.scoring-profiles{align-self:stretch}

.ou-read{
  padding:2px 0 5px;
  border-bottom:1px solid rgba(42,86,108,.26)
}
.ou-hero b{font-size:20px}
.ou-hero strong{color:#78c9b5}
.ou-gauge{height:40px;margin:1px 0 0}
.ou-gauge .base{top:24px}
.ou-gauge .edge{top:22px}
.ou-gauge .mark{top:17px}
.ou-gauge .lab{
  top:1px;padding:2px 4px;border-radius:999px;
  background:#071c26;border:1px solid rgba(55,103,126,.38);
  font-size:3.8px;letter-spacing:.03em
}
.ou-gauge .lab.market{left:51%;color:#82bdd2}
.ou-gauge .lab.model{left:69%;color:#8bd2c0}
.ou-scale{margin-top:-2px;color:#5b7481}
.dist-focus{fill:#12303b;opacity:.30}
.goals-card .dist-svg{margin-top:5px}
.read{border-color:rgba(48,111,137,.34);border-left-color:#63c4ad}
.audit-line,.footer-note{letter-spacing:.01em}
@media(max-width:740px){
  .shell{display:block;border:0;max-width:none}.rail{display:none}.topbar{height:44px;padding:0 10px}
  .hero{grid-template-columns:minmax(0,1fr) 104px;padding:9px 10px 9px}.crest{width:50px;height:50px}.shield{width:31px;height:38px}.team b{font-size:10.8px;max-width:118px}
  .quality{border-radius:7px}.qrow{padding:6px}.qrow span{font-size:4.8px}.qrow b{font-size:5.2px}
  .tabs{padding:6px 7px}.tabs button{padding:6px 8px;font-size:6px}
  .content{padding:7px}.grid.top{grid-template-columns:1.15fr .72fr .86fr}.grid.mid{grid-template-columns:.92fr .93fr 1.15fr}
  .visual-strip{grid-template-columns:repeat(4,1fr)}.visual-stat{padding:5px}.visual-stat b{font-size:7px}
  .panel{padding:7px}.panel-head h3{font-size:6.8px}.prob{padding:7px 2px}.prob b{font-size:10.5px}.prob.home b{font-size:12.5px}.prob-summary strong{font-size:4.6px}.xg b{font-size:18px}
  .profile-row{grid-template-columns:58px minmax(0,1fr) 29px}.profile-row span{font-size:4.35px}.profile-row b{font-size:4.4px}
  .distribution{height:63px}
}
@media(max-width:430px){
  .topbar .back{font-size:6.2px}.live-wrap{font-size:5.5px}.hero{grid-template-columns:minmax(0,1fr) 94px;gap:7px;padding:8px 8px 8px}
  .crest{width:47px;height:47px}.team b{font-size:9.8px;max-width:105px}.hero-meta{font-size:5.2px}.qrow{padding:5px}
  .content{padding:6px}.section-title h1{font-size:10.8px}.section-title p{font-size:5px}
  .visual-strip{grid-template-columns:repeat(4,minmax(0,1fr))}.visual-stat{padding:5px 4px}.visual-stat span{font-size:3.8px}.visual-stat b{font-size:6.2px}
  .grid{gap:5px}.grid.top{grid-template-columns:1.18fr .82fr}.grid.top .edge-card{grid-column:1/-1}
  .grid.mid{grid-template-columns:1fr 1fr}.grid.mid .goals-card{grid-column:1/-1}
  .top-deck{grid-template-columns:1.16fr .84fr}.top-deck .edge-card{grid-column:1/-1;border-left:0;border-top:1px solid rgba(38,82,104,.5)}
  .mid-deck{grid-template-columns:1fr}
  .mid-deck .module{grid-column:1/-1}
  .mid-deck .module+.module{border-left:0;border-top:1px solid rgba(38,82,104,.5)}
  .module{padding:8px}.module+.module{border-left:1px solid rgba(38,82,104,.5)}
  .edge-card .panel-head{margin-bottom:2px}
  .edge-body{grid-template-columns:minmax(0,1fr) 108px;gap:0}.edge-card .edge-svg{height:82px;margin:0}
  .edge-side-stat{padding:7px 7px}.edge-side-stat .k{font-size:4px}.edge-side-stat .v{font-size:11px}.edge-side-stat .sub{font-size:3.8px}

  .profile-card{padding-bottom:10px}
  .profile-card .panel-head{margin-bottom:8px}
  .profile{grid-template-columns:repeat(2,minmax(0,1fr));gap:0 12px}
  .profile-row{
    grid-template-columns:minmax(0,1fr) auto;gap:4px 6px;padding:7px 0;
    border-bottom:1px solid rgba(40,82,102,.32)
  }
  .profile-row:nth-last-child(-n+2){border-bottom:0}
  .profile-row span{grid-column:1;grid-row:1;font-size:4.8px;color:#78919e}
  .profile-row b{grid-column:2;grid-row:1;font-size:4.6px;align-self:center}
  .profile-row div{grid-column:1/-1;grid-row:2;height:6px}
  .profile-row i{box-shadow:0 0 8px rgba(75,216,163,.08)}

  .matrix-card{padding-bottom:10px}
  .matrix-card .panel-head{margin-bottom:8px}
  .matrix-layout{display:grid;grid-template-columns:minmax(0,1.18fr) minmax(132px,.82fr);gap:8px;align-items:center}
  .matrix-main{min-width:0;align-self:center}
  .matrix-main .matrix{grid-template-columns:13px repeat(5,minmax(0,1fr));gap:2.5px;max-width:none;margin:0}
  .matrix-main .axis{font-size:4.2px}
  .matrix-main .cell{aspect-ratio:1.34/1;font-size:4.7px;border-radius:3px}
  .matrix-note{font-size:4.4px;margin-top:5px}
  .scoring-profiles{display:grid;grid-template-columns:1fr;gap:5px;margin:0;padding-left:7px;border-left:1px solid rgba(43,88,109,.42)}
  .team-profile{padding:4px;border-radius:6px}
  .team-profile-head{margin-bottom:3px}.team-profile-head b{font-size:4.9px}.team-profile-head span{font-size:3.4px}
  .mini-matrix{grid-template-columns:7px repeat(5,minmax(0,1fr));gap:1.3px}
  .mini-axis{font-size:3.1px}
  .mini-cell{aspect-ratio:1.22/1;font-size:3.2px;border-radius:2px}
  .profile-caption{font-size:3.1px;margin-top:2px}

  .xg-summary{gap:4px}.xg-team b{font-size:18px}.xg-meta div{padding:4px 2px}.xg-trend{height:28px}
  .goals-card{display:block}.goals-card .panel-head{margin-bottom:7px}.ou-read{grid-template-columns:104px minmax(0,1fr);gap:9px}.ou-hero{padding-right:9px}.ou-hero b{font-size:17.5px}.ou-gauge{height:38px}.ou-gauge .lab{font-size:3.6px}.ou-gauge .lab.market{left:49%}.ou-gauge .lab.model{left:70%}.goals-card .dist-svg{height:76px;margin-top:4px}.goals-card .legend{margin-top:3px;font-size:4.7px}
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
            <div class="prob-summary">
              <span class="lean"><i></i>HOME LEADS MODEL</span>
              <strong>67.1% · +46.8 pp vs draw</strong>
            </div>
          </article>

          <article class="module xg-card">
            <div class="panel-head"><h3>Expected Goals (λ)</h3><span>SPORT</span></div>
            <div class="xg-summary">
              <div class="xg-team"><span>Arsenal</span><b>2.08</b></div>
              <div class="xg-sep">—</div>
              <div class="xg-team"><span>Brighton</span><b>0.91</b></div>
            </div>
            <div class="xg-meta">
              <div><span>TOTAL XG</span><b>2.99</b></div>
              <div><span>HOME DELTA</span><b class="good">+1.17</b></div>
            </div>
            <svg class="chart-svg xg-trend" viewBox="0 0 180 32" aria-label="Illustrative attacking trend">
              <defs><linearGradient id="xgArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#63b9a7" stop-opacity=".18"/><stop offset="100%" stop-color="#63b9a7" stop-opacity="0"/></linearGradient></defs>
              <line x1="4" y1="26" x2="176" y2="26" class="gridline"/>
              <path d="M4 25 L31 21 L58 22 L85 15 L112 17 L140 10 L176 7 L176 29 L4 29 Z" class="area"/>
              <polyline points="4,25 31,21 58,22 85,15 112,17 140,10 176,7" class="line"/>
              <circle cx="176" cy="7" r="2.2" class="dot"/>
            </svg>
            <div class="xg-caption">Illustrative attacking trend · design lab</div>
          </article>

          <article class="module edge-card">
            <div class="panel-head"><h3>Edge Gap</h3><span>EXAMPLE MARKET</span></div>
            <div class="edge-body">
              <div class="edge-visual">
                <svg class="chart-svg edge-svg" viewBox="0 0 260 84" aria-label="Model versus market">
                  <rect x="101" y="2" width="72" height="25" rx="12.5" class="edge-gap-pill"/>
                  <text x="137" y="14" class="edge-gap-pill-value">+11.7 pp</text>
                  <text x="137" y="21.5" class="edge-gap-pill-label">EDGE GAP</text>

                  <line x1="22" y1="54" x2="95" y2="54" class="edge-base"/>
                  <line x1="179" y1="54" x2="238" y2="54" class="edge-base"/>
                  <line x1="22" y1="49" x2="22" y2="59" class="edge-tick"/>
                  <line x1="94" y1="49" x2="94" y2="59" class="edge-tick"/>
                  <line x1="166" y1="49" x2="166" y2="59" class="edge-tick"/>
                  <line x1="238" y1="49" x2="238" y2="59" class="edge-tick"/>
                  <text x="22" y="75" class="edge-scale">50%</text>
                  <text x="94" y="75" class="edge-scale">60%</text>
                  <text x="166" y="75" class="edge-scale">70%</text>
                  <text x="238" y="75" class="edge-scale">80%</text>

                  <line x1="95" y1="54" x2="179" y2="54" class="edge-gap-glow"/>
                  <line x1="95" y1="54" x2="179" y2="54" class="edge-gap-focus"/>
                  <line x1="98" y1="54" x2="176" y2="54" class="edge-gap-core"/>
                  <circle cx="95" cy="54" r="6.2" class="edge-market-dot"/>
                  <circle cx="179" cy="54" r="6.2" class="edge-model-dot"/>
                  <text x="95" y="37" class="edge-point-label market">MARKET 60.1</text>
                  <text x="179" y="37" class="edge-point-label model">MODEL 71.8</text>
                </svg>
              </div>
              <aside class="edge-side-stats" aria-label="Edge context">
                <div class="edge-side-stat">
                  <span class="k">FAIR PRICE</span><strong class="v">1.39</strong>
                  <span class="sub"><b>Model</b> · Market 1.66</span>
                </div>
                <div class="edge-side-stat conf">
                  <span class="k">CONFIDENCE</span><strong class="v">84</strong>
                  <span class="sub"><b>High</b> · signal confidence</span>
                </div>
              </aside>
            </div>
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
            <div class="panel-head"><h3>Score Matrix (FT)</h3><span>FT + SCORING PROFILE</span></div>
            <div class="matrix-layout">
              <div class="matrix-main">
                <div class="matrix">
                  <span></span><span class="axis">0</span><span class="axis">1</span><span class="axis">2</span><span class="axis">3</span><span class="axis">4+</span>
                  <span class="axis">0</span><div class="cell c1">2</div><div class="cell c2">5</div><div class="cell c2">6</div><div class="cell c1">3</div><div class="cell c1">1</div>
                  <span class="axis">1</span><div class="cell c2">5</div><div class="cell c4">9</div><div class="cell c5 hot">11</div><div class="cell c3">7</div><div class="cell c1">2</div>
                  <span class="axis">2</span><div class="cell c2">4</div><div class="cell c5">10</div><div class="cell c4">9</div><div class="cell c3">6</div><div class="cell c1">2</div>
                  <span class="axis">3</span><div class="cell c1">2</div><div class="cell c3">5</div><div class="cell c3">6</div><div class="cell c2">4</div><div class="cell c1">1</div>
                  <span class="axis">4+</span><div class="cell c1">1</div><div class="cell c1">2</div><div class="cell c2">3</div><div class="cell c1">2</div><div class="cell c1">1</div>
                </div>
                <div class="matrix-note">Most likely: <b>2–1 · 11.4%</b></div>
              </div>

              <div class="scoring-profiles" aria-label="Illustrative goals for versus goals against profiles">
                <div class="team-profile">
                  <div class="team-profile-head"><b>Arsenal</b><span>GF × GA</span></div>
                  <div class="mini-matrix">
                    <span></span><span class="mini-axis">0</span><span class="mini-axis">1</span><span class="mini-axis">2</span><span class="mini-axis">3</span><span class="mini-axis">4+</span>
                    <span class="mini-axis">0</span><div class="mini-cell home-p1">1</div><div class="mini-cell home-p2">3</div><div class="mini-cell home-p3">4</div><div class="mini-cell home-p2">2</div><div class="mini-cell home-p1">1</div>
                    <span class="mini-axis">1</span><div class="mini-cell home-p2">2</div><div class="mini-cell home-p3">5</div><div class="mini-cell home-p5">7</div><div class="mini-cell home-p2">3</div><div class="mini-cell home-p1">1</div>
                    <span class="mini-axis">2</span><div class="mini-cell home-p1">1</div><div class="mini-cell home-p3">4</div><div class="mini-cell home-p4">6</div><div class="mini-cell home-p2">2</div><div class="mini-cell home-p1">1</div>
                    <span class="mini-axis">3</span><div class="mini-cell home-p1">1</div><div class="mini-cell home-p2">2</div><div class="mini-cell home-p2">3</div><div class="mini-cell home-p1">1</div><div class="mini-cell home-p1">0</div>
                    <span class="mini-axis">4+</span><div class="mini-cell home-p1">0</div><div class="mini-cell home-p1">1</div><div class="mini-cell home-p1">1</div><div class="mini-cell home-p1">0</div><div class="mini-cell home-p1">0</div>
                  </div>
                  <div class="profile-caption">X = GF · Y = GA</div>
                </div>

                <div class="team-profile">
                  <div class="team-profile-head"><b>Brighton</b><span>GF × GA</span></div>
                  <div class="mini-matrix">
                    <span></span><span class="mini-axis">0</span><span class="mini-axis">1</span><span class="mini-axis">2</span><span class="mini-axis">3</span><span class="mini-axis">4+</span>
                    <span class="mini-axis">0</span><div class="mini-cell away-p1">1</div><div class="mini-cell away-p2">2</div><div class="mini-cell away-p2">3</div><div class="mini-cell away-p1">1</div><div class="mini-cell away-p1">0</div>
                    <span class="mini-axis">1</span><div class="mini-cell away-p2">2</div><div class="mini-cell away-p3">4</div><div class="mini-cell away-p5">5</div><div class="mini-cell away-p2">2</div><div class="mini-cell away-p1">1</div>
                    <span class="mini-axis">2</span><div class="mini-cell away-p1">1</div><div class="mini-cell away-p2">3</div><div class="mini-cell away-p4">4</div><div class="mini-cell away-p2">2</div><div class="mini-cell away-p1">1</div>
                    <span class="mini-axis">3</span><div class="mini-cell away-p1">0</div><div class="mini-cell away-p2">2</div><div class="mini-cell away-p2">2</div><div class="mini-cell away-p1">1</div><div class="mini-cell away-p1">0</div>
                    <span class="mini-axis">4+</span><div class="mini-cell away-p1">0</div><div class="mini-cell away-p1">1</div><div class="mini-cell away-p1">1</div><div class="mini-cell away-p1">0</div><div class="mini-cell away-p1">0</div>
                  </div>
                  <div class="profile-caption">X = GF · Y = GA</div>
                </div>
              </div>
            </div>
          </article>

          <article class="module goals-card">
            <div class="panel-head"><h3>Over / Under 2.5 Goals</h3><span>MODEL VS MARKET</span></div>
            <div class="ou-read">
              <div class="ou-hero">
                <span>MODEL OVER 2.5</span>
                <b>64.4%</b>
                <strong>+8.9 pp edge</strong>
                <small>Market fair 55.3%</small>
              </div>
              <div>
                <div class="ou-gauge">
                  <div class="base"></div>
                  <div class="edge"></div>
                  <span class="lab market">MARKET 55.3</span>
                  <span class="lab model">MODEL 64.4</span>
                  <i class="mark market"></i>
                  <i class="mark model"></i>
                </div>
                <div class="ou-scale"><span>40%</span><span>50%</span><span>60%</span><span>70%</span><span>80%</span></div>
              </div>
            </div>
            <svg class="chart-svg dist-svg" viewBox="0 0 300 92" aria-label="Illustrative goal distribution">
              <defs>
                <linearGradient id="homeBar" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#68b3cb"/><stop offset="100%" stop-color="#3c849e"/></linearGradient>
                <linearGradient id="awayBar" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#82cbb8"/><stop offset="100%" stop-color="#519886"/></linearGradient>
              </defs>
              <rect x="102" y="4" width="70" height="68" rx="6" class="dist-focus"/>
              <line x1="18" y1="20" x2="292" y2="20" class="dist-grid" opacity=".35"/>
              <line x1="18" y1="46" x2="292" y2="46" class="dist-grid" opacity=".55"/>
              <line x1="18" y1="72" x2="292" y2="72" class="dist-grid"/>
              <g transform="translate(28,0)"><rect x="0" y="51" width="10" height="21" rx="3" class="dist-home"/><rect x="12" y="38" width="10" height="34" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">0</text></g>
              <g transform="translate(70,0)"><rect x="0" y="28" width="10" height="44" rx="3" class="dist-home"/><rect x="12" y="15" width="10" height="57" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">1</text></g>
              <g transform="translate(112,0)"><rect x="0" y="8" width="10" height="64" rx="3" class="dist-home"/><rect x="12" y="35" width="10" height="37" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">2</text></g>
              <g transform="translate(154,0)"><rect x="0" y="24" width="10" height="48" rx="3" class="dist-home"/><rect x="12" y="54" width="10" height="18" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">3</text></g>
              <g transform="translate(196,0)"><rect x="0" y="42" width="10" height="30" rx="3" class="dist-home"/><rect x="12" y="65" width="10" height="7" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">4</text></g>
              <g transform="translate(238,0)"><rect x="0" y="56" width="10" height="16" rx="3" class="dist-home"/><rect x="12" y="69" width="10" height="3" rx="3" class="dist-away"/><text x="11" y="86" class="dist-label">5+</text></g>
            </svg>
            <div class="legend"><span><i class="h"></i>Arsenal</span><span><i class="a"></i>Brighton</span></div>
          </article>
        </section>

        <section class="read">
          <div class="read-mark">SE</div>
          <div><b>Primary read <span class="read-tag">SPORT FIRST</span></b><p>Sporting projection is built first. Market value is assessed only after the football case is established.</p></div>
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
