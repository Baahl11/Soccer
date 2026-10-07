from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_FRONTEND_V2_PREVIEW_1.0.0"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def render() -> str:
    auth = supabase_auth_v4.public_auth_config()
    config = _safe_json(
        {
            "supabase_url": auth.get("project_url"),
            "publishable_key": auth.get("publishable_key"),
            "auth_configured": bool(auth.get("configured")),
            "api_base": "/app/api/v2",
        }
    )
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Soccer Edge</title>
<meta name="theme-color" content="#06111a">
<style>
:root{{
  --bg:#050b11;--bg2:#07121b;--panel:#091a27;--panel2:#0b2130;--line:#15364a;--line2:#23516b;
  --text:#eff7fb;--muted:#7892a3;--muted2:#526c7d;--green:#4de1ad;--green2:#0d3a30;
  --blue:#5fbfff;--amber:#e9ba62;--red:#ff6e7e;--chip:#0b2637;--shadow:0 20px 55px rgba(0,0,0,.24);
  --safe-top:max(12px,env(safe-area-inset-top));--safe-bottom:max(12px,env(safe-area-inset-bottom))
}}
*{{box-sizing:border-box}}html{{background:var(--bg)}}body{{margin:0;background:
radial-gradient(circle at 44% -12%,#103148 0,#07151f 25%,var(--bg) 62%);color:var(--text);
font:13px Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}}
button,input,select{{font:inherit}}button{{cursor:pointer}}a{{color:inherit}}.hidden{{display:none!important}}
.shell{{min-height:100vh;display:grid;grid-template-columns:218px minmax(0,1fr)}}.side{{position:sticky;top:0;height:100vh;
padding:22px 13px;background:linear-gradient(180deg,#07141f,#050c12);border-right:1px solid #123044;z-index:20}}
.brand{{display:flex;align-items:center;gap:10px;padding:2px 8px 24px}}.brand-mark{{width:36px;height:36px;border-radius:10px;
display:grid;place-items:center;background:#0a2a3f;border:1px solid #1f506b;color:var(--blue);font-weight:950;font-size:13px;box-shadow:0 8px 24px #0007}}
.brand-copy b{{display:block;font-size:14px;letter-spacing:.01em}}.brand-copy span{{display:block;margin-top:2px;color:var(--green);font-size:9px;font-weight:850;letter-spacing:.08em}}
.nav-label{{margin:15px 10px 6px;color:#496879;font-size:8px;letter-spacing:.14em;text-transform:uppercase;font-weight:800}}
.nav{{display:grid;gap:3px}}.nav button{{width:100%;display:flex;gap:10px;align-items:center;border:1px solid transparent;background:transparent;
color:#7892a3;padding:10px 11px;border-radius:8px;text-align:left;font-weight:760}}.nav button:hover{{color:#dcebf3;background:#091e2b}}
.nav button.active{{color:#fff;background:linear-gradient(90deg,#0d3148,#0a2231);border-color:#16435d}}.nav i{{width:17px;text-align:center;color:#59bfff;font-style:normal}}
.side-foot{{position:absolute;left:13px;right:13px;bottom:18px;padding:11px;border:1px solid #153448;border-radius:10px;background:#071720;
color:#658091;font-size:9px;line-height:1.45}}.side-foot b{{color:#bcd0db}}
.main{{min-width:0;padding:22px 26px 64px}}.topbar{{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:22px}}
.context{{min-width:0}}.context b{{font-size:11px}}.context span{{display:block;color:var(--muted);font-size:9px;margin-top:3px}}
.account-dock{{display:flex;align-items:center;gap:8px}}.plan-chip,.fresh-chip{{display:inline-flex;align-items:center;border:1px solid #1f6b54;background:#0a2a23;color:#63e5b7;
border-radius:999px;padding:5px 8px;font-size:8px;font-weight:900;white-space:nowrap}}.fresh-chip{{border-color:#23526c;background:#0a2130;color:#87b7d1}}
.icon-btn,.primary,.secondary{{border:1px solid #21465d;background:#091d2a;color:#dcebf3;border-radius:8px;padding:8px 11px;font-weight:820;font-size:9px}}
.icon-btn:hover,.secondary:hover{{border-color:#347493}}.primary{{background:#0b3b30;border-color:#286f5a;color:#7ff0c9}}
.page{{display:none}}.page.active{{display:block}}.page-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:15px;margin-bottom:17px}}
.page-head h1{{font-size:27px;line-height:1.05;margin:0 0 4px;letter-spacing:-.025em}}.sub{{color:var(--muted);font-size:10px;line-height:1.5}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin-bottom:12px}}.kpi{{border:1px solid var(--line);background:linear-gradient(180deg,#0b1e2c,#081620);
border-radius:11px;padding:13px}}.kpi b{{display:block;font-size:22px;line-height:1.05}}.kpi span{{display:block;margin-top:4px;color:#668394;font-size:8px;text-transform:uppercase;letter-spacing:.08em}}
.kpi.good b{{color:var(--green)}}.kpi.lean b{{color:var(--blue)}}.kpi.watch b{{color:var(--amber)}}.panel{{border:1px solid var(--line);background:
linear-gradient(180deg,rgba(10,28,41,.97),rgba(7,20,29,.98));border-radius:12px;padding:14px;box-shadow:var(--shadow);margin-bottom:12px}}
.panel-head{{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:11px}}.panel-head h2,.panel-head h3{{font-size:12px;margin:0}}
.link-btn{{border:0;background:none;color:var(--blue);font-size:9px;font-weight:750;padding:3px}}.hero{{padding:0;overflow:hidden}}.hero-grid{{display:grid;grid-template-columns:minmax(300px,.95fr) minmax(360px,1.05fr);min-height:260px}}
.hero-left{{padding:24px;display:grid;align-content:center;border-right:1px solid #173b50;background:radial-gradient(circle at 40% 48%,rgba(18,82,112,.30),transparent 56%)}}
.eyebrow{{color:var(--green);font-size:9px;letter-spacing:.1em;text-transform:uppercase;font-weight:900}}.faceoff{{margin:18px 0 12px;display:grid;grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr);align-items:center;gap:10px}}
.team{{display:grid;justify-items:center;gap:10px;min-width:0}}.crest{{width:74px;height:74px;display:grid;place-items:center;position:relative}}.crest img{{width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 5px 8px #0008)}}
.crest .fallback{{position:absolute;inset:0;display:grid;place-items:center;border:1px solid #25516c;border-radius:18px;background:#0a2b3e;color:#d6edf8;font-size:24px;font-weight:950}}
.crest.has-img .fallback{{opacity:0}}.team strong{{font-size:17px;line-height:1.15;text-align:center;overflow-wrap:anywhere}}.vs{{color:#617e90;text-align:center;font-weight:900}}
.kickoff{{color:#7794a7;text-align:center;font-size:9px}}.hero-right{{padding:25px;display:grid;align-content:center;gap:15px}}.market-title{{font-size:22px;line-height:1.15;font-weight:950;letter-spacing:-.015em}}
.decision-row{{display:flex;gap:7px;flex-wrap:wrap}}.decision{{border-radius:999px;padding:6px 9px;font-size:8px;font-weight:950;border:1px solid #2a5c76;color:#a5c6d8;background:#0b2433}}
.decision.bet{{border-color:#27745c;background:#0a3228;color:#72ebbe}}.decision.lean{{border-color:#275f88;background:#0b2944;color:#74c7ff}}.decision.watch{{border-color:#735a27;background:#30250d;color:#efc46f}}
.projection-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid #184258;border-radius:10px;overflow:hidden;background:#09202e}}
.projection{{padding:11px;border-right:1px solid #17394c;min-width:0}}.projection:last-child{{border-right:0}}.projection small{{display:block;color:#698899;font-size:7px;text-transform:uppercase;letter-spacing:.05em}}
.projection b{{display:block;font-size:18px;margin-top:3px;white-space:nowrap}}.projection.edge b{{color:var(--green)}}.badges{{display:flex;gap:6px;flex-wrap:wrap}}.badge{{border:1px solid #1d4359;background:#091d29;
border-radius:6px;padding:5px 7px;color:#8ea9b8;font-size:8px}}.badge.good{{border-color:#21624f;color:#64dfb3;background:#09281f}}.badge.warn{{border-color:#6c5522;color:#e8bd67;background:#2b220e}}
.empty{{padding:26px;text-align:center;border:1px dashed #28485b;border-radius:10px;color:#7690a1;background:#071722}}.empty b{{display:block;color:#cbdde6;font-size:12px;margin-bottom:5px}}
.section-title{{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:19px 1px 9px}}.section-title h2{{font-size:15px;margin:0}}.section-title span{{color:var(--muted);font-size:9px}}
.card-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.pick-card{{border:1px solid var(--line);background:linear-gradient(180deg,#0a1e2b,#081720);
border-radius:12px;padding:14px;min-width:0;box-shadow:0 12px 32px #0003}}.pick-card:hover{{border-color:#245c79}}.card-top{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}
.card-match{{min-width:0}}.card-match b{{font-size:12px;display:block;overflow-wrap:anywhere}}.card-match small{{display:block;color:#6c899a;margin-top:3px;font-size:8px}}.card-market{{font-size:16px;font-weight:900;margin:13px 0 8px}}
.price-line{{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-top:1px solid #112f41;border-bottom:1px solid #112f41;color:#89a4b3;font-size:9px}}.price-line b{{color:#e7f3f8}}
.mini-proj{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:9px}}.mini-proj div{{padding:7px;background:#0a2230;border-radius:7px;min-width:0}}.mini-proj small{{display:block;color:#638294;font-size:7px}}.mini-proj b{{display:block;margin-top:2px;font-size:11px;white-space:nowrap}}
.reason{{margin-top:9px;color:#809aa9;font-size:9px;line-height:1.45}}.reason b{{color:#bcd0db}}.lock{{border:1px dashed #2c5064;background:#071821;border-radius:12px;padding:26px;text-align:center;color:#7894a4}}.lock b{{display:block;color:#d4e4eb;margin-bottom:5px}}
.watch-list,.slate-list{{display:grid}}.watch-row,.slate-row{{display:grid;grid-template-columns:92px minmax(0,1fr) auto;align-items:center;gap:12px;padding:11px 5px;border-bottom:1px solid #102c3c}}
.watch-row:last-child,.slate-row:last-child{{border-bottom:0}}.time{{color:#7b98a9;font-size:9px}}.time b{{display:block;color:#cbdbe4;font-size:11px}}.row-teams b{{display:block;font-size:10px}}.row-teams small{{display:block;color:#688596;font-size:8px;margin-top:3px}}.row-state{{justify-self:end;text-align:right}}
.status{{display:inline-flex;border-radius:999px;border:1px solid #2b536a;color:#8db3c7;padding:4px 7px;font-size:7px;font-weight:900;text-transform:uppercase}}.status.bet{{border-color:#27674f;color:#65e5b6;background:#09291f}}.status.lean{{border-color:#2a6084;color:#6fc4ff;background:#0a2840}}.status.watch{{border-color:#6b5525;color:#e7bd68;background:#2b220e}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:9px;min-width:720px}}th{{text-align:left;color:#668596;font-size:7px;text-transform:uppercase;letter-spacing:.06em;padding:8px;border-bottom:1px solid #17384b}}td{{padding:10px 8px;border-bottom:1px solid #102b3b}}
.metric-quiet{{color:#718d9d}}.match-detail{{display:grid;gap:10px}}.detail-head{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}.detail-head h2{{font-size:19px;margin:0}}.detail-tabs{{display:flex;gap:6px;flex-wrap:wrap}}
.detail-tab{{border:1px solid #1e4257;background:#081b27;color:#7895a6;border-radius:7px;padding:6px 9px;font-size:8px}}.detail-tab.active{{color:#fff;border-color:#286689;background:#0b3046}}
.detail-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}}.detail-card{{border:1px solid #15384c;background:#081a26;border-radius:10px;padding:12px}}.detail-card h3{{margin:0 0 8px;font-size:10px}}.big{{font-size:22px;font-weight:950}}
.account-page{{max-width:760px}}.account-card{{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center}}.account-card h2{{margin:0 0 4px;font-size:18px}}.account-card p{{margin:0;color:#7893a3;font-size:9px;line-height:1.5}}
.modal{{position:fixed;inset:0;display:none;place-items:center;padding:18px;background:#02070bcc;backdrop-filter:blur(8px);z-index:100}}.modal.open{{display:grid}}.modal-card{{width:min(450px,100%);border:1px solid #21485f;
background:linear-gradient(180deg,#0a1f2d,#07131c);border-radius:15px;padding:20px;box-shadow:0 30px 100px #000d}}.modal-card h2{{margin:0 0 4px;font-size:20px}}.modal-card p{{color:#7893a3;font-size:9px;line-height:1.5}}
.modal-card input{{width:100%;padding:10px;margin:7px 0 0;border-radius:8px;border:1px solid #1a3d52;background:#061620;color:#eef7fb}}.modal-actions{{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}}.modal-status{{min-height:15px;color:var(--amber);font-size:9px;margin-top:8px}}
.boot{{position:fixed;inset:0;z-index:200;display:grid;place-items:center;background:radial-gradient(circle at 50% 18%,#103148,#06111a 40%,#050b11 78%);padding:20px}}.boot-card{{width:min(400px,calc(100vw - 30px));text-align:center;border:1px solid #1c435a;
background:#081b27;border-radius:16px;padding:28px;box-shadow:0 30px 100px #000c}}.boot-card .brand-mark{{margin:0 auto 13px}}.boot-title{{font-size:18px;font-weight:950}}.boot-copy{{color:#7792a2;font-size:9px;margin-top:6px;line-height:1.5}}.boot-line{{height:3px;border-radius:99px;overflow:hidden;background:#0f2b3b;margin-top:18px}}.boot-line i{{display:block;width:38%;height:100%;background:var(--green);animation:load 1.05s ease-in-out infinite alternate}}@keyframes load{{from{{transform:translateX(-20%)}}to{{transform:translateX(185%)}}}}
.mobile-nav{{display:none}}
@media(max-width:1050px){{
 .shell{{grid-template-columns:1fr}}.side{{display:none}}.main{{padding:calc(62px + var(--safe-top)) 13px calc(82px + var(--safe-bottom))}}.topbar{{position:fixed;top:0;left:0;right:0;height:calc(52px + env(safe-area-inset-top));padding:var(--safe-top) 12px 8px;background:#06131de8;backdrop-filter:blur(12px);border-bottom:1px solid #123247;z-index:50;margin:0}}
 .context span{{display:none}}.account-dock .fresh-chip{{display:none}}.mobile-nav{{display:flex;position:fixed;bottom:0;left:0;right:0;z-index:55;overflow-x:auto;gap:4px;padding:7px 8px calc(7px + env(safe-area-inset-bottom));background:#06131df2;backdrop-filter:blur(12px);border-top:1px solid #153448;scrollbar-width:none}}
 .mobile-nav button{{flex:0 0 auto;min-width:68px;border:0;background:transparent;color:#718c9c;padding:7px 9px;border-radius:8px;font-size:8px;font-weight:800}}.mobile-nav button.active{{background:#0b2b3d;color:var(--green)}}
 .hero-grid{{grid-template-columns:1fr}}.hero-left{{border-right:0;border-bottom:1px solid #173b50}}.projection-grid{{grid-template-columns:repeat(2,1fr)}}.projection:nth-child(2){{border-right:0}}.projection:nth-child(-n+2){{border-bottom:1px solid #17394c}}
}}
@media(max-width:700px){{
 .page-head{{display:block}}.page-head>div:last-child{{margin-top:8px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.kpi:last-child{{grid-column:1/-1}}.card-grid{{grid-template-columns:1fr}}
 .faceoff{{grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr)}}.crest{{width:64px;height:64px}}.team strong{{font-size:14px}}.hero-left,.hero-right{{padding:18px}}.market-title{{font-size:19px}}
 .mini-proj{{grid-template-columns:repeat(2,1fr)}}.watch-row,.slate-row{{grid-template-columns:64px minmax(0,1fr)}}.row-state{{grid-column:2;justify-self:stretch;text-align:left}}.detail-grid{{grid-template-columns:1fr}}
 .account-card{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div id="boot" class="boot" role="status" aria-live="polite">
  <div class="boot-card"><div class="brand-mark">SE</div><div class="boot-title">Soccer Edge</div>
  <div id="bootCopy" class="boot-copy">Loading the latest verified snapshot. Missing evidence remains NOT VERIFIED.</div>
  <div class="boot-line"><i></i></div></div>
</div>
<div class="shell">
<aside class="side">
  <div class="brand"><div class="brand-mark">SE</div><div class="brand-copy"><b>Soccer Edge</b><span>MODEL VS MARKET</span></div></div>
  <div class="nav-label">Decision desk</div>
  <nav class="nav">
    <button class="active" data-page="today"><i>◉</i>Today</button>
    <button data-page="picks"><i>▲</i>Picks</button>
    <button data-page="leans"><i>◆</i>Leans</button>
    <button data-page="matches"><i>◎</i>Matches</button>
  </nav>
  <div class="nav-label">Trust</div>
  <nav class="nav">
    <button data-page="performance"><i>▦</i>Performance</button>
    <button data-page="myedge"><i>☆</i>My Edge</button>
    <button data-page="account"><i>○</i>Account</button>
  </nav>
  <div class="side-foot"><b>Sport first. Market second.</b><br>Zero bets is a valid result. Only canonical engine decisions are shown as picks.</div>
</aside>
<main class="main">
  <div class="topbar">
    <div class="context"><b>Soccer Edge Intelligence</b><span id="snapshotText">Waiting for persisted snapshot</span></div>
    <div class="account-dock"><span id="freshChip" class="fresh-chip">VERIFYING</span><span id="planChip" class="plan-chip">EXPLORER</span><button id="accountBtn" class="icon-btn">Sign in</button></div>
  </div>

  <section id="today" class="page active">
    <div class="page-head"><div><h1>Today</h1><div class="sub">Canonical BETS first, transparent LEANS second, verified WATCH states after that.</div></div><div id="todayStamp" class="fresh-chip">LATEST SNAPSHOT</div></div>
    <div class="kpis">
      <div class="kpi"><b id="kpiFixtures">—</b><span>fixtures</span></div>
      <div class="kpi good"><b id="kpiPicks">—</b><span>canonical bets</span></div>
      <div class="kpi lean"><b id="kpiLeans">—</b><span>leans</span></div>
      <div class="kpi watch"><b id="kpiWatches">—</b><span>watch</span></div>
      <div class="kpi"><b id="kpiPasses">—</b><span>pass</span></div>
    </div>
    <div id="hero" class="panel hero"></div>
    <div class="section-title"><div><h2>Best Bets</h2><span>Only explicit persisted BET classifications</span></div><button class="link-btn" data-go="picks">View picks →</button></div>
    <div id="todayPicks" class="card-grid"></div>
    <div class="section-title"><div><h2>Leans</h2><span>Interesting, but not promoted to BET</span></div><button class="link-btn" data-go="leans">View leans →</button></div>
    <div id="todayLeans" class="card-grid"></div>
    <div class="section-title"><div><h2>Watch</h2><span>Waiting for price, XI, goalkeeper or other required evidence</span></div></div>
    <div class="panel"><div id="todayWatches" class="watch-list"></div></div>
    <div class="section-title"><div><h2>Full Slate</h2><span>Every persisted fixture remains visible</span></div></div>
    <div class="panel"><div id="todaySlate" class="slate-list"></div></div>
  </section>

  <section id="picks" class="page">
    <div class="page-head"><div><h1>Picks</h1><div class="sub">BET means the canonical engine emitted BET. READY or STRONG alone never qualifies.</div></div><span class="decision bet">BET ONLY</span></div>
    <div id="picksPage" class="card-grid"></div>
  </section>

  <section id="leans" class="page">
    <div class="page-head"><div><h1>Leans</h1><div class="sub">Sporting interest without enough verified market edge or decision confidence to become a BET.</div></div><span class="decision lean">LEAN</span></div>
    <div id="leansPage" class="card-grid"></div>
  </section>

  <section id="matches" class="page">
    <div class="page-head"><div><h1>Matches</h1><div class="sub">Browse the verified slate. Premium match intelligence opens from persisted fixture evidence only.</div></div></div>
    <div class="panel"><div id="matchesList" class="slate-list"></div></div>
    <div id="matchDetail" class="panel hidden"></div>
  </section>

  <section id="performance" class="page">
    <div class="page-head"><div><h1>Performance</h1><div class="sub">Validation/OOS evidence and realized BET performance are kept conceptually separate.</div></div></div>
    <div id="performanceBody" class="panel"><div class="empty"><b>Open this page to load verified evidence.</b>Research/OOS metrics are never presented as a customer BET ledger.</div></div>
  </section>

  <section id="myedge" class="page">
    <div class="page-head"><div><h1>My Edge</h1><div class="sub">Saved decisions and tracked matches. Server-side subscriber persistence follows in FE-6.</div></div></div>
    <div id="savedBody" class="panel"><div class="empty"><b>No saved items yet.</b>Open a pick or match to start building your personal slate.</div></div>
  </section>

  <section id="account" class="page account-page">
    <div class="page-head"><div><h1>Account</h1><div class="sub">Authentication and entitlement state are resolved server-side.</div></div></div>
    <div class="panel account-card"><div><h2 id="accountPlan">Explorer</h2><p id="accountCopy">Browse the verified slate and watch states. Premium decision evidence stays redacted until entitlement permits it.</p></div><button id="accountPageBtn" class="primary">Sign in</button></div>
    <div class="panel"><div class="panel-head"><h3>Commercial readiness</h3></div><div class="sub">Subscription infrastructure already exists, but this preview does not initiate billing. Checkout/paywall polish is governed by FE-7.</div></div>
  </section>
</main>
</div>
<nav class="mobile-nav">
  <button class="active" data-page="today">Today</button><button data-page="picks">Picks</button><button data-page="leans">Leans</button><button data-page="matches">Matches</button><button data-page="performance">Results</button><button data-page="account">Account</button>
</nav>

<div id="authModal" class="modal" aria-hidden="true"><div class="modal-card">
  <h2>Soccer Edge</h2><p>Sign in to resolve your server-side plan and unlock premium evidence when entitled.</p>
  <input id="authEmail" type="email" autocomplete="email" placeholder="Email">
  <input id="authPassword" type="password" autocomplete="current-password" placeholder="Password">
  <div class="modal-actions"><button id="signIn" class="primary">Sign in</button><button id="signUp" class="secondary">Create account</button><button id="signOut" class="secondary">Sign out</button><button id="closeAuth" class="secondary">Close</button></div>
  <div id="authStatus" class="modal-status"></div>
</div></div>

<script id="soccer-edge-v2-config" type="application/json">{config}</script>
<script id="soccer-edge-v2-app">
(() => {{
  const AK='soccer_edge_access_token',RK='soccer_edge_refresh_token';
  const cfg=(()=>{{try{{return JSON.parse(document.getElementById('soccer-edge-v2-config')?.textContent||'{{}}')}}catch(_){{return {{}}}}}})();
  let DATA=null,ACCESS=null,PERF_LOADED=false;
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  const token=()=>localStorage.getItem(AK)||'';
  const pct=v=>{{const n=Number(v);if(!Number.isFinite(n))return '—';const p=Math.abs(n)<=1?n*100:n;return p.toFixed(1)+'%'}};
  const num=v=>{{const n=Number(v);return Number.isFinite(n)?n:null}};
  const pp=v=>{{const n=num(v);return n===null?'—':(n>=0?'+':'')+n.toFixed(1)+' pp'}};
  const ev=v=>{{const n=num(v);return n===null?'—':(n>=0?'+':'')+(Math.abs(n)<=1?n*100:n).toFixed(1)+'%'}};
  const initials=name=>String(name||'?').split(/\\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase().slice(0,3);
  const dt=raw=>{{if(!raw)return ['TBD','Kickoff'];const d=new Date(raw);if(Number.isNaN(d.getTime()))return [String(raw),'Kickoff'];return [new Intl.DateTimeFormat(undefined,{{hour:'2-digit',minute:'2-digit'}}).format(d),new Intl.DateTimeFormat(undefined,{{month:'short',day:'numeric'}}).format(d)]}};
  const api=async(path)=>{{const h={{}};if(token())h.Authorization='Bearer '+token();const r=await fetch((cfg.api_base||'/app/api/v2')+path,{{headers:h,cache:'no-store'}});const d=await r.json().catch(()=>({{}}));if(!r.ok)throw Object.assign(new Error(d.error||('HTTP '+r.status)),{{status:r.status,data:d}});return d}};
  const authFetch=async(path,body)=>{{if(!cfg.auth_configured)throw new Error('AUTH_NOT_CONFIGURED');const r=await fetch(cfg.supabase_url+path,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify(body)}});const d=await r.json().catch(()=>({{}}));if(!r.ok)throw new Error(d.error_description||d.msg||d.error||('HTTP '+r.status));return d}};
  const classification=c=>String(c?.decision?.classification||'').toUpperCase();
  const fixture=c=>c?.fixture||{{}},market=c=>c?.market||{{}},proj=c=>c?.projections||{{}},avail=c=>c?.availability||{{}};
  const premium=()=>!!ACCESS?.premium_unlocked;
  function crest(url,name){{const src=String(url||'');return '<span class="crest '+(src?'has-img':'')+'">'+(src?'<img loading="lazy" decoding="async" src="'+esc(src)+'" alt="'+esc(name)+' crest" onerror="this.style.display=\\'none\\';this.parentElement.classList.remove(\\'has-img\\')">':'')+'<span class="fallback">'+esc(initials(name))+'</span></span>'}}
  function priceText(c){{const p=market(c).price||{{}},v=num(p.value),f=String(p.format||'NOT VERIFIED');if(v===null)return 'Price NOT VERIFIED';if(f==='DECIMAL')return v.toFixed(2)+(p.bookmaker?' · '+p.bookmaker:'');return String(v)+' · format '+f}}
  function evidence(c){{const e=c?.evidence||{{}},s=(e.sporting_reasons||[])[0],m=(e.market_reasons||[])[0],b=(e.blockers||[])[0];if(s)return '<b>Sport:</b> '+esc(s);if(m)return '<b>Market:</b> '+esc(m);if(b)return '<b>Blocker:</b> '+esc(b);return 'No additional persisted explanation in this snapshot.'}}
  function decisionClass(c){{const x=classification(c);return x==='BET'?'bet':x==='LEAN'?'lean':'watch'}}
  function decisionLabel(c){{return classification(c)||String(c?.decision?.execution_status||'WATCH').replaceAll('_',' ')}}
  function card(c){{const f=fixture(c),m=market(c),p=proj(c),a=avail(c),d=c?.decision||{{}},price=m.price||{{}},stamp=price.captured_at||c?.freshness?.provider_update||'NOT VERIFIED';
    return '<article class="pick-card" data-fixture="'+esc(f.fixture_id||'')+'"><div class="card-top"><div class="card-match"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(f.league||f.country||'Competition')+' · '+esc(f.kickoff||'Kickoff N/V')+'</small></div><span class="status '+decisionClass(c)+'">'+esc(decisionLabel(c))+(d.tier?' · '+esc(d.tier):'')+'</span></div>'+
    '<div class="card-market">'+esc(m.selection||m.name||m.family||'Market')+(m.line!==null&&m.line!==undefined?' · '+esc(m.line):'')+'</div>'+
    '<div class="price-line"><span>'+esc(priceText(c))+'</span><span>Updated '+esc(stamp)+'</span></div>'+
    '<div class="mini-proj"><div><small>RAW SPORT</small><b>'+pct(p.raw_sport_probability)+'</b></div><div><small>SHRUNK</small><b>'+pct(p.market_shrunk_probability)+'</b></div><div><small>MARKET FAIR</small><b>'+pct(p.fair_market_probability)+'</b></div><div><small>EDGE</small><b style="color:var(--green)">'+pp(p.probability_edge_pp)+'</b></div></div>'+
    '<div class="badges" style="margin-top:9px"><span class="badge">Availability '+pct(a.confidence)+'</span><span class="badge">Data '+esc(a.data_tier||'NOT VERIFIED')+'</span><span class="badge">XI '+esc(a.starting_xi_status||'NOT VERIFIED')+'</span><span class="badge">GK '+esc(a.goalkeeper_status||'NOT VERIFIED')+'</span></div>'+
    '<div class="reason">'+evidence(c)+'</div></article>'}}
  function empty(title,copy){{return '<div class="empty"><b>'+esc(title)+'</b>'+esc(copy)+'</div>'}}
  function lock(resource,count){{return '<div class="lock"><b>'+esc(resource)+' · Edge Pro</b>'+esc(count||0)+' canonical item'+((count||0)===1?'':'s')+' in this snapshot. Premium probability, price and evidence fields remain server-redacted.</div>'}}
  function renderKpis(){{const c=DATA?.counts||{{}};$('kpiFixtures').textContent=c.fixtures??'—';$('kpiPicks').textContent=c.picks??'—';$('kpiLeans').textContent=c.leans??'—';$('kpiWatches').textContent=c.watches??'—';$('kpiPasses').textContent=c.passes??'—'}}
  function renderHero(){{const picks=DATA?.picks||{{}},leans=DATA?.leans||{{}},watches=DATA?.watches||{{}},hero=$('hero');if(!hero)return;
    if(picks.locked&&picks.total>0){{hero.innerHTML='<div style="padding:24px">'+lock('Verified picks',picks.total)+'</div>';return}}
    const c=(picks.rows||[])[0]||(leans.rows||[])[0]||(watches.rows||[])[0];if(!c){{hero.innerHTML='<div style="padding:24px">'+empty('No canonical BET right now','Zero bets is a valid result. The system will not promote a READY or STRONG row into a wager.')+'</div>';return}}
    if(!c.projections){{hero.innerHTML='<div style="padding:24px">'+empty('Watch state active',String(c?.decision?.reason||c?.decision?.execution_status||'Waiting for required evidence'))+'</div>';return}}
    const f=fixture(c),m=market(c),p=proj(c),a=avail(c),[time,date]=dt(f.kickoff);
    hero.innerHTML='<div class="hero-grid"><div class="hero-left"><div class="eyebrow">'+esc(f.league||f.country||'Football Intelligence')+'</div><div class="faceoff"><div class="team">'+crest(f.home_team_logo,f.home_team)+'<strong>'+esc(f.home_team||'Home')+'</strong></div><div class="vs">VS</div><div class="team">'+crest(f.away_team_logo,f.away_team)+'<strong>'+esc(f.away_team||'Away')+'</strong></div></div><div class="kickoff">'+esc(date)+' · '+esc(time)+'</div></div>'+
    '<div class="hero-right"><div class="decision-row"><span class="decision '+decisionClass(c)+'">'+esc(decisionLabel(c))+'</span>'+(c?.decision?.tier?'<span class="decision">TIER '+esc(c.decision.tier)+'</span>':'')+'</div><div class="market-title">'+esc(m.selection||m.name||m.family||'Market')+(m.line!==null&&m.line!==undefined?' · '+esc(m.line):'')+'</div>'+
    '<div class="projection-grid"><div class="projection"><small>RAW SPORT</small><b>'+pct(p.raw_sport_probability)+'</b></div><div class="projection"><small>MARKET SHRUNK</small><b>'+pct(p.market_shrunk_probability)+'</b></div><div class="projection"><small>MARKET FAIR</small><b>'+pct(p.fair_market_probability)+'</b></div><div class="projection edge"><small>PROB EDGE</small><b>'+pp(p.probability_edge_pp)+'</b></div></div>'+
    '<div class="badges"><span class="badge good">'+esc(priceText(c))+'</span><span class="badge">Availability '+pct(a.confidence)+'</span><span class="badge">XI '+esc(a.starting_xi_status||'NOT VERIFIED')+'</span><span class="badge">GK '+esc(a.goalkeeper_status||'NOT VERIFIED')+'</span></div><div class="reason">'+evidence(c)+'</div></div></div>'}}
  function renderCards(){{const picks=DATA?.picks||{{}},leans=DATA?.leans||{{}};
    $('todayPicks').innerHTML=picks.locked?lock('Best Bets',picks.total):(picks.rows||[]).slice(0,4).map(card).join('')||empty('No BETS','No explicit persisted BET classifications in the latest snapshot.');
    $('todayLeans').innerHTML=leans.locked?lock('Leans',leans.total):(leans.rows||[]).slice(0,4).map(card).join('')||empty('No LEANS','The engine has not emitted an explicit LEAN in the latest snapshot.');
    $('picksPage').innerHTML=picks.locked?lock('Picks',picks.total):(picks.rows||[]).map(card).join('')||empty('No BETS','Zero bets is a valid outcome.');
    $('leansPage').innerHTML=leans.locked?lock('Leans',leans.total):(leans.rows||[]).map(card).join('')||empty('No LEANS','Nothing currently meets the canonical LEAN classification.');
    document.querySelectorAll('.pick-card[data-fixture]').forEach(el=>el.onclick=()=>openMatch(el.dataset.fixture));
  }}
  function watchRow(c){{const f=fixture(c),m=market(c),[time,date]=dt(f.kickoff),d=c?.decision||{{}};return '<div class="watch-row"><div class="time"><b>'+esc(time)+'</b>'+esc(date)+'</div><div class="row-teams"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(m.name||m.family||'Market')+(d.reason?' · '+esc(d.reason):'')+'</small></div><div class="row-state"><span class="status watch">'+esc(d.execution_status||d.classification||'WATCH')+'</span></div></div>'}}
  function slateRow(r){{const f=r?.fixture||{{}},s=r?.state||{{}},[time,date]=dt(f.kickoff);return '<div class="slate-row" data-fixture="'+esc(f.fixture_id||'')+'"><div class="time"><b>'+esc(time)+'</b>'+esc(date)+'</div><div class="row-teams"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(f.league||f.country||'Competition')+'</small></div><div class="row-state"><span class="status">'+esc(s.status||'NOT VERIFIED')+'</span></div></div>'}}
  function renderWatchSlate(){{const w=DATA?.watches?.rows||[],s=DATA?.slate?.rows||[];$('todayWatches').innerHTML=w.map(watchRow).join('')||empty('No active WATCH states','No persisted wait state is active in this snapshot.');
    const slate=s.map(slateRow).join('')||empty('No fixtures','No persisted fixtures in the latest slate.');$('todaySlate').innerHTML=slate;$('matchesList').innerHTML=slate;document.querySelectorAll('.slate-row[data-fixture]').forEach(el=>el.onclick=()=>openMatch(el.dataset.fixture));
  }}
  function renderPlan(){{ACCESS=DATA?.access||ACCESS||{{}};const plan=ACCESS?.display_role||ACCESS?.effective_plan||(token()?'FREE':'EXPLORER');$('planChip').textContent=String(plan).toUpperCase();$('accountBtn').textContent=token()?'Account':'Sign in';$('accountPageBtn').textContent=token()?'Manage account':'Sign in';$('accountPlan').textContent=premium()?'Edge Pro':'Explorer';$('accountCopy').textContent=premium()?'Premium decision evidence is unlocked for this account.':'Browse the verified slate and watch states. Premium probability, price and evidence fields remain redacted.';$('signOut').classList.toggle('hidden',!token())}}
  function renderMeta(){{const raw=DATA?.generated_at_utc||DATA?.generated_at_local||'persisted';$('snapshotText').textContent='Persisted snapshot · '+raw;$('todayStamp').textContent=raw;$('freshChip').textContent=DATA?.status==='SUBSCRIBER_CONTRACT_V2_READY'?'VERIFIED SNAPSHOT':'CHECK STATE'}}
  async function loadToday(){{DATA=await api('/today');ACCESS=DATA.access||{{}};renderMeta();renderPlan();renderKpis();renderHero();renderCards();renderWatchSlate()}}
  async function openMatch(fid){{if(!fid)return;activate('matches');const box=$('matchDetail');box.classList.remove('hidden');if(!premium()){{box.innerHTML=lock('Match Intelligence',1);return}}box.innerHTML=empty('Loading match intelligence','Reading persisted fixture evidence…');try{{const d=await api('/match/'+encodeURIComponent(fid));renderMatch(d)}}catch(e){{box.innerHTML=empty('Match detail unavailable',e.message)}}}}
  function renderMatch(d){{const box=$('matchDetail'),f=d?.fixture||{{}},c=d?.selected_candidate||{{}},p=c?.projections||{{}},a=c?.availability||{{}},ctx=d?.sport_context||{{}},xg=ctx.expected_goals||null,probs=ctx.outcome_probabilities||null;
    box.innerHTML='<div class="match-detail"><div class="detail-head"><div><div class="eyebrow">'+esc(f.league||f.country||'Competition')+'</div><h2>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</h2><div class="sub">'+esc(f.kickoff||'Kickoff NOT VERIFIED')+'</div></div><span class="decision '+decisionClass(c)+'">'+esc(decisionLabel(c))+'</span></div>'+
    '<div class="detail-tabs"><button class="detail-tab active">Overview</button><button class="detail-tab">Sport</button><button class="detail-tab">Availability</button><button class="detail-tab">Market</button><button class="detail-tab">Model</button></div>'+
    '<div class="detail-grid"><div class="detail-card"><h3>Raw Sport → Shrunk → Market</h3><div class="mini-proj"><div><small>RAW</small><b>'+pct(p.raw_sport_probability)+'</b></div><div><small>SHRUNK</small><b>'+pct(p.market_shrunk_probability)+'</b></div><div><small>FAIR MKT</small><b>'+pct(p.fair_market_probability)+'</b></div><div><small>EDGE</small><b>'+pp(p.probability_edge_pp)+'</b></div></div></div>'+
    '<div class="detail-card"><h3>Availability</h3><div class="big">'+pct(a.confidence)+'</div><div class="sub">Data '+esc(a.data_tier||'NOT VERIFIED')+' · XI '+esc(a.starting_xi_status||'NOT VERIFIED')+' · GK '+esc(a.goalkeeper_status||'NOT VERIFIED')+' · Injuries '+esc(a.injury_status||'NOT VERIFIED')+'</div></div>'+
    '<div class="detail-card"><h3>Expected Goals</h3><div class="big">'+(xg?esc((xg.home??'—')+' – '+(xg.away??'—')):'—')+'</div><div class="sub">'+(probs?'H '+pct(probs.home)+' · D '+pct(probs.draw)+' · A '+pct(probs.away):'Outcome probabilities NOT VERIFIED')+'</div></div></div>'+
    '<div class="detail-card"><h3>Evidence</h3><div class="reason">'+evidence(c)+'</div></div></div>'}}
  async function loadPerformance(){{if(PERF_LOADED)return;PERF_LOADED=true;const box=$('performanceBody');if(!premium()){{box.innerHTML=lock('Verified Performance',0);return}}box.innerHTML=empty('Loading performance','Reading persisted validation evidence…');try{{const d=await api('/performance'),rows=d.rows||[];box.innerHTML='<div class="panel-head"><h3>Persisted Validation Evidence</h3><span class="fresh-chip">'+esc(d.status||'N/V')+'</span></div><div class="sub" style="margin-bottom:10px">'+esc(d.note||'')+'</div><div class="table-wrap"><table><thead><tr><th>Market</th><th>Sample</th><th>Settled</th><th>ROI units</th><th>True CLV</th><th>Brier</th><th>Status</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+esc(r.label)+'</td><td>'+esc(r.sample_n??'—')+'</td><td>'+esc(r.settled??'—')+'</td><td>'+esc(r.roi_units??'—')+'</td><td>'+esc(r.avg_clv_pp??'—')+'</td><td>'+esc(r.brier??'—')+'</td><td>'+esc(r.status||'N/V')+'</td></tr>').join('')+'</tbody></table></div>'}}catch(e){{box.innerHTML=empty('Performance unavailable',e.message)}}}}
  function activate(id){{document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===id));window.scrollTo({{top:0,behavior:'smooth'}});if(id==='performance')loadPerformance()}}
  document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>activate(b.dataset.page));document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>activate(b.dataset.go));
  const modal=$('authModal'),openAuth=()=>{{modal.classList.add('open');modal.setAttribute('aria-hidden','false')}},closeAuth=()=>{{modal.classList.remove('open');modal.setAttribute('aria-hidden','true')}};
  $('accountBtn').onclick=()=>token()?activate('account'):openAuth();$('accountPageBtn').onclick=openAuth;$('closeAuth').onclick=closeAuth;
  $('signOut').onclick=()=>{{localStorage.removeItem(AK);localStorage.removeItem(RK);location.reload()}};
  $('signIn').onclick=async()=>{{try{{$('authStatus').textContent='Signing in…';const d=await authFetch('/auth/v1/token?grant_type=password',{{email:$('authEmail').value,password:$('authPassword').value}});localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);location.reload()}}catch(e){{$('authStatus').textContent=e.message}}}};
  $('signUp').onclick=async()=>{{try{{$('authStatus').textContent='Creating account…';const d=await authFetch('/auth/v1/signup',{{email:$('authEmail').value,password:$('authPassword').value}});if(d.access_token)localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);$('authStatus').textContent=d.access_token?'Account created. Reloading…':'Account created. Check your email if confirmation is required.';if(d.access_token)location.reload()}}catch(e){{$('authStatus').textContent=e.message}}}};
  (async()=>{{try{{await loadToday();$('boot').classList.add('hidden')}}catch(e){{$('bootCopy').textContent='Snapshot unavailable · '+e.message+' · no values were fabricated.';const line=document.querySelector('.boot-line');if(line)line.classList.add('hidden')}}}})();
}})();
</script>
</body>
</html>'''


async def app_page(request: Request) -> HTMLResponse:
    return HTMLResponse(render(), headers={"Cache-Control": "no-store"})


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "route": "/app-v2",
        "data_source": "/app/api/v2",
        "mock_data": False,
        "single_bootstrap_controller": True,
        "primary_navigation": [
            "Today",
            "Picks",
            "Leans",
            "Matches",
            "Performance",
            "My Edge",
            "Account",
        ],
        "operator_surfaces_in_primary_navigation": False,
        "billing_mutations_enabled": False,
        "frontend_creates_bet_or_lean": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
