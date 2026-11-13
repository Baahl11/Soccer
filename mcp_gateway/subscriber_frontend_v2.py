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
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Soccer Edge</title>
<meta name="theme-color" content="#06111a">
<link rel="manifest" href="/app.webmanifest"><link rel="icon" href="/pwa/icon.svg" type="image/svg+xml"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230a2a3f'/%3E%3Ctext x='32' y='39' text-anchor='middle' font-family='Arial' font-size='22' font-weight='700' fill='%235fbfff'%3ESE%3C/text%3E%3C/svg%3E">
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
border-radius:999px;padding:5px 8px;font-size:8px;font-weight:900;white-space:nowrap;max-width:100%;overflow:hidden;text-overflow:ellipsis}}.fresh-chip{{border-color:#23526c;background:#0a2130;color:#87b7d1}}
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
.edge-radar-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:10px}}
.edge-radar-cell{{border:1px solid #21485d;border-radius:9px;padding:11px;background:#081d2a;min-width:0}}
.edge-radar-cell b{{display:block;font-size:21px;color:#a7d9ed;line-height:1.15}}
.edge-radar-cell span{{display:block;font-size:9px;color:#9fb9c8;margin-top:5px;line-height:1.35}}
.edge-family-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:9px}}
.edge-family-row{{padding:9px;border:1px solid #234459;border-radius:8px;background:#071b27}}
.edge-family-row b{{font-size:11px;display:block;color:#b7e0ed}}
.edge-family-row span{{font-size:10px;display:block;color:#a7b9c7;margin-top:4px;line-height:1.4}}
@media(max-width:850px){{.edge-family-grid{{grid-template-columns:repeat(1,minmax(0,1fr))}}}}

.radar-reason{{display:block;color:#bbd1de;margin-top:5px;font-size:9px;line-height:1.35}}
.radar-evidence{{display:block;color:#7394a7;margin-top:3px;font-size:8px;line-height:1.35}}
@media(max-width:850px){{.edge-radar-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.section-title{{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:19px 1px 9px}}.section-title h2{{font-size:15px;margin:0}}.section-title span{{color:var(--muted);font-size:9px}}
/* Player markets are RESEARCH ONLY. Large labels and one explicit warning
   replace raw English sportsbook keys and repeated 8px NOT VERIFIED lines. */
.sgp-player-guide{{display:grid;grid-template-columns:1fr;gap:9px;margin:10px 0 12px;padding:15px;
  border:1px solid #285169;border-radius:12px;background:#0a2231}}
.sgp-player-guide-head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.sgp-player-guide-head strong{{font-size:13px;line-height:1.4;color:#e9f5fb}}
.sgp-player-note{{margin:0;color:#adc6d3;font-size:12px;line-height:1.55}}
.sgp-guide-steps{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.sgp-guide-step{{border:1px solid #1c455b;border-radius:9px;padding:10px 11px;background:#081b28}}
.sgp-guide-step b{{display:block;font-size:11px;color:#e0eff6;margin-bottom:5px}}
.sgp-guide-step span{{font-size:10px;line-height:1.45;color:#9db9c8}}
.sgp-caution-tag{{display:inline-flex;align-items:center;gap:5px;padding:5px 9px;border-radius:999px;
  border:1px solid #85612a;background:#322510;color:#f3c778;font-weight:850;font-size:10px;white-space:nowrap}}
.sgp-player-grid{{align-items:stretch}}
.sgp-leg-card{{margin:0;min-width:0;display:flex;flex-direction:column;gap:12px;
  border:1px solid #1b465e;border-radius:13px;padding:16px;
  background:linear-gradient(160deg,#0b2636,#081a26 70%);box-shadow:0 10px 28px #0003}}
.sgp-leg-card.selected{{border-color:#4daee1;box-shadow:0 0 0 1px #2f83ae,0 14px 34px #0005}}
.sgp-leg-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}}
.sgp-leg-player{{min-width:0}}.sgp-leg-player b{{display:block;font-size:16px;line-height:1.25;overflow-wrap:anywhere}}
.sgp-leg-team{{display:block;margin-top:5px;color:#a1bac8;font-size:11px;line-height:1.3}}
.sgp-leg-match{{color:#a3c0cf;font-size:11px;line-height:1.5;overflow-wrap:anywhere}}
.sgp-leg-market{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}
.sgp-leg-pill{{display:inline-flex;border:1px solid #25526a;border-radius:6px;padding:5px 8px;color:#a7d6ed;
  background:#092536;font-size:11px;font-weight:730}}
.sgp-leg-pill.muted{{color:#a7bac4;border-color:#274356;background:#0b1b27}}
.sgp-leg-main{{display:flex;gap:16px;justify-content:space-between;align-items:end;flex-wrap:wrap}}
.sgp-leg-count{{display:grid;gap:4px;min-width:0}}
.sgp-leg-count strong{{font-size:22px;line-height:1.1;color:#f5fbff;overflow-wrap:anywhere}}
.sgp-leg-count small{{color:#93afbe;font-size:10px;line-height:1.45;max-width:260px}}
.sgp-leg-price{{display:grid;gap:4px;justify-items:end;text-align:right}}
.sgp-leg-price strong{{font-size:24px;color:#a8e3ff;font-variant-numeric:tabular-nums}}
.sgp-leg-price small{{font-size:10px;color:#9ebdcd}}
.sgp-leg-review{{display:flex;flex-wrap:wrap;gap:7px;align-items:center;color:#9eb7c4;font-size:10px;line-height:1.5}}
.sgp-leg-review .sgp-caution-tag{{font-size:9px;padding:4px 7px}}
.sgp-leg-details{{margin:0;padding:11px 12px;border:1px solid #29495b;border-radius:8px;
  background:#071821;color:#9db8c7;font-size:11px;line-height:1.65}}
.sgp-leg-details b{{color:#f3c878}}
.sgp-leg-source{{margin-top:-6px;color:#809eb0;font-size:10px;line-height:1.4}}
.sgp-leg-select{{display:flex;align-items:center;gap:10px;margin-top:auto;padding-top:11px;
  border-top:1px solid #214357;color:#e4f5fc;font-size:12px;font-weight:800;cursor:pointer}}
.sgp-leg-select input{{appearance:auto;width:17px;height:17px;accent-color:#4abdf1;flex:0 0 auto;margin:0;cursor:pointer}}
.sgp-leg-select span{{flex:1}}
.sgp-slip{{margin-bottom:11px;padding:16px;background:#0a2331;border:1px solid #25516a;border-radius:12px}}
.sgp-slip-title{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}}
.sgp-slip-title b{{font-size:13px;color:#e5f4fa}}.sgp-slip-description{{margin-top:8px;font-size:11px;line-height:1.55;color:#a1bac8}}
.sgp-slip-items{{display:grid;gap:6px;margin-top:12px}}
.sgp-slip-item{{padding:9px 10px;border:1px solid #214458;background:#071b28;border-radius:7px;
  color:#d6eaf4;font-size:11px;overflow-wrap:anywhere}}
.sgp-slip-warning{{margin-top:11px;color:#efc677;font-size:11px;line-height:1.5}}
.sgp-slip-help{{font-size:11px;color:#a7bdc9;line-height:1.55}}
.sport-model-summary{{padding:14px;border:1px solid #245069;background:#092434;border-radius:12px;margin-bottom:12px}}
.sport-model-summary strong{{font-size:13px;color:#edf7ff}}
.sport-model-sub{{color:#aac2cf;font-size:11px;line-height:1.6;margin-top:7px}}
.sport-model-projection{{border:1px solid #2a4c64;border-radius:12px;padding:14px;background:#071f2d;
display:flex;flex-direction:column;gap:9px}}
.sport-model-top{{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}}
.sport-model-top b{{font-size:13px;line-height:1.4;overflow-wrap:anywhere}}
.sport-model-big{{font-size:21px;font-weight:800;color:#e2f4ff;font-variant-numeric:tabular-nums}}
.sport-model-stats{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}}
.sport-model-stat{{padding:8px;border-radius:8px;background:#112e40;min-width:0}}
.sport-model-stat b{{display:block;font-size:12px;color:#e1f1fa}}
.sport-model-stat small{{font-size:10px;line-height:1.4;color:#9bb4c4}}
.sport-model-foot{{font-size:10px;color:#a5b7c2;line-height:1.5}}

@media(max-width:700px){{.sgp-guide-steps{{grid-template-columns:1fr}}.sgp-leg-card{{padding:13px}}
 .sgp-leg-main{{align-items:start}}.sgp-leg-count strong{{font-size:20px}}.sgp-leg-price strong{{font-size:22px}}
 .sgp-leg-player b{{font-size:14px}}.sgp-player-guide{{padding:12px}}}}

.card-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.pick-card{{border:1px solid var(--line);background:linear-gradient(180deg,#0a1e2b,#081720);
border-radius:12px;padding:14px;min-width:0;box-shadow:0 12px 32px #0003}}.pick-card:hover{{border-color:#245c79}}.card-top{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}
.card-match{{min-width:0}}.card-match b{{font-size:12px;display:block;overflow-wrap:anywhere}}.card-match small{{display:block;color:#6c899a;margin-top:3px;font-size:8px}}.card-market{{font-size:16px;font-weight:900;margin:13px 0 8px}}
.price-line{{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-top:1px solid #112f41;border-bottom:1px solid #112f41;color:#89a4b3;font-size:9px}}.price-line b{{color:#e7f3f8}}
.mini-proj{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:9px}}.mini-proj div{{padding:7px;background:#0a2230;border-radius:7px;min-width:0}}.mini-proj small{{display:block;color:#638294;font-size:7px}}.mini-proj b{{display:block;margin-top:2px;font-size:11px;white-space:nowrap}}
.reason{{margin-top:9px;color:#809aa9;font-size:9px;line-height:1.45}}.reason b{{color:#bcd0db}}.card-actions{{display:flex;justify-content:flex-end;gap:6px;margin-top:10px}}.save-btn{{padding:6px 9px}}.saved-list{{display:grid;gap:8px}}.saved-row{{display:grid;grid-template-columns:92px minmax(0,1fr) auto;gap:12px;align-items:center;border:1px solid #15384c;background:#081a26;border-radius:9px;padding:10px}}.saved-row small{{display:block;color:#6f8b9b;margin-top:3px;font-size:8px}}.saved-row .remove{{border-color:#5a2c35;color:#ff8995;background:#291119}}@media(max-width:700px){{.saved-row{{grid-template-columns:1fr}}.saved-row>div:last-child{{justify-self:start}}}}.lock{{border:1px dashed #2c5064;background:#071821;border-radius:12px;padding:26px;text-align:center;color:#7894a4}}.lock b{{display:block;color:#d4e4eb;margin-bottom:5px}}
.watch-list,.slate-list{{display:grid}}.watch-row,.slate-row{{display:grid;grid-template-columns:92px minmax(0,1fr) auto;align-items:center;gap:12px;padding:11px 5px;border-bottom:1px solid #102c3c;min-width:0}}.slate-row{{cursor:pointer;border-radius:8px;transition:background .14s ease,border-color .14s ease;color:inherit;text-decoration:none}}.slate-row:hover{{background:#0a2030}}
.watch-row:last-child,.slate-row:last-child{{border-bottom:0}}.time{{color:#7b98a9;font-size:9px}}.time b{{display:block;color:#cbdbe4;font-size:11px}}.row-teams{{min-width:0;overflow:hidden}}.row-teams b{{display:block;font-size:10px;overflow-wrap:anywhere;word-break:break-word}}.row-teams small{{display:block;color:#688596;font-size:8px;margin-top:3px;overflow-wrap:anywhere;word-break:break-word}}.row-state{{justify-self:end;text-align:right;min-width:0}}
.status{{display:inline-flex;border-radius:999px;border:1px solid #2b536a;color:#8db3c7;padding:4px 7px;font-size:7px;font-weight:900;text-transform:uppercase}}.status.bet{{border-color:#27674f;color:#65e5b6;background:#09291f}}.status.lean{{border-color:#2a6084;color:#6fc4ff;background:#0a2840}}.status.watch{{border-color:#6b5525;color:#e7bd68;background:#2b220e}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:9px;min-width:720px}}th{{text-align:left;color:#668596;font-size:7px;text-transform:uppercase;letter-spacing:.06em;padding:8px;border-bottom:1px solid #17384b}}td{{padding:10px 8px;border-bottom:1px solid #102b3b}}
.metric-quiet{{color:#718d9d}}.match-detail{{display:grid;gap:10px}}.detail-head{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}.detail-head h2{{font-size:19px;margin:0}}.detail-tabs{{display:flex;gap:6px;flex-wrap:wrap}}
.detail-tab{{border:1px solid #1e4257;background:#081b27;color:#7895a6;border-radius:7px;padding:6px 9px;font-size:8px}}.detail-tab.active{{color:#fff;border-color:#286689;background:#0b3046}}
.detail-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}}.detail-card{{border:1px solid #15384c;background:#081a26;border-radius:10px;padding:12px}}.detail-card h3{{margin:0 0 8px;font-size:10px}}.big{{font-size:22px;font-weight:950}}
.detail-section{{display:none}}.detail-section.active{{display:block}}.detail-stack{{display:grid;gap:10px}}.detail-grid.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}.detail-grid.four{{grid-template-columns:repeat(4,minmax(0,1fr))}}
.detail-list{{display:grid;gap:6px}}.detail-list .item{{display:flex;justify-content:space-between;gap:14px;padding:8px 0;border-bottom:1px solid #112e3e;font-size:9px}}.detail-list .item:last-child{{border-bottom:0}}.detail-list .item span:first-child{{color:#7290a1}}.detail-list .item span:last-child{{color:#e4eff4;text-align:right;overflow-wrap:anywhere}}
.score-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}}.score-chip{{border:1px solid #173b50;background:#09202d;border-radius:8px;padding:9px;text-align:center}}.score-chip b{{display:block;font-size:14px}}.score-chip small{{display:block;color:#6d899a;font-size:7px;margin-top:2px}}
.profile-row{{display:grid;grid-template-columns:88px minmax(0,1fr) 42px;gap:8px;align-items:center;margin:7px 0}}.profile-row span{{font-size:8px;color:#7893a4}}.profile-bar{{height:6px;background:#102b3b;border-radius:99px;overflow:hidden}}.profile-bar i{{display:block;height:100%;background:#4de1ad;border-radius:99px}}
.missing-list{{display:flex;gap:6px;flex-wrap:wrap}}.missing-list .badge{{color:#e6bb68;border-color:#6b5423;background:#2a210d}}.market-group{{margin-top:10px}}.market-group h4{{font-size:9px;margin:0 0 6px;color:#8ba8b8;text-transform:uppercase;letter-spacing:.07em}}
@media(max-width:700px){{.detail-grid.two,.detail-grid.four{{grid-template-columns:1fr}}.score-grid{{grid-template-columns:repeat(3,1fr)}}.profile-row{{grid-template-columns:72px minmax(0,1fr) 36px}}}}

/* Match Intelligence — customer-facing information hierarchy */
#matches:has(.match-detail) .page-head{{display:none}}
#matchDetail{{padding:0;overflow:hidden;background:linear-gradient(180deg,#081925,#06131c);border-color:#16394d}}
.match-detail{{gap:0}}
.intel-hero{{position:relative;padding:16px 18px 16px;border-bottom:1px solid #15384b;background:
radial-gradient(circle at 50% -25%,rgba(40,129,166,.22),transparent 44%),linear-gradient(180deg,#071b27,#06151f)}}
.match-brand-row{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}}
.match-brand{{display:flex;align-items:center;gap:7px}}.match-brand>span{{width:27px;height:27px;border-radius:8px;display:grid;place-items:center;background:#0b5e9b;color:white;font-size:9px;font-weight:950;box-shadow:0 0 18px #0e6ba455}}
.match-brand b{{font-size:8px;line-height:.9;letter-spacing:.02em}}.match-context-row{{display:flex;justify-content:space-between;gap:10px;color:#7895a5;font-size:7px;margin-bottom:10px}}
.match-context-row span{{font-weight:800;color:#a5bac5}}.match-context-row b{{font-size:7px;color:#829baa}}
.intel-hero-top{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px}}
.intel-back{{display:inline-flex;align-items:center;gap:6px;border:0;background:none;color:#88a8b9;font-weight:800;font-size:9px;padding:0}}
.intel-back:hover{{color:#fff}}.intel-meta{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;justify-content:flex-end}}
.intel-kicker{{font-size:8px;text-transform:uppercase;letter-spacing:.12em;color:#6f91a3;font-weight:900}}
.intel-matchup{{display:grid;grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr);align-items:center;gap:14px;max-width:760px;margin:0 auto}}
.intel-team{{display:flex;align-items:center;gap:14px;min-width:0}}.intel-team.away{{justify-content:flex-end;text-align:right}}
.intel-team .crest{{width:64px;height:64px;flex:0 0 64px}}.intel-team-name{{min-width:0}}.intel-team-name b{{display:block;font-size:20px;line-height:1.08;letter-spacing:-.025em;overflow-wrap:anywhere}}
.intel-team-name span{{display:block;margin-top:5px;color:#6e8a9b;font-size:9px}}.intel-vs{{text-align:center;color:#577487;font-weight:950;font-size:10px}}
.intel-scoreline{{text-align:center;margin-top:16px;color:#7896a7;font-size:9px}}
.intel-summary{{padding:8px 12px 9px;background:#06141d;border-bottom:1px solid #143347}}
.intel-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}}
.intel-metric{{min-width:0;border:1px solid #15394d;background:#081a25;border-radius:8px;padding:8px 9px}}
.intel-metric span{{display:block;color:#668495;font-size:7px;text-transform:uppercase;letter-spacing:.08em;font-weight:850}}
.intel-metric b{{display:block;margin-top:3px;font-size:11px;line-height:1.12;overflow-wrap:anywhere}}.intel-metric.good b{{color:var(--green)}}.intel-metric.warn b{{color:var(--amber)}}.intel-metric.info b{{color:var(--blue)}}
.intel-lens{{margin-top:10px;border:1px solid #1a4358;border-radius:11px;background:linear-gradient(180deg,#0a2030,#081923);padding:13px 14px;display:grid;grid-template-columns:auto minmax(0,1fr);gap:11px;align-items:start}}
.intel-lens-mark{{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;background:#0c332a;border:1px solid #23654f;color:#72e8bc;font-size:12px;font-weight:950}}
.intel-lens b{{display:block;font-size:10px}}.intel-lens p{{margin:4px 0 0;color:#8aa4b3;font-size:9px;line-height:1.5}}
.intel-tabs{{display:flex;gap:6px;padding:10px 14px;border-bottom:1px solid #143447;background:#071721;overflow-x:auto;scrollbar-width:none;position:sticky;top:0;z-index:3}}
.intel-tabs::-webkit-scrollbar{{display:none}}.intel-tab{{flex:0 0 auto;border:1px solid transparent;background:transparent;color:#708d9e;border-radius:7px;padding:7px 10px;font-size:8px;font-weight:850}}
.intel-tab.active{{background:#0c2b3d;border-color:#21516b;color:#eaf5fa}}
.intel-body{{padding:14px 16px 18px}}.intel-section{{display:none}}.intel-section.active{{display:block}}
.intel-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.intel-card{{border:1px solid #15384b;background:#081a25;border-radius:11px;padding:13px;min-width:0}}
.intel-card h3{{margin:0 0 9px;font-size:10px;letter-spacing:.005em}}.intel-card h4{{margin:12px 0 6px;font-size:8px;color:#7491a2;text-transform:uppercase;letter-spacing:.08em}}
.intel-pills{{display:flex;flex-wrap:wrap;gap:6px}}.intel-pill{{display:inline-flex;align-items:center;max-width:100%;border:1px solid #20465b;background:#0a2230;color:#abc0cb;border-radius:999px;padding:6px 9px;font-size:8px;line-height:1.1}}
.intel-pill.market{{border-color:#285978;background:#0b2638;color:#8dccf2}}.intel-pill.missing{{border-color:#674f22;background:#2a210d;color:#e0b861}}.intel-pill.good{{border-color:#23664f;background:#0a2b22;color:#64dfb2}}
.intel-kv{{display:grid;gap:0}}.intel-kv-row{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;padding:9px 0;border-bottom:1px solid #102d3d;align-items:start}}
.intel-kv-row:last-child{{border-bottom:0}}.intel-kv-row span{{color:#728e9f;font-size:8px}}.intel-kv-row b{{font-size:9px;text-align:right;max-width:58vw;overflow-wrap:anywhere}}
.intel-section-title{{display:flex;justify-content:space-between;gap:12px;align-items:end;margin:2px 0 9px}}.intel-section-title h3{{margin:0;font-size:12px}}.intel-section-title span{{color:#688596;font-size:8px}}
.intel-accordion{{display:grid;gap:8px}}.intel-accordion details{{border:1px solid #15384b;background:#081a25;border-radius:10px;overflow:hidden}}
.intel-accordion summary{{list-style:none;cursor:pointer;padding:12px 13px;display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:9px;font-weight:850}}
.intel-accordion summary::-webkit-details-marker{{display:none}}.intel-accordion summary::after{{content:'+';color:#6b8b9d;font-size:14px;font-weight:500}}.intel-accordion details[open] summary::after{{content:'–'}}
.intel-accordion-body{{border-top:1px solid #133448;padding:10px 12px;overflow:hidden}}
.intel-snapshot-list{{display:grid;gap:7px}}.intel-snapshot{{border:1px solid #143548;background:#071721;border-radius:8px;padding:10px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:start}}
.intel-snapshot b{{font-size:9px}}.intel-snapshot small{{display:block;color:#708d9e;font-size:7px;margin-top:3px;line-height:1.4}}.intel-snapshot .snap-right{{text-align:right;color:#a7bbc6;font-size:8px}}
.intel-empty{{border:1px dashed #25485b;border-radius:9px;padding:18px;text-align:center;color:#708b9c;font-size:8px;background:#071720}}
.intel-footnote{{margin-top:10px;color:#657f90;font-size:8px;line-height:1.5}}
.visual-grid{{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(220px,.8fr);gap:10px;margin-bottom:10px}}
.visual-card{{border:1px solid #173d52;background:linear-gradient(180deg,#0a1f2c,#071720);border-radius:12px;padding:14px;min-width:0;overflow:hidden}}
.visual-card-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:12px}}
.visual-card-head h3{{margin:0;font-size:11px}}.visual-card-head span{{font-size:7px;color:#6d899a;text-transform:uppercase;letter-spacing:.08em;font-weight:850;text-align:right}}
.rate-bars{{display:grid;gap:11px}}.rate-row{{display:grid;grid-template-columns:88px minmax(0,1fr) 42px;gap:8px;align-items:center}}
.rate-row label{{font-size:8px;color:#8ca5b3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.rate-track{{height:8px;border-radius:99px;background:#102b3b;overflow:hidden;box-shadow:inset 0 0 0 1px #17384a}}
.rate-fill{{height:100%;min-width:2px;border-radius:99px;background:linear-gradient(90deg,#2389bd,#55bfe9)}}
.rate-fill.away{{background:linear-gradient(90deg,#20715c,#53d3a8)}}.rate-fill.total{{background:linear-gradient(90deg,#7a652f,#d7aa46)}}
.rate-row b{{font-size:10px;text-align:right}}.rate-source{{margin-top:10px;color:#678495;font-size:7px;line-height:1.45}}
.coverage-wrap{{display:grid;grid-template-columns:86px minmax(0,1fr);gap:13px;align-items:center}}
.coverage-gauge{{--coverage:0%;width:82px;height:82px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--green) var(--coverage),#123144 0);position:relative}}
.coverage-gauge::after{{content:'';position:absolute;inset:8px;border-radius:50%;background:#081a25;border:1px solid #173c50}}
.coverage-gauge div{{position:relative;z-index:1;text-align:center}}.coverage-gauge b{{display:block;font-size:18px}}.coverage-gauge span{{display:block;font-size:6px;color:#7794a4;text-transform:uppercase;letter-spacing:.07em}}
.coverage-copy b{{display:block;font-size:10px}}.coverage-copy p{{margin:4px 0 0;color:#7895a5;font-size:8px;line-height:1.5}}
.form-compare{{display:grid;gap:10px}}.form-team{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center}}
.form-team b{{font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.form-dots{{display:flex;gap:4px;justify-content:flex-end}}
.form-dot{{width:21px;height:21px;border-radius:6px;display:grid;place-items:center;font-size:7px;font-weight:950;border:1px solid #27475a;background:#102633;color:#8aa4b3}}
.form-dot.w{{border-color:#25654f;background:#0c2d24;color:#62dfb1}}.form-dot.d{{border-color:#6b5728;background:#2c240f;color:#e0b85f}}.form-dot.l{{border-color:#66363b;background:#2b161b;color:#ee8892}}
.prob-tiles{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}}.prob-tile{{padding:11px 8px;border:1px solid #173a4e;border-radius:10px;background:#081b27;text-align:center}}
.prob-tile b{{display:block;font-size:18px}}.prob-tile span{{display:block;margin-top:3px;color:#728f9f;font-size:7px;text-transform:uppercase;letter-spacing:.07em}}
.prob-stack{{height:9px;margin-top:10px;border-radius:99px;overflow:hidden;display:flex;background:#112d3d}}.prob-stack i{{display:block;height:100%}}.prob-stack .home{{background:#42ce9c}}.prob-stack .draw{{background:#d2aa4f}}.prob-stack .away{{background:#45a7dc}}
.compare-bars{{display:grid;gap:10px}}.compare-line{{display:grid;grid-template-columns:minmax(0,1fr) 44px;gap:9px;align-items:center}}
.compare-meta{{display:flex;justify-content:space-between;gap:8px;font-size:8px;margin-bottom:5px}}.compare-meta span{{color:#7e9bab}}.compare-meta b{{font-size:9px}}
.compare-track{{height:7px;border-radius:99px;background:#112c3b;overflow:hidden}}.compare-fill{{height:100%;border-radius:99px;background:linear-gradient(90deg,#2c8ab7,#58c3ec)}}
.takeaway-list{{display:grid;gap:7px}}.takeaway{{display:grid;grid-template-columns:24px minmax(0,1fr);gap:9px;padding:9px 10px;border:1px solid #15384b;background:#071923;border-radius:9px}}
.takeaway-num{{width:24px;height:24px;border-radius:7px;display:grid;place-items:center;background:#0b2e3e;color:#68d9b1;font-size:8px;font-weight:950}}
.takeaway b{{display:block;font-size:9px}}.takeaway small{{display:block;margin-top:3px;color:#718d9d;font-size:7px;line-height:1.4}}
.sport-stat-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.sport-stat{{border:1px solid #15384b;background:#071923;border-radius:9px;padding:10px}}
.sport-stat span{{display:block;color:#718e9e;font-size:7px;line-height:1.3}}.sport-stat b{{display:block;margin-top:4px;font-size:13px}}.sport-stat small{{display:block;margin-top:3px;color:#607d8e;font-size:6px}}
.goal-dist{{display:grid;gap:8px}}.goal-dist-legend{{display:flex;gap:10px;flex-wrap:wrap;color:#7894a4;font-size:6.5px}}.goal-dist-legend span{{display:flex;align-items:center;gap:4px}}.goal-dist-legend i{{width:6px;height:6px;border-radius:50%;display:inline-block}}.goal-dist-legend .h,.goal-dist-pair .h{{background:#42aee2}}.goal-dist-legend .a,.goal-dist-pair .a{{background:#4bd39f}}
.goal-dist-bars{{height:98px;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;align-items:end;border-bottom:1px solid #17384a;padding:6px 2px 0}}.goal-dist-col{{height:100%;display:grid;grid-template-rows:1fr auto;gap:4px;align-items:end;text-align:center}}.goal-dist-col b{{font-size:6.5px;color:#7793a3}}.goal-dist-pair{{height:100%;display:flex;align-items:flex-end;justify-content:center;gap:2px}}.goal-dist-pair i{{display:block;width:38%;min-height:2px;border-radius:3px 3px 0 0}}
.sport-top-card{{margin-bottom:8px}}.sport-two-up{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:8px}}.sport-two-up .visual-card{{padding:10px;min-height:165px}}.sport-profile-card{{margin-bottom:8px}}
@media(max-width:420px){{.sport-two-up{{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}}.sport-two-up .visual-card{{padding:8px;min-height:152px}}.score-grid{{gap:4px}}.score-chip{{padding:6px 3px}}.score-chip b{{font-size:9px}}.score-chip small{{font-size:5.5px}}.goal-dist-bars{{height:82px;gap:3px}}}}
.visual-empty{{min-height:118px;display:grid;place-items:center;text-align:center;border:1px dashed #24495d;border-radius:10px;background:#071720;color:#738f9f;font-size:8px;line-height:1.5;padding:18px}}
.visual-empty.compact{{min-height:72px;padding:11px}}
.overview-feature{{position:relative;border-left:3px solid #36d7a0;background:linear-gradient(135deg,#0b2430,#081923 55%)}}
.overview-feature::before{{content:'▥';position:absolute;right:14px;top:12px;color:#3ad9a4;font-size:18px;opacity:.85}}
.mini-visual-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:8px}}
.mini-visual-grid .visual-card{{padding:11px;min-height:154px}}
.mini-visual-grid .visual-card-head{{margin-bottom:8px}}.mini-visual-grid .visual-card-head h3{{font-size:9px}}
.goal-card{{padding:12px 13px;margin-bottom:8px}}.goal-card .visual-card-head{{margin-bottom:9px}}
.goal-card .rate-row{{grid-template-columns:82px minmax(0,1fr) 38px;gap:7px}}.goal-card .rate-track{{height:7px}}
.form-chart-lines{{height:58px;border-left:1px solid #17384a;border-bottom:1px solid #17384a;background:repeating-linear-gradient(to top,transparent 0,transparent 18px,#102d3c 19px);position:relative;margin-bottom:8px}}
.form-spark{{position:absolute;inset:4px 4px 2px;width:calc(100% - 8px);height:calc(100% - 6px);color:#42aee2}}.form-spark.away{{color:#48d19f;opacity:.9}}
.form-compare{{gap:6px}}.form-team{{grid-template-columns:minmax(0,1fr) auto}}.form-team b{{font-size:7px}}
.form-dot{{width:18px;height:18px;border-radius:5px;font-size:6px}}
.takeaway{{padding:7px 8px;grid-template-columns:21px minmax(0,1fr);gap:7px}}.takeaway-num{{width:21px;height:21px;border-radius:6px;font-size:7px}}
.takeaway b{{font-size:8px}}.takeaway small{{font-size:6.5px}}
.compact-inputs .intel-snapshot{{padding:8px}}.compact-inputs .intel-snapshot b{{font-size:8px}}.compact-inputs .intel-snapshot small{{font-size:6px}}
@media(max-width:700px){{
 .visual-grid{{grid-template-columns:1fr;gap:8px}}.visual-card{{padding:11px}}
 .mini-visual-grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}}
 .mini-visual-grid .visual-card{{min-height:142px;padding:9px}}
 .rate-row{{grid-template-columns:68px minmax(0,1fr) 34px}}.coverage-wrap{{grid-template-columns:1fr;justify-items:center;text-align:center;gap:7px}}.coverage-gauge{{width:68px;height:68px}}
 .coverage-copy b{{font-size:8px}}.coverage-copy p{{font-size:6.5px;margin-top:2px}}
 .prob-tile b{{font-size:16px}}.sport-stat-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}
 .goal-card .rate-row{{grid-template-columns:68px minmax(0,1fr) 34px}}
 body:has(#matches.active .match-detail) .topbar{{display:none}}
 body:has(#matches.active .match-detail) .main{{padding-top:calc(10px + env(safe-area-inset-top))}}
 .intel-hero{{border-radius:12px 12px 0 0}}.intel-summary{{padding-top:9px}}
}}
@media(max-width:700px){{
 #matchDetail{{margin-left:-1px;margin-right:-1px;border-radius:11px}}
 .intel-hero{{padding:11px 10px 12px}}.intel-hero-top{{margin-bottom:10px}}
 .intel-matchup{{grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr);gap:7px}}
 .intel-team{{display:grid;justify-items:center;gap:7px;text-align:center}}.intel-team.away{{justify-content:initial;text-align:center}}
 .intel-team .crest{{width:46px;height:46px;flex-basis:46px}}.intel-team-name b{{font-size:13px}}.intel-team-name span{{font-size:7px;margin-top:3px}}
 .intel-summary{{padding:7px 8px 8px}}.intel-metrics{{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}}.intel-metric{{flex:1 0 92px;padding:7px 8px}}.intel-metric b{{font-size:10px}}
 .intel-lens{{grid-template-columns:24px minmax(0,1fr);padding:11px;gap:9px}}.intel-lens-mark{{width:24px;height:24px}}
 .intel-tabs{{margin:0;padding:6px 8px;top:0}}.intel-tab{{padding:6px 9px;font-size:7px}}
 .intel-body{{padding:8px 8px 12px}}.intel-grid{{grid-template-columns:1fr;gap:8px}}.intel-card{{padding:10px}}
 .intel-kv-row{{grid-template-columns:minmax(0,1fr) minmax(90px,45%)}}.intel-kv-row b{{max-width:none}}
 .intel-pills{{gap:5px}}.intel-pill{{font-size:7px;padding:5px 7px}}
 .table-wrap{{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}}.table-wrap table{{min-width:640px}}
}}
.account-page{{max-width:760px}}.account-card{{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center}}.account-card h2{{margin:0 0 4px;font-size:18px}}.account-card p{{margin:0;color:#7893a3;font-size:9px;line-height:1.5}}
.modal{{position:fixed;inset:0;display:none;place-items:center;padding:18px;background:#02070bcc;backdrop-filter:blur(8px);z-index:100}}.modal.open{{display:grid}}.modal-card{{width:min(450px,100%);border:1px solid #21485f;
background:linear-gradient(180deg,#0a1f2d,#07131c);border-radius:15px;padding:20px;box-shadow:0 30px 100px #000d}}.modal-card h2{{margin:0 0 4px;font-size:20px}}.modal-card p{{color:#7893a3;font-size:9px;line-height:1.5}}
.modal-card input{{width:100%;padding:10px;margin:7px 0 0;border-radius:8px;border:1px solid #1a3d52;background:#061620;color:#eef7fb}}.modal-actions{{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}}.modal-status{{min-height:15px;color:var(--amber);font-size:9px;margin-top:8px}}
.boot{{position:fixed;inset:0;z-index:200;display:grid;place-items:center;background:radial-gradient(circle at 50% 18%,#103148,#06111a 40%,#050b11 78%);padding:20px}}.boot-card{{width:min(400px,calc(100vw - 30px));text-align:center;border:1px solid #1c435a;
background:#081b27;border-radius:16px;padding:28px;box-shadow:0 30px 100px #000c}}.boot-card .brand-mark{{margin:0 auto 13px}}.boot-title{{font-size:18px;font-weight:950}}.boot-copy{{color:#7792a2;font-size:9px;margin-top:6px;line-height:1.5}}.boot-line{{height:3px;border-radius:99px;overflow:hidden;background:#0f2b3b;margin-top:18px}}.boot-line i{{display:block;width:38%;height:100%;background:var(--green);animation:load 1.05s ease-in-out infinite alternate}}@keyframes load{{from{{transform:translateX(-20%)}}to{{transform:translateX(185%)}}}}
.mobile-nav{{display:none}}
@media(max-width:1050px){{
 .shell{{grid-template-columns:1fr}}.side{{display:none}}.main{{padding:calc(62px + var(--safe-top)) 13px calc(82px + var(--safe-bottom));overflow-x:hidden}}.topbar{{position:fixed;top:0;left:0;right:0;height:calc(52px + env(safe-area-inset-top));padding:var(--safe-top) 12px 8px;background:#06131de8;backdrop-filter:blur(12px);border-bottom:1px solid #123247;z-index:50;margin:0}}
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
/* Sport-first landing: preserve legacy expert surfaces in other pages but
   do not make an empty market laboratory the homepage. */
#today > :not(.page-head):not(.kpis):not(#sportFocus){{display:none!important}}
#sportFocus{{padding:24px;background:linear-gradient(145deg,#0b2432,#071722);border:1px solid #27516b;box-shadow:none}}
.focus-heading{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}}
.focus-heading h2{{font-size:25px;margin:4px 0 6px;letter-spacing:-.02em}}
.focus-heading p,.focus-subhead p{{color:#9eb7c6;font-size:13px;line-height:1.5;margin:0}}
.focus-eyebrow{{color:#4de1ad;font-size:10px;letter-spacing:.12em;font-weight:900}}
.focus-status{{margin:18px 0 14px;background:#0e3444;border-left:3px solid #5dc3df;padding:14px 16px;color:#cfe8f2;border-radius:6px;font-size:13px;line-height:1.5}}
.focus-forecast-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:14px 0}}
.focus-card{{border:1px solid #29536a;background:#0a202f;border-radius:12px;padding:18px}}
.focus-card h3{{font-size:16px;line-height:1.4;margin:8px 0 5px}}
.focus-card .focus-prob{{font-size:32px;font-weight:850;color:#fff;letter-spacing:-.03em;margin-top:14px}}
.focus-card p{{font-size:12px;line-height:1.5;color:#a9c2d0;margin:8px 0}}
.focus-card .focus-tag{{font-size:10px;font-weight:850;color:#8fdac7;letter-spacing:.03em}}
.focus-card.past .focus-tag{{color:#dfbc75}}
.focus-forecast-grid .focus-empty{{grid-column:1/-1;padding:20px;border:1px dashed #2b566c;border-radius:11px;font-size:13px;color:#b2ccda;line-height:1.7}}
.focus-forecast-grid .focus-empty-compact{{padding:12px 14px;line-height:1.55}}
.focus-subhead{{border-top:1px solid #284a5b;margin-top:23px;padding-top:20px}}
.focus-subhead h3{{font-size:20px;margin:0 0 6px}}
.focus-filters{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 8px}}
.focus-filters button{{background:#0a2a3a;color:#b2d0df;border:1px solid #275369;border-radius:8px;font-size:12px;font-weight:800;padding:8px 12px}}
.focus-filters button.active{{background:#145271;color:#fff;border-color:#4db5e0}}
.focus-fixture{{display:flex;justify-content:space-between;align-items:center;gap:16px;border-bottom:1px solid #204458;padding:14px 0;color:inherit;text-decoration:none}}
.focus-fixture strong{{display:block;font-size:14px;line-height:1.4}}
.focus-fixture span{{display:block;font-size:11px;color:#98b8c8;margin-top:5px;line-height:1.4}}
.focus-fixture .focus-state{{white-space:nowrap;font-size:11px;color:#b8d5e3;border:1px solid #28506a;border-radius:7px;padding:7px 10px}}
.focus-honesty{{margin-top:18px;font-size:12px;color:#94b0c1;line-height:1.6}}
.focus-disclosure{{border-top:1px solid #244559;padding-top:13px;margin-top:15px;color:#a9c5d5;font-size:12px}}
.focus-disclosure summary{{cursor:pointer;font-size:12px;font-weight:800;color:#a9daec}}
@media(max-width:800px){{#sportFocus{{padding:16px}}.focus-heading{{flex-direction:column}}.focus-heading h2{{font-size:22px}}.focus-forecast-grid{{grid-template-columns:1fr}}.focus-card .focus-prob{{font-size:28px}}.kpi span{{font-size:10px}}}}
.focus-prob-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:14px}}
.focus-prob-strip.submarkets{{grid-template-columns:repeat(2,1fr);margin-top:7px}}
.focus-metric{{border:1px solid #2b5367;border-radius:9px;padding:10px 8px;background:#0c2a39}}
.focus-metric small{{display:block;font-size:10px;color:#a3c5d4;line-height:1.3}}
.focus-metric b{{display:block;font-size:22px;color:#eaf9ff;margin-top:5px}}
@media(max-width:680px){{.focus-metric b{{font-size:18px}}.focus-metric small{{font-size:9px}}}}
.nav button{{font-size:12px}}
.context b{{font-size:13px}}
.context span{{font-size:11px}}
.kpi span{{font-size:10px}}
@media(max-width:1050px){{.mobile-nav button{{font-size:11px;min-width:75px}}}}

/* Accessible Match Intelligence: forecast-first rather than technical dashboard. */
#matchDetail .intel-metric{{padding:12px 13px}}
#matchDetail .intel-metric span{{font-size:11px;color:#a9c5d4;line-height:1.35}}
#matchDetail .intel-metric b{{font-size:14px;line-height:1.35;margin-top:5px}}
#matchDetail .intel-tab{{font-size:12px;padding:11px 14px}}
#matchDetail .intel-body{{padding:20px 22px 28px}}
#matchDetail .match-reading-banner{{border-left:4px solid #4de1ad;background:#0b2532;border-radius:10px;padding:18px 20px;margin-bottom:14px}}
#matchDetail .match-reading-banner span{{font-size:11px;color:#89d9bb;font-weight:850;letter-spacing:.07em}}
#matchDetail .match-reading-banner h2{{font-size:23px;line-height:1.2;margin:8px 0}}
#matchDetail .match-reading-banner p{{font-size:14px;line-height:1.55;margin:0;color:#c3d9e5}}
#matchDetail .overview-primary-grid{{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:12px;margin-bottom:12px}}
#matchDetail .overview-primary-grid>.visual-card{{min-height:220px}}
#matchDetail .visual-card{{padding:17px}}
#matchDetail .visual-card-head h3{{font-size:16px;line-height:1.3}}
#matchDetail .visual-card-head span{{font-size:11px;color:#a6c2d1;text-transform:none;letter-spacing:0;line-height:1.4}}
#matchDetail .prob-tile b{{font-size:27px}}
#matchDetail .prob-tile span{{font-size:12px;color:#b3cbd9;text-transform:none;letter-spacing:0}}
#matchDetail .overview-explainer{{font-size:12px;line-height:1.5;color:#a4bbc9;margin:14px 0 0}}
#matchDetail .takeaway{{padding:12px;grid-template-columns:26px minmax(0,1fr);gap:10px}}
#matchDetail .takeaway-num{{width:26px;height:26px;font-size:12px}}
#matchDetail .takeaway b{{font-size:13px}}
#matchDetail .takeaway small{{font-size:12px;color:#afc6d3;line-height:1.5}}
#matchDetail .rate-row{{grid-template-columns:minmax(100px,160px) minmax(0,1fr) 50px;gap:10px}}
#matchDetail .rate-row label{{font-size:12px;color:#bad1df}}
#matchDetail .rate-row b{{font-size:13px}}
#matchDetail .rate-source{{font-size:11px;color:#aac1cf;line-height:1.55}}
#matchDetail .overview-assurance{{display:flex;gap:12px;padding:13px 16px;margin-bottom:13px;background:#0a2030;border:1px solid #25516a;border-radius:10px}}
#matchDetail .overview-assurance b{{flex:0 0 auto;font-size:12px;color:#8bd4b9}}
#matchDetail .overview-assurance span{{font-size:12px;line-height:1.5;color:#bcd0dc}}
#matchDetail .intel-accordion summary{{font-size:13px;padding:15px 16px;color:#e3f0f7;line-height:1.45}}
#matchDetail .intel-accordion summary span{{font-size:11px;color:#9db8c9}}
#matchDetail .intel-accordion-body{{padding:15px}}
#matchDetail .mini-visual-grid .visual-card{{min-height:0}}
#matchDetail .coverage-copy b{{font-size:13px}}
#matchDetail .coverage-copy p{{font-size:11px;color:#a0b9c8;line-height:1.5}}
#matchDetail .coverage-gauge span{{font-size:9px}}
#matchDetail .form-team b{{font-size:12px}}
#matchDetail .form-dot{{width:27px;height:27px;font-size:11px}}
#matchDetail .form-legend{{font-size:11px;color:#adc3d1;line-height:1.5;margin-top:12px}}
#matchDetail .intel-snapshot b{{font-size:12px}}
#matchDetail .intel-snapshot small{{font-size:10px;color:#a3bbca}}
#matchDetail .intel-snapshot .snap-right{{font-size:11px}}
#matchDetail .intel-kv-row span,#matchDetail .intel-kv-row b{{font-size:12px;line-height:1.45}}
#matchDetail .visual-empty{{font-size:13px;min-height:100px;color:#b9ccda}}
#matchDetail .intel-pill,#matchDetail .intel-footnote{{font-size:11px;line-height:1.5}}
@media(max-width:800px){{#matchDetail .overview-primary-grid{{grid-template-columns:1fr}}}}
@media(max-width:700px){{
 #matchDetail .intel-body{{padding:12px}}
 #matchDetail .match-reading-banner{{padding:14px}}
 #matchDetail .match-reading-banner h2{{font-size:20px}}
 #matchDetail .match-reading-banner p{{font-size:13px}}
 #matchDetail .intel-tab{{font-size:12px;padding:10px 11px}}
 #matchDetail .intel-metric{{flex:1 0 125px;padding:10px}}
 #matchDetail .intel-metric span{{font-size:10px}}
 #matchDetail .intel-metric b{{font-size:12px}}
 #matchDetail .mini-visual-grid,#matchDetail .sport-two-up{{grid-template-columns:1fr}}
 #matchDetail .overview-assurance{{display:block}}
 #matchDetail .overview-assurance span{{display:block;margin-top:5px}}
 #matchDetail .visual-card-head{{display:block}}
 #matchDetail .visual-card-head span{{display:block;text-align:left;margin-top:4px}}
 #matchDetail .rate-row{{grid-template-columns:80px minmax(0,1fr) 44px;gap:7px}}
 #matchDetail .rate-row label{{font-size:11px}}
}}

/* A fixture always has an independent, accessible sporting reading. */
#matchDetail .universal-match{{padding:20px;display:grid;gap:14px}}
#matchDetail .public-match-heading{{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}}
#matchDetail .public-match-heading span{{font-size:11px;color:#8bd4b9;letter-spacing:.07em;font-weight:750}}
#matchDetail .universal-match h2{{font-size:24px;line-height:1.2;margin:0}}
#matchDetail .universal-match p{{font-size:13px;line-height:1.55;color:#b8d1de;margin:0}}
#matchDetail .universal-status-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
#matchDetail .universal-status-grid strong{{font-size:18px;color:#e8f5f8}}
#matchDetail .universal-missing{{border:1px solid #23495f;border-radius:9px;padding:11px 15px}}
#matchDetail .universal-missing summary{{cursor:pointer;font-size:13px;font-weight:650}}
#matchDetail .universal-missing-body{{padding:12px 0;font-size:12px;color:#b6cedc;line-height:1.6}}
@media(max-width:700px){{#matchDetail .universal-match{{padding:12px}}#matchDetail .universal-status-grid{{grid-template-columns:1fr}}#matchDetail .universal-match h2{{font-size:20px}}}}

#matchDetail .analyst-notebook{{margin-top:15px;padding:16px;border-radius:12px;border:1px solid #244e64;background:#0a2030;display:grid;gap:12px}}
#matchDetail .analyst-notebook p{{font-size:12px;line-height:1.5;color:#b5cad8;margin:0}}
#matchDetail .analyst-notebook textarea{{width:100%;min-height:115px;box-sizing:border-box;padding:12px;border:1px solid #326078;border-radius:8px;background:#081b28;color:#e8f5f8;font:inherit;font-size:14px;resize:vertical}}
#matchDetail .analyst-note-actions{{display:flex;gap:9px;align-items:center;flex-wrap:wrap}}
#matchDetail #analystNoteStatus{{font-size:11px;color:#aec5d3;line-height:1.5}}

.focus-queue-summary{{display:grid;gap:6px;margin-top:12px;padding:13px 15px;border:1px solid #295068;border-radius:11px;background:#091c2b}}
.focus-queue-summary strong{{font-size:14px;line-height:1.4;color:#e4f1f7}}
.focus-queue-summary span{{font-size:13px;line-height:1.5;color:#bad3df}}
.focus-queue-summary small{{font-size:11px;line-height:1.5;color:#9fbacb}}

/* Mobile match intelligence: no horizontal body overflow and no empty dashboard walls. */
#matchDetail,#matchDetail .match-detail,#matchDetail .intel-hero,
#matchDetail .intel-body,#matchDetail .intel-summary,#matchDetail .visual-card,
#matchDetail .match-event-block,#matchDetail .match-event-alert,
#matchDetail .match-event-facts,#matchDetail .analyst-notebook{{
  box-sizing:border-box;min-width:0;max-width:100%;
}}
#matchDetail .match-event-block{{padding:14px;display:grid;gap:14px}}
#matchDetail .match-event-alert{{border:1px solid #916c39;border-left:5px solid #f0b75c;
  background:#2b231b;border-radius:12px;padding:20px;display:grid;gap:10px}}
#matchDetail .match-event-alert span{{font-size:12px;font-weight:800;color:#ffc16c;letter-spacing:.04em}}
#matchDetail .match-event-alert h2{{font-size:23px;line-height:1.2;margin:0;color:#ffe3c0}}
#matchDetail .match-event-alert p,#matchDetail .match-event-facts p{{
  font-size:14px;line-height:1.55;color:#d2dfe8;margin:0;overflow-wrap:anywhere}}
#matchDetail .match-event-facts{{border:1px solid #284a5d;border-radius:12px;padding:18px;display:grid;gap:12px}}
#matchDetail .match-event-facts h3{{font-size:17px;margin:0}}
#matchDetail .status-source{{font-size:12px!important;color:#b5c7d5!important;margin-top:6px!important}}
#matchDetail .verified-history{{border:1px solid #244b63;border-radius:10px;padding:10px}}
#matchDetail .verified-history summary{{font-size:13px;font-weight:700;cursor:pointer}}
#matchDetail .intel-tabs{{max-width:100%;box-sizing:border-box}}
#matchDetail .intel-section,#matchDetail .overview-primary-grid,#matchDetail .sport-two-up{{
  min-width:0;max-width:100%}}
@media(max-width:700px){{
  #matchDetail .intel-metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  #matchDetail .intel-matchup{{grid-template-columns:minmax(0,1fr) 22px minmax(0,1fr);gap:5px}}
  #matchDetail .intel-team{{flex-direction:column;align-items:center;gap:5px;text-align:center}}
  #matchDetail .intel-team.away{{flex-direction:column-reverse;align-items:center;text-align:center}}
  #matchDetail .intel-team .crest{{width:52px;height:52px;flex:0 0 52px}}
  #matchDetail .intel-team-name b{{font-size:15px;line-height:1.2}}
  #matchDetail .intel-scoreline{{font-size:11px;overflow-wrap:anywhere}}
  #matchDetail .intel-summary{{padding:10px}}
  #matchDetail .universal-status-grid{{grid-template-columns:minmax(0,1fr)}}
  #matchDetail .visual-grid,#matchDetail .overview-primary-grid,#matchDetail .sport-two-up{{
    grid-template-columns:minmax(0,1fr)}}
  #matchDetail .visual-card{{min-height:0!important;padding:14px}}
  #matchDetail .intel-body{{padding:12px}}
  #matchDetail .intel-tabs{{overflow-x:auto;overscroll-behavior-x:contain}}
  #matchDetail .intel-tab{{white-space:nowrap}}
  #matchDetail .analyst-notebook{{padding:14px}}
}}

#matchDetail .season-facts-owner{{margin:10px 0 16px;border:1px solid #2a5765;background:#0a2431;border-radius:12px;padding:16px;display:grid;gap:13px;min-width:0}}
#matchDetail .season-facts-head{{display:flex;justify-content:space-between;align-items:start;gap:10px;flex-wrap:wrap}}
#matchDetail .season-facts-head h3{{margin:0;font-size:19px}}
#matchDetail .season-facts-head span{{font-size:10px;color:#80dfb1;font-weight:800;letter-spacing:.05em}}
#matchDetail .season-facts-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
#matchDetail .season-fact-card{{padding:13px;background:#091b27;border:1px solid #295069;border-radius:10px;min-width:0}}
#matchDetail .season-fact-card h4{{font-size:17px;margin:0 0 6px}}
#matchDetail .season-fact-card p,#matchDetail .season-facts-owner>p{{font-size:12px;line-height:1.5;margin:0;color:#aac6d5}}
#matchDetail .season-fact-stats{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:12px 0}}
#matchDetail .season-fact-stats span{{display:flex;flex-direction:column;font-size:11px;color:#a5bfd0}}
#matchDetail .season-fact-stats b{{font-size:21px;color:#ebf3f9}}
#matchDetail .season-fact-card .season-split{{font-size:12px;line-height:1.5;color:#bed8e6}}
@media(max-width:700px){{#matchDetail .season-facts-grid{{grid-template-columns:minmax(0,1fr)}}
#matchDetail .season-facts-owner{{padding:12px}}
}}

/* Source-backed descriptive sport reading is first-class even when P0 is blocked. */
#matchDetail .verified-sport-read{{padding:16px;border:1px solid #277469;border-left:5px solid #20be99;background:#0a2832;border-radius:12px;display:grid;gap:13px;min-width:0;overflow-wrap:anywhere}}
#matchDetail .verified-sport-heading{{display:grid;gap:8px}}
#matchDetail .verified-sport-heading>span{{font-size:11px;color:#7fe1bf;letter-spacing:.035em;font-weight:800}}
#matchDetail .verified-sport-heading h2{{font-size:21px;line-height:1.25;margin:0;color:#e8f7f6}}
#matchDetail .verified-sport-heading p{{font-size:13px;line-height:1.55;margin:0;color:#b7d5db}}
#matchDetail .verified-sport-fact-list{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}
#matchDetail .sport-fact{{padding:13px;border:1px solid #22546b;border-radius:10px;background:#091e2c;display:flex;gap:10px;align-items:flex-start;min-width:0}}
#matchDetail .sport-fact>b{{font-size:12px;color:#89f2cd;flex:none}}
#matchDetail .sport-fact p{{font-size:13px;color:#e6f2f5;line-height:1.55;margin:0}}
#matchDetail .verified-sport-caveats{{padding:12px;background:#102936;border-radius:9px;display:grid;gap:6px}}
#matchDetail .verified-sport-caveats strong{{font-size:13px;color:#ffd89a}}
#matchDetail .verified-sport-caveats p{{font-size:12px;color:#bed0d9;line-height:1.5;margin:0}}
#matchDetail .verified-sport-source{{font-size:11px;color:#96bcc8;line-height:1.5}}
@media(max-width:700px){{#matchDetail .verified-sport-fact-list{{grid-template-columns:1fr}}#matchDetail .verified-sport-heading h2{{font-size:19px}}}}

/* Cross-competition recent results: verified descriptive facts, no odds or RAW model. */
#matchDetail .verified-recent-form{{padding:16px;border:1px solid #28647f;border-left:5px solid #4bafd1;background:#0b2332;border-radius:12px;display:grid;gap:14px;min-width:0;overflow-wrap:anywhere}}
#matchDetail .recent-form-teams{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
#matchDetail .recent-form-team{{padding:13px;border:1px solid #234e68;border-radius:10px;background:#081d2a;display:grid;gap:9px;min-width:0}}
#matchDetail .recent-form-team h3{{font-size:17px;line-height:1.3;margin:0;color:#e5f1f6}}
#matchDetail .recent-form-summary{{font-size:13px;font-weight:700;color:#96e7d5}}
#matchDetail .recent-form-rows{{display:grid;gap:5px}}
#matchDetail .recent-form-row{{display:grid;grid-template-columns:24px minmax(0,1fr) 52px;gap:8px;align-items:center;padding:8px;border-radius:7px;background:#112a39;font-size:11px;color:#d1e6eb}}
#matchDetail .recent-form-row strong{{font-size:12px;color:#abddf2}}
#matchDetail .recent-form-row span{{line-height:1.45;overflow-wrap:anywhere}}
#matchDetail .recent-form-row b{{text-align:right;color:#effbfb;font-size:13px}}
#matchDetail .recent-form-team>p{{font-size:12px;color:#acc7d4;line-height:1.5;margin:0}}
@media(max-width:700px){{#matchDetail .recent-form-teams{{grid-template-columns:1fr}}#matchDetail .verified-recent-form{{padding:12px}}}}

.focus-league-detail{{margin:10px 0 4px;padding:13px 15px;border:1px solid #2a5467;border-radius:10px;background:#0a1d2b}}
.focus-league-detail summary{{cursor:pointer;font-size:13px;color:#c2e4ed;font-weight:750}}
.focus-league-rows{{display:grid;gap:7px;max-height:440px;overflow:auto;margin:11px 0}}
.focus-league-row{{display:flex;flex-wrap:wrap;gap:4px 14px;justify-content:space-between;border-bottom:1px solid #244456;padding:8px 0}}
.focus-league-row strong{{font-size:12px;color:#e8f3f5}}
.focus-league-row span{{font-size:12px;line-height:1.5;color:#abc5d2}}
.focus-league-detail small{{font-size:11px;color:#a8bfcb;display:block;line-height:1.5}}

#matchDetail .owner-frozen-research{{border:1px solid #276658;border-left:5px solid #39c59b;border-radius:12px;padding:16px;display:grid;gap:13px;background:#0a2a31;min-width:0;overflow-wrap:anywhere}}
#matchDetail .owner-frozen-title{{display:grid;gap:6px}}
#matchDetail .owner-frozen-title span{{font-size:11px;color:#76e0bc;font-weight:800;letter-spacing:.045em}}
#matchDetail .owner-frozen-title h3{{font-size:21px;line-height:1.3;margin:0}}
#matchDetail .owner-frozen-title p{{font-size:13px;line-height:1.5;margin:0;color:#b8d6df}}
#matchDetail .owner-frozen-probs{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}}
#matchDetail .owner-frozen-probs>div{{border:1px solid #2e5d68;border-radius:9px;padding:13px 7px;text-align:center;display:grid;gap:5px}}
#matchDetail .owner-frozen-probs small{{font-size:11px;line-height:1.35;color:#b0ced6}}
#matchDetail .owner-frozen-probs b{{font-size:23px;color:#edfbf5}}
#matchDetail .owner-frozen-extra{{font-size:13px;font-weight:700;color:#ddf5eb}}
#matchDetail .owner-frozen-source{{font-size:11px;line-height:1.5;color:#a8c8d1;margin:0}}
@media(max-width:700px){{#matchDetail .owner-frozen-research{{padding:12px}}#matchDetail .owner-frozen-probs b{{font-size:19px}}}}
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
  <div class="brand"><div class="brand-mark">SE</div><div class="brand-copy"><b>Soccer Edge</b><span>FÚTBOL PRIMERO</span></div></div>
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
    <div class="context"><b>Soccer Edge · Análisis</b><span id="snapshotText">Waiting for persisted snapshot</span></div>
    <div class="account-dock"><span id="freshChip" class="fresh-chip">VERIFYING</span><span id="planChip" class="plan-chip">EXPLORER</span><button id="accountBtn" class="icon-btn">Sign in</button></div>
  </div>

  <section id="today" class="page active">
    <div class="page-head"><div><h1>Hoy · Análisis deportivo</h1><div class="sub">Primero lo que dice el fútbol. Después, y solo con pruebas, el mercado.</div></div><div id="todayStamp" class="fresh-chip">LATEST SNAPSHOT</div></div>
    <div class="kpis">
      <div class="kpi"><b id="kpiFixtures">—</b><span>Partidos</span></div>
      <div class="kpi good"><b id="kpiPicks">—</b><span>Pronósticos previos</span></div>
      <div class="kpi lean"><b id="kpiLeans">—</b><span>Estimaciones guardadas</span></div>
      <div class="kpi watch"><b id="kpiWatches">—</b><span>Sin análisis</span></div>
      <div class="kpi"><b id="kpiPasses">—</b><span>Apuestas verificadas</span></div>
    </div>
    <section id="sportFocus" class="panel" aria-label="Pronósticos deportivos y jornada">
      <div class="focus-heading"><div><span class="focus-eyebrow">SOCCER EDGE · FÚTBOL PRIMERO</span>
        <h2>Pronósticos y partidos de hoy</h2><p>Probabilidades deportivas guardadas antes del inicio. Nunca las inventamos a partir de una cuota.</p></div>
        <button id="focusRefresh" class="secondary" type="button">Actualizar datos</button></div>
      <div id="focusStatus" class="focus-status" role="status" aria-live="polite">Revisando qué pronósticos hay realmente guardados.</div>
      <div id="focusForecasts" class="focus-forecast-grid"></div>
      <details id="focusHistorical" class="focus-disclosure hidden"><summary>Ver análisis guardados de partidos ya iniciados</summary><div id="focusHistoricalRows" class="focus-forecast-grid"></div></details>
      <div class="focus-subhead"><div><h3>Jornada completa</h3><p>Por qué cada partido tiene o no tiene un pronóstico</p></div></div>
      <div id="focusFilters" class="focus-filters"></div>
      <div id="focusSlate" class="focus-slate"></div>
      <div class="focus-honesty">Una proyección parcial no es una apuesta. Si faltan estadísticas prepartido, alineaciones o precios, el sistema debe decirlo claramente.</div>
      <details id="focusTechnical" class="focus-disclosure"><summary>Ver diagnóstico técnico y mercados experimentales</summary>
        <p>Para revisar fuentes, modelos y estados de investigación, consulta las secciones avanzadas del sistema. Los cálculos sin validación no se promocionan a BET.</p>
      </details>
    </section>
    <section id="edgeSportRadar" class="panel" aria-label="Edge Radar · análisis deportivo">
      <div class="panel-head"><h2>Edge Radar · Cobertura deportiva real</h2><span class="badge warn">INVESTIGACIÓN · NO BET</span></div>
      <div id="edgeRadarStatus" class="sub" role="status" aria-live="polite">Leyendo únicamente datos de partidos persistidos; no se inventan estadísticas.</div>
      <div id="edgeRadarStats" class="edge-radar-grid"></div>
      <div id="edgeRadarFamilyPanel" class="hidden" style="margin-top:12px">
        <details>
          <summary style="cursor:pointer;font-size:12px;color:#c5e7f5">Modelos por mercado · 12 familias · investigación (NO BET)</summary>
          <p class="sub">Estado del código, NO del rendimiento en vivo. Se necesitan estadísticas prepartido, OOS, cuotas ejecutables y licencias por mercado.</p>
          <div id="edgeRadarFamilyList" class="edge-family-grid"></div>
        </details>
      </div>

    </section>
    <div id="hero" class="panel hero"></div>
    <div class="section-title"><div><h2>Best Bets</h2><span>Only explicit persisted BET classifications</span></div><button class="link-btn" data-go="picks">View picks →</button></div>
    <div id="todayPicks" class="card-grid"></div>
    <div class="section-title"><div><h2>Leans</h2><span>Interesting, but not promoted to BET</span></div><button class="link-btn" data-go="leans">View leans →</button></div>
    <div id="todayLeans" class="card-grid"></div>
    <div class="section-title"><div><h2>Radar de precios · Investigación</h2><span>Datos prospectivos; NO BET y sin notificaciones push habilitadas</span></div></div>
    <div class="panel"><div id="researchStatus" class="sub">Verificando conexión del radar de precios...</div></div>
    <div class="section-title"><div><h2>EDGE -110 o mejor</h2><span>Incluye +100, +110, +150 y superiores. Precio no equivale a valor.</span></div></div>
    <div id="researchPlus100" class="card-grid"></div>
    <div class="section-title"><div><h2>EDGE REGULAR</h2><span>Precios inferiores a -110 para investigación</span></div></div>
    <div id="researchRegular" class="card-grid"></div>
    <div class="section-title"><div><h2>Same Game Parlay (SGP) · Señales RAW (NO BET)</h2><span>Máximo una hipótesis experimental por partido; EV RAW positivo no demuestra valor real.</span></div></div>
    <div class="panel"><div id="sgpResearchStatus" class="sub">Revisando probabilidades conjuntas y precios observados...</div></div>
    <div id="sgpQuotedComposites" class="card-grid"></div>
    <details id="sgpMarketCatalogDetails" class="panel">
      <summary>Ver catálogo de cuotas combinadas (NO son picks)</summary>
      <div class="sub">Aquí se muestran alternativas incompatibles, sin modelo o con EV RAW negativo solamente para comparar precios; nunca son recomendaciones.</div>
      <div id="sgpMarketCatalog" class="card-grid"></div>
    </details>
    <div class="section-title"><div><h2>Jugadores · Análisis estadístico primero</h2><span>Solo candidatos con evidencia deportiva y mercado validado. No se elige por cuota.</span></div></div>
    <div class="sgp-player-guide" aria-label="Requisitos estadísticos para proponer jugadores">
      <div class="sgp-player-guide-head"><span class="sgp-caution-tag">SIN STATS VERIFICADAS = SIN PICK</span><strong>La cuota NO selecciona jugadores</strong></div>
      <p class="sgp-player-note">Retiramos la lista que priorizaba ofertas Bet365 cercanas a +110. Un momio no indica cuántos tiros hará un futbolista. <b>Primero debe existir una proyección deportiva independiente</b> del precio. No se envían apuestas.</p>
      <div class="sgp-guide-steps">
        <div class="sgp-guide-step"><b>1. Estadísticas reales</b><span>Identidad confirmada, tiros o tiros a puerta por 90 minutos, tamaño de muestra, rol y minutos jugados.</span></div>
        <div class="sgp-guide-step"><b>2. Proyección del jugador</b><span>Rival y contexto ajustados, minutos esperados, probabilidad RAW y evaluación fuera de muestra.</span></div>
        <div class="sgp-guide-step"><b>3. Mercado, al final</b><span>XI confirmada, umbral y reglas exactas, cuota vigente y comparación de valor. Si falta algo, WATCH.</span></div>
      </div>
    </div>
    <div id="playerSportModelSummary" class="sport-model-summary" role="status" aria-live="polite"><strong>Proyecciones deportivas independientes</strong><div class="sport-model-sub">Verificando historial, minutos, XI oficial y predicciones anteriores al partido. No se consultan momios para seleccionar jugadores.</div></div>
    <div id="playerSportModelRows" class="card-grid"></div>
    <div id="sgpBuilderSlip" class="sgp-slip" role="status" aria-live="polite">Esperando candidatos respaldados por estadísticas; ninguna cuota por sí sola crea una selección.</div>
    <div id="sgpPlayerLegs" class="card-grid sgp-player-grid"></div>
    <div class="section-title"><div><h2>Watch</h2><span>Waiting for price, XI, goalkeeper or other required evidence</span></div></div>
    <div class="panel"><div id="todayWatches" class="watch-list"></div></div>
    <div class="section-title"><div><h2>Full Slate</h2><span>Every eligible fixture stays visible, even when analysis is incomplete</span></div></div>
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
    <div class="page-head"><div><h1>Matches</h1><div class="sub">Browse the complete eligible slate. Each match shows what is analyzed, which markets have data, and what is still missing.</div></div></div>
    <div id="matchDetail" class="panel hidden"></div>
    <div id="matchesBrowsePanel" class="panel"><div id="matchesList" class="slate-list"></div></div>
  </section>

  <section id="performance" class="page">
    <div class="page-head"><div><h1>Performance</h1><div class="sub">Validation/OOS evidence and realized BET performance are kept conceptually separate.</div></div></div>
    <div id="performanceBody" class="panel"><div class="empty"><b>Open this page to load verified evidence.</b>Research/OOS metrics are never presented as a customer BET ledger.</div></div>
  </section>

  <section id="myedge" class="page">
    <div class="page-head"><div><h1>My Edge</h1><div class="sub">Saved decisions and tracked matches, persisted to your authenticated account.</div></div><span class="decision">EDGE PRO</span></div>
    <div id="savedBody" class="panel"><div class="empty"><b>Open My Edge to load saved items.</b>Your saved product state is isolated from all model inputs.</div></div>
  </section>

  <section id="account" class="page account-page">
    <div class="page-head"><div><h1>Account</h1><div class="sub">Authentication, entitlement and subscription lifecycle are resolved server-side.</div></div></div>
    <div class="panel account-card"><div><h2 id="accountPlan">Explorer</h2><p id="accountCopy">Browse the verified slate and watch states. Premium decision evidence stays redacted until entitlement permits it.</p></div><button id="accountPageBtn" class="primary">Sign in</button></div>
    <div id="billingBody" class="panel"><div class="empty"><b>Open Account to load subscription state.</b>Checkout remains disabled until the public billing launch gate is explicitly enabled.</div></div>
  </section>
</main>
</div>
<nav class="mobile-nav">
  <button class="active" data-page="today">Today</button><button data-page="picks">Picks</button><button data-page="leans">Leans</button><button data-page="matches">Matches</button><button data-page="performance">Results</button><button data-page="myedge">My Edge</button><button data-page="account">Account</button>
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
  let DATA=null,ACCESS=null,PERF_LOADED=false,MYEDGE_LOADED=false,SAVED_ROWS=[],ACCOUNT_LOADED=false,ACCOUNT_DATA=null;
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  const token=()=>localStorage.getItem(AK)||'';
  const pct=v=>{{if(v===null||v===undefined||v==='')return '—';const n=Number(v);if(!Number.isFinite(n))return '—';const p=Math.abs(n)<=1?n*100:n;return p.toFixed(1)+'%'}};
  const num=v=>{{if(v===null||v===undefined||v==='')return null;const n=Number(v);return Number.isFinite(n)?n:null}};
  // Presentation only: canonical prices and model calculations are unchanged.
  // Decimal 2.10 => American +110; decimal 1.91 => approximately -110.
  function americanOddsFromDecimal(value){{
    const decimal=num(value);
    if(decimal===null||decimal<=1)return 'NOT VERIFIED';
    const american=decimal>=2?
      Math.round((decimal-1)*100):-Math.round(100/(decimal-1));
    if(!Number.isSafeInteger(american)||Math.abs(american)<100)return 'NOT VERIFIED';
    return american>0?'+'+american:String(american);
  }}
  function americanOdds(value,format='DECIMAL'){{
    const f=String(format||'').toUpperCase(),v=num(value);
    if(v===null)return 'NOT VERIFIED';
    if(f==='DECIMAL')return americanOddsFromDecimal(v);
    if(f==='AMERICAN'&&Math.abs(v)>=100&&Number.isSafeInteger(v))
      return v>0?'+'+v:String(v);
    return 'NOT VERIFIED';
  }}
  const pp=v=>{{const n=num(v);return n===null?'—':(n>=0?'+':'')+n.toFixed(1)+' pp'}};
  const ev=v=>{{const n=num(v);return n===null?'—':(n>=0?'+':'')+(Math.abs(n)<=1?n*100:n).toFixed(1)+'%'}};
  const units=v=>{{const n=num(v);return n===null?'—':(n>=0?'+':'')+n.toFixed(2)+'u'}};
  const initials=name=>String(name||'?').split(/\\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase().slice(0,3);
  const APP_TIMEZONE='America/Mexico_City';
  const dt=raw=>{{if(!raw)return ['TBD','Kickoff'];const d=new Date(raw);if(Number.isNaN(d.getTime()))return [String(raw),'Kickoff'];return [new Intl.DateTimeFormat('es-MX',{{timeZone:APP_TIMEZONE,hour:'numeric',minute:'2-digit',hour12:true}}).format(d),new Intl.DateTimeFormat('es-MX',{{timeZone:APP_TIMEZONE,month:'short',day:'numeric'}}).format(d)]}};
  const api=async(path)=>{{const h={{}};if(token())h.Authorization='Bearer '+token();const r=await fetch((cfg.api_base||'/app/api/v2')+path,{{headers:h,cache:'no-store'}});const d=await r.json().catch(()=>({{}}));if(!r.ok)throw Object.assign(new Error(d.error||('HTTP '+r.status)),{{status:r.status,data:d}});return d}};
  const apiWrite=async(path,method,body)=>{{const h={{Authorization:'Bearer '+token(),'Content-Type':'application/json'}},opts={{method,headers:h,cache:'no-store'}};if(body!==undefined&&body!==null)opts.body=JSON.stringify(body);const r=await fetch((cfg.api_base||'/app/api/v2')+path,opts),d=await r.json().catch(()=>({{}}));if(!r.ok)throw Object.assign(new Error(d.error||('HTTP '+r.status)),{{status:r.status,data:d}});return d}};
  const billingCall=async(slug,body={{}})=>{{if(!token())throw new Error('AUTH_REQUIRED');if(!cfg.supabase_url||!cfg.publishable_key)throw new Error('BILLING_CONFIGURATION_UNAVAILABLE');const r=await fetch(cfg.supabase_url+'/functions/v1/'+slug,{{method:'POST',headers:{{Authorization:'Bearer '+token(),apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify(body)}}),d=await r.json().catch(()=>({{}}));if(!r.ok)throw Object.assign(new Error(d.error||('HTTP '+r.status)),{{status:r.status,data:d}});return d}};
  const publicJson=async(path)=>{{const r=await fetch(path,{{cache:'no-store'}});const d=await r.json().catch(()=>({{}}));if(!r.ok)throw new Error(d.error||('HTTP '+r.status));return d}};
  function candidateRows(){{return [...(DATA?.picks?.rows||[]),...(DATA?.leans?.rows||[]),...(DATA?.watches?.rows||[])]}}
  function fixtureTargets(){{return [...(DATA?.slate?.rows||[]),...candidateRows()]}}
  async function enrichIdentities(){{
    const targets=fixtureTargets(),ids=[...new Set(targets.map(x=>x?.fixture?.fixture_id).filter(x=>x!==null&&x!==undefined&&String(x).length))];
    if(!ids.length)return;
    try{{
      const map=new Map();
      for(let i=0;i<ids.length;i+=80){{
        const chunk=ids.slice(i,i+80),d=await publicJson('/app/fixture-identities?ids='+encodeURIComponent(chunk.join(',')));
        for(const row of (d.rows||[]))map.set(String(row.fixture_id),row);
      }}
      for(const item of targets){{const f=item?.fixture||{{}},identity=map.get(String(f.fixture_id||''));if(!identity)continue;item.fixture={{...f,...identity}}}}
    }}catch(_){{/* presentation enrichment is optional; canonical decision payload remains untouched */}}
  }}
  const authFetch=async(path,body)=>{{if(!cfg.auth_configured)throw new Error('AUTH_NOT_CONFIGURED');const r=await fetch(cfg.supabase_url+path,{{method:'POST',headers:{{apikey:cfg.publishable_key,'Content-Type':'application/json'}},body:JSON.stringify(body)}});const d=await r.json().catch(()=>({{}}));if(!r.ok)throw new Error(d.error_description||d.msg||d.error||('HTTP '+r.status));return d}};
  const classification=c=>String(c?.decision?.classification||'').toUpperCase();
  const fixture=c=>c?.fixture||{{}},market=c=>c?.market||{{}},proj=c=>c?.projections||{{}},avail=c=>c?.availability||{{}};
  const premium=()=>!!ACCESS?.premium_unlocked;
  function crest(url,name){{const src=String(url||'');return '<span class="crest '+(src?'has-img':'')+'">'+(src?'<img loading="lazy" decoding="async" src="'+esc(src)+'" alt="'+esc(name)+' crest" onerror="this.style.display=\\'none\\';this.parentElement.classList.remove(\\'has-img\\')">':'')+'<span class="fallback">'+esc(initials(name))+'</span></span>'}}
  function priceText(c){{const p=market(c).price||{{}},price=americanOdds(p.value,p.format);return price==='NOT VERIFIED'?'Cuota NO VERIFICADA':price+(p.bookmaker?' · '+p.bookmaker:'')}}
  function priceVerified(c){{const p=market(c).price||{{}};return americanOdds(p.value,p.format)!=='NOT VERIFIED'&&!!p.bookmaker&&!!p.captured_at}}
  function evidence(c){{const e=c?.evidence||{{}},s=(e.sporting_reasons||[])[0],m=(e.market_reasons||[])[0],b=(e.blockers||[])[0];if(s)return '<b>Sport:</b> '+esc(s);if(m)return '<b>Market:</b> '+esc(m);if(b)return '<b>Blocker:</b> '+esc(b);return 'No additional persisted explanation in this snapshot.'}}
  function decisionClass(c){{const x=classification(c);return x==='BET'?'bet':x==='LEAN'?'lean':'watch'}}
  function decisionLabel(c){{const d=c?.decision||{{}};return classification(c)||d.display_bucket||String(d.execution_status||'WATCH').replaceAll('_',' ')}}
  function candidateKey(c){{const f=fixture(c),m=market(c),d=c?.decision||{{}};return [f.fixture_id||'',m.family||'',m.name||'',m.selection||'',m.line??'',classification(c)||d.display_bucket||'WATCH'].join('|')}}
  function candidateSavedPayload(c){{const f=fixture(c),m=market(c),d=c?.decision||{{}};return {{item_type:classification(c)||d.display_bucket||'WATCH',fixture_id:f.fixture_id,market_family:m.family||null,market_name:m.name||null,selection:m.selection||null,line:m.line??null,source_snapshot_at:DATA?.generated_at_utc||DATA?.generated_at_local||null,payload:{{home_team:f.home_team||null,away_team:f.away_team||null,league:f.league||f.country||null,kickoff:f.kickoff||null,classification:classification(c)||d.display_bucket||null,tier:d.tier||null,reason_display:d.reason_display||null,price_display:priceText(c)}}}}}}
  function card(c){{const f=fixture(c),m=market(c),p=proj(c),a=avail(c),d=c?.decision||{{}},price=m.price||{{}},stamp=price.captured_at||c?.freshness?.provider_update||'NOT VERIFIED';
    return '<article class="pick-card" data-fixture="'+esc(f.fixture_id||'')+'"><div class="card-top"><div class="card-match"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(f.league||f.country||'Competition')+' · '+esc(f.kickoff||'Kickoff N/V')+'</small></div><span class="status '+decisionClass(c)+'">'+esc(decisionLabel(c))+(classification(c)==='BET'&&d.tier?' · '+esc(d.tier):'')+'</span></div>'+
    '<div class="card-market">'+esc(m.selection||m.name||m.family||'Market')+(m.line!==null&&m.line!==undefined?' · '+esc(m.line):'')+'</div>'+
    '<div class="price-line"><span>'+esc(priceText(c))+'</span><span>Updated '+esc(stamp)+'</span></div>'+
    '<div class="mini-proj"><div><small>RAW SPORT</small><b>'+pct(p.raw_sport_probability)+'</b></div><div><small>SHRUNK</small><b>'+pct(p.market_shrunk_probability)+'</b></div><div><small>MARKET FAIR</small><b>'+pct(p.fair_market_probability)+'</b></div><div><small>EDGE</small><b style="color:var(--green)">'+pp(p.probability_edge_pp)+'</b></div></div>'+
    '<div class="badges" style="margin-top:9px"><span class="badge">Availability '+pct(a.confidence)+'</span><span class="badge">Data '+esc(a.data_tier||'NOT VERIFIED')+'</span><span class="badge">XI '+esc(a.starting_xi_status||'NOT VERIFIED')+'</span><span class="badge">GK '+esc(a.goalkeeper_status||'NOT VERIFIED')+'</span></div>'+
    '<div class="reason">'+evidence(c)+'</div><div class="card-actions">'+(premium()?'<button type="button" class="secondary save-btn" data-save-key="'+esc(candidateKey(c))+'">☆ Save</button>':'')+'</div></article>'}}
  function empty(title,copy){{return '<div class="empty"><b>'+esc(title)+'</b>'+esc(copy)+'</div>'}}
  function lock(resource,count){{return '<div class="lock"><b>'+esc(resource)+' · Edge Pro</b>'+esc(count||0)+' canonical item'+((count||0)===1?'':'s')+' in this snapshot. Premium probability, price and evidence fields remain server-redacted.</div>'}}

  let focusFilter='all';
  function sportForecastEntries(){{
    const rows=[...(DATA?.picks?.rows||[]),...(DATA?.leans?.rows||[]),...(DATA?.watches?.rows||[])];
    const seen=new Set(),entries=[];
    for(const row of rows){{
      const f=fixture(row),m=market(row),p=proj(row);
      const probability=p.raw_sport_probability;
      if(!f.fixture_id||typeof probability!=='number'||!Number.isFinite(probability)||probability<0||probability>1)continue;
      if(!m.selection&&!m.name)continue;
      const key=[f.fixture_id,m.family,m.name,m.selection,m.line??''].join('|');
      if(seen.has(key))continue;
      seen.add(key);
      const kickoff=Date.parse(f.kickoff||'');
      const snap=Date.parse(DATA?.generated_at_utc||DATA?.generated_at_local||'');
      const before=Number.isFinite(kickoff)&&Number.isFinite(snap)&&snap<=kickoff-5*60*1000;
      const future=Number.isFinite(kickoff)&&kickoff>Date.now()+5*60*1000;
      entries.push({{row,kickoff,before,future,active:before&&future}});
    }}
    return entries.sort((a,b)=>Number(b.active)-Number(a.active)||(a.kickoff-b.kickoff));
  }}
  function sportSelection(row){{
    const m=market(row),s=String(m.selection||'').toLowerCase().trim();
    if(s==='home'||s==='1')return 'Victoria local';
    if(s==='away'||s==='2')return 'Victoria visitante';
    if(s==='draw'||s==='x')return 'Empate';
    if(s==='yes')return 'Ambos anotan: Sí';
    if(s==='no')return 'Ambos anotan: No';
    if(/^over/i.test(s))return 'Más de '+(m.line??s.replace(/[^0-9.]/g,''))+' goles';
    if(/^under/i.test(s))return 'Menos de '+(m.line??s.replace(/[^0-9.]/g,''))+' goles';
    return m.selection||m.name||'Resultado deportivo';
  }}
  function focusForecastCard(entry){{
    const row=entry.row,f=fixture(row),p=proj(row),[time,date]=dt(f.kickoff);
    const label=entry.active?'Pronóstico prepartido · MODELO LIMITADO':'Registro antiguo · NO VIGENTE';
    return '<div class="focus-card '+(entry.active?'':'past')+'"><span class="focus-tag">'+esc(label)+'</span>'+
      '<h3>'+esc(f.home_team||'Local')+' vs '+esc(f.away_team||'Visitante')+'</h3>'+
      '<p>'+esc(f.league||f.country||'Competición')+' · '+esc(date)+' · '+esc(time)+'</p>'+
      '<div class="focus-prob">'+pct(p.raw_sport_probability)+'</div>'+
      '<div>'+esc(sportSelection(row))+'</div>'+
      '<p>'+esc(entry.active?'Estimación deportiva RAW; todavía no calibrada para apuestas. No hay cuota ejecutable verificada.':
        'Conservado para auditoría. El partido comenzó, la fecha expiró o no consta el corte prepartido.')+'</p>'+
      '<button class="secondary focus-open" data-fixture="'+esc(f.fixture_id)+'" type="button">Ver datos del partido</button></div>';
  }}
  function focusCoverageState(row,entries){{
    const f=row?.fixture||{{}},sport=row?.sport_analysis||{{}},n=Number(row?.coverage?.analysis_rows||0);
    if(entries.some(e=>e.active&&String(e.row?.fixture?.fixture_id)===String(f.fixture_id)))return 'forecast';
    if(['MODELED_LIMITED','PARTIAL','SPORT_EVIDENCE_ONLY'].includes(sport.status))return 'partial';
    return n>0?'partial':'pending';
  }}
  function focusMissing(row){{
    const codes=row?.screening?.missing_evidence_codes||[];
    if(codes.includes('RAW_SPORT_NOT_RECORDED'))return 'Sin proyección deportiva guardada';
    if(codes.includes('SPORT_FEATURE_SNAPSHOT_NOT_RECORDED'))return 'Faltan estadísticas deportivas comprobadas';
    if(codes.includes('LINEUP_SNAPSHOT_NOT_RECORDED'))return 'Falta alineación confirmada';
    return 'Sin datos suficientes para un pronóstico';
  }}

  function p0Entries(){{
    const payload=DATA?.sport_previews||{{}};
    if(payload.status!=='PREMATCH_LIMITED_RESEARCH')return [];
    return (Array.isArray(payload.rows)?payload.rows:[]).filter(x=>{{
      const p=x.raw_probabilities||{{}},at=Date.parse(x.captured_at_utc||''),kick=Date.parse(x.kickoff_utc||'');
      return x.bet_eligible===false&&
        x.model_validation==='NOT_VERIFIED_OOS'&&
        Number.isFinite(at)&&Number.isFinite(kick)&&
        at<=kick-5*60*1000&&kick>Date.now()+5*60*1000&&
        ['raw_home_win_prob','raw_draw_prob','raw_away_win_prob',
         'raw_btts_yes_prob','raw_over_2_5_prob'].every(k=>typeof p[k]==='number'&&p[k]>=0&&p[k]<=1);
    }});
  }}
  function p0Card(x){{
    const f=x.fixture||{{}},p=x.raw_probabilities||{{}},[time,date]=dt(x.kickoff_utc);
    const metric=(label,value)=>'<div class="focus-metric"><small>'+esc(label)+'</small><b>'+pct(value)+'</b></div>';
    return '<article class="focus-card"><div class="focus-tag">MODELO DE GOLES · INVESTIGACIÓN PREPARTIDO</div>'+
      '<h3>'+esc(f.home_team||'Local')+' vs '+esc(f.away_team||'Visitante')+'</h3>'+
      '<p>'+esc(f.league||f.country||'Competición')+' · '+esc(date)+' · '+esc(time)+'</p>'+
      '<div class="focus-prob-strip">'+metric('Gana local',p.raw_home_win_prob)+
      metric('Empate',p.raw_draw_prob)+metric('Gana visita',p.raw_away_win_prob)+'</div>'+
      '<div class="focus-prob-strip submarkets">'+metric('Más de 2.5',p.raw_over_2_5_prob)+
      metric('Ambos anotan',p.raw_btts_yes_prob)+'</div>'+
      '<p>Base de goles observados, NO xG · '+esc(x.min_split_sample)+' juegos mínimos por división local/visitante · sin alineaciones confirmadas ni calibración OOS.</p>'+
      '<button type="button" class="secondary focus-open" data-fixture="'+esc(x.fixture_id)+'">Ver partido</button></article>';
  }}

  function renderSportFocus(){{
    const entries=sportForecastEntries(),saved=p0Entries();
    const savedIds=new Set(saved.map(x=>String(x.fixture_id)));
    const active=entries.filter(e=>e.active&&!savedIds.has(String(e.row.fixture.fixture_id)));
    const previous=entries.filter(e=>!e.active);
    const forecastCount=new Set([...saved.map(x=>String(x.fixture_id)),...active.map(x=>String(x.row.fixture.fixture_id))]).size;
    const slate=DATA?.slate?.rows||[],counts={{pending:0,partial:0,forecast:0}};
    for(const row of slate)counts[savedIds.has(String(row?.fixture?.fixture_id))?'forecast':focusCoverageState(row,entries)]++;
    $('kpiFixtures').textContent=DATA?.counts?.fixtures??slate.length;
    $('kpiPicks').textContent=forecastCount;
    $('kpiLeans').textContent=new Set([...savedIds,...entries.map(x=>String(x.row.fixture.fixture_id))]).size;
    $('kpiWatches').textContent=counts.pending;
    $('kpiPasses').textContent=DATA?.counts?.picks??0;
    const status=$('focusStatus'),board=$('focusForecasts'),history=$('focusHistorical');
    if(!status||!board||!history)return;
    const screen=DATA?.sport_screening||{{}};
    const hasTickScreen=screen.status==='SPORT_FIRST_SCREEN_RECORDED';
    const screenSummary=hasTickScreen?
      'En el último ciclo: '+esc(screen.attempted)+' partidos examinados, '+esc(screen.modeled_research)+
      ' proyecciones deportivas limitadas, '+esc(screen.inputs_insufficient)+
      ' sin muestra suficiente y '+esc(screen.quota_deferred)+' diferidos por cuota. Estos conteos NO son apuestas ni representan toda la jornada.':
      'La ejecución del screening deportivo de este ciclo todavía no está verificada.';
    const canReviewPrivateResearch=ACCESS?.owner===true||ACCESS?.admin===true;
    status.innerHTML=forecastCount?
      '<strong>'+forecastCount+' partido(s) con proyección deportiva previa</strong> entre '+slate.length+
      ' · '+counts.pending+' sin análisis guardado. Las probabilidades NO equivalen a apuestas. <span>'+screenSummary+'</span>':
      '<strong>'+esc(hasTickScreen&&screen.modeled_research>0?'Investigación deportiva registrada, sin pronósticos públicos autorizados.':
         'Sin proyecciones deportivas publicables para esta sesión.')+'</strong> '+
      screenSummary+' Hay '+slate.length+' partidos en el calendario; '+counts.pending+
      ' siguen sin análisis completo. Una estimación de investigación sin fuente publicable o validación suficiente no se publica como pronóstico.';
    const queueOverview=DATA?.sport_coverage_queue||{{}},queueCounts=queueOverview.by_sport_queue_status||{{}};
    const countOrUnknown=v=>Number.isInteger(v)&&v>=0?String(v):'NOT VERIFIED';
    const queueSummary=queueOverview.all_fixtures_accounted_for===true?
      '<div class="focus-queue-summary"><strong>Estado deportivo acumulado de la jornada</strong>'+
      '<span>'+esc(countOrUnknown(queueOverview.total_eligible_fixtures))+' registrados · '+
      esc(countOrUnknown(queueCounts.MODELED_LIMITED))+' con modelo deportivo limitado · '+
      esc(countOrUnknown(queueCounts.DATA_INSUFFICIENT))+' examinados sin muestra suficiente · '+
      esc(countOrUnknown(queueCounts.QUEUED))+' pendientes de screening · '+
      esc(countOrUnknown(queueCounts.RETRY_REQUIRED))+' por reintentar · '+
      esc(countOrUnknown(queueCounts.NOT_VERIFIED))+' sin estado comprobado.</span>'+
      '<small>La cobertura cuenta estados persistidos, NO predicciones publicables ni recomendaciones. Cada partido tiene su expediente.</small></div>':
      '<div class="focus-queue-summary"><strong>Estado de cobertura no verificado</strong>'+
      '<span>No se puede afirmar cuántos partidos han sido examinados. Consulta la ficha individual.</span></div>';
    const leagueRows=Array.isArray(queueOverview.by_league)?queueOverview.by_league:[];
    const leagueSummary=queueOverview.all_fixtures_accounted_for===true&&leagueRows.length?
      '<details class="focus-league-detail"><summary>Cobertura deportiva por competición ('+
      esc(queueOverview.by_league_total??leagueRows.length)+')</summary>'+
      '<div class="focus-league-rows">'+leagueRows.slice(0,25).map(group=>{{
        const c=group.by_status||{{}};
        const pending=(c.QUEUED||0)+(c.RETRY_REQUIRED||0)+(c.NOT_VERIFIED||0);
        return '<div class="focus-league-row"><strong>'+esc(group.league||'Competición no identificada')+
          '</strong><span>'+esc(group.eligible??'–')+' partidos · '+
          esc(c.MODELED_LIMITED||0)+' modelados · '+esc(c.DATA_INSUFFICIENT||0)+
          ' sin datos suficientes · '+esc(pending)+' pendientes/no verificados</span></div>';
      }}).join('')+'</div>'+
      (queueOverview.by_league_truncated||leagueRows.length>25?
        '<small>Se muestran las primeras 25 competiciones. Consulta Partidos para el resto.</small>':'')+
      '<small>Conteos de la cola de screening; no son pronósticos publicables ni un indicador de precisión.</small></details>':'';
    status.innerHTML+=queueSummary+leagueSummary;
    board.innerHTML=forecastCount?(saved.slice(0,8).map(p0Card).join('')+active.slice(0,Math.max(0,8-saved.length)).map(focusForecastCard).join('')):
      '<div class="focus-empty focus-empty-compact">'+
      esc(canReviewPrivateResearch?
        'No hay probabilidades deportivas verificables para estos partidos en la cuenta actual. Selecciona Todos o Datos parciales para revisar lo guardado.':
        'Las proyecciones de investigación no autorizadas para publicación no aparecen en modo FREE. Abre Todos para consultar los partidos y sus datos verificados.')+
      '</div>';
    history.classList.toggle('hidden',previous.length===0);
    $('focusHistoricalRows').innerHTML=previous.slice(0,10).map(focusForecastCard).join('');
    const filters=[['pending','Por analizar'],['partial','Datos parciales'],['forecast','Con pronóstico'],['all','Todos']];
    $('focusFilters').innerHTML=filters.map(([key,label])=>
      '<button type="button" data-focus-filter="'+key+'" class="'+(focusFilter===key?'active':'')+'">'+label+
      ' ('+(key==='all'?slate.length:counts[key])+')</button>').join('');
    $('focusFilters').querySelectorAll('button').forEach(b=>b.onclick=()=>{{focusFilter=b.dataset.focusFilter;renderSportFocus();}});
    const time=row=>Date.parse(row?.fixture?.kickoff||'')||0;
    const order=[...slate].sort((a,b)=>{{
      const ao=time(a)>Date.now()?0:1,bo=time(b)>Date.now()?0:1;
      return ao-bo||time(a)-time(b);
    }});
    const stateOf=r=>savedIds.has(String(r?.fixture?.fixture_id))?'forecast':focusCoverageState(r,entries);
    const chosen=order.filter(r=>focusFilter==='all'||stateOf(r)===focusFilter);
    const card=r=>{{
      const f=r?.fixture||{{}},[clock,date]=dt(f.kickoff),state=stateOf(r);
      const sa=r?.sport_analysis||{{}},op=r?.opportunity||{{}},queue=r?.screen_queue||{{}};
      const label=state==='forecast'?'Pronóstico deportivo':state==='partial'?'Análisis deportivo':'Por analizar';
      const reason=[sa.explanation||focusMissing(r),valueLabels[op.status]||'Mercado no evaluado',
        queue.attempt_count!==null&&queue.attempt_count!==undefined?'Intentos: '+queue.attempt_count:null].filter(Boolean).join(' · ');
      return '<a class="focus-fixture" href="'+esc(matchPath(f.fixture_id||''))+'" data-focus-fixture="'+esc(f.fixture_id||'')+'">'+
        '<div><strong>'+esc(f.home_team||'Local')+' vs '+esc(f.away_team||'Visitante')+'</strong>'+
        '<span>'+esc(f.league||f.country||'Competición')+' · '+esc(date)+' · '+esc(clock)+'</span>'+
        '<span>'+esc(reason)+'</span></div><div class="focus-state">'+esc(label)+'</div></a>';
    }};
    $('focusSlate').innerHTML=chosen.length?chosen.slice(0,12).map(card).join('')+
      (chosen.length>12?'<p class="sub">Mostrando 12 de '+chosen.length+'. Abre Partidos para el resto.</p>':''):
      '<div class="focus-empty">No hay partidos en esta categoría.</div>';
    document.querySelectorAll('.focus-open').forEach(x=>x.onclick=()=>openMatch(x.dataset.fixture));
    document.querySelectorAll('[data-focus-fixture]').forEach(x=>x.onclick=e=>{{e.preventDefault();openMatch(x.dataset.focusFixture);}});
    $('focusRefresh').onclick=()=>loadToday();
    // Translate the product navigation without altering routing or permissions.
    const translations={{today:'Hoy',picks:'Apuestas',leans:'En observación',matches:'Partidos',performance:'Resultados',myedge:'Guardados',account:'Cuenta'}};
    document.querySelectorAll('.nav button[data-page],.mobile-nav button[data-page]').forEach(x=>{{
      const code=x.dataset.page;
      if(!(code in translations))return;
      const sym=x.querySelector('i');x.textContent=translations[code];
      if(sym)x.prepend(sym);
    }});
  }}

  function renderKpis(){{const c=DATA?.counts||{{}};$('kpiFixtures').textContent=c.fixtures??'—';$('kpiPicks').textContent=c.picks??'—';$('kpiLeans').textContent=c.leans??'—';$('kpiWatches').textContent=c.watches??'—';$('kpiPasses').textContent=c.passes??'—'}}
  function renderHero(){{const picks=DATA?.picks||{{}},leans=DATA?.leans||{{}},watches=DATA?.watches||{{}},hero=$('hero');if(!hero)return;
    if(picks.locked&&picks.total>0){{hero.innerHTML='<div style="padding:24px">'+lock('Verified picks',picks.total)+'</div>';return}}
    const c=(picks.rows||[])[0]||(leans.rows||[])[0]||(watches.rows||[])[0];if(!c){{hero.innerHTML='<div style="padding:24px">'+empty('No canonical BET right now','Zero bets is a valid result. The system will not promote a READY or STRONG row into a wager.')+'</div>';return}}
    if(!c.projections){{const f=fixture(c),m=market(c),[time,date]=dt(f.kickoff),d=c?.decision||{{}};hero.innerHTML='<div class="hero-grid"><div class="hero-left"><div class="eyebrow">'+esc(f.league||f.country||'Football Intelligence')+'</div><div class="faceoff"><div class="team">'+crest(f.home_team_logo,f.home_team)+'<strong>'+esc(f.home_team||'Home')+'</strong></div><div class="vs">VS</div><div class="team">'+crest(f.away_team_logo,f.away_team)+'<strong>'+esc(f.away_team||'Away')+'</strong></div></div><div class="kickoff">'+esc(date)+' · '+esc(time)+'</div></div><div class="hero-right"><div class="decision-row"><span class="decision watch">'+esc(decisionLabel(c))+'</span></div><div class="market-title">'+esc(m.name||m.family||'Market watch')+'</div><div class="badges"><span class="badge warn">No BET yet</span><span class="badge">Exact price '+(m.price?.value!==null&&m.price?.value!==undefined?esc(priceText(c)):'NOT VERIFIED')+'</span></div><div class="reason">'+esc(d.reason_display||'Waiting for the remaining verified conditions required for action.')+'</div></div></div>';return}}
    const f=fixture(c),m=market(c),p=proj(c),a=avail(c),[time,date]=dt(f.kickoff);
    hero.innerHTML='<div class="hero-grid"><div class="hero-left"><div class="eyebrow">'+esc(f.league||f.country||'Football Intelligence')+'</div><div class="faceoff"><div class="team">'+crest(f.home_team_logo,f.home_team)+'<strong>'+esc(f.home_team||'Home')+'</strong></div><div class="vs">VS</div><div class="team">'+crest(f.away_team_logo,f.away_team)+'<strong>'+esc(f.away_team||'Away')+'</strong></div></div><div class="kickoff">'+esc(date)+' · '+esc(time)+'</div></div>'+
    '<div class="hero-right"><div class="decision-row"><span class="decision '+decisionClass(c)+'">'+esc(decisionLabel(c))+'</span>'+(classification(c)==='BET'&&c?.decision?.tier?'<span class="decision">TIER '+esc(c.decision.tier)+'</span>':'')+'</div><div class="market-title">'+esc(m.selection||m.name||m.family||'Market')+(m.line!==null&&m.line!==undefined?' · '+esc(m.line):'')+'</div>'+
    '<div class="projection-grid"><div class="projection"><small>RAW SPORT</small><b>'+pct(p.raw_sport_probability)+'</b></div><div class="projection"><small>MARKET SHRUNK</small><b>'+pct(p.market_shrunk_probability)+'</b></div><div class="projection"><small>MARKET FAIR</small><b>'+pct(p.fair_market_probability)+'</b></div><div class="projection edge"><small>PROB EDGE</small><b>'+pp(p.probability_edge_pp)+'</b></div></div>'+
    '<div class="badges"><span class="badge '+(priceVerified(c)?'good':'warn')+'">'+esc(priceText(c))+'</span><span class="badge">Availability '+pct(a.confidence)+'</span><span class="badge">XI '+esc(a.starting_xi_status||'NOT VERIFIED')+'</span><span class="badge">GK '+esc(a.goalkeeper_status||'NOT VERIFIED')+'</span></div><div class="reason">'+evidence(c)+'</div></div></div>'}}
  function renderCards(){{const picks=DATA?.picks||{{}},leans=DATA?.leans||{{}};
    $('todayPicks').innerHTML=(picks.locked&&picks.total>0)?lock('Best Bets',picks.total):(picks.rows||[]).slice(0,4).map(card).join('')||empty('No BETS','No explicit persisted BET classifications in the latest snapshot.');
    $('todayLeans').innerHTML=(leans.locked&&leans.total>0)?lock('Leans',leans.total):(leans.rows||[]).slice(0,4).map(card).join('')||empty('No LEANS','The engine has not emitted an explicit LEAN in the latest snapshot.');
    $('picksPage').innerHTML=(picks.locked&&picks.total>0)?lock('Picks',picks.total):(picks.rows||[]).map(card).join('')||empty('No BETS','Zero bets is a valid outcome.');
    $('leansPage').innerHTML=(leans.locked&&leans.total>0)?lock('Leans',leans.total):(leans.rows||[]).map(card).join('')||empty('No LEANS','Nothing currently meets the canonical LEAN classification.');
    document.querySelectorAll('.pick-card[data-fixture]').forEach(el=>el.onclick=()=>openMatch(el.dataset.fixture));wireSaveButtons();
  }}
  function watchRow(c){{const f=fixture(c),m=market(c),[time,date]=dt(f.kickoff),d=c?.decision||{{}};return '<div class="watch-row"><div class="time"><b>'+esc(time)+'</b>'+esc(date)+'</div><div class="row-teams"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(m.name||m.family||'Market')+(d.reason_display?' · '+esc(d.reason_display):'')+'</small></div><div class="row-state"><span class="status watch">'+esc(d.classification||d.display_bucket||'WATCH')+'</span></div></div>'}}
  function renderEdgeSportRadar(){{const r=DATA?.radar||{{}},status=$('edgeRadarStatus'),stats=$('edgeRadarStats');if(!status||!stats)return;
    const familyPanel=$('edgeRadarFamilyPanel'),familyList=$('edgeRadarFamilyList');
    const marketResearch=r?.market_family_research||{{}},families=Array.isArray(marketResearch.families)?marketResearch.families:[];
    const owner=!!(ACCESS?.owner||ACCESS?.admin);
    if(familyPanel&&familyList){{
      familyPanel.classList.toggle('hidden',!owner||families.length===0);
      familyList.innerHTML=(owner?families:[]).map(f=>
        '<div class="edge-family-row"><b>'+esc(f.family)+' · '+esc(f.group)+'</b>'+
        '<span>Modelo: código de investigación, no validado en OOS</span>'+
        '<span>Datos requeridos: '+esc(f.sport_inputs_required||'NOT VERIFIED')+'</span>'+
        '<span>Cuota ejecutable: NO VERIFICADA · BET: BLOQUEADO</span></div>'
      ).join('');
    }}

    if(r.status!=='PERSISTED_EVIDENCE_AUDIT_READY'||r.reconciliation_complete!==true){{status.textContent='Cobertura del registro NO VERIFICADA: no hay recuento fiable para esta jornada.';stats.innerHTML='';return}}
    const count=x=>Number.isSafeInteger(x)&&x>=0?String(x):'—';
    status.textContent='Solo calendario elegible persistido · '+(r.slate_date||'Fecha no verificada')+' · sin consultas nuevas a la API · una proyección guardada NO es BET.';
    const items=[['En calendario',r.eligible_registry_count],['RAW registrado',r.raw_sport_recorded_count],['Análisis parcial',r.partial_analysis_count],['Evidencia deportiva',r.sport_only_count],['Sin análisis guardado',r.no_persisted_analysis_count]];
    stats.innerHTML=items.map(([label,n])=>'<div class="edge-radar-cell"><b>'+count(n)+'</b><span>'+esc(label)+'</span></div>').join('');
    if(r.market_only_count)status.textContent+=' · '+count(r.market_only_count)+' con únicamente datos de mercado.';
    const h=r.highlightly_research||{{}};
    if((ACCESS?.owner||ACCESS?.admin)&&h.status==='OWNER_RESEARCH_COVERAGE_AVAILABLE'){{
      status.textContent+=' · Highlightly BASIC (investigación privada): '+count(h.fixture_research_rows_on_slate)+' con registro; '+count(h.prematch_research_limited_count)+' capturas prepartido limitadas. Sin licencia de publicación, sin BET.';
    }}else if((ACCESS?.owner||ACCESS?.admin)&&h.status==='SOURCE_NOT_VERIFIED'){{
      status.textContent+=' · Highlightly: origen de investigación aún NO VERIFICADO.';
    }}

  }}
  function slateCoverage(r){{const c=r?.coverage||{{}},sport=Number(c.sport_evidence_count||0),market=Number(c.market_evidence_count||0),rows=Number(c.analysis_rows||0),parts=[];if(rows>0)parts.push(rows+' current analysis row'+(rows===1?'':'s'));if(sport>0)parts.push('Sport data: '+sport+' snapshots');else parts.push('Sport data: NOT VERIFIED');if(market>0)parts.push('Market data: '+market+' snapshots');if(c.raw_sport_projection_present===false&&rows>0)parts.push('Raw sport projection NOT VERIFIED');return parts.join(' · ')}}
  function slateRow(r){{const f=r?.fixture||{{}},s=r?.state||{{}},c=r?.coverage||{{}},e=r?.screening||{{}},[time,date]=dt(f.kickoff),hasIntel=Number(c.analysis_rows||0)>0,status=String(s.display_status||c.coverage_status||'NOT VERIFIED').toUpperCase(),tone=status==='BET'?'bet':status==='LEAN'?'lean':status==='WATCH'?'watch':'',href=matchPath(f.fixture_id||'');const screenReason=e.reason||s.reason_display||'Estado de análisis: NOT VERIFIED';const missing=Array.isArray(e.missing_evidence_codes)?e.missing_evidence_codes:[];const labels={{RAW_SPORT_NOT_RECORDED:'proyección RAW',SPORT_FEATURE_SNAPSHOT_NOT_RECORDED:'estadísticas deportivas guardadas',LINEUP_SNAPSHOT_NOT_RECORDED:'registro de alineación',AVAILABILITY_SNAPSHOT_NOT_RECORDED:'registro de disponibilidad',MARKET_SNAPSHOT_NOT_RECORDED:'precio guardado'}};const missingText=missing.map(k=>labels[k]||k);const sources=Array.isArray(c.data_sources)?c.data_sources:[];const h=e?.supplementary_sport_research||{{}};const hReason=(ACCESS?.owner||ACCESS?.admin)&&h.source==='HIGHLIGHTLY_BASIC_RESEARCH_ONLY'?'Highlightly BASIC · '+(h.identity_status==='IDENTITY_EXACT_VERIFIED'?'partido validado':'identidad pendiente')+' · '+(h.evidence_status==='PREMATCH_RESEARCH_LIMITED'?'captura prepartido experimental':'sin evidencia apta para modelar')+' · datos: '+((h.families_research||[]).join(', ')||'NO VERIFICADOS')+' · NO BET · licencia pendiente':null;return '<a class="slate-row" href="'+esc(href)+'" data-fixture="'+esc(f.fixture_id||'')+'" data-intel="'+(hasIntel?'1':'0')+'" title="Open every persisted fact available for this fixture"><div class="time"><b>'+esc(time)+'</b>'+esc(date)+'</div><div class="row-teams"><b>'+esc((f.home_team||'Home')+' vs '+(f.away_team||'Away'))+'</b><small>'+esc(f.league||f.country||'Competition')+' · '+esc(slateCoverage(r))+'</small><span class="radar-reason">'+esc(screenReason)+'</span><span class="radar-evidence">'+esc('Fuente: '+(sources.join(', ')||'registro de fixtures')+' · Faltan: '+(missingText.join(', ')||'sin faltantes comprobados'))+'</span>'+(hReason?'<span class="radar-reason">'+esc(hReason)+'</span>':'')+'</div><div class="row-state"><span class="status '+tone+'">'+esc(status)+'</span><span class="link-btn">Open data →</span></div></a>'}}
  function renderWatchSlate(){{renderEdgeSportRadar();const w=DATA?.watches?.rows||[],s=DATA?.slate?.rows||[];$('todayWatches').innerHTML=w.map(watchRow).join('')||empty('No active WATCH states','No persisted wait state is active in this snapshot.');
    const slate=s.map(slateRow).join('')||empty('No eligible fixtures','No eligible fixtures remain in the current slate.');$('todaySlate').innerHTML=slate;$('matchesList').innerHTML=slate;
  }}
  function renderPlan(){{ACCESS=DATA?.access||ACCESS||{{}};const plan=ACCESS?.display_role||ACCESS?.effective_plan||(token()?'FREE':'EXPLORER');$('planChip').textContent=String(plan).toUpperCase();$('accountBtn').textContent=token()?'Account':'Sign in';$('accountPageBtn').textContent=token()?'Manage account':'Sign in';$('accountPlan').textContent=premium()?'Edge Pro':'Explorer';$('accountCopy').textContent=premium()?'Premium decision evidence is unlocked for this account.':'Browse the verified slate and watch states. Premium probability, price and evidence fields remain redacted.';$('signOut').classList.toggle('hidden',!token())}}
  function renderMeta(){{const raw=DATA?.generated_at_utc||DATA?.generated_at_local||'',d=new Date(raw);const readable=raw&&!Number.isNaN(d.getTime())?new Intl.DateTimeFormat('es-MX',{{timeZone:'America/Mexico_City',dateStyle:'medium',timeStyle:'short'}}).format(d):'Fecha sin verificar';$('snapshotText').textContent='Datos guardados · '+readable;$('todayStamp').textContent=readable;$('todayStamp').title=raw||'Sin registro';$('freshChip').textContent=DATA?.status==='SUBSCRIBER_CONTRACT_V2_READY'?'DATOS GUARDADOS':'REVISAR DATOS'}}
  async function recoverSession(){{
    const refresh=localStorage.getItem(RK)||'';
    if(refresh&&cfg.auth_configured){{
      try{{
        const d=await authFetch('/auth/v1/token?grant_type=refresh_token',{{refresh_token:refresh}});
        if(d.access_token){{
          localStorage.setItem(AK,d.access_token);
          if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);
          return true;
        }}
      }}catch(_e){{/* invalid/expired refresh falls through to anonymous recovery */}}
    }}
    localStorage.removeItem(AK);
    localStorage.removeItem(RK);
    return false;
  }}
  async function loadToday(){{
    try{{DATA=await api('/today')}}
    catch(e){{
      if(e?.status===401&&token()){{
        await recoverSession();
        DATA=await api('/today');
      }}else throw e
    }}
    ACCESS=DATA.access||{{}};renderMeta();renderPlan();renderKpis();renderHero();renderCards();renderWatchSlate();renderSportFocus();void enrichIdentities().then(()=>{{renderSportFocus();renderCards();renderWatchSlate()}})
  }}
  function researchPriceCard(x){{
    const market=esc(x.market_family)+(x.line===null?'':' · '+esc(x.line));
    const oddsText=americanOddsFromDecimal(x.price_decimal);
    const evText=x.hypothetical_ev_pct===null?'NOT VERIFIED':((Number(x.hypothetical_ev_pct)>=0?'+':'')+Number(x.hypothetical_ev_pct).toFixed(2)+'%');
    const link='/app/match/'+encodeURIComponent(String(x.fixture_id||''));
    return '<div class="panel"><div><b>'+esc(x.home_team)+' vs '+esc(x.away_team)+'</b></div>'+
      '<div class="sub">'+esc(x.league)+' · '+esc(x.market_family)+' · '+esc(x.selection)+'</div>'+
      '<div><b>'+market+' @ '+oddsText+'</b> · '+esc(x.bookmaker)+'</div>'+
      '<div class="sub">RAW '+pct(x.raw_sport_probability)+' · Shrunk '+pct(x.market_shrunk_probability)+
      ' · Breakeven '+pct(x.breakeven_probability)+'</div>'+
      '<div class="sub">EV hipotético: '+evText+' · No es EV validado</div>'+
      '<div class="sub">Cuota actualizada por proveedor: '+esc(x.quote_provider_updated_at||'NOT VERIFIED')+'</div>'+
      '<div class="sub">WATCH · NO BET · confirmar precio y XI antes de actuar</div>'+
      '<a href="'+link+'" class="link-btn">Match Center</a></div>'
  }}
  let sgpPlayerObservations=[];
  let sgpSelectedPlayerIndexes=[];
  function sgpComboCard(x){{
    const odds=num(x.price_decimal),p=num(x.raw_joint_probability),ev=num(x.raw_hypothetical_ev_pct);
    const target=odds!==null&&odds>=2.1?'+110 o mejor':'-110 a +110';
    const stale=!x.source_quote_fresh;
    const sourceState=stale?'CUOTA OBSERVADA HISTÓRICA · NECESITA RECONFIRMAR':'Fuente reciente · RECONFIRMAR ANTES DE ACTUAR';
    return '<div class="panel"><div class="sub"><b>ÚNICA SEÑAL RAW DEL PARTIDO · NO BET</b></div><b>'+esc(x.home_team)+' vs '+esc(x.away_team)+'</b>'+
      '<div class="sub">'+esc(x.provider_market)+' · '+esc(x.selection)+'</div>'+
      '<div><b>Bet365 @ '+americanOddsFromDecimal(x.price_decimal)+'</b> · '+target+'</div>'+
      '<div class="sub">Probabilidad conjunta RAW: '+pct(p)+' · EV RAW hipotético: '+
       (ev===null?'NOT VERIFIED':(ev>=0?'+':'')+ev.toFixed(2)+'%')+'</div>'+
      '<div class="sub">Fuente actualizada: '+esc(x.quote_updated_at||'NOT VERIFIED')+'</div>'+
      '<div class="sub">'+sourceState+'</div>'+
      '<div class="sub">WATCH · Mercado combinado COTIZADO por casa, NO SGP personalizado, no BET</div>'+
      '<a href="/app/match/'+encodeURIComponent(String(x.fixture_id))+'" class="link-btn">Ver partido</a></div>';
  }}
  function sgpCatalogCard(x){{
    const ev=num(x.raw_hypothetical_ev_pct);
    const reason=String(x.catalog_reason||'NO_JOINT_PROBABILITY_MODEL');
    const message=reason==='NON_POSITIVE_RAW_HYPOTHETICAL_EV'?'DESCARTADA: EV RAW no positivo':
      reason==='NO_JOINT_PROBABILITY_MODEL'?'SIN MODELO CONJUNTO · NO CANDIDATA':
      reason==='SUPPRESSED_ALTERNATIVE_SAME_FIXTURE'?'ALTERNATIVA SUPRIMIDA: máximo una señal por partido':
      'Ya aparece como señal RAW experimental arriba';
    return '<div class="panel"><div><b>'+esc(x.home_team)+' vs '+esc(x.away_team)+'</b></div>'+
      '<div class="sub">'+esc(x.provider_market)+' · '+esc(x.selection)+'</div>'+
      '<div>Cuota Bet365 observada: '+americanOddsFromDecimal(x.price_decimal)+'</div>'+
      '<div class="sub">EV RAW hipotético: '+(ev===null?'NO CALCULADO':((ev>=0?'+':'')+ev.toFixed(2)+'%'))+'</div>'+
      '<div class="sub">'+esc(message)+' · '+(x.source_quote_fresh?'fuente reciente':'cuota histórica, reconfirmar')+'</div>'+
      '<div class="sub">CATÁLOGO DE MERCADO, NO PICK · NO BET</div></div>';
  }}
  function playerSportModelCard(x){{
    const family=x.family==='SOT'?'Tiros a puerta':'Tiros';
    const projected=num(x.projected_count),rate=num(x.historical_rate_per90);
    const played=num(x.historical_played_matches),minutes=num(x.expected_minutes);
    const observedMinutes=num(x.historical_observed_minutes);
    const opponent=num(x.opponent_adjustment_factor);
    const val=n=>n===null?'NOT VERIFIED':n.toFixed(2);
    return '<article class="sport-model-projection">'+
      '<div class="sport-model-top"><b>'+esc(x.player||'Jugador')+'</b>'+
      '<span class="sgp-caution-tag">RESEARCH ONLY · OOS PENDIENTE</span></div>'+
      '<div class="sgp-leg-match">'+esc(x.home_team||'Local')+' vs '+
      esc(x.away_team||'Visitante')+' · '+esc(family)+'</div>'+
      '<div><span class="sport-model-big">'+esc(val(projected))+'</span>'+
      '<span class="sport-model-sub"> · '+esc(x.family==='SOT'?'tiros a puerta previstos':'tiros previstos')+', estimación bruta sin calibrar</span></div>'+
      '<div class="sport-model-stats">'+
      '<div class="sport-model-stat"><b>'+esc(val(rate))+'</b><small>Histórico por 90 minutos</small></div>'+
      '<div class="sport-model-stat"><b>'+esc(played===null?'NOT VERIFIED':String(played))+'</b><small>Partidos con minutos verificados</small></div>'+
      '<div class="sport-model-stat"><b>'+esc(minutes===null?'NOT VERIFIED':String(minutes))+'</b><small>Minutos esperados</small></div>'+
      '</div><div class="sport-model-foot">Minutos históricos: '+esc(observedMinutes===null?'NOT VERIFIED':String(Math.round(observedMinutes)))+
      ' · Factor rival: '+esc(val(opponent))+
      ' · ID jugador: '+esc(x.player_id||'NOT VERIFIED')+
      ' · Predicción: '+esc(sgpPlayerTime(x.pregame_snapshot_at_utc))+
      '. Sin probabilidad OOS aprobada ni apuesta sugerida.</div></article>';
  }}
  function renderPlayerSportResearch(value){{
    const area=$('playerSportModelSummary'),box=$('playerSportModelRows');
    if(!area||!box)return;
    const root=value&&typeof value==='object'?value:{{}};
    const sport=root.sport&&typeof root.sport==='object'?root.sport:{{}};
    const matches=root.market_alignment&&typeof root.market_alignment==='object'?root.market_alignment:{{}};
    const oos=root.oos&&typeof root.oos==='object'?root.oos:{{}};
    const families=oos.families&&typeof oos.families==='object'?oos.families:{{}};
    const example=(key)=>{{
      const item=families[key]||{{}};
      const count=num(item.unique_player_fixtures);
      return count===null?'NO VERIFICADO':String(Math.floor(count));
    }};
    const sportRows=Array.isArray(sport.rows)?sport.rows:[];
    const quoteMatches=num(matches.raw_model_exact_matches)||0;
    const headline=sportRows.length?
      'Proyecciones deportivas independientes: '+sportRows.length:
      'Sin proyecciones completas verificadas para los próximos partidos';
    area.innerHTML='<strong>'+esc(headline)+'</strong>'+
      '<div class="sport-model-sub">Base de validación fuera de muestra: tiros '+
      esc(example('SHOTS'))+'/500; tiros a puerta '+esc(example('SOT'))+
      '/500. Para revisión accionable se requieren al menos 2,000 observaciones y validación de precisión. <b>Actualmente ninguna apuesta está autorizada.</b></div>'+
      '<div class="sport-model-sub">Ofertas enlazadas por ID, XI, línea exacta y tiempo: '+
      esc(quoteMatches)+'. Estas coincidencias son diagnósticas, no picks. '+
      'No se usan cuotas para generar ni ordenar predicciones.</div>'+
      '<div class="sport-model-sub">Estado OOS: '+esc(oos.status||'NOT VERIFIED')+
      ' · Estado de fuentes prepartido: '+esc(sport.status||root.status||'NOT VERIFIED')+'</div>';
    box.innerHTML=sportRows.length?sportRows.slice(0,12).map(playerSportModelCard).join(''):'';
  }}
  function sgpPlayerMarketName(x){{
    const raw=String(x.provider_market||'Mercado individual');
    if(/shots? on target|on target shots?|sot/i.test(raw))return 'Tiros a puerta';
    if(/shots?|disparos?/i.test(raw))return 'Tiros del jugador';
    if(/goalscorer|anytime goal|player to score/i.test(raw))return 'Jugador goleador';
    return raw;
  }}
  function sgpPlayerQuantity(x){{
    const raw=String(x.selection||'');
    const m=raw.match(/-\\s*(\\d+(?:\\.\\d+)?)\\s*$/);
    if(!m)return null;
    const n=Number(m[1]);
    return Number.isFinite(n)&&n>=0&&n<=100?m[1]:null;
  }}
  function sgpPlayerShortSelection(x){{
    const quantity=sgpPlayerQuantity(x),family=sgpPlayerMarketName(x);
    if(quantity!==null){{
      const singular=Number(quantity)===1;
      if(family==='Tiros del jugador')return quantity+(singular?' tiro':' tiros');
      if(family==='Tiros a puerta')return quantity+(singular?' tiro a puerta':' tiros a puerta');
      if(family==='Jugador goleador')return quantity+' (referencia del mercado)';
      return quantity+' · selección publicada';
    }}
    return String(x.selection||'Selección no verificada');
  }}
  function sgpPlayerTime(raw){{
    const d=new Date(raw||'');
    if(Number.isNaN(d.getTime()))return 'Fecha no verificada';
    return new Intl.DateTimeFormat('es-MX',{{
      timeZone:'America/Mexico_City',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'
    }}).format(d)+' CDMX';
  }}
  // Defense-in-depth: stale or raw price-only source JSON never becomes
  // a player candidate in the UI. Independent sport evidence must precede
  // line/price comparison and have OOS and starting-XI verification.
  function playerStatEvidenceEligible(x){{
    const e=x?.independent_player_sport_evidence;
    if(!e||typeof e!=='object')return false;
    const modelProbability=num(e.raw_sport_probability);
    const rate=num(e.player_shots_or_sot_per90),sample=num(e.verified_sample_matches);
    const expectedMinutes=num(e.expected_minutes);
    return x?.player_sport_gate==='VERIFIED_SPORT_FIRST_OOS_MARKET_ALIGNED'&&
      e.source==='CANONICAL_INDEPENDENT_PLAYER_SPORT_MODEL'&&
      e.player_identity_verified===true&&e.historical_stats_verified===true&&
      e.opponent_adjusted===true&&e.minutes_projection_verified===true&&
      e.starting_xi_verified===true&&e.exact_market_operator_verified===true&&
      e.oos_validation_complete===true&&e.projection_before_market===true&&
      modelProbability!==null&&modelProbability>0&&modelProbability<1&&
      rate!==null&&rate>=0&&sample!==null&&sample>0&&
      expectedMinutes!==null&&expectedMinutes>0&&expectedMinutes<=120&&
      x?.source_quote_fresh===true;
  }}
  function sgpPlayerCard(x,index){{
    const recent=x.source_quote_fresh===true;
    const quantity=sgpPlayerQuantity(x),marketName=sgpPlayerMarketName(x);
    const role=String(x.observed_team_side||'').toUpperCase()==='HOME'?'Equipo local':
      String(x.observed_team_side||'').toUpperCase()==='AWAY'?'Equipo visitante':'Equipo por confirmar';
    const selected=sgpSelectedPlayerIndexes.includes(index);
    const price=americanOddsFromDecimal(x.price_decimal);
    const reference=quantity!==null?sgpPlayerShortSelection(x):'Ver selección original';
    return '<article class="sgp-leg-card'+(selected?' selected':'')+'">'+
      '<div class="sgp-leg-head"><div class="sgp-leg-player">'+
      '<b>'+esc(x.player_label||'Jugador sin identificar')+'</b>'+
      '<span class="sgp-leg-team">'+esc(role)+'</span></div>'+
      '<span class="sgp-caution-tag">NO BET · pendiente</span></div>'+
      '<div class="sgp-leg-match">'+esc(x.home_team||'Local')+' vs '+esc(x.away_team||'Visitante')+'</div>'+
      '<div class="sgp-leg-market"><span class="sgp-leg-pill">'+esc(marketName)+'</span>'+
      '<span class="sgp-leg-pill muted">Bet365 · selección individual</span></div>'+
      '<div class="sgp-leg-main"><div class="sgp-leg-count">'+
      '<strong>'+esc(reference)+'</strong>'+
      '<small>Referencia publicada. La condición exacta («más de», «al menos» o exacta) no está confirmada.</small></div>'+
      '<div class="sgp-leg-price"><strong>'+esc(price)+'</strong>'+
      '<small>Cuota americana individual</small></div></div>'+
      '<div class="sgp-leg-review"><span class="sgp-caution-tag">'+
      (recent?'Precio reciente · confirmar':'Precio anterior · reconfirmar')+'</span>'+
      '<span>Actualización: '+esc(sgpPlayerTime(x.quote_updated_at))+'</span></div>'+
      '<div class="sgp-leg-details"><b>Falta comprobar:</b> titularidad y minutos, reglas de liquidación de este mercado y precio conjunto de una combinada.</div>'+
      '<details class="sgp-leg-source"><summary>Ver selección original del proveedor</summary>'+
      '<div>'+esc(x.provider_market||'Mercado no verificado')+' · '+esc(x.selection||'Sin selección verificada')+
      '</div></details>'+
      '<label class="sgp-leg-select"><input type="checkbox" data-sgp-player-index="'+index+'" '+
      (selected?'checked':'')+' aria-label="Incluir '+esc(x.player_label||'jugador')+
      ' en borrador de investigación" />'+
      '<span>Incluir en borrador de investigación</span></label></article>';
  }}
  function renderSgpPlayerSlip(message=''){{
    const slip=$('sgpBuilderSlip');
    if(!slip)return;
    const legs=sgpSelectedPlayerIndexes.map(i=>sgpPlayerObservations[i]).filter(Boolean);
    const count=legs.length;
    const head='<div class="sgp-slip-title"><b>Borrador de investigación · '+count+'/3 jugadores</b>'+
      '<span class="sgp-caution-tag">NO BET · sin cuota combinada</span></div>';
    if(!count){{
      slip.innerHTML=head+
        '<div class="sgp-slip-description">Marca hasta tres mercados de jugadores del <b>mismo partido</b> para compararlos. No se suman ni multiplican cuotas y no se envía ninguna apuesta.</div>'+
        (message?'<div class="sgp-slip-warning">'+esc(message)+'</div>':'');
      return;
    }}
    const allSameGame=legs.every(x=>x.fixture_id===legs[0].fixture_id);
    const selectionList='<div class="sgp-slip-items">'+legs.map(x=>
      '<div class="sgp-slip-item"><b>'+esc(x.player_label||'Jugador')+'</b> · '+
      esc(sgpPlayerShortSelection(x))+'</div>'
    ).join('')+'</div>';
    slip.innerHTML=head+
      '<div class="sgp-slip-description">'+esc(legs[0].home_team||'Local')+' vs '+
      esc(legs[0].away_team||'Visitante')+
      ' · Comparación de mercados individuales, no un boleto de la casa.</div>'+
      selectionList+
      '<div class="sgp-slip-warning"><b>CUOTA COMBINADA: NOT VERIFIED.</b> La casa debe cotizar el boleto real; tampoco está validada la correlación entre jugadores.</div>'+
      '<div class="sgp-slip-help">Antes de considerar cualquier acción faltan alineaciones, minutos y reglas verificadas. Este borrador no recomienda ni realiza apuestas.</div>'+
      (!allSameGame?'<div class="sgp-slip-warning">Partidos diferentes: no pueden formar la misma SGP.</div>':'')+
      (message?'<div class="sgp-slip-warning">'+esc(message)+'</div>':'');
  }}
  function wireSgpPlayerSelection(){{
    const container=$('sgpPlayerLegs');
    if(!container)return;
    container.querySelectorAll('input[data-sgp-player-index]').forEach(input=>{{
      input.addEventListener('change',()=>{{
        const idx=Number(input.getAttribute('data-sgp-player-index'));
        const row=sgpPlayerObservations[idx];
        if(!row)return;
        if(input.checked){{
          const first=sgpSelectedPlayerIndexes.length?sgpPlayerObservations[sgpSelectedPlayerIndexes[0]]:null;
          if((first&&first.fixture_id!==row.fixture_id)||sgpSelectedPlayerIndexes.length>=3){{
            input.checked=false;
            renderSgpPlayerSlip('Solo puedes elegir hasta tres jugadores del mismo partido. Quita uno para cambiar de encuentro.');
            return;
          }}
          sgpSelectedPlayerIndexes.push(idx);
        }}else sgpSelectedPlayerIndexes=sgpSelectedPlayerIndexes.filter(i=>i!==idx);
        const card=input.closest('.sgp-leg-card');
        if(card)card.classList.toggle('selected',input.checked);
        renderSgpPlayerSlip();
      }});
    }});
  }}
  async function loadResearchDesk(){{
    const status=$('researchStatus'),plus=$('researchPlus100'),regular=$('researchRegular');
    const comboStatus=$('sgpResearchStatus'),comboBox=$('sgpQuotedComposites'),playerBox=$('sgpPlayerLegs'),catalogBox=$('sgpMarketCatalog');
    if(!status||!plus||!regular)return;
    if(!(ACCESS?.owner||ACCESS?.admin)){{
      status.textContent='Radar experimental restringido a owner/admin; no es una lista pública de apuestas.';
      plus.innerHTML='';regular.innerHTML='';
      if(comboStatus)comboStatus.textContent='Investigación SGP solo para owner/admin.';
      if(comboBox)comboBox.innerHTML='';
      if(catalogBox)catalogBox.innerHTML='';
      if(playerBox)playerBox.innerHTML='';
      return;
    }}
    status.textContent='Consultando cuotas recientes y proyecciones registradas...';
    try{{
      const data=await api('/research-price-desk');
      const rows=Array.isArray(data.rows)?data.rows:[];
      const high=rows.filter(x=>x.price_bucket==='EDGE_MINUS110_OR_BETTER');
      const low=rows.filter(x=>x.price_bucket==='EDGE_REGULAR');
      plus.innerHTML=high.map(researchPriceCard).join('');
      regular.innerHTML=low.map(researchPriceCard).join('');
      const combos=Array.isArray(data.sgp_composites)?data.sgp_composites:[];
      const catalog=Array.isArray(data.sgp_market_catalog)?data.sgp_market_catalog:[];
      renderPlayerSportResearch(data.player_sport_research);
      const rawPlayerRows=Array.isArray(data.sgp_player_components)?data.sgp_player_components:[];
      const playerRows=rawPlayerRows.filter(playerStatEvidenceEligible);
      const priceOnlyCount=num(data.sgp_player_quote_catalog_count)||0;
      sgpPlayerObservations=playerRows;
      sgpSelectedPlayerIndexes=[];
      if(comboBox)comboBox.innerHTML=combos.length?combos.map(sgpComboCard).join(''):
        '<div class="panel"><b>Sin señales SGP con EV RAW positivo</b><div class="sub">Las cuotas pueden consultarse en el catálogo, pero NO son picks.</div></div>';
      if(catalogBox)catalogBox.innerHTML=catalog.map(sgpCatalogCard).join('');
      if(playerBox)playerBox.innerHTML=playerRows.length?
        playerRows.slice(0,42).map(sgpPlayerCard).join(''):
        '<div class="panel"><b>0 jugadores con ventaja estadística verificada</b>'+
        '<div class="sub">Se detectaron '+esc(priceOnlyCount)+' cotizaciones individuales, pero ninguna acredita todavía estadísticas del jugador, minutos/XI, proyección propia, línea exacta y validación OOS. No aparecen como picks. No se inventarán cuotas ni probabilidades. Una cuota +110 NO es una proyección.</div></div>';
      if(comboStatus){{
        const fresh=combos.filter(x=>x.source_quote_fresh).length;
        const rejected=catalog.filter(x=>x.catalog_reason==='NON_POSITIVE_RAW_HYPOTHETICAL_EV').length;
        const unmodeled=catalog.filter(x=>x.catalog_reason==='NO_JOINT_PROBABILITY_MODEL').length;
        comboStatus.textContent=('Señales RAW experimentales: '+combos.length+
          ' (cuotas recientes: '+fresh+'). Descartadas por EV RAW no positivo: '+rejected+
          '. Sin probabilidad conjunta: '+unmodeled+
          '. Máximo UNA señal por encuentro. Todas son WATCH, NO BET. '+
          'Mercados descartados disponibles solo en el catálogo de cuotas.');
      }}
      if(playerRows.length){{
        renderSgpPlayerSlip();
        wireSgpPlayerSelection();
      }}else if($('sgpBuilderSlip')){{
        $('sgpBuilderSlip').innerHTML='<div class="sgp-slip-title"><b>Sin selecciones de jugadores verificadas</b>'+
          '<span class="sgp-caution-tag">SPORT FIRST · NO BET</span></div>'+
          '<div class="sgp-slip-description">No se crea borrador con cotizaciones aisladas. Hasta contar con estadísticas y proyección deportiva independiente, la lista permanece vacía.</div>';
      }}
      status.textContent=rows.length?
       'INVESTIGACIÓN · '+rows.length+' precios vigentes · WATCH solamente · Push próximamente':
       'No hay cuotas vigentes verificadas para partidos futuros (ventana 90 min); no mostrar precios viejos.';
    }}catch(e){{
      status.textContent='Radar no disponible: NO VERIFICADO. La app canónica funciona por separado.';
      plus.innerHTML='';regular.innerHTML='';
      if(comboStatus)comboStatus.textContent='SGP: fuente NO VERIFICADA.';
      if(comboBox)comboBox.innerHTML='';
      if(catalogBox)catalogBox.innerHTML='';
      if(playerBox)playerBox.innerHTML='';
    }}
  }}
  function matchPath(fid){{return '/app/match/'+encodeURIComponent(fid)}}
  function routeMatchId(){{const parts=location.pathname.split('/').filter(Boolean);return parts.length===3&&parts[0]==='app'&&parts[1]==='match'&&/^[0-9]+$/.test(parts[2])?parts[2]:null}}
  function setMatchRoute(fid,replace=false){{const url=matchPath(fid);if(location.pathname===url)return;(replace?history.replaceState:history.pushState).call(history,{{page:'match',fixture_id:String(fid)}},'',url)}}
  function showMatchBrowse(show){{const panel=$('matchesBrowsePanel');if(panel)panel.classList.toggle('hidden',!show)}}
  const analysisLabels={{MODELED_LIMITED:'Proyección RAW registrada',PARTIAL:'Análisis parcial',SPORT_EVIDENCE_ONLY:'Evidencia deportiva',NOT_ANALYZED:'Aún sin screening',DATA_INSUFFICIENT:'Datos insuficientes',CANCELLED:'Partido cancelado',POSTPONED:'Partido aplazado'}};
  const valueLabels={{RECOMENDACION_DE_VALOR:'Recomendación de valor',TENDENCIA_FAVORABLE:'Tendencia favorable',EN_OBSERVACION:'En observación',SIN_VALOR:'Sin valor identificado',SPORT_SCREEN_PASS:'Sin shortlist deportiva',MERCADO_NO_EVALUADO:'Mercado no evaluado',VERIFICACION_PENDIENTE:'Verificación pendiente',EVENTO_NO_VIGENTE:'Evento cancelado/aplazado'}};
  function publicMatchCard(row){{
    const f=row?.fixture||{{}},s=row?.sport_analysis||{{}},o=row?.opportunity||{{}},
      [clock,date]=dt(f.kickoff),m=Array.isArray(s.missing_evidence_codes)?s.missing_evidence_codes:[],
      tracks=s.tracks||{{}};
    return '<div class="universal-match"><div class="public-match-heading"><span>ANÁLISIS DEPORTIVO · REGISTRO VERIFICADO</span>'+
      '<button type="button" id="publicMatchBack" class="secondary">Volver a los partidos</button></div>'+
      '<h2>'+esc((f.home_team||'Local')+' vs '+(f.away_team||'Visitante'))+'</h2>'+
      '<p>'+esc(f.league||f.country||'Competición')+' · '+esc(date)+' · '+esc(clock)+'</p>'+
      '<div class="universal-status-grid"><div class="visual-card"><div class="visual-card-head"><h3>Estado deportivo</h3></div>'+
      '<strong>'+esc(analysisLabels[s.status]||'NOT VERIFIED')+'</strong><p>'+esc(s.explanation||'Sin evidencia suficiente.')+'</p>'+
      '<p>Modelos registrados: '+esc(s.model_run_count??'NOT VERIFIED')+' · Snapshots estadísticos: '+esc(s.verified_feature_snapshot_count??'NOT VERIFIED')+'</p>'+
      '</div><div class="visual-card"><div class="visual-card-head"><h3>Evaluación de oportunidad</h3></div>'+
      '<strong>'+esc(valueLabels[o.status]||'Mercado no evaluado')+'</strong><p>'+esc(o.explanation||'No hay evaluación verificable.')+'</p>'+
      '<p>Decisión histórica de análisis · Sin precio ejecutable actual verificado.</p></div></div>'+
      '<div class="visual-card"><h3>Tracks deportivos</h3><div class="sport-stat-grid">'+
      ['SIDE','GOALS_BTTS','CORNERS'].map(k=>'<div class="sport-stat"><span>'+esc(k)+'</span><b>'+esc(tracks[k]||'NOT VERIFIED')+'</b></div>').join('')+'</div>'+
      '<p>Proyección numérica y datos de proveedores sujetos a licencia y validación OOS; no se inventan porcentajes.</p></div>'+
      '<details class="universal-missing"><summary>Qué falta por verificar ('+m.length+')</summary><div class="universal-missing-body">'+
      (m.length?m.map(x=>'<div>'+esc(x)+'</div>').join(''):'Sin faltantes documentados; no implica cobertura completa.')+'</div></details>'+
      '<p class="intel-footnote">Un PASS no oculta el análisis. La decisión de mercado no es lo mismo que el estado deportivo.</p>'+
      (token()?'<section class="analyst-notebook"><h3>Mi lectura del partido</h3>'+
        '<p>Nota privada · No modifica el modelo</p>'+
        '<textarea id="analystNoteText" maxlength="2000" rows="4" placeholder="Mi hipótesis deportiva…"></textarea>'+
        '<div class="analyst-note-actions"><button type="button" class="secondary" id="saveAnalystNote">Guardar nota</button>'+
        '<button type="button" class="secondary" id="deleteAnalystNote">Eliminar</button>'+
        '<span id="analystNoteStatus">Solo visible para esta cuenta.</span></div></section>':
        '<section class="analyst-notebook"><h3>Tu lectura del partido</h3>'+
        '<p>Accede con tu cuenta para guardar hipótesis privadas. No necesitas un plan PRO para crear notas.</p>'+
        '<button type="button" class="secondary" id="publicNoteLogin">Iniciar sesión</button></section>')+
      '</div>';
  }}
  function publicMatchFromSlate(fid){{return (DATA?.slate?.rows||[]).find(r=>String(r?.fixture?.fixture_id)===String(fid))||null}}
  function showPublicMatch(fid){{const box=$('matchDetail'),row=publicMatchFromSlate(fid);box.innerHTML=row?publicMatchCard(row):
    empty('Partido no encontrado','No consta este partido en el calendario elegible guardado.');
    const back=$('publicMatchBack');if(back)back.onclick=closeMatchRoute;
    const login=$('publicNoteLogin');if(login)login.onclick=openAuth;
    if(token())void loadAnalystNote(fid);
    box.scrollIntoView({{block:'start'}})}}
  async function openMatch(fid,opts={{}}){{if(!fid)return;activate('matches');setMatchRoute(fid,!!opts.replace);showMatchBrowse(false);const box=$('matchDetail');box.classList.remove('hidden');if(!premium()){{showPublicMatch(fid);return}}box.innerHTML=empty('Cargando análisis deportivo','Consultando los registros persistidos…');box.scrollIntoView({{block:'start'}});try{{const d=await api('/match/'+encodeURIComponent(fid));renderMatch(d);box.scrollIntoView({{block:'start'}})}}catch(e){{if(e?.status===403)showPublicMatch(fid);else box.innerHTML=empty('Detalle no disponible',e.message)}}}}

  async function loadAnalystNote(fid){{
    const area=$('analystNoteText'),status=$('analystNoteStatus'),save=$('saveAnalystNote'),del=$('deleteAnalystNote');
    if(!area||!status||!save||!del||!fid)return;
    save.onclick=async()=>{{const text=area.value.trim();if(!text||text.length>2000){{status.textContent='La nota requiere 1–2000 caracteres.';return}}
      try{{await apiWrite('/my-edge','POST',{{item_type:'NOTE',fixture_id:Number(fid),note_text:text}});status.textContent='Nota guardada en tu cuenta. No afecta al modelo.'}}
      catch(e){{status.textContent='No se pudo guardar: '+e.message}}}};
    del.onclick=async()=>{{try{{await apiWrite('/my-edge?resource=notes&fixture_id='+encodeURIComponent(fid),'DELETE');
      area.value='';status.textContent='Nota eliminada.'}}catch(e){{status.textContent='No se pudo eliminar: '+e.message}}}};
    try{{const d=await api('/my-edge?resource=notes&fixture_id='+encodeURIComponent(fid));
      if(String(routeMatchId())!==String(fid))return;
      const row=(d.rows||[])[0];if(row&&typeof row.note_text==='string')area.value=row.note_text;
      status.textContent=row?'Nota privada recuperada.':'Sin notas guardadas todavía.';}}
    catch(e){{status.textContent='Notas no disponibles: '+e.message}}
  }}
  function closeMatchRoute(){{history.pushState({{page:'matches'}},'','/app');showMatchBrowse(true);$('matchDetail').classList.add('hidden');activate('matches')}}
  function detailKV(label,value){{return '<div class="item"><span>'+esc(label)+'</span><span>'+esc(value??'NOT VERIFIED')+'</span></div>'}}
  function marketTable(rows){{if(!rows?.length)return empty('No persisted market rows','This market family has no persisted candidate rows for the fixture.');
    return '<div class="table-wrap"><table><thead><tr><th>Decision</th><th>Market</th><th>Selection</th><th>Line</th><th>Price</th><th>Book</th><th>Raw</th><th>Shrunk</th><th>Fair Mkt</th><th>Edge</th></tr></thead><tbody>'+
      rows.map(c=>{{const d=c?.decision||{{}},m=c?.market||{{}},p=c?.projections||{{}},pr=m.price||{{}};return '<tr><td>'+esc(d.classification||d.display_bucket||d.execution_status||'WATCH')+'</td><td>'+esc(m.name||m.family||'—')+'</td><td>'+esc(m.selection||'—')+'</td><td>'+esc(m.line??'—')+'</td><td>'+esc(pr.value??'NOT VERIFIED')+'</td><td>'+esc(pr.bookmaker||'NOT VERIFIED')+'</td><td>'+pct(p.raw_sport_probability)+'</td><td>'+pct(p.market_shrunk_probability)+'</td><td>'+pct(p.fair_market_probability)+'</td><td>'+pp(p.probability_edge_pp)+'</td></tr>'}}).join('')+
      '</tbody></table></div>'}}
  function compactJson(value){{try{{const s=JSON.stringify(value??null);return s.length>260?s.slice(0,257)+'…':s}}catch(_){{return 'NOT VERIFIED'}}}}
  function persistedMarketTable(rows){{if(!rows?.length)return empty('No persisted market snapshots','No relational market snapshots are stored for this fixture.');
    return '<div class="table-wrap"><table><thead><tr><th>Captured</th><th>Market</th><th>Book</th><th>Values</th><th>Provider update</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+esc(r.captured_at||'NOT VERIFIED')+'</td><td>'+esc(r.market||'NOT VERIFIED')+'</td><td>'+esc(r.bookmaker||'NOT VERIFIED')+'</td><td>'+esc(compactJson(r.values||[]))+'</td><td>'+esc(r.provider_update||'NOT VERIFIED')+'</td></tr>').join('')+'</tbody></table></div>'}}
  function persistedModelTable(rows){{if(!rows?.length)return empty('No persisted model runs','No relational model run is stored for this fixture.');
    return '<div class="table-wrap"><table><thead><tr><th>Run</th><th>Type</th><th>Class</th><th>Model P</th><th>Market fair</th><th>Edge</th><th>EV</th><th>Availability</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+esc(r.run_timestamp||'NOT VERIFIED')+'</td><td>'+esc(r.run_type||'NOT VERIFIED')+'</td><td>'+esc(r.classification||'NOT VERIFIED')+'</td><td>'+pct(r.model_probability)+'</td><td>'+pct(r.market_fair_probability)+'</td><td>'+pp(r.probability_edge_pp)+'</td><td>'+ev(r.estimated_ev)+'</td><td>'+pct(r.availability_confidence)+'</td></tr>').join('')+'</tbody></table></div>'}}
  function persistedFeatureTable(rows){{if(!rows?.length)return empty('No persisted feature snapshots','No relational feature snapshot is stored for this fixture.');
    return '<div class="table-wrap"><table><thead><tr><th>Captured</th><th>Stage</th><th>Tier</th><th>Features</th><th>Missing</th><th>Model</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+esc(r.captured_at||'NOT VERIFIED')+'</td><td>'+esc(r.stage||'NOT VERIFIED')+'</td><td>'+esc(r.data_tier||'NOT VERIFIED')+'</td><td>'+esc(r.feature_count??'—')+'</td><td>'+esc(r.missing_feature_count??'—')+'</td><td>'+esc(r.model_version||'NOT VERIFIED')+'</td></tr>').join('')+'</tbody></table></div>'}}
  function scoreMatrix(rows){{if(!rows?.length)return '<div class="visual-empty compact">Score matrix is NOT VERIFIED.</div>';
    return '<div class="score-grid">'+rows.slice(0,9).map(x=>'<div class="score-chip"><b>'+esc(x.score||'—')+'</b><small>'+pct(x.probability)+'</small></div>').join('')+'</div>'}}
  function sportProfile(rows){{if(!rows?.length)return '<div class="visual-empty compact">Sport profile is NOT VERIFIED.</div>';
    return rows.map(x=>{{const n=Math.max(0,Math.min(100,Number(x.score)||0));return '<div class="profile-row"><span>'+esc(x.label||'Metric')+'</span><div class="profile-bar"><i style="width:'+n+'%"></i></div><span>'+esc(Number.isFinite(Number(x.score))?Number(x.score).toFixed(0):'—')+'</span></div>'}}).join('')}}
  function evidenceList(e){{const groups=[['Sporting reasons',e?.sporting_reasons],['Market reasons',e?.market_reasons],['Blockers',e?.blockers],['Invalidation conditions',e?.invalidation_conditions]];const rows=[];for(const [label,items] of groups){{for(const item of (items||[]))rows.push(detailKV(label,item))}}return rows.join('')||detailKV('Evidence','No additional persisted explanation in this snapshot.')}}
  function prettyStamp(raw){{if(!raw)return 'NOT VERIFIED';const d=new Date(raw);if(Number.isNaN(d.getTime()))return String(raw);const now=Date.now(),diff=Math.round((now-d.getTime())/60000);if(diff>=0&&diff<60)return diff<=1?'just now':diff+'m ago';if(diff>=60&&diff<1440)return Math.round(diff/60)+'h ago';return new Intl.DateTimeFormat(undefined,{{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}}).format(d)}}
  function infoPills(items,kind=''){{const rows=[...new Set((items||[]).filter(Boolean).map(x=>String(x).replaceAll('_',' ')))];return rows.length?'<div class="intel-pills">'+rows.map(x=>'<span class="intel-pill '+kind+'">'+esc(x)+'</span>').join('')+'</div>':'<div class="intel-empty">Nothing verified here yet.</div>'}}
  function intelKV(rows){{return '<div class="intel-kv">'+rows.map(([k,v])=>'<div class="intel-kv-row"><span>'+esc(k)+'</span><b>'+esc(v??'NOT VERIFIED')+'</b></div>').join('')+'</div>'}}
  function evidenceTotal(counts){{return ['refresh_events','market_snapshots','model_runs','feature_snapshots','lineup_snapshots','availability_snapshots'].reduce((n,k)=>n+Number(counts?.[k]||0),0)}}
  function latestFeatureSnapshot(rel){{return (rel?.feature_snapshots||[])[0]||{{}}}}
  function sportFeatures(rel){{return latestFeatureSnapshot(rel)?.payload?.features||{{}}}}
  function featureValue(features,key){{const row=features?.[key];return row&&row.value!==null&&row.value!==undefined?row.value:null}}
  function featureSample(features,key){{const n=Number(features?.[key]?.sample_n);return Number.isFinite(n)?n:null}}
  function featureVerified(features,key){{const row=features?.[key];if(!row||row.value===null||row.value===undefined)return false;const source=String(row.source||'');if(source==='API_FOOTBALL_TEAM_STATS'){{let n=featureSample(features,key);if(n===null){{const side=key.includes('.home_')?'home':key.includes('.away_')?'away':null;if(side)n=featureSample(features,'team_performance.'+side+'_form')}}if(n!==null&&n<=0)return false;}}return true}}
  const CUSTOMER_SPORT_FEATURES={{
    'team_performance.home_goal_rate_blend':'Home goal-rate baseline',
    'team_performance.away_goal_rate_blend':'Away goal-rate baseline',
    'team_performance.total_goal_rate_blend':'Total goal-rate baseline',
    'team_performance.home_form':'Home form',
    'team_performance.away_form':'Away form',
    'team_performance.home_played_split':'Home-team home sample',
    'team_performance.away_played_split':'Away-team away sample',
    'team_performance.home_wins_total':'Home-team wins',
    'team_performance.home_draws_total':'Home-team draws',
    'team_performance.home_losses_total':'Home-team losses',
    'team_performance.away_wins_total':'Away-team wins',
    'team_performance.away_draws_total':'Away-team draws',
    'team_performance.away_losses_total':'Away-team losses',
    'team_performance.home_goals_for_avg':'Home-team goals for / match',
    'team_performance.home_goals_against_avg':'Home-team goals against / match',
    'team_performance.away_goals_for_avg':'Away-team goals for / match',
    'team_performance.away_goals_against_avg':'Away-team goals against / match',
    'team_performance.home_clean_sheets':'Home-team clean sheets',
    'team_performance.away_clean_sheets':'Away-team clean sheets',
    'team_performance.home_failed_to_score':'Home-team failed to score',
    'team_performance.away_failed_to_score':'Away-team failed to score',
    'team_performance.home_recent_matches':'Home recent matches captured',
    'team_performance.away_recent_matches':'Away recent matches captured',
    'availability.both_xi_confirmed':'Both starting XIs confirmed',
    'availability.both_goalkeepers_confirmed':'Both goalkeepers confirmed',
    'availability.home_formation':'Home formation',
    'availability.away_formation':'Away formation',
    'availability.injury_report_count':'Injury reports',
    'xg.xgf':'xG for',
    'xg.xga':'xG against',
    'xg.npxg':'Non-penalty xG',
    'xg.npxga':'Non-penalty xGA',
    'territory.ppda':'PPDA',
    'territory.field_tilt':'Field tilt',
    'territory.box_entries':'Box entries',
    'territory.final_third_entries':'Final-third entries',
    'context.venue':'Venue',
    'context.city':'City',
    'context.referee':'Referee',
    'context.weather':'Weather',
    'context.rest_days_home':'Home rest days',
    'context.rest_days_away':'Away rest days'
  }};
  function customerSportRows(features){{return Object.entries(CUSTOMER_SPORT_FEATURES).map(([key,label])=>{{const row=features?.[key]||{{}},v=row.value;if(!featureVerified(features,key))return null;let display=v;if(typeof v==='boolean')display=v?'YES':'NO';return {{key,label,value:display,source:row.source||'NOT VERIFIED',freshness:row.freshness||'NOT VERIFIED',sample_n:row.sample_n}}}}).filter(Boolean)}}
  function sportFeatureCount(features){{return customerSportRows(features).length}}
  function sportFeatureMissing(features){{return Object.keys(CUSTOMER_SPORT_FEATURES).filter(key=>!featureVerified(features,key)).length}}
  function verifiedSportInputs(features){{const rows=customerSportRows(features);if(!rows.length)return '<div class="intel-empty">No customer-facing sporting inputs are verified for this fixture yet.</div>';return '<div class="intel-snapshot-list">'+rows.map(r=>'<div class="intel-snapshot"><div><b>'+esc(r.label)+'</b><small>'+esc(r.source)+(r.sample_n!==null&&r.sample_n!==undefined?' · sample '+esc(r.sample_n):'')+'</small></div><div class="snap-right">'+esc(r.value)+'<br>'+esc(r.freshness)+'</div></div>').join('')+'</div>'}}
  function freshnessLabel(counts){{const stamps=[counts?.latest_refresh_at,counts?.latest_market_at,counts?.latest_model_at,counts?.latest_feature_at,counts?.latest_lineup_at,counts?.latest_availability_at].filter(Boolean).map(x=>new Date(x).getTime()).filter(Number.isFinite);return stamps.length?prettyStamp(new Date(Math.max(...stamps)).toISOString()):'NOT VERIFIED'}}
  function coverageTone(status){{const s=String(status||'').toUpperCase();return s.includes('BET')||s.includes('LEAN')||s.includes('DATA AVAILABLE')?'good':s.includes('WATCH')||s.includes('PARTIAL')?'info':'warn'}}
  function snapshotCards(rows,type){{if(!rows?.length)return '<div class="intel-empty">No persisted '+esc(type)+' records.</div>';return '<div class="intel-snapshot-list">'+rows.slice(0,12).map(r=>{{if(type==='feature')return '<div class="intel-snapshot"><div><b>'+esc(r.stage||'Feature snapshot')+'</b><small>'+esc(prettyStamp(r.captured_at))+' · '+esc(r.data_tier||'tier N/V')+'</small></div><div class="snap-right">'+esc(r.feature_count??'—')+' features<br>'+esc(r.missing_feature_count??'—')+' missing</div></div>';if(type==='model')return '<div class="intel-snapshot"><div><b>'+esc(r.classification||r.run_type||'Model run')+'</b><small>'+esc(prettyStamp(r.run_timestamp))+' · '+esc(r.model_version||'model N/V')+'</small></div><div class="snap-right">'+pct(r.model_probability)+' model<br>'+pp(r.probability_edge_pp)+' edge</div></div>';return '<div class="intel-snapshot"><div><b>'+esc(r.market||'Market snapshot')+'</b><small>'+esc(r.bookmaker||'Book N/V')+' · '+esc(prettyStamp(r.captured_at))+'</small></div><div class="snap-right">'+esc((r.values||[]).length)+' prices<br>'+esc(prettyStamp(r.provider_update))+'</div></div>'}}).join('')+'</div>'}}
  const CORE_SPORT_KEYS=[
    'team_performance.home_form','team_performance.away_form',
    'team_performance.home_goals_for_avg','team_performance.home_goals_against_avg',
    'team_performance.away_goals_for_avg','team_performance.away_goals_against_avg',
    'team_performance.home_goal_rate_blend','team_performance.away_goal_rate_blend',
    'availability.home_formation','availability.away_formation',
    'availability.both_xi_confirmed','availability.both_goalkeepers_confirmed'
  ];
  function nval(v){{if(v===null||v===undefined||v===''||typeof v==='boolean')return null;const n=Number(v);return Number.isFinite(n)?n:null}}
  function coreCoverage(features){{const verified=CORE_SPORT_KEYS.filter(k=>featureVerified(features,k)).length,total=CORE_SPORT_KEYS.length;return {{verified,total,pct:total?Math.round((verified/total)*100):0}}}}
  function coverageGauge(features){{const c=coreCoverage(features);return '<div class="coverage-wrap"><div class="coverage-gauge" style="--coverage:'+esc(c.pct)+'%"><div><b>'+esc(c.pct)+'%</b><span>campos</span></div></div><div class="coverage-copy"><b>'+esc(c.verified)+' / '+esc(c.total)+' core sport inputs</b><p>Porcentaje de campos disponibles en ESTE partido; no mide confianza del modelo ni porcentaje de partidos analizados. Market rows do not increase it.</p></div></div>'}}
  function formTokens(raw){{const s=String(raw||'').toUpperCase().replace(/[^WDL]/g,'');return [...s].slice(-5)}}
  function formDots(raw){{const rows=formTokens(raw);return rows.length?'<div class="form-dots">'+rows.map(x=>'<span class="form-dot '+x.toLowerCase()+'">'+x+'</span>').join('')+'</div>':'<span class="intel-pill missing">NOT VERIFIED</span>'}}
  function formSpark(raw,cls='home'){{const rows=formTokens(raw);if(!rows.length)return '';const pts=rows.map((x,i)=>{{const y=x==='W'?12:x==='D'?27:42;const xx=8+(i*(84/Math.max(1,rows.length-1)));return xx+','+y}}).join(' ');return '<svg class="form-spark '+cls+'" viewBox="0 0 100 54" role="img" aria-label="Verified form sequence"><polyline points="'+pts+'" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'+rows.map((x,i)=>{{const y=x==='W'?12:x==='D'?27:42,xx=8+(i*(84/Math.max(1,rows.length-1)));return '<circle cx="'+xx+'" cy="'+y+'" r="3.4" fill="currentColor"/>'}}).join('')+'</svg>'}}
  function formCompare(features,f){{const home=featureVerified(features,'team_performance.home_form')?featureValue(features,'team_performance.home_form'):null,away=featureVerified(features,'team_performance.away_form')?featureValue(features,'team_performance.away_form'):null;if(!home&&!away)return '<div class="visual-empty compact">Forma reciente: NOT VERIFIED.</div>';return '<div class="form-compare"><div class="form-team"><b>'+esc(f.home_team||'Local')+'</b>'+formDots(home)+'</div><div class="form-team"><b>'+esc(f.away_team||'Visitante')+'</b>'+formDots(away)+'</div></div><p class="form-legend">W = victoria · D = empate · L = derrota. Últimos cinco resultados disponibles, no probabilidades.</p>'}}
  function rateBars(features,f){{let home=featureVerified(features,'team_performance.home_goal_rate_blend')?nval(featureValue(features,'team_performance.home_goal_rate_blend')):null,away=featureVerified(features,'team_performance.away_goal_rate_blend')?nval(featureValue(features,'team_performance.away_goal_rate_blend')):null,total=featureVerified(features,'team_performance.total_goal_rate_blend')?nval(featureValue(features,'team_performance.total_goal_rate_blend')):null,source='Tasa deportiva de goles';if(home===null&&away===null){{home=featureVerified(features,'team_performance.home_goals_for_avg')?nval(featureValue(features,'team_performance.home_goals_for_avg')):null;away=featureVerified(features,'team_performance.away_goals_for_avg')?nval(featureValue(features,'team_performance.away_goals_for_avg')):null;total=home!==null&&away!==null?home+away:null;source='Promedio histórico de goles anotados';}}if(home===null&&away===null)return '<div class="visual-empty compact">No hay tasas de gol verificadas.</div>';const vals=[home,away,total].filter(x=>x!==null),mx=Math.max(...vals,1),row=(label,v,cls='')=>v===null?'':'<div class="rate-row"><label>'+esc(label)+'</label><div class="rate-track"><div class="rate-fill '+cls+'" style="width:'+esc(Math.max(3,Math.min(100,(v/mx)*100)).toFixed(1))+'%"></div></div><b>'+esc(v.toFixed(2))+'</b></div>';return '<div class="rate-bars">'+row(f.home_team||'Local',home,'')+row(f.away_team||'Visitante',away,'away')+row('Combinado',total,'total')+'</div><div class="rate-source">'+esc(source)+' · values are not labeled xG unless a verified xG source exists.</div>'}}
  function p100(v){{const n=nval(v);if(n===null)return null;return Math.max(0,Math.min(100,Math.abs(n)<=1?n*100:n))}}
  function probabilityVisual(probs,f){{const h=p100(probs?.home),d=p100(probs?.draw),a=p100(probs?.away);if(h===null||d===null||a===null)return '<div class="visual-empty">Match-result probabilities are NOT VERIFIED.</div>';const total=h+d+a||1,hs=h/total*100,ds=d/total*100,as=a/total*100;return '<div class="prob-tiles"><div class="prob-tile"><b>'+hs.toFixed(1)+'%</b><span>'+esc(f.home_team||'Home')+'</span></div><div class="prob-tile"><b>'+ds.toFixed(1)+'%</b><span>Draw</span></div><div class="prob-tile"><b>'+as.toFixed(1)+'%</b><span>'+esc(f.away_team||'Away')+'</span></div></div><div class="prob-stack"><i class="home" style="width:'+hs.toFixed(2)+'%"></i><i class="draw" style="width:'+ds.toFixed(2)+'%"></i><i class="away" style="width:'+as.toFixed(2)+'%"></i></div>'}}
  function goalDistribution(rows,f){{if(!rows?.length)return '<div class="visual-empty compact">Goal distribution is NOT VERIFIED.</div>';const buckets=['0','1','2','3','4+'],home=Object.fromEntries(buckets.map(x=>[x,0])),away=Object.fromEntries(buckets.map(x=>[x,0]));for(const row of rows){{const m=String(row?.score||'').match(/^(\\d+)\\s*[-:]\\s*(\\d+)$/);const p=p100(row?.probability);if(!m||p===null)continue;const hg=Number(m[1]),ag=Number(m[2]),hb=hg>=4?'4+':String(hg),ab=ag>=4?'4+':String(ag);home[hb]=(home[hb]||0)+p;away[ab]=(away[ab]||0)+p;}}const mx=Math.max(1,...Object.values(home),...Object.values(away));return '<div class="goal-dist"><div class="goal-dist-legend"><span><i class="h"></i>'+esc(f.home_team||'Home')+'</span><span><i class="a"></i>'+esc(f.away_team||'Away')+'</span></div><div class="goal-dist-bars">'+buckets.map(b=>'<div class="goal-dist-col"><div class="goal-dist-pair"><i class="h" style="height:'+Math.max(3,(home[b]/mx)*100).toFixed(1)+'%"></i><i class="a" style="height:'+Math.max(3,(away[b]/mx)*100).toFixed(1)+'%"></i></div><b>'+b+'</b></div>').join('')+'</div></div>'}}
  function xgVisual(xg,f){{const h=nval(xg?.home),a=nval(xg?.away),t=nval(xg?.total);if(h===null||a===null)return '<div class="visual-empty">Expected goals are NOT VERIFIED for this fixture.</div>';const mx=Math.max(h,a,t||0,1),line=(label,v)=>'<div class="compare-line"><div><div class="compare-meta"><span>'+esc(label)+'</span><b>'+esc(v.toFixed(2))+'</b></div><div class="compare-track"><div class="compare-fill" style="width:'+esc(Math.max(3,(v/mx)*100).toFixed(1))+'%"></div></div></div><b>'+esc(v.toFixed(2))+'</b></div>';return '<div class="compare-bars">'+line(f.home_team||'Home',h)+line(f.away_team||'Away',a)+(t!==null?line('Total',t):'')+'</div>'}}
  function scoringStats(features,f){{const defs=[[f.home_team+' form','team_performance.home_form'],[f.away_team+' form','team_performance.away_form'],[f.home_team+' GF / match','team_performance.home_goals_for_avg'],[f.home_team+' GA / match','team_performance.home_goals_against_avg'],[f.away_team+' GF / match','team_performance.away_goals_for_avg'],[f.away_team+' GA / match','team_performance.away_goals_against_avg']];const rows=defs.filter(x=>featureVerified(features,x[1])).map(x=>[x[0],featureValue(features,x[1]),featureSample(features,x[1])]);return rows.length?'<div class="sport-stat-grid">'+rows.map(x=>'<div class="sport-stat"><span>'+esc(x[0])+'</span><b>'+esc(x[1])+'</b><small>verified'+(x[2]!==null?' · n='+esc(x[2]):'')+'</small></div>').join('')+'</div>':'<div class="visual-empty compact">Team form and scoring statistics are NOT VERIFIED.</div>'}}
  function keyTakeaways(features,a,f){{const rows=[],hf=featureVerified(features,'team_performance.home_form')?featureValue(features,'team_performance.home_form'):null,af=featureVerified(features,'team_performance.away_form')?featureValue(features,'team_performance.away_form'):null,hgf=featureVerified(features,'team_performance.home_goals_for_avg')?nval(featureValue(features,'team_performance.home_goals_for_avg')):null,agf=featureVerified(features,'team_performance.away_goals_for_avg')?nval(featureValue(features,'team_performance.away_goals_for_avg')):null,hga=featureVerified(features,'team_performance.home_goals_against_avg')?nval(featureValue(features,'team_performance.home_goals_against_avg')):null,aga=featureVerified(features,'team_performance.away_goals_against_avg')?nval(featureValue(features,'team_performance.away_goals_against_avg')):null,hbase=nval(featureValue(features,'team_performance.home_goal_rate_blend')),abase=nval(featureValue(features,'team_performance.away_goal_rate_blend'));if(hf||af)rows.push(['Recent form',(f.home_team||'Home')+': '+(hf||'N/V')+' · '+(f.away_team||'Away')+': '+(af||'N/V')]);if(hgf!==null||agf!==null)rows.push(['Verified scoring',(f.home_team||'Home')+': '+(hgf!==null?hgf.toFixed(2):'N/V')+' GF/match · '+(f.away_team||'Away')+': '+(agf!==null?agf.toFixed(2):'N/V')+' GF/match']);else if(hbase!==null||abase!==null)rows.push(['Goal-rate baseline',(f.home_team||'Home')+': '+(hbase!==null?hbase.toFixed(2):'N/V')+' · '+(f.away_team||'Away')+': '+(abase!==null?abase.toFixed(2):'N/V')]);if(hga!==null||aga!==null)rows.push(['Verified goals allowed',(f.home_team||'Home')+': '+(hga!==null?hga.toFixed(2):'N/V')+' · '+(f.away_team||'Away')+': '+(aga!==null?aga.toFixed(2):'N/V')]);if(a?.confidence!==null&&a?.confidence!==undefined)rows.push(['Availability','Confidence '+pct(a.confidence)+' · confirmation status remains separate']);return rows.length?'<div class="takeaway-list">'+rows.slice(0,4).map((x,i)=>'<div class="takeaway"><div class="takeaway-num">'+(i+1)+'</div><div><b>'+esc(x[0])+'</b><small>'+esc(x[1])+'</small></div></div>').join('')+'</div>':'<div class="visual-empty compact">No verified sporting takeaway is supported yet.</div>'}}

  function analystLens(decisionText,markets,missing,counts,features){{const state=String(decisionText||'').toUpperCase(),sportN=sportFeatureCount(features),marketN=Number(counts?.market_snapshots||0);if(state==='BET')return 'Sporting evidence produced a canonical BET and the market layer also passed. Review availability and exact price before acting.';if(state==='LEAN')return 'The sporting case is interesting, but the market layer did not justify BET. Sport remains the primary reason to inspect this match.';if(state==='WATCH')return 'The match has a WATCH state because sporting evidence exists but at least one required condition is unresolved.';if(sportN>0)return 'Sport-first evidence exists for this match. Start with team performance, formation and availability. Market snapshots are secondary context, not proof of value.';if(marketN>0)return 'Market data exists, but verified sporting evidence is not sufficient. Do not treat the number of available markets as an edge.';return 'Only fixture identity is verified right now. There is not enough sporting evidence to support a conclusion.'}}
  function fe11ProfileMetric(profile,where,key,label){{
    const v=profile?.splits?.[where]?.features?.[key]||{{}},n=Number(v.n||0),avg=num(v.mean);
    const value=n>0&&avg!==null?avg.toFixed(2):'NOT VERIFIED';
    const enough=v.minimum_descriptive_sample_met===true;
    return '<div class="item"><span>'+esc(label)+'</span><span>'+value+
      ' · n='+n+(enough?'':' · MUESTRA BAJA')+'</span></div>';
  }}
  function fe11TeamProfile(profile,title,venue){{
    if(!profile||profile.descriptive_only!==true||profile.bet_eligible!==false)
      return '<div class="intel-card"><h3>'+esc(title)+'</h3><div class="intel-footnote">Perfil deportivo histórico NOT VERIFIED.</div></div>';
    const all=profile.splits?.all||{{}},split=profile.splits?.[venue]||{{}};
    const share=num(split.period_goals_share_scored_second_half);
    const period=share===null?'NO VERIFICADO':(share*100).toFixed(1)+'%';
    const n=Number(split.matches_with_valid_ft||0);
    return '<div class="intel-card"><h3>'+esc(title)+' · '+esc(venue==='home'?'Local':'Visitante')+'</h3>'+
      '<div class="intel-footnote">Muestra localía: '+n+
      ' partidos FT históricos. Media y frecuencia observadas; NO son pronósticos.</div>'+
      '<div class="detail-list">'+
      fe11ProfileMetric(profile,venue,'first_half_goals_for','Goles a favor 1H / partido')+
      fe11ProfileMetric(profile,venue,'second_half_goals_for','Goles a favor 2H / partido')+
      fe11ProfileMetric(profile,venue,'first_half_goals_total','Goles totales 1H / partido')+
      fe11ProfileMetric(profile,venue,'second_half_goals_total','Goles totales 2H / partido')+
      fe11ProfileMetric(profile,venue,'team_corners_ft','Córners a favor FT')+
      fe11ProfileMetric(profile,venue,'opponent_corners_ft','Córners concedidos FT')+
      fe11ProfileMetric(profile,venue,'total_corners_ft','Córners del partido FT')+
      '<div class="item"><span>Goles propios en segundo tiempo (share histórico)</span><span>'+
      esc(period)+' · n='+Number(split.second_half_goal_share_sample_fixtures||0)+'</span></div>'+
      '<div class="item"><span>Partidos totales válidos</span><span>'+Number(all.matches_with_valid_ft||0)+'</span></div>'+
      '</div></div>';
  }}
  function fe11PatternCard(pattern,title,venue){{
    const a=pattern?.ASIAN||null,t=pattern?.THREE_WAY||null;
    if(!a||!t)return '<div class="intel-card"><h3>'+esc(title)+'</h3><div class="intel-footnote">No existe una muestra histórica completa para el patrón.</div></div>';
    const rows=(type,name)=>{{
      const x=pattern[type]||{{}},n=Number(x.complete_historical_rows||0),
      wins=Number(x.observed_joint_hits||0),push=Number(x.asian_handicap_push_records||0);
      return '<div class="item"><span>'+esc(name)+'</span><span>'+wins+'/'+n+
        ' coincidencias históricas'+(type==='ASIAN'?' · '+push+' push de hándicap':'')+
        (n<8?' · MUESTRA BAJA':'')+'</span></div>';
    }};
    return '<div class="intel-card"><h3>'+esc(title)+'</h3>'+
      '<div class="intel-footnote">Ejemplo histórico: Over 1.5 goles FT + Over 6.5 córners FT + equipo +1. '+
      'Las tres condiciones se comprueban en LOS MISMOS PARTIDOS, sin multiplicar estadísticas.</div>'+
      '<div class="detail-list">'+rows('ASIAN','Hándicap asiático +1')+
      rows('THREE_WAY','Hándicap tres vías +1')+'</div>'+
      '<div class="intel-footnote">El push asiático NO equivale al empate de hándicap 3 vías. '+
      'El operador puede liquidar builders de forma distinta. NO es una probabilidad conjunta futura.</div></div>';
  }}
  function fe11MultimarketSection(research,f){{
    const section='<div class="intel-section-title"><h3>Periodos, córners y hándicap</h3>'+
      '<span>SPORT FIRST · investigación histórica · NO BET</span></div>';
    if(!research||research.status!=='HISTORICAL_DESCRIPTIVE_ONLY'||research.history_fresh!==true)
      return '<div class="intel-card">'+section+'<div class="visual-empty compact">'+
        'NOT VERIFIED: no hay perfil histórico validado y reciente para este fixture. '+
        'No se infieren porcentajes, córners ni picks. Fuente: '+esc(research?.reason||research?.status||'NO_RESEARCH_FEED')+
        '</div></div>';
    return '<div class="intel-card" style="margin-bottom:10px">'+section+
      '<div class="intel-footnote">Fuente del análisis: '+esc(research.source_generated_at_utc||'NOT VERIFIED')+
      ' · Modelo '+esc(research.model_version||'NOT VERIFIED')+
      ' · Aún sin calibración OOS ni comparación de precio.</div>'+
      '<div class="intel-footnote">Cada n es una observación histórica; un partido con córners no documentados '+
      'no cuenta para la intersección del builder. Ninguna tarjeta constituye recomendación de apuesta.</div></div>'+
      '<div class="intel-grid">'+
      fe11TeamProfile(research.home,f.home_team||'Local','home')+
      fe11TeamProfile(research.away,f.away_team||'Visitante','away')+
      '</div><div class="intel-card" style="margin-top:10px"><h3>Constructor de 3 selecciones (historial, sin cuota)</h3>'+
      '<div class="intel-footnote">Un boleto completo necesita modelo conjunto calibrado, XI y precio exacto del bookmaker. '+
      'El perfil histórico no predice que se cumplan los tres eventos.</div></div>'+
      '<div class="intel-grid">'+
      fe11PatternCard(research.home_pattern,f.home_team||'Local','home')+
      fe11PatternCard(research.away_pattern,f.away_team||'Visitante','away')+
      '</div><div class="intel-footnote" style="margin-top:10px">Bloqueos: correlación y joint OOS NOT VERIFIED; cuota del Bet Builder NOT VERIFIED; '+
      'hándicap y reglas de liquidación requieren confirmación de la casa. Estado WATCH, NO BET.</div>';
  }}
  // Join EXISTING PostgreSQL feature snapshots already unioned by the match
  // contract. OWNER only: no second collector, no provider calls, no prices.
  // Never present a post-kickoff observation as pregame evidence.
  function ownerPersistedSportEvidence(packet,fixture){{
    if(ACCESS?.owner!==true&&ACCESS?.admin!==true)return '';
    const groups=Array.isArray(packet?.private_pregame_evidence_sections)?packet.private_pregame_evidence_sections:[],
      kickoff=new Date(fixture?.kickoff||'').getTime(),now=Date.now();
    const validTime=Number.isFinite(kickoff);
    const allowed=new Set(['TEAMS','GOALS','STATS','CORNERS','CARDS','EVENTS','PLAYERS','AVAILABILITY','CONTEXT','OTHER']);
    const sections=[];
    if(validTime)for(const group of groups){{
      if(!allowed.has(group?.category)||!Array.isArray(group.items))continue;
      const rows=group.items.filter(item=>{{
        const captured=new Date(item?.captured_at||'').getTime(),sample=item?.sample_n,
          value=item?.value;
        return item?.status==='PERSISTED'&&!!item?.source&&
          Number.isFinite(captured)&&captured<kickoff&&captured<=now&&
          value!==null&&value!==undefined&&typeof value!=='object'&&
          (sample===null||sample===undefined||(Number.isInteger(sample)&&sample>0));
      }}).slice(0,24);
      if(rows.length)sections.push('<details><summary>'+esc(group.category)+' · '+rows.length+
        ' observaciones verificadas</summary><div class="intel-snapshot-list">'+rows.map(item=>
        '<div class="intel-snapshot"><div><b>'+esc(item.label||item.key||'Dato deportivo')+'</b>'+
        '<small>'+esc(item.source)+' · '+esc(item.captured_at)+
        (item.sample_n!==null&&item.sample_n!==undefined?' · n='+esc(item.sample_n):' · n=NO VERIFICADO')+
        '</small></div><div class="snap-right">'+esc(item.value)+'</div></div>'
      ).join('')+'</div></details>');
    }}
    return '<section class="owner-persisted-sport-evidence"><details><summary>'+
      'Investigación deportiva ya guardada por partido · '+sections.length+' familias con datos'+
      '</summary><p class="intel-footnote">Solo OWNER. Unión de registros deportivos existentes por fixture_id; '+
      'cada valor conserva fuente, captura y muestra. Solo observaciones anteriores al inicio. '+
      'No son predicciones RAW, OOS ni recomendaciones. NO BET.</p>'+
      (sections.join('')||'<p class="intel-footnote">Sin observaciones prepartido adicionales verificadas en PostgreSQL para este fixture. NOT VERIFIED no significa cero.</p>')+
      '</details></section>';
  }}
  function internalSeasonFacts(packet){{
    const ctx=packet?.private_team_context;
    if(ctx?.status!=='SPORTING_SEASON_CONTEXT_ONLY'||!Array.isArray(ctx.teams)||!ctx.teams.length)return '';
    const cards=ctx.teams.filter(x=>Number.isInteger(x?.played)&&x.played>0).map(x=>
      '<article class="season-fact-card"><h4>'+esc(x.team_name||'Equipo')+'</h4>'+
      '<p>'+esc(x.competition||'Competición')+' · '+esc(x.season||'')+' · '+esc(x.played)+' partidos</p>'+
      '<div class="season-fact-stats"><span><b>'+esc(x.wins??'—')+'</b> Ganados</span>'+
      '<span><b>'+esc(x.draws??'—')+'</b> Empatados</span>'+
      '<span><b>'+esc(x.losses??'—')+'</b> Perdidos</span>'+
      '<span><b>'+esc(x.goals_for??'—')+'</b> Goles a favor</span>'+
      '<span><b>'+esc(x.goals_against??'—')+'</b> Goles contra</span></div>'+
      '<p class="season-split">'+esc(x.venue==='home'?'Como local':'Como visitante')+': '+
      esc(x.played_venue??'—')+' partidos; '+esc(x.goals_for_per_match_venue??'—')+
      ' goles marcados y '+esc(x.goals_against_per_match_venue??'—')+' recibidos por encuentro.</p></article>'
    ).join('');
    if(!cards)return '';
    return '<section class="season-facts-owner"><div class="season-facts-head"><h3>Datos históricos verificados</h3>'+
      '<span>INVESTIGACIÓN PRIVADA</span></div><div class="season-facts-grid">'+cards+'</div>'+
      '<p>Fuente: API-Football, estadísticas equipo/temporada · capturadas: '+esc(prettyStamp(ctx.fetched_at_utc))+
      '. Son datos históricos, NO xG ni un modelo 1X2. No tienen validación OOS ni permiso de redistribución comercial.'+
      ' Las competiciones o rivales pueden ser distintos y las estadísticas no garantizan un pronóstico.</p></section>';
  }}
  function recentVerifiedForm(packet){{
    const r=packet?.private_recent_form||{{}};
    if(!['VERIFIED_RECENT_FORM','PARTIAL_RECENT_FORM'].includes(r.status))return '';
    const teams=Array.isArray(r.teams)?r.teams.filter(x=>x&&typeof x==='object'):[];
    return '<section class="verified-recent-form"><div class="verified-sport-heading">'+
      '<span>DATOS NUEVOS · ÚLTIMOS PARTIDOS OFICIALES</span>'+
      '<h2>Forma reciente de los equipos</h2>'+
      '<p>Resultados registrados antes del inicio del encuentro. Incluye distintas competiciones; no equivale a una proyección calibrada.</p></div>'+
      '<div class="recent-form-teams">'+teams.map(t=>{{
        const matches=Array.isArray(t.recent_fixtures)?t.recent_fixtures.slice(0,6):[];
        return '<div class="recent-form-team"><h3>'+esc(t.team_name||'Equipo')+'</h3>'+
          '<div class="recent-form-summary">'+esc(t.sample??0)+' partidos · '+
          esc(t.wins??'–')+'V / '+esc(t.draws??'–')+'E / '+esc(t.losses??'–')+'D · '+
          esc(t.goals_for??'–')+' GF / '+esc(t.goals_against??'–')+' GC</div>'+
          '<div class="recent-form-rows">'+matches.map(m=>'<div class="recent-form-row">'+
          '<strong>'+esc(m.result||'–')+'</strong><span>'+esc(m.opponent||'Rival no verificado')+
          ' · '+esc(m.venue||'')+' · '+esc(m.competition||'')+'</span><b>'+
          esc(m.goals_for??'–')+'–'+esc(m.goals_against??'–')+'</b></div>').join('')+'</div>'+
          '<p>Como local: '+esc(t.home_sample??0)+' · Como visitante: '+esc(t.away_sample??0)+'</p></div>';
      }}).join('')+'</div>'+
      '<div class="verified-sport-caveats"><strong>Limitaciones</strong>'+
      '<p>Competición y fuerza de los rivales no ajustadas. Sin xG, once confirmado ni calidad del oponente verificable. Muestra exclusivamente histórica; no produce un pick ni una recomendación.</p></div>'+
      '<div class="verified-sport-source">Fuente: '+esc(r.source||'NOT VERIFIED')+
      ' · Captura: '+esc(prettyStamp(r.captured_at_utc))+
      ' · Acceso: investigación privada · Proyección 1X2: no calculada</div></section>';
  }}
  function verifiedSportAssessment(packet){{
    const read=packet?.private_verified_sport_read||{{}};
    if(!['VERIFIED_SEASON_CONTEXT','PARTIAL_SEASON_CONTEXT'].includes(read.status))return '';
    const facts=Array.isArray(read.facts)?read.facts.filter(v=>typeof v==='string'):[];
    const warnings=Array.isArray(read.warnings)?read.warnings.filter(v=>typeof v==='string'):[];
    const headline=read.headline||'Contexto deportivo observado';
    return '<section class="verified-sport-read" aria-label="Lectura deportiva histórica">'+
      '<div class="verified-sport-heading"><span>LECTURA DEPORTIVA · INVESTIGACIÓN PRIVADA</span>'+
      '<h2>'+esc(headline)+'</h2>'+
      '<p>Estos son hechos de ambos equipos, no solamente un calendario. La muestra se explica antes de interpretar los resultados.</p></div>'+
      '<div class="verified-sport-fact-list">'+facts.map((v,i)=>'<div class="sport-fact"><b>0'+(i+1)+'</b><p>'+esc(v)+'</p></div>').join('')+'</div>'+
      '<div class="verified-sport-caveats"><strong>Límites de esta lectura</strong>'+
      warnings.map(v=>'<p>'+esc(v)+'</p>').join('')+'</div>'+
      '<div class="verified-sport-source">Fuente: '+esc(read.source||'NOT VERIFIED')+
      ' · Consulta: '+esc(prettyStamp(read.observed_at_utc))+
      ' · Proyección 1X2: no calculada · Uso comercial: no autorizado</div></section>';
  }}
  function ownerMatchupResearch(packet,fixture,raw,details,blocked){{
    if(blocked||(ACCESS?.owner!==true&&ACCESS?.admin!==true))return '';
    const research=packet?.multi_market_research, kickoff=Date.parse(fixture?.kickoff||''),
      captured=Date.parse(research?.source_generated_at_utc||'');
    const sameHistory=Number.isFinite(kickoff)&&Number.isFinite(captured)&&
      captured<kickoff&&captured<=Date.now()&&!!fixture?.fixture_id&&
      String(research?.fixture_id)===String(fixture.fixture_id)&&Date.parse(research?.kickoff||'')===kickoff&&
      research?.status==='HISTORICAL_DESCRIPTIVE_ONLY'&&research?.history_fresh===true&&
      research?.observations_not_forecasts===true&&research?.bet_eligible===false;
    const probability=x=>typeof x==='number'&&Number.isFinite(x)&&x>=0&&x<=1;
    const resultKeys=['raw_home_win_prob','raw_draw_prob','raw_away_win_prob'];
    const score=resultKeys.every(k=>probability(raw?.[k]))?
      resultKeys.map(k=>pct(raw[k])).join(' / '):null;
    const sample=(metric)=>{{
      if(!sameHistory)return 'Sin perfil FE11 vigente verificado';
      const n=['home','away'].map(side=>research[side]?.splits?.[side]?.features?.[metric]?.n);
      if(!n.some(x=>Number.isInteger(x)&&x>0))return 'Sin muestra FE11 por localía';
      return 'Histórico FE11 · n local '+(n[0]>0?n[0]:'N/D')+' · n visita '+(n[1]>0?n[1]:'N/D');
    }};
    const saved=(key)=>probability(raw?.[key])?'RAW preliminar: '+pct(raw[key]):'Sin RAW verificado en este panel';
    const unavailable='Sin proyección específica verificada en este panel';
    const observations=(category)=>{{
      const groups=packet?.private_pregame_evidence_sections;
      const count=Array.isArray(groups)?groups.find(g=>g?.category===category)?.items?.length:0;
      return Number.isInteger(count)&&count>0?
        count+' campos prepartido guardados · ver fuentes abajo · sin RAW verificado aquí':
        'Sin evidencia prepartido verificada en este panel · sin RAW';
    }};
    const rates=packet?.private_frozen_sport_projection?.raw_goal_rates;
    const ratesValid=raw===packet?.private_frozen_sport_projection?.raw_probabilities&&
      [rates?.raw_home_goal_rate,rates?.raw_away_goal_rate].every(x=>
        typeof x==='number'&&Number.isFinite(x)&&x>=0.15&&x<=4.5);
    const rows=[
      ['1X2 · local / empate / visita',score?'RAW preliminar: '+score:unavailable],
      ['Altas/bajas FT · más de 2.5',saved('raw_over_2_5_prob')],
      ['BTTS · ambos anotan',saved('raw_btts_yes_prob')],
      ['Goles por equipo',ratesValid?'Tasas RAW guardadas (λ, no xG) · local '+
        rates.raw_home_goal_rate.toFixed(2)+' · visita '+rates.raw_away_goal_rate.toFixed(2)+
        ' · sin proyección verificada por línea':unavailable],
      ['DNB · doble oportunidad',unavailable],
      ['Primer tiempo · goles',sample('first_half_goals_total')],
      ['Segundo tiempo · goles',sample('second_half_goals_total')],
      ['Córners totales',sample('total_corners_ft')],
      ['Córners por equipo',sample('team_corners_ft')],
      ['Hándicap y Bet Builder',sameHistory?'Consultar muestra del patrón abajo · sin modelo conjunto':unavailable],
      ['Tarjetas y disciplina',observations('CARDS')],
      ['Jugadores · contexto y disponibilidad',observations('PLAYERS')],
      ['Props · tiros, SOT, goles, asistencias, tarjetas y atajadas',unavailable],
    ];
    return '<section class="intel-section" data-detail-section="matchup-research">'+
      '<div class="intel-card"><h3>Expediente deportivo del matchup</h3>'+
      '<p class="intel-footnote">Investigación privada. Un histórico no es RAW; RAW no equivale a calibración OOS. '+
      'Esta vista no certifica OOS ni precio ejecutable. Sin habilitación BET.</p>'+
      intelKV(rows)+'</div>'+details+
      fe11MultimarketSection(sameHistory?research:{{reason:research?.status==='HISTORICAL_DESCRIPTIVE_ONLY'?
        'IDENTIDAD_O_TIEMPO_NO_VERIFICADO':research?.reason||research?.status}},fixture)+'</section>';
  }}
  function renderMatch(d){{const box=$('matchDetail'),rawF=d?.fixture||{{}},known=fixtureTargets().find(x=>String(x?.fixture?.fixture_id||'')===String(rawF.fixture_id||''))?.fixture||{{}},f={{...known,...rawF}},c=d?.selected_candidate||{{}},p=d?.projection_ladder||c?.projections||{{}},a=d?.availability||c?.availability||{{}},ctx=d?.sport_context||{{}},decision=d?.decision_summary||c?.decision||{{}},marketCtx=d?.market_context||{{}},modelCtx=d?.model_context||{{}},review=d?.analyst_review||{{}},rel=d?.relational_evidence||{{}},counts=rel.counts||{{}},disc=d?.data_disclosure||{{}},e=d?.evidence||c?.evidence||{{}};
    const selectedMarket=marketCtx.selected||c?.market||{{}},missing=(disc.missing_sections||review.missing_sections||[]),currentMarkets=(review.available_markets||[]),persistedMarkets=(counts.market_names||[]),markets=[...new Set([...currentMarkets,...persistedMarkets].filter(Boolean))],verifiedPrices=(review.verified_price_markets||[]),features=sportFeatures(rel),sportN=sportFeatureCount(features),sportMissing=sportFeatureMissing(features),decisionText=decision.classification||decision.display_bucket||decision.execution_status||review.status||'INSUFFICIENT DATA',fresh=freshnessLabel(counts),coverage=review.status||decisionText||'NOT VERIFIED',tone=coverageTone(coverage),[kickTime,kickDate]=dt(f.kickoff);
    const actualStatus=String(d?.live_fixture_status?.source==='API_FOOTBALL_FIXTURE_STATUS_BY_ID'?
      d.live_fixture_status.status:(f.fixture_status||f.status||'')).toUpperCase();
    const blockedEvent=['CANC','ABD','AWD','WO','PST','SUSP'].includes(actualStatus);
    const statusLabel=['CANC','ABD','AWD','WO'].includes(actualStatus)?'CANCELADO':
      ['PST','SUSP'].includes(actualStatus)?'APLAZADO / SUSPENDIDO':
      ['FT','AET','PEN'].includes(actualStatus)?'FINALIZADO':actualStatus;
    const statusObserved=d?.live_fixture_status?.source==='API_FOOTBALL_FIXTURE_STATUS_BY_ID'?
      d.live_fixture_status.observed_at_utc:f.status_observed_at;
    const statusProvenance=d?.live_fixture_status?.source==='API_FOOTBALL_FIXTURE_STATUS_BY_ID'?
      'API-Football · verificación actual':'Registro persistido · puede estar desactualizado';
    const hero='<div class="intel-hero"><div class="match-brand-row"><div class="match-brand"><span>SE</span><b>SOCCER<br>EDGE</b></div><button id="backToMatches" type="button" class="intel-back">← Matches</button></div><div class="match-context-row"><span>'+esc(f.league||f.country||'Competition')+'</span><b>'+esc(kickDate)+' · '+esc(kickTime)+'</b></div><div class="intel-matchup"><div class="intel-team">'+crest(f.home_team_logo,f.home_team)+'<div class="intel-team-name"><b>'+esc(f.home_team||'Home')+'</b><span>Home</span></div></div><div class="intel-vs">VS</div><div class="intel-team away"><div class="intel-team-name"><b>'+esc(f.away_team||'Away')+'</b><span>Away</span></div>'+crest(f.away_team_logo,f.away_team)+'</div></div><div class="intel-scoreline">'+esc(f.country||'')+
      (statusLabel?' · Estado: '+esc(statusLabel):'')+'</div></div>';
    // Owners may inspect an independently validated frozen pre-kickoff
    // research preview that was already delivered by the owner-only slate API.
    // Never reinterpret it as a canonical BET, current price or OOS model.
    const matchFrozen=d?.private_frozen_sport_projection;
    const ownerP0=(ACCESS?.owner===true||ACCESS?.admin===true)?
      [matchFrozen,...p0Entries()].find(x=>x&&String(x.fixture_id)===String(f.fixture_id)&&
        x.market_queried===false&&x.bet_eligible===false&&
        x.commercial_display_allowed===false&&
        Number.isFinite(Date.parse(f.kickoff||''))&&
        Math.abs(Date.parse(x.kickoff_utc)-Date.parse(f.kickoff))<60*1000):null;
    const ownerRaw=ownerP0?.raw_probabilities||{{}};
    const ownerP0Valid=!!ownerP0&&
      ['raw_home_win_prob','raw_draw_prob','raw_away_win_prob'].every(k=>
        typeof ownerRaw[k]==='number'&&Number.isFinite(ownerRaw[k])&&
        ownerRaw[k]>=0&&ownerRaw[k]<=1)&&
      Math.abs(ownerRaw.raw_home_win_prob+ownerRaw.raw_draw_prob+
        ownerRaw.raw_away_win_prob-1)<0.01;
    const ownerResearchCard=ownerP0Valid?
      '<section class="owner-frozen-research"><div class="owner-frozen-title">'+
      '<span>MODELO DEPORTIVO PRELIMINAR · SOLO INVESTIGACIÓN PRIVADA</span>'+
      '<h3>Probabilidades RAW guardadas antes del partido</h3>'+
      '<p>Modelo de tasas de goles observados (no xG). No validado fuera de muestra, sin XI confirmado ni recomendación comercial.</p></div>'+
      '<div class="owner-frozen-probs"><div><small>Gana '+esc(f.home_team||'local')+'</small><b>'+pct(ownerRaw.raw_home_win_prob)+'</b></div>'+
      '<div><small>Empate</small><b>'+pct(ownerRaw.raw_draw_prob)+'</b></div>'+
      '<div><small>Gana '+esc(f.away_team||'visita')+'</small><b>'+pct(ownerRaw.raw_away_win_prob)+'</b></div>'+
      '</div><div class="owner-frozen-extra">Más de 2.5: '+pct(ownerRaw.raw_over_2_5_prob)+
      ' · Ambos anotan: '+pct(ownerRaw.raw_btts_yes_prob)+'</div>'+
      '<p class="owner-frozen-source">Capturado: '+esc(prettyStamp(ownerP0.captured_at_utc))+
      ' · Modelo: '+esc(ownerP0.model_version||'NOT VERIFIED')+
      (ownerP0.storage==='POSTGRES_ISOLATED_SPORT_FROZEN_RESEARCH'?' · Fuente: PostgreSQL, proyección original':'')+
      ' · Muestra mínima: '+esc(ownerP0.min_split_sample??'NOT VERIFIED')+
      ' · Precio de mercado: NO CONSULTADO · Validación OOS: PENDIENTE</p></section>':'';
    const summary='<div class="intel-summary"><div class="intel-metrics"><div class="intel-metric '+(ctx.outcome_probabilities||ownerP0Valid?'good':'warn')+'"><span>Pronóstico 1X2</span><b>'+esc(ctx.outcome_probabilities?'Modelo guardado':ownerP0Valid?'RAW PRELIMINAR':'NO VERIFICADO')+'</b></div><div class="intel-metric"><span>Alineaciones</span><b>'+esc(featureValue(features,'availability.both_xi_confirmed')===true?'XI confirmados':'Sin confirmar')+'</b></div><div class="intel-metric"><span>Datos actualizados</span><b>'+esc(fresh)+'</b></div></div></div>';
    const privateMatchup=ACCESS?.owner===true||ACCESS?.admin===true;
    const tabs=[['overview','Resumen'],...(privateMatchup?[['matchup-research','Expediente deportivo']]:[]),...(ctx.outcome_probabilities||sportN>0?[['sport','Pronóstico y goles']]:[]),...(!privateMatchup&&d?.multi_market_research?.status&&d.multi_market_research.status!=='NOT_VERIFIED'?[['periods','1H / 2H / Córners']]:[]),['availability','Alineaciones'],...(markets.length?[['markets','Cuotas']]:[]),['advanced','Fuentes y datos']];
    const tabHtml='<div class="intel-tabs">'+tabs.map((x,i)=>'<button class="intel-tab '+(i===0?'active':'')+'" data-detail-tab="'+x[0]+'">'+x[1]+'</button>').join('')+'</div>';

    const coreCov=coreCoverage(features),probs=ctx.outcome_probabilities||null,xg=ctx.expected_goals||null,scoreRows=ctx.score_matrix||[];
    const hasSportProb=['home','draw','away'].every(k=>p100(probs?.[k])!==null);
    const goalRateLabel=ctx.goal_rate_semantics==='POISSON_LAMBDA_NOT_XG'?'Poisson λ, no xG':'Tasa de goles del modelo';
    const ownerSeasonResearch=internalSeasonFacts(d);
    const ownerSportRead=verifiedSportAssessment(d),ownerRecent=recentVerifiedForm(d),
      hasOwnerSportRead=!!ownerSportRead||!!ownerRecent;
    const overview='<section class="intel-section active" data-detail-section="overview">'+
      '<div class="match-reading-banner"><span>EL PARTIDO, ANTES QUE LAS CUOTAS</span><h2>'+esc(hasSportProb?'Pronóstico deportivo registrado':ownerP0Valid?'Modelo deportivo preliminar guardado':sportN>0||hasOwnerSportRead?'Análisis deportivo histórico disponible':'Partido identificado · análisis pendiente')+'</h2><p>'+
      esc(hasSportProb?'Existe una proyección histórica que debe revisarse junto con la disponibilidad.':
        ownerP0Valid?'Hay una proyección RAW auténtica capturada antes del inicio. Es investigación no calibrada, NO una recomendación ejecutable.':
        sportN>0||hasOwnerSportRead?'Hay estadísticas verificadas para lectura deportiva. Los resultados observados no equivalen a probabilidades de un modelo 1X2.':
        'La ficha identifica el encuentro, pero aún no contiene estadísticas deportivas persistidas. No se infieren porcentajes ni recomendaciones a partir del calendario.')+
      '</p><p class="status-source">Estado '+esc(statusLabel||'NO VERIFICADO')+
      ' · '+esc(statusProvenance)+' · '+esc(statusObserved?prettyStamp(statusObserved):'Sin fecha de última verificación')+'</p></div>'+
      ownerResearchCard+ownerRecent+ownerSportRead+ownerSeasonResearch+
      ownerPersistedSportEvidence(d,f)+
      (hasSportProb?'<div class="visual-card primary-forecast"><h3>Victoria · Empate · Derrota</h3>'+
        probabilityVisual(probs,f)+'</div>':'')+
      (sportN>0?'<div class="visual-card primary-takeaways"><h3>Datos deportivos que sí tenemos</h3>'+
        keyTakeaways(features,a,f)+'</div>'+
        '<div class="visual-card goal-card"><h3>Contexto de goles registrado</h3>'+rateBars(features,f)+'</div>':
        (hasOwnerSportRead?'':
        '<div class="intel-footnote">No hay estadísticas verificadas en el expediente de este partido. Los datos históricos de los equipos requieren una captura independiente y validada.</div>'))+
      '<div class="overview-assurance"><b>Sport-first read</b><span>'+esc(analystLens(decisionText,markets,missing,counts,features))+'</span></div>'+
      '<div class="intel-accordion"><details><summary>Forma reciente y datos disponibles <span>Input coverage · diagnóstico</span></summary><div class="intel-accordion-body"><div class="mini-visual-grid"><div class="visual-card"><div class="visual-card-head"><h3>Datos disponibles</h3><span>'+esc(coreCov.verified)+' / '+esc(coreCov.total)+'</span></div>'+coverageGauge(features)+'</div><div class="visual-card"><div class="visual-card-head"><h3>Forma reciente</h3><span>Form momentum · W/D/L</span></div>'+formCompare(features,f)+'</div></div></div></details>'+
      '<details><summary>Qué falta por verificar <span>'+esc(sportMissing)+' campos pendientes</span></summary><div class="intel-accordion-body">'+infoPills([!featureVerified(features,'xg.xgf')?'xG / xGA':null,!hasSportProb?'1X2 probabilities':null,!ctx.sport_profile?.length?'Perfil deportivo':null,!featureVerified(features,'team_performance.home_form')||!featureVerified(features,'team_performance.away_form')?'Forma reciente':null,!featureVerified(features,'availability.both_xi_confirmed')?'XI':null,!featureVerified(features,'availability.both_goalkeepers_confirmed')?'Porteros':null],'missing')+'</div></details>'+
      '<details><summary>Verified sporting inputs <span>'+esc(sportN)+' campos</span></summary><div class="intel-accordion-body compact-inputs">'+verifiedSportInputs(features)+'</div></details>'+
      '<details><summary>Model & evidence quality <span>'+esc(counts.model_runs??0)+' ejecuciones</span></summary><div class="intel-accordion-body">'+intelKV([['Clasificación',decisionText],['Raw sport',pct(p.raw_sport_probability)],['Market shrunk',pct(p.market_shrunk_probability)],['Market fair',pct(p.fair_market_probability)],['Probability edge',pp(p.probability_edge_pp)],['Core sport coverage',coreCov.verified+' / '+coreCov.total],['Feature snapshots',counts.feature_snapshots??0]])+'</div></details></div></section>';

    const marketsSection='<section class="intel-section" data-detail-section="markets"><div class="intel-card"><div class="intel-section-title"><h3>Market layer</h3><span>secondary · after sport</span></div>'+infoPills(markets,'market')+'</div><div class="intel-card" style="margin-top:10px"><div class="intel-section-title"><h3>Latest market activity</h3><span>context only · not an edge by itself</span></div>'+snapshotCards(rel.market_snapshots||[],'market')+'</div></section>';

    const sportSection='<section class="intel-section" data-detail-section="sport">'+
      '<div class="visual-card sport-top-card"><div class="visual-card-head"><h3>Match result probability</h3><span>verified model only</span></div>'+probabilityVisual(probs,f)+'</div>'+
      '<div class="visual-card sport-top-card"><div class="visual-card-head"><h3>Goles modelados (λ)</h3><span>'+esc(goalRateLabel)+' · Expected goals (xG) NOT VERIFIED sin fuente independiente</span></div>'+xgVisual(xg,f)+'</div>'+
      '<div class="sport-two-up"><div class="visual-card"><div class="visual-card-head"><h3>Score matrix</h3><span>FT probability</span></div>'+scoreMatrix(scoreRows)+'</div>'+
      '<div class="visual-card"><div class="visual-card-head"><h3>Goal distribution</h3><span>from score matrix</span></div>'+goalDistribution(scoreRows,f)+'</div></div>'+
      '<div class="visual-card sport-profile-card"><div class="visual-card-head"><h3>Team sport profile</h3><span>defined model scores only</span></div>'+sportProfile(ctx.sport_profile||[])+'</div>'+
      '<div class="visual-card sport-top-card"><div class="visual-card-head"><h3>Team form & scoring</h3><span>verified team stats</span></div>'+scoringStats(features,f)+'</div>'+
      '<div class="intel-accordion"><details><summary>Formation & squad context <span>availability-linked</span></summary><div class="intel-accordion-body">'+intelKV([[f.home_team+' formation',featureValue(features,'availability.home_formation')||'NOT VERIFIED'],[f.away_team+' formation',featureValue(features,'availability.away_formation')||'NOT VERIFIED'],['Both XI confirmed',featureValue(features,'availability.both_xi_confirmed')===null?'NOT VERIFIED':(featureValue(features,'availability.both_xi_confirmed')?'YES':'NO')],['Both goalkeepers confirmed',featureValue(features,'availability.both_goalkeepers_confirmed')===null?'NOT VERIFIED':(featureValue(features,'availability.both_goalkeepers_confirmed')?'YES':'NO')],['Injury report count',featureValue(features,'availability.injury_report_count')??'NOT VERIFIED']])+'</div></details>'+
      '<details><summary>Verified sporting inputs <span>'+esc(sportN)+'</span></summary><div class="intel-accordion-body compact-inputs">'+verifiedSportInputs(features)+'</div></details></div></section>';

    const availabilitySection='<section class="intel-section" data-detail-section="availability"><div class="intel-grid"><div class="intel-card"><h3>Availability</h3>'+intelKV([['Confidence',a.confidence!==null&&a.confidence!==undefined?pct(a.confidence):'NOT VERIFIED'],['Data tier',a.data_tier||'NOT VERIFIED'],['Starting XI',a.starting_xi_status||'NOT VERIFIED'],['Goalkeepers',a.goalkeeper_status||'NOT VERIFIED'],['Injuries',a.injury_status||'NOT VERIFIED'],['Weather',a.weather_status||'NOT VERIFIED']])+'</div><div class="intel-card"><h3>Latest persisted availability</h3>'+intelKV([['Latest lineup',prettyStamp(counts.latest_lineup_at)],['Latest availability',prettyStamp(counts.latest_availability_at)],['Latest refresh',prettyStamp(counts.latest_refresh_at)]])+'<div class="intel-footnote">Material unverified availability continues to block BET eligibility. Missing data is never treated as healthy or confirmed.</div></div></div></section>';

    const advanced='<section class="intel-section" data-detail-section="advanced"><div class="intel-accordion"><details><summary>Feature snapshots <span>'+esc(counts.feature_snapshots??0)+'</span></summary><div class="intel-accordion-body">'+snapshotCards(rel.feature_snapshots||[],'feature')+'</div></details><details><summary>Model runs <span>'+esc(counts.model_runs??0)+'</span></summary><div class="intel-accordion-body">'+snapshotCards(rel.model_runs||[],'model')+'</div></details><details><summary>Market snapshots <span>'+esc(counts.market_snapshots??0)+'</span></summary><div class="intel-accordion-body">'+snapshotCards(rel.market_snapshots||[],'market')+'</div></details><details><summary>Raw current-snapshot market table <span>'+esc((marketCtx.candidates||[]).length)+'</span></summary><div class="intel-accordion-body">'+marketTable(marketCtx.candidates||[])+'</div></details><details><summary>Missing / not verified <span>'+esc(missing.length)+'</span></summary><div class="intel-accordion-body">'+infoPills(missing,'missing')+'</div></details></div></section>';

    const multiMarket='<section class="intel-section" data-detail-section="periods">'+
       fe11MultimarketSection(d?.multi_market_research||null,f)+'</section>';
    const matchupResearch=ownerMatchupResearch(d,f,ownerP0Valid?ownerRaw:null,
      ownerResearchCard+ownerRecent+ownerSportRead+ownerSeasonResearch+ownerPersistedSportEvidence(d,f),
      blockedEvent);
    const sportState=d?.sport_analysis||{{}},opportunityState=d?.opportunity||{{}};
    const independentStatus='<div class="universal-status-grid" style="padding:12px 16px 0">'+
      '<div class="visual-card"><h3>Lectura deportiva</h3><strong>'+esc(ownerP0Valid?'Proyección RAW preliminar':analysisLabels[sportState.status]||sportState.status||'NOT VERIFIED')+
      '</strong><p>'+esc(ownerP0Valid?'Existe una investigación previa al partido, sin validación OOS ni licencia para publicación comercial.':sportState.reason||'El análisis deportivo se evalúa con independencia del mercado.')+'</p></div>'+
      '<div class="visual-card"><h3>Evaluación de valor</h3><strong>'+esc(valueLabels[opportunityState.status]||opportunityState.status||'Mercado no evaluado')+
      '</strong><p>'+esc(opportunityState.reason||'No consta evaluación verificable de valor.')+'</p></div></div>';
    const notesHtml='<section class="analyst-notebook"><div class="visual-card-head"><h3>Mi lectura del partido</h3><span>Nota privada · No modifica el modelo</span></div>'+
      '<p>Registra información que el sistema pudo pasar por alto. No se convierte en estadística verificada ni altera ninguna proyección histórica.</p>'+
      '<textarea id="analystNoteText" maxlength="2000" rows="4" placeholder="Mi hipótesis deportiva (no sustituye fuentes verificadas)…"></textarea>'+
      '<div class="analyst-note-actions"><button type="button" class="secondary" id="saveAnalystNote">Guardar nota</button>'+
      '<button type="button" class="secondary" id="deleteAnalystNote">Eliminar</button><span id="analystNoteStatus">Solo visible para esta cuenta.</span></div></section>';
    if(blockedEvent){{
      const pastFacts=sportN>0?
        '<details class="verified-history"><summary>Ver estadísticas deportivas históricas verificadas ('+sportN+')</summary>'+
        '<div class="intel-accordion-body">'+verifiedSportInputs(features)+'</div></details>':
        '<div class="intel-footnote">Este expediente todavía no contiene estadísticas deportivas históricas guardadas. No equivale a que los clubes no tengan datos.</div>';
      const headline=statusLabel==='CANCELADO'?'Este partido está cancelado':'Este partido está aplazado o suspendido';
      box.innerHTML='<div class="match-detail">'+hero+
        '<section class="match-event-block"><div class="match-event-alert">'+
        '<span>ESTADO DEL ENCUENTRO · '+esc(actualStatus)+'</span>'+
        '<h2>'+esc(headline)+'</h2>'+
        '<p>No corresponde emitir una proyección vigente ni una recomendación de valor sobre un partido que no se disputará según el estado disponible.</p>'+
        '<p class="status-source">'+esc(statusProvenance)+' · '+esc(statusObserved?prettyStamp(statusObserved):'Fecha de estado no verificada')+'</p></div>'+
        '<div class="match-event-facts"><h3>Información que sí conocemos</h3>'+
        '<p>'+esc(f.home_team||'Local')+' vs '+esc(f.away_team||'Visitante')+' · '+esc(f.league||'Competición')+
        ' · Inicio originalmente previsto: '+esc(kickDate)+' '+esc(kickTime)+'</p>'+
        '<p>Estado de oportunidad: evento no vigente. Los antecedentes deportivos, si están guardados, siguen consultables sin promocionarlos a una predicción.</p>'+
        ownerRecent+ownerSportRead+ownerSeasonResearch+pastFacts+'</div>'+notesHtml+'</section></div>';
      void loadAnalystNote(f.fixture_id);
      const back=$('backToMatches');if(back)back.onclick=closeMatchRoute;
      return;
    }}
    box.innerHTML='<div class="match-detail">'+hero+independentStatus+summary+tabHtml+
      '<div class="intel-body">'+overview+sportSection+(privateMatchup?matchupResearch:multiMarket)+marketsSection+availabilitySection+advanced+notesHtml+'</div></div>';
    void loadAnalystNote(f.fixture_id);
    box.querySelectorAll('[data-detail-tab]').forEach(btn=>btn.onclick=()=>{{const id=btn.dataset.detailTab;box.querySelectorAll('[data-detail-tab]').forEach(x=>x.classList.toggle('active',x===btn));box.querySelectorAll('[data-detail-section]').forEach(x=>x.classList.toggle('active',x.dataset.detailSection===id))}});
    const back=$('backToMatches');if(back)back.onclick=closeMatchRoute;
  }}
  async function saveCandidate(c){{if(!token()){{openAuth();return}}if(!premium()){{activate('account');return}}try{{await apiWrite('/my-edge','POST',candidateSavedPayload(c));MYEDGE_LOADED=false;await loadMyEdge(true)}}catch(e){{$('savedBody').innerHTML=empty('Could not save item',e.message)}}}}
  async function saveMatch(fid){{if(!token()){{openAuth();return}}if(!premium()){{activate('account');return}}const target=(DATA?.slate?.rows||[]).find(x=>String(x?.fixture?.fixture_id||'')===String(fid)),f=target?.fixture||{{}};if(!f.fixture_id)return;try{{await apiWrite('/my-edge','POST',{{item_type:'MATCH',fixture_id:f.fixture_id,source_snapshot_at:DATA?.generated_at_utc||DATA?.generated_at_local||null,payload:{{home_team:f.home_team||null,away_team:f.away_team||null,league:f.league||f.country||null,kickoff:f.kickoff||null}}}});MYEDGE_LOADED=false;await loadMyEdge(true)}}catch(e){{$('savedBody').innerHTML=empty('Could not save match',e.message)}}}}
  function wireSaveButtons(){{const map=new Map(candidateRows().map(x=>[candidateKey(x),x]));document.querySelectorAll('[data-save-key]').forEach(btn=>btn.onclick=e=>{{e.stopPropagation();const c=map.get(btn.dataset.saveKey);if(c)saveCandidate(c)}})}}
  function wireMatchSaveButtons(){{document.querySelectorAll('[data-save-match]').forEach(btn=>btn.onclick=e=>{{e.stopPropagation();saveMatch(btn.dataset.saveMatch)}})}}
  function savedRow(row){{const p=row?.payload||{{}},kind=String(row?.item_type||'SAVED'),match=((p.home_team||'Home')+' vs '+(p.away_team||'Away')),market=[row?.selection,row?.line,row?.market_name||row?.market_family].filter(x=>x!==null&&x!==undefined&&String(x).length).join(' · ');return '<div class="saved-row"><div><span class="status '+(kind==='BET'?'bet':kind==='LEAN'?'lean':'watch')+'">'+esc(kind)+'</span></div><div><b>'+esc(match)+'</b><small>'+esc(p.league||'Competition')+(market?' · '+esc(market):'')+(p.price_display?' · '+esc(p.price_display):'')+'</small></div><div><button type="button" class="secondary remove" data-remove-saved="'+esc(row?.item_key||'')+'">Remove</button></div></div>'}}
  function renderMyEdge(rows){{SAVED_ROWS=rows||[];const box=$('savedBody');box.innerHTML=SAVED_ROWS.length?'<div class="panel-head"><div><h3>Saved Edge</h3><div class="sub">Account-persisted product state only; never a model input.</div></div><span class="fresh-chip">'+SAVED_ROWS.length+' SAVED</span></div><div class="saved-list">'+SAVED_ROWS.map(savedRow).join('')+'</div>':empty('No saved items yet','Use ☆ Save on a pick, lean or match to build your personal slate.');box.querySelectorAll('[data-remove-saved]').forEach(btn=>btn.onclick=()=>removeSaved(btn.dataset.removeSaved))}}
  async function removeSaved(key){{if(!key)return;try{{await apiWrite('/my-edge?item_key='+encodeURIComponent(key),'DELETE',null);SAVED_ROWS=SAVED_ROWS.filter(x=>x.item_key!==key);renderMyEdge(SAVED_ROWS)}}catch(e){{$('savedBody').innerHTML=empty('Could not remove item',e.message)}}}}
  async function loadMyEdge(force=false){{const box=$('savedBody');if(!token()){{box.innerHTML=empty('Sign in to use My Edge','Saved items are attached to your authenticated account.');return}}if(!premium()){{box.innerHTML=lock('My Edge',0);return}}if(MYEDGE_LOADED&&!force){{renderMyEdge(SAVED_ROWS);return}}box.innerHTML=empty('Loading My Edge','Reading your account-saved items…');try{{const d=await api('/my-edge');MYEDGE_LOADED=true;renderMyEdge(d.rows||[])}}catch(e){{box.innerHTML=empty('My Edge unavailable',e.message)}}}}

  function checkoutQueryStatus(){{const q=new URLSearchParams(location.search),value=q.get('checkout');return value==='success'?'Checkout completed. Your PRO access updates only after the verified Stripe webhook persists the entitlement.':value==='cancel'?'Checkout canceled. No subscription change was assumed.':null}}
  function renderBillingState(account){{ACCOUNT_DATA=account||{{}};const box=$('billingBody'),state=account?.billing_state||{{}},billing=state.billing||{{}},markets=billing.markets||{{}},owner=state.owner_or_admin_override===true,status=state.status||state.effective_plan_reason||'NOT VERIFIED',queryNote=checkoutQueryStatus();
    if(!account?.access?.authenticated){{box.innerHTML='<div class="panel-head"><h3>Edge Pro</h3><span class="decision">SIGN IN</span></div><div class="sub">Sign in before managing a subscription. Billing market and price are never inferred from language or location.</div>';return}}
    if(owner){{box.innerHTML='<div class="panel-head"><h3>Internal Pro access</h3><span class="decision bet">OWNER / ADMIN</span></div><div class="sub">This account has verified internal Pro access. It is never prompted to purchase a subscription.</div>';return}}
    const entries=Object.entries(markets),options='<option value="">Choose billing market</option>'+entries.map(([key,v])=>'<option value="'+esc(key)+'">'+esc(key==='US'?'United States':'Mexico / LATAM')+' — '+esc(v.display_price||'Price NOT VERIFIED')+'</option>').join('');
    const lifecycle='<div class="detail-list">'+detailKV('Effective plan',state.effective_plan||'FREE')+detailKV('Subscription status',status)+detailKV('Current period end',state.current_period_end||'NOT VERIFIED')+detailKV('Cancel at period end',state.cancel_at_period_end===null||state.cancel_at_period_end===undefined?'NOT VERIFIED':state.cancel_at_period_end?'YES':'NO')+'</div>';
    let action='';
    if(state.manage_subscription_eligible)action='<button id="manageBillingBtn" type="button" class="primary">Manage subscription</button>';
    else if(state.upgrade_eligible)action='<select id="billingMarket" style="width:100%;margin:10px 0 8px;padding:9px;border-radius:8px;border:1px solid #1d4359;background:#071722;color:#eff7fb">'+options+'</select><button id="upgradeBillingBtn" type="button" class="primary">Upgrade with Stripe Checkout</button>';
    else if(!billing.public_launch_enabled)action='<div class="badge warn">Billing infrastructure ready · public launch disabled</div>';
    else if(String(state.effective_plan||'').toUpperCase()==='PRO')action='<div class="badge good">Edge Pro active</div>';
    box.innerHTML='<div class="panel-head"><div><h3>Edge Pro subscription</h3><div class="sub">Stripe-hosted Checkout · Customer Portal · webhook-authoritative entitlement</div></div><span class="fresh-chip">'+esc(billing.status||'BILLING STATE')+'</span></div>'+lifecycle+'<div style="margin-top:12px">'+action+'</div>'+(queryNote?'<div class="reason"><b>Checkout:</b> '+esc(queryNote)+'</div>':'')+'<div class="reason"><b>Pricing:</b> '+entries.map(([k,v])=>esc((k==='US'?'US':'MX/LATAM')+' '+(v.display_price||'NOT VERIFIED'))).join(' · ')+'<br>Choose your billing market explicitly. The browser never sends a Stripe Price ID.</div>';
    const upgrade=$('upgradeBillingBtn');if(upgrade)upgrade.onclick=async()=>{{const market=$('billingMarket')?.value||'';if(!['US','MX_LATAM'].includes(market)){{upgrade.textContent='Choose billing market first';return}}upgrade.disabled=true;upgrade.textContent='Opening secure Checkout…';try{{const d=await billingCall(billing.checkout_function||'create-checkout-session',{{billing_market:market}});if(!d.url)throw new Error('CHECKOUT_URL_MISSING');location.href=d.url}}catch(e){{upgrade.disabled=false;upgrade.textContent=e.message}}}};
    const manage=$('manageBillingBtn');if(manage)manage.onclick=async()=>{{manage.disabled=true;manage.textContent='Opening portal…';try{{const d=await billingCall(billing.portal_function||'create-customer-portal',{{}});if(!d.url)throw new Error('PORTAL_URL_MISSING');location.href=d.url}}catch(e){{manage.disabled=false;manage.textContent=e.message}}}}
  }}
  async function loadAccount(force=false){{const box=$('billingBody');if(ACCOUNT_LOADED&&!force){{renderBillingState(ACCOUNT_DATA);return}}box.innerHTML=empty('Loading subscription state','Reading verified entitlement and billing lifecycle…');try{{const d=await api('/account');ACCOUNT_LOADED=true;renderBillingState(d)}}catch(e){{box.innerHTML=empty('Account unavailable',e.message)}}}}

  function performanceFamilyRows(rows){{if(!rows?.length)return '<tr><td colspan="7">No verified BET-family settlement sample.</td></tr>';return rows.map(r=>'<tr><td>'+esc(r.market_family||'N/V')+'</td><td>'+esc(r.settled??'—')+'</td><td>'+esc((r.win??0)+'-'+(r.loss??0)+'-'+(r.push??0))+'</td><td>'+pct(r.hit_rate_ex_push)+'</td><td>'+units(r.roi_units)+'</td><td>'+esc(r.ungraded??'—')+'</td><td>'+esc(r.status||'N/V')+'</td></tr>').join('')}}
  function validationRows(rows){{if(!rows?.length)return '<tr><td colspan="7">No persisted OOS/validation rows available.</td></tr>';return rows.map(r=>'<tr><td>'+esc(r.label||'N/V')+'</td><td>'+esc(r.sample_n??'—')+'</td><td>'+esc(r.settled??'—')+'</td><td>'+esc(r.roi_units??'—')+'</td><td>'+esc(r.avg_clv_pp??'—')+'</td><td>'+esc(r.brier??'—')+'</td><td>'+esc(r.status||'N/V')+'</td></tr>').join('')}}
  async function loadPerformance(){{if(PERF_LOADED)return;PERF_LOADED=true;const box=$('performanceBody');if(!premium()){{box.innerHTML=lock('Verified Performance',0);return}}box.innerHTML=empty('Loading performance','Reading canonical settlement and validation evidence…');try{{const d=await api('/performance'),bet=d.bet_track_record||{{}},lean=d.research_lean||{{}},settlement=d.settlement||{{}},clv=d.true_clv||{{}},validation=d.validation_evidence||{{}},notes=d.notes||[];
    const record=(bet.settled===null||bet.settled===undefined)?'—':(bet.win||0)+'-'+(bet.loss||0)+'-'+(bet.push||0),sample=bet.sample_status||'SAMPLE NOT VERIFIED';
    const headline='<div class="panel-head"><div><h3>BET-only Track Record</h3><div class="sub">Canonical settled BET rows only. LEANS and research rows cannot improve this headline.</div></div><span class="fresh-chip">'+esc(sample)+'</span></div>'+
      '<div class="kpis"><div class="kpi good"><b>'+esc(bet.settled??'—')+'</b><span>graded bets</span></div><div class="kpi"><b>'+esc(record)+'</b><span>W-L-P</span></div><div class="kpi"><b>'+pct(bet.hit_rate_ex_push)+'</b><span>hit rate</span></div><div class="kpi"><b>'+units(bet.roi_units)+'</b><span>ROI units</span></div><div class="kpi lean"><b>'+esc(clv.rows??'—')+'</b><span>true CLV rows</span></div></div>';
    const trust='<div class="detail-grid two"><div class="detail-card"><h3>Settlement integrity</h3><div class="detail-list">'+detailKV('Ledger rows',settlement.ledger_rows??'NOT VERIFIED')+detailKV('Actionable rows',settlement.source_actionable_rows??'NOT VERIFIED')+detailKV('Coverage',settlement.coverage_rate!==null&&settlement.coverage_rate!==undefined?pct(settlement.coverage_rate):'NOT VERIFIED')+detailKV('Backlog',settlement.backlog_rows??'NOT VERIFIED')+'</div></div><div class="detail-card"><h3>True CLV</h3><div class="detail-list">'+detailKV('Status',clv.status||'NOT VERIFIED')+detailKV('Rows',clv.rows??'NOT VERIFIED')+detailKV('Average',clv.avg_probability_pp!==null&&clv.avg_probability_pp!==undefined?pp(clv.avg_probability_pp):'NOT VERIFIED')+detailKV('Positive / Negative / Flat',(clv.positive??'—')+' / '+(clv.negative??'—')+' / '+(clv.flat??'—'))+detailKV('Same-book rows',clv.same_book_rows??'NOT VERIFIED')+'</div></div></div>';
    const betFamilies='<div class="section-title"><div><h2>BET performance by market</h2><span>Headline-eligible canonical BET settlements only</span></div></div><div class="table-wrap"><table><thead><tr><th>Market</th><th>Settled</th><th>W-L-P</th><th>Hit rate</th><th>ROI</th><th>Ungraded</th><th>Status</th></tr></thead><tbody>'+performanceFamilyRows(bet.families||[])+'</tbody></table></div>';
    const leanResearch='<div class="section-title"><div><h2>LEAN research record</h2><span>Shown separately; never included in BET headline</span></div></div><div class="detail-grid two"><div class="detail-card"><h3>LEAN research sample</h3><div class="detail-list">'+detailKV('Settled',lean.settled??'NOT VERIFIED')+detailKV('W-L-P',(lean.win??0)+'-'+(lean.loss??0)+'-'+(lean.push??0))+detailKV('Hit rate',lean.hit_rate_ex_push!==null&&lean.hit_rate_ex_push!==undefined?pct(lean.hit_rate_ex_push):'NOT VERIFIED')+detailKV('ROI units',units(lean.roi_units))+detailKV('Headline eligible',lean.headline_eligible===false?'NO':'NOT VERIFIED')+'</div></div><div class="detail-card"><h3>Performance policy</h3><div class="reason">'+esc((d.performance_policy||{{}}).headline_scope||'CANONICAL_BET_SETTLEMENT_ONLY')+' · LEANS excluded from headline · no historical reconstruction/backfill by this view.</div></div></div>';
    const validationBlock='<div class="section-title"><div><h2>Research / OOS validation</h2><span>Evidence quality, not realized customer BET performance</span></div></div><div class="table-wrap"><table><thead><tr><th>Market</th><th>Sample</th><th>Settled</th><th>ROI units</th><th>True CLV</th><th>Brier</th><th>Status</th></tr></thead><tbody>'+validationRows(validation.rows||[])+'</tbody></table></div>';
    const noteHtml='<div class="detail-card" style="margin-top:12px"><h3>Trust notes</h3><div class="detail-list">'+notes.map(x=>detailKV('Policy',x)).join('')+'</div></div>';
    box.innerHTML=headline+trust+betFamilies+leanResearch+validationBlock+noteHtml
  }}catch(e){{box.innerHTML=empty('Performance unavailable',e.message)}}}}

  function activate(id){{document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===id));window.scrollTo({{top:0,behavior:'smooth'}});if(id==='performance')loadPerformance();if(id==='myedge')loadMyEdge();if(id==='account')loadAccount()}}
  function openSection(id){{if(routeMatchId())history.pushState({{page:id}},'','/app');if(id==='matches')showMatchBrowse(true);$('matchDetail').classList.add('hidden');activate(id)}}
  document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>openSection(b.dataset.page));document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>openSection(b.dataset.go));
  window.addEventListener('popstate',()=>{{const fid=routeMatchId();if(fid)openMatch(fid,{{replace:true}});else{{showMatchBrowse(true);$('matchDetail').classList.add('hidden');activate('matches')}}}});
  const modal=$('authModal'),openAuth=()=>{{modal.classList.add('open');modal.setAttribute('aria-hidden','false')}},closeAuth=()=>{{modal.classList.remove('open');modal.setAttribute('aria-hidden','true')}};
  $('accountBtn').onclick=()=>token()?activate('account'):openAuth();$('accountPageBtn').onclick=openAuth;$('closeAuth').onclick=closeAuth;
  $('signOut').onclick=()=>{{localStorage.removeItem(AK);localStorage.removeItem(RK);location.reload()}};
  $('signIn').onclick=async()=>{{try{{$('authStatus').textContent='Signing in…';const d=await authFetch('/auth/v1/token?grant_type=password',{{email:$('authEmail').value,password:$('authPassword').value}});localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);location.reload()}}catch(e){{$('authStatus').textContent=e.message}}}};
  $('signUp').onclick=async()=>{{try{{$('authStatus').textContent='Creating account…';const d=await authFetch('/auth/v1/signup',{{email:$('authEmail').value,password:$('authPassword').value}});if(d.access_token)localStorage.setItem(AK,d.access_token);if(d.refresh_token)localStorage.setItem(RK,d.refresh_token);$('authStatus').textContent=d.access_token?'Account created. Reloading…':'Account created. Check your email if confirmation is required.';if(d.access_token)location.reload()}}catch(e){{$('authStatus').textContent=e.message}}}};
  async function registerPwa(){{if(!('serviceWorker' in navigator))return;try{{await navigator.serviceWorker.register('/sw.js',{{scope:'/'}})}}catch(_e){{/* PWA installability is optional; live customer data remains network-only */}}}}
  (async()=>{{try{{await loadToday();$('boot').classList.add('hidden');registerPwa();const fid=routeMatchId();if(fid)await openMatch(fid,{{replace:true}})}}catch(e){{$('bootCopy').textContent='Snapshot unavailable · '+e.message+' · no values were fabricated.';const line=document.querySelector('.boot-line');if(line)line.classList.add('hidden')}}}})();
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
        "route": "/app",
        "preview_alias": "/app-v2",
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
        "performance_headline_scope": "CANONICAL_BET_SETTLEMENT_ONLY",
        "research_oos_separate_from_realized_bets": True,
        "my_edge_persistence": "SUPABASE_RLS_SUBSCRIBER_SAVED_ITEMS",
        "my_edge_model_input_allowed": False,
        "billing_surface": "GUARDED_STRIPE_HOSTED_CHECKOUT_CUSTOMER_PORTAL",
        "browser_sends_stripe_price_id": False,
        "public_billing_launch_default": False,
        "pwa_manifest": "/app.webmanifest",
        "service_worker": "/sw.js",
        "live_data_cache_allowed": False,
        "push_notifications_enabled": False,
        "frontend_creates_bet_or_lean": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
