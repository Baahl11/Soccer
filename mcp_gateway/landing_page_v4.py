from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import HTMLResponse

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_LANDING_PAGE_V4_1.0.0"

_COPY = {
    "en": {
        "title": "Soccer Edge — Model vs Market",
        "description": "Transparent soccer market intelligence built around prices, probabilities, calibration and verified performance.",
        "eyebrow": "MODEL VS MARKET",
        "hero": "Stop betting blind.",
        "sub": "Soccer Edge compares market prices with calibrated model probabilities and shows the evidence behind every signal — including when the right answer is NO BET.",
        "cta": "Open Explorer",
        "cta2": "See how it works",
        "trust1": "Price-aware",
        "trust1b": "The question is not only who wins. It is whether the market price is wrong.",
        "trust2": "Evidence-gated",
        "trust2b": "Research markets stay research until enough verified evidence exists.",
        "trust3": "Transparent",
        "trust3b": "Settlements, sample size and closing-line evidence are not hidden when results are inconvenient.",
        "section": "What Soccer Edge shows you",
        "f1": "Model vs Market",
        "f1b": "Compare market-implied probability with the calibrated model probability and the measured gap.",
        "f2": "Market readiness",
        "f2b": "Know whether a signal is ready, waiting for price, waiting for lineup confirmation, or a no-bet.",
        "f3": "Verified performance",
        "f3b": "Track settled BET-only performance and sample size instead of screenshots with missing losses.",
        "f4": "Premium match detail",
        "f4b": "Edge Pro unlocks price provenance, calibrated probability, edge, blockers and advanced market families.",
        "free": "Explorer — Free",
        "freeb": "Verified slate, market radar, model maturity and public read-only intelligence.",
        "pro": "Edge Pro",
        "prob": "Strong signals, calibrated value, premium match detail, advanced markets and verified history.",
        "open": "Create free account",
        "footer": "Soccer Edge provides analytical information, not guaranteed outcomes. Betting involves risk. Use only where legal and only if you meet the legal age requirements in your jurisdiction.",
        "lang": "Language",
    },
    "es": {
        "title": "Soccer Edge — Modelo vs Mercado",
        "description": "Inteligencia transparente del mercado de fútbol basada en precios, probabilidades, calibración y rendimiento verificable.",
        "eyebrow": "MODELO VS MERCADO",
        "hero": "Deja de apostar a ciegas.",
        "sub": "Soccer Edge compara los precios del mercado con probabilidades calibradas del modelo y muestra la evidencia detrás de cada señal, incluso cuando la respuesta correcta es NO BET.",
        "cta": "Abrir Explorer",
        "cta2": "Ver cómo funciona",
        "trust1": "Consciente del precio",
        "trust1b": "La pregunta no es solo quién gana. Es si el precio del mercado está equivocado.",
        "trust2": "Basado en evidencia",
        "trust2b": "Los mercados de investigación siguen en investigación hasta acumular evidencia verificada suficiente.",
        "trust3": "Transparente",
        "trust3b": "No ocultamos settlements, tamaño de muestra ni evidencia contra el cierre cuando los resultados no convienen.",
        "section": "Qué te muestra Soccer Edge",
        "f1": "Modelo vs Mercado",
        "f1b": "Compara la probabilidad implícita del mercado con la probabilidad calibrada del modelo y la diferencia medida.",
        "f2": "Estado del mercado",
        "f2b": "Sabe si una señal está lista, esperando precio, esperando alineación o es un no-bet.",
        "f3": "Rendimiento verificable",
        "f3b": "Sigue rendimiento BET-only liquidado y tamaño de muestra, no capturas donde desaparecen las pérdidas.",
        "f4": "Detalle premium del partido",
        "f4b": "Edge Pro desbloquea procedencia del precio, probabilidad calibrada, edge, bloqueadores y mercados avanzados.",
        "free": "Explorer — Gratis",
        "freeb": "Cartelera verificada, radar de mercado, maduración del modelo e inteligencia pública de solo lectura.",
        "pro": "Edge Pro",
        "prob": "Señales fuertes, valor calibrado, detalle premium, mercados avanzados e historial verificable.",
        "open": "Crear cuenta gratis",
        "footer": "Soccer Edge ofrece información analítica, no resultados garantizados. Apostar implica riesgo. Úsalo solo donde sea legal y si cumples la edad legal requerida en tu jurisdicción.",
        "lang": "Idioma",
    },
}


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def render_landing() -> str:
    copy = _safe_json(_COPY)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soccer Edge — Model vs Market</title><meta id="desc" name="description" content="Transparent soccer market intelligence built around prices, probabilities, calibration and verified performance.">
<meta property="og:title" id="ogTitle" content="Soccer Edge — Model vs Market"><meta property="og:description" id="ogDesc" content="Transparent soccer market intelligence built around prices, probabilities, calibration and verified performance.">
<style>
:root{{--bg:#040a10;--panel:#08141e;--panel2:#0b1d29;--line:#17364a;--text:#f1f8fb;--muted:#8199aa;--green:#5de3b0;--blue:#75c8ff;--amber:#f3c86f}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 18% 0,#123149 0,#07131d 30%,#040a10 70%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}a{{color:inherit;text-decoration:none}}button{{font:inherit}}.wrap{{max-width:1180px;margin:auto;padding:20px}}.nav{{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:8px 0 28px}}.brand{{font-weight:950;font-size:24px;letter-spacing:-.04em}}.brand span{{color:var(--green)}}.navright{{display:flex;align-items:center;gap:10px}}.langs{{display:flex;border:1px solid var(--line);border-radius:999px;padding:2px;background:#07131d}}.langs button{{border:0;background:transparent;color:var(--muted);font-size:10px;font-weight:900;padding:6px 9px;border-radius:999px;cursor:pointer}}.langs button.active{{background:#13384f;color:#fff}}.login{{border:1px solid var(--line);padding:8px 11px;border-radius:9px;font-size:11px;font-weight:850}}
.hero{{min-height:570px;display:grid;grid-template-columns:1.15fr .85fr;gap:28px;align-items:center;padding:40px 0 60px}}.eyebrow{{font-size:11px;font-weight:950;letter-spacing:.14em;color:var(--green)}}h1{{font-size:72px;line-height:.94;letter-spacing:-.065em;margin:12px 0 20px;max-width:750px}}.lead{{font-size:18px;line-height:1.6;color:#9ab0bf;max-width:760px}}.actions{{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}}.btn{{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:10px;border:1px solid #28536e;font-weight:900;font-size:12px;background:#0b2230}}.btn.primary{{background:#0e6448;border-color:#17835f;color:white}}.terminal{{border:1px solid var(--line);border-radius:20px;background:linear-gradient(160deg,#0d2230,#07121b);padding:18px;box-shadow:0 25px 80px rgba(0,0,0,.35)}}.termhead{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #163245;padding-bottom:11px;margin-bottom:13px}}.signal{{border:1px solid #18394d;border-radius:12px;background:#07141e;padding:14px;margin-top:9px}}.row{{display:flex;justify-content:space-between;gap:15px;align-items:center}}.teams{{font-size:13px;font-weight:900}}.label{{font-size:10px;color:var(--muted)}}.big{{font-size:26px;font-weight:950;color:var(--green)}}.pill{{border:1px solid #28526b;border-radius:999px;padding:6px 9px;font-size:9px;font-weight:900;color:var(--blue)}}
.trust,.features,.plans{{display:grid;gap:12px}}.trust{{grid-template-columns:repeat(3,1fr);margin-bottom:80px}}.features{{grid-template-columns:repeat(2,1fr)}}.plans{{grid-template-columns:repeat(2,1fr);margin-top:32px}}.card{{border:1px solid var(--line);border-radius:14px;background:rgba(8,20,30,.88);padding:18px}}.card h3{{font-size:15px;margin:4px 0 8px}}.card p{{margin:0;color:var(--muted);line-height:1.55;font-size:12px}}.section{{padding:20px 0 78px}}.section h2{{font-size:38px;letter-spacing:-.04em;margin:9px 0 22px}}.plan{{min-height:190px;display:flex;flex-direction:column;justify-content:space-between}}.plan.pro{{border-color:#24684f;background:linear-gradient(150deg,#0b291f,#091822)}}.footer{{border-top:1px solid var(--line);padding:24px 0 36px;color:#667f90;font-size:10px;line-height:1.6}}
@media(max-width:850px){{.hero{{grid-template-columns:1fr;min-height:auto;padding-top:20px}}h1{{font-size:52px}}.trust{{grid-template-columns:1fr}}.features,.plans{{grid-template-columns:1fr}}}}@media(max-width:520px){{.wrap{{padding:14px}}h1{{font-size:43px}}.lead{{font-size:15px}}.nav{{align-items:flex-start}}.navright{{flex-direction:column-reverse;align-items:flex-end}}}}
</style></head><body><div class="wrap">
<nav class="nav"><div class="brand">Soccer <span>Edge</span></div><div class="navright"><div class="langs" aria-label="Language"><button id="enBtn">EN</button><button id="esBtn">ES</button></div><a class="login" href="/app">Explorer</a></div></nav>
<section class="hero"><div><div class="eyebrow" data-k="eyebrow"></div><h1 data-k="hero"></h1><p class="lead" data-k="sub"></p><div class="actions"><a class="btn primary" href="/app" data-k="cta"></a><a class="btn" href="#how" data-k="cta2"></a></div></div>
<div class="terminal"><div class="termhead"><div><div class="eyebrow">SOCCER EDGE</div><div class="teams">MODEL vs MARKET</div></div><div class="pill">READ-ONLY DEMO</div></div><div class="signal"><div class="row"><div><div class="label">MARKET</div><div class="teams">Fair probability</div></div><div class="big">53.5%</div></div></div><div class="signal"><div class="row"><div><div class="label">MODEL</div><div class="teams">Calibrated probability</div></div><div class="big">61.4%</div></div></div><div class="signal"><div class="row"><div><div class="label">DECISION</div><div class="teams">Evidence first</div></div><div class="pill">WAIT / BET / NO BET</div></div></div></div></section>
<section class="trust"><article class="card"><div class="eyebrow">01</div><h3 data-k="trust1"></h3><p data-k="trust1b"></p></article><article class="card"><div class="eyebrow">02</div><h3 data-k="trust2"></h3><p data-k="trust2b"></p></article><article class="card"><div class="eyebrow">03</div><h3 data-k="trust3"></h3><p data-k="trust3b"></p></article></section>
<section class="section" id="how"><div class="eyebrow">SOCCER EDGE</div><h2 data-k="section"></h2><div class="features"><article class="card"><h3 data-k="f1"></h3><p data-k="f1b"></p></article><article class="card"><h3 data-k="f2"></h3><p data-k="f2b"></p></article><article class="card"><h3 data-k="f3"></h3><p data-k="f3b"></p></article><article class="card"><h3 data-k="f4"></h3><p data-k="f4b"></p></article></div><div class="plans"><article class="card plan"><div><div class="eyebrow">FREE</div><h3 data-k="free"></h3><p data-k="freeb"></p></div><div class="actions"><a class="btn" href="/app" data-k="open"></a></div></article><article class="card plan pro"><div><div class="eyebrow">PRO</div><h3 data-k="pro"></h3><p data-k="prob"></p></div><div class="actions"><a class="btn primary" href="/app" data-k="cta"></a></div></article></div></section>
<footer class="footer" data-k="footer"></footer></div>
<script id="landingCopy" type="application/json">{copy}</script><script>
const copy=JSON.parse(document.getElementById('landingCopy').textContent);const key='soccer_edge_locale';let locale=localStorage.getItem(key)||((navigator.language||'en').toLowerCase().startsWith('es')?'es':'en');if(!copy[locale])locale='en';
function apply(){{const d=copy[locale];document.documentElement.lang=locale;document.title=d.title;document.getElementById('desc').content=d.description;document.getElementById('ogTitle').content=d.title;document.getElementById('ogDesc').content=d.description;document.querySelectorAll('[data-k]').forEach(el=>{{const k=el.dataset.k;if(d[k])el.textContent=d[k]}});document.getElementById('enBtn').classList.toggle('active',locale==='en');document.getElementById('esBtn').classList.toggle('active',locale==='es')}}document.getElementById('enBtn').onclick=()=>{{locale='en';localStorage.setItem(key,locale);apply()}};document.getElementById('esBtn').onclick=()=>{{locale='es';localStorage.setItem(key,locale);apply()}};apply();
</script></body></html>"""


async def landing_page(request: Request) -> HTMLResponse:
    return HTMLResponse(render_landing())
