from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse

from mcp_gateway import subscriber_billing_v2
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_LANDING_V2_1.0.0"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def render() -> str:
    auth = supabase_auth_v4.public_auth_config()
    billing = subscriber_billing_v2.public_billing_contract()
    config = _safe_json({
        "supabase_url": auth.get("project_url"),
        "publishable_key": auth.get("publishable_key"),
        "analytics_enabled": bool(auth.get("configured")),
        "billing_launch_enabled": bool(billing.get("public_launch_enabled")),
        "app_url": "/app",
    })
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soccer Edge — Sport First. Market Second.</title>
<meta name="description" content="Transparent soccer intelligence with canonical BETS, LEANS, WATCH states, exact market context and verified performance.">
<style>
:root{{--bg:#040a0f;--panel:#081722;--line:#17374a;--text:#eff7fb;--muted:#8098a8;--green:#4de1ad;--blue:#64c1ff;--amber:#e9bc68}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 50% -10%,#12364e,#07131d 34%,#040a0f 70%);color:var(--text);font:14px Inter,system-ui,-apple-system,Segoe UI,sans-serif}}
a{{color:inherit;text-decoration:none}}button{{font:inherit}}.wrap{{max-width:1180px;margin:auto;padding:20px}}.nav{{display:flex;justify-content:space-between;align-items:center;padding:6px 0 22px;gap:14px}}.brand{{font-size:21px;font-weight:950;letter-spacing:-.04em}}.brand span{{color:var(--green)}}.nav-actions{{display:flex;gap:8px}}.btn{{display:inline-flex;align-items:center;justify-content:center;border:1px solid #26516a;background:#09202e;color:#eef7fb;border-radius:9px;padding:10px 13px;font-weight:850;font-size:11px}}.btn.primary{{background:#0c5b43;border-color:#1b7a5c}}.btn:hover{{filter:brightness(1.08)}}.hero{{display:grid;grid-template-columns:1.1fr .9fr;gap:34px;align-items:center;min-height:610px;padding:38px 0 70px}}.eyebrow{{color:var(--green);font-size:10px;font-weight:950;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:66px;line-height:.96;letter-spacing:-.055em;margin:12px 0 18px;max-width:760px}}.lead{{color:#98adba;font-size:17px;line-height:1.65;max-width:720px}}.hero-actions{{display:flex;gap:9px;flex-wrap:wrap;margin-top:24px}}.principles{{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}}.chip{{border:1px solid #1d455a;background:#081b27;border-radius:999px;padding:6px 9px;color:#8fb0c1;font-size:9px;font-weight:800}}.terminal{{border:1px solid #1c4258;background:linear-gradient(160deg,#0c2230,#07131c);border-radius:18px;padding:18px;box-shadow:0 28px 90px #0008}}.term-head{{display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid #15374b;padding-bottom:11px;margin-bottom:12px}}.term-head b{{font-size:12px}}.state{{display:grid;grid-template-columns:84px 1fr;gap:10px;align-items:start;padding:11px 0;border-bottom:1px solid #102d3d}}.state:last-child{{border-bottom:0}}.badge{{display:inline-flex;border-radius:999px;padding:5px 7px;font-size:8px;font-weight:950;border:1px solid #2a566f;color:#9dc4d7}}.badge.bet{{border-color:#2b7158;background:#0a3025;color:#72e6bb}}.badge.lean{{border-color:#2a6087;background:#0a2840;color:#75c8ff}}.badge.watch{{border-color:#6a5426;background:#2b220e;color:#e8bd68}}.state h3{{font-size:12px;margin:0 0 4px}}.state p{{margin:0;color:#718b9b;font-size:9px;line-height:1.45}}.section{{padding:72px 0}}.section-head{{max-width:760px;margin-bottom:24px}}.section h2{{font-size:38px;letter-spacing:-.04em;margin:8px 0 10px}}.section p{{color:var(--muted);line-height:1.6}}.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.grid2{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.card{{border:1px solid var(--line);background:rgba(8,23,34,.9);border-radius:13px;padding:18px}}.card h3{{font-size:14px;margin:7px 0 8px}}.card p{{margin:0;color:var(--muted);font-size:11px;line-height:1.55}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.flow .card{{min-height:145px}}.num{{color:var(--blue);font-size:10px;font-weight:950}}.plan{{display:flex;flex-direction:column;justify-content:space-between;min-height:230px}}.plan.pro{{border-color:#26654f;background:linear-gradient(155deg,#0a2b21,#081822)}}.feature-list{{display:grid;gap:7px;margin:12px 0 16px;color:#8ea5b3;font-size:10px}}.feature-list span:before{{content:'✓';color:var(--green);margin-right:7px}}.trust-note{{border:1px solid #4d4223;background:#221d0c;color:#d9bf7b;border-radius:10px;padding:12px;font-size:10px;line-height:1.5;margin-top:15px}}.footer{{border-top:1px solid var(--line);padding:24px 0 38px;color:#667f8f;font-size:10px;line-height:1.6}}.lang{{border:1px solid #24495f;background:#081b27;color:#9ab5c4;border-radius:8px;padding:8px 9px;font-size:10px}}
@media(max-width:900px){{.hero{{grid-template-columns:1fr;min-height:auto;padding-top:30px}}h1{{font-size:52px}}.grid3,.flow{{grid-template-columns:1fr 1fr}}}}
@media(max-width:620px){{.wrap{{padding:14px}}h1{{font-size:42px}}.lead{{font-size:15px}}.grid3,.grid2,.flow{{grid-template-columns:1fr}}.nav{{align-items:flex-start}}.nav-actions{{flex-wrap:wrap;justify-content:flex-end}}}}
</style>
</head>
<body><div class="wrap">
<nav class="nav"><div class="brand">Soccer <span>Edge</span></div><div class="nav-actions"><select id="lang" class="lang"><option value="en">EN</option><option value="es">ES</option></select><a class="btn" href="/app" data-track="explorer_cta">Open Explorer</a></div></nav>
<section class="hero"><div>
<div class="eyebrow" data-en="Sport first. Market second." data-es="Deporte primero. Mercado después.">Sport first. Market second.</div>
<h1 data-en="Know what the model likes. Know when the price is wrong." data-es="Sabe qué le gusta al modelo. Sabe cuándo el precio está mal.">Know what the model likes. Know when the price is wrong.</h1>
<p class="lead" data-en="Soccer Edge separates the raw sporting projection from the betting market, then shows canonical BETS, LEANS and WATCH states with the evidence behind each decision." data-es="Soccer Edge separa la proyección deportiva cruda del mercado de apuestas y después muestra BETS, LEANS y WATCH con la evidencia detrás de cada decisión.">Soccer Edge separates the raw sporting projection from the betting market, then shows canonical BETS, LEANS and WATCH states with the evidence behind each decision.</p>
<div class="hero-actions"><a class="btn primary" href="/app" data-track="explorer_cta" data-en="Explore today's slate" data-es="Explorar la cartelera de hoy">Explore today's slate</a><a class="btn" href="#how" data-en="See the decision process" data-es="Ver el proceso de decisión">See the decision process</a></div>
<div class="principles"><span class="chip">RAW SPORT ≠ MARKET SHRUNK</span><span class="chip">EXACT PRICE + TIMESTAMP</span><span class="chip">NOT VERIFIED stays NOT VERIFIED</span><span class="chip">ZERO BETS IS VALID</span></div>
</div>
<div class="terminal"><div class="term-head"><div><div class="eyebrow">DECISION STATES</div><b>No fabricated picks</b></div><span class="chip">CANONICAL ENGINE ONLY</span></div>
<div class="state"><span class="badge bet">BET</span><div><h3 data-en="Actionable verified edge" data-es="Edge verificado y accionable">Actionable verified edge</h3><p data-en="Exact market, line, price, bookmaker, model probability, market fair probability, Availability Confidence and evidence." data-es="Mercado, línea, precio, bookmaker, probabilidad del modelo, probabilidad justa del mercado, Availability Confidence y evidencia.">Exact market, line, price, bookmaker, model probability, market fair probability, Availability Confidence and evidence.</p></div></div>
<div class="state"><span class="badge lean">LEAN</span><div><h3 data-en="Interesting, not promoted" data-es="Interesante, sin promover">Interesting, not promoted</h3><p data-en="The sporting case can be strong while price, availability or evidence keeps it below BET." data-es="El caso deportivo puede ser fuerte, pero precio, disponibilidad o evidencia lo mantienen debajo de BET.">The sporting case can be strong while price, availability or evidence keeps it below BET.</p></div></div>
<div class="state"><span class="badge watch">WATCH</span><div><h3 data-en="Waiting for a required condition" data-es="Esperando una condición requerida">Waiting for a required condition</h3><p data-en="XI, goalkeeper, price, fresh quote or another material verification can remain unresolved." data-es="XI, portero, precio, cotización fresca u otra verificación material puede seguir pendiente.">XI, goalkeeper, price, fresh quote or another material verification can remain unresolved.</p></div></div>
</div></section>

<section class="section" id="how"><div class="section-head"><div class="eyebrow">HOW IT WORKS</div><h2 data-en="A visible path from sport to decision." data-es="Un camino visible del deporte a la decisión.">A visible path from sport to decision.</h2><p data-en="The market is inspected after the sporting thesis is built. The customer sees the distinction instead of one vague confidence number." data-es="El mercado se revisa después de construir la tesis deportiva. El usuario ve la diferencia en lugar de un solo número ambiguo de confianza.">The market is inspected after the sporting thesis is built. The customer sees the distinction instead of one vague confidence number.</p></div>
<div class="flow"><article class="card"><div class="num">01</div><h3>Raw Sport Projection</h3><p data-en="Team strength, matchup, style, availability, venue, schedule and validated sport features." data-es="Fuerza, matchup, estilo, disponibilidad, sede, calendario y variables deportivas validadas.">Team strength, matchup, style, availability, venue, schedule and validated sport features.</p></article>
<article class="card"><div class="num">02</div><h3>Market Comparison</h3><p data-en="Exact price, fair market probability, breakeven and current instrument." data-es="Precio exacto, probabilidad justa del mercado, breakeven e instrumento actual.">Exact price, fair market probability, breakeven and current instrument.</p></article>
<article class="card"><div class="num">03</div><h3>Decision Gate</h3><p data-en="BET, LEAN, WATCH or PASS. No forced card and no frontend-created recommendation." data-es="BET, LEAN, WATCH o PASS. Sin picks forzados ni recomendaciones creadas por el frontend.">BET, LEAN, WATCH or PASS. No forced card and no frontend-created recommendation.</p></article>
<article class="card"><div class="num">04</div><h3>Verified Record</h3><p data-en="BET-only realized performance stays separate from LEAN research and OOS validation." data-es="El rendimiento real BET-only permanece separado de investigación LEAN y validación OOS.">BET-only realized performance stays separate from LEAN research and OOS validation.</p></article></div></section>

<section class="section"><div class="section-head"><div class="eyebrow">WHY IT FEELS DIFFERENT</div><h2 data-en="More evidence, less storytelling." data-es="Más evidencia, menos narrativa.">More evidence, less storytelling.</h2></div>
<div class="grid3"><article class="card"><h3 data-en="Exact market context" data-es="Contexto exacto del mercado">Exact market context</h3><p data-en="Actionable decisions require the exact threshold, price, source and freshness context." data-es="Las decisiones accionables requieren umbral, precio, fuente y frescura exactos.">Actionable decisions require the exact threshold, price, source and freshness context.</p></article>
<article class="card"><h3 data-en="Availability is visible" data-es="La disponibilidad es visible">Availability is visible</h3><p data-en="XI, goalkeeper, injuries and other material unknowns do not disappear behind a confidence score." data-es="XI, portero, lesiones y otras incógnitas materiales no desaparecen detrás de un score de confianza.">XI, goalkeeper, injuries and other material unknowns do not disappear behind a confidence score.</p></article>
<article class="card"><h3 data-en="Performance is not cherry-picked" data-es="El rendimiento no se selecciona a conveniencia">Performance is not cherry-picked</h3><p data-en="Settled BET results, sample warnings, CLV and research evidence are labeled by what they actually are." data-es="Resultados BET liquidados, advertencias de muestra, CLV y evidencia de investigación se etiquetan por lo que realmente son.">Settled BET results, sample warnings, CLV and research evidence are labeled by what they actually are.</p></article></div></section>

<section class="section"><div class="section-head"><div class="eyebrow">ACCESS</div><h2>Explorer + Edge Pro</h2></div><div class="grid2">
<article class="card plan"><div><div class="eyebrow">FREE</div><h3>Explorer</h3><div class="feature-list"><span data-en="Verified slate" data-es="Cartelera verificada">Verified slate</span><span>WATCH / market readiness</span><span data-en="Public intelligence" data-es="Inteligencia pública">Public intelligence</span></div></div><a class="btn" href="/app" data-track="explorer_cta" data-en="Open Explorer" data-es="Abrir Explorer">Open Explorer</a></article>
<article class="card plan pro"><div><div class="eyebrow">PRO</div><h3>Edge Pro</h3><div class="feature-list"><span>BETS + LEANS</span><span>Match Intelligence</span><span data-en="Verified performance" data-es="Rendimiento verificado">Verified performance</span><span>My Edge</span></div><div class="trust-note" data-en="Public paid launch is gated. No subscription price is presented here until the commercial launch flag and pricing strategy are explicitly approved." data-es="El lanzamiento público de pago está bloqueado por un gate. No mostramos precio de suscripción aquí hasta aprobar explícitamente el lanzamiento comercial y la estrategia de precios.">Public paid launch is gated. No subscription price is presented here until the commercial launch flag and pricing strategy are explicitly approved.</div></div><a class="btn primary" href="/app" data-track="explorer_cta" data-en="View Edge Pro inside the app" data-es="Ver Edge Pro dentro de la app">View Edge Pro inside the app</a></article>
</div></section>
<footer class="footer" data-en="Soccer Edge provides analytical information, not guaranteed outcomes. Betting involves risk. Use only where legal and only if you meet the legal age requirements in your jurisdiction." data-es="Soccer Edge ofrece información analítica, no resultados garantizados. Apostar implica riesgo. Úsalo solo donde sea legal y si cumples la edad legal requerida en tu jurisdicción.">Soccer Edge provides analytical information, not guaranteed outcomes. Betting involves risk. Use only where legal and only if you meet the legal age requirements in your jurisdiction.</footer>
</div>
<script id="landing-v2-config" type="application/json">{config}</script>
<script>
(()=>{{
 const cfg=JSON.parse(document.getElementById('landing-v2-config')?.textContent||'{{}}'),sel=document.getElementById('lang'),key='soccer_edge_locale';
 let lang=localStorage.getItem(key)||((navigator.language||'en').toLowerCase().startsWith('es')?'es':'en');if(!['en','es'].includes(lang))lang='en';
 const apply=()=>{{document.documentElement.lang=lang;sel.value=lang;document.querySelectorAll('[data-en][data-es]').forEach(el=>el.textContent=el.dataset[lang]||el.dataset.en)}};
 sel.onchange=()=>{{lang=sel.value;localStorage.setItem(key,lang);apply();track('language_change',{{language:lang}})}};apply();
 const uuid=()=>crypto.randomUUID?crypto.randomUUID():('xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{{const r=Math.random()*16|0,v=c==='x'?r:(r&3|8);return v.toString(16)}}));
 let anon=localStorage.getItem('soccer_edge_anon_id');if(!anon){{anon=uuid();localStorage.setItem('soccer_edge_anon_id',anon)}};let session=sessionStorage.getItem('soccer_edge_session_id');if(!session){{session=uuid();sessionStorage.setItem('soccer_edge_session_id',session)}}
 async function track(event_name,properties={{}}){{if(!cfg.analytics_enabled||!cfg.supabase_url||!cfg.publishable_key)return;try{{const q=new URLSearchParams(location.search),token=localStorage.getItem('soccer_edge_access_token')||'',h={{apikey:cfg.publishable_key,'Content-Type':'application/json'}};if(token)h.Authorization='Bearer '+token;await fetch(cfg.supabase_url+'/functions/v1/track-product-event',{{method:'POST',headers:h,body:JSON.stringify({{event_id:uuid(),event_name,anonymous_id:anon,session_id:session,locale:lang,path:location.pathname,utm_source:q.get('utm_source')||'',utm_medium:q.get('utm_medium')||'',utm_campaign:q.get('utm_campaign')||'',utm_content:q.get('utm_content')||'',properties}})}})}}catch(_){{}}}}
 track('landing_view',{{surface:'landing_v2'}});document.querySelectorAll('[data-track]').forEach(el=>el.addEventListener('click',()=>track(el.dataset.track,{{destination:el.getAttribute('href')||''}})));
}})();
</script>
</body></html>'''


async def landing_page(request: Request) -> HTMLResponse:
    return HTMLResponse(render(), headers={"Cache-Control": "no-store"})


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "route": "/",
        "preview_alias": "/landing-v2",
        "app_destination": "/app",
        "mock_probabilities": False,
        "fixed_accuracy_claims": False,
        "public_subscription_price_claims": False,
        "billing_launch_gate_respected": True,
        "primary_message": "SPORT_FIRST_MARKET_SECOND",
        "canonical_decision_states": ["BET", "LEAN", "WATCH", "PASS"],
        "responsible_risk_copy_present": True,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
