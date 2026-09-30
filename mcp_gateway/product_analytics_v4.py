from __future__ import annotations

import json

from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PRODUCT_ANALYTICS_V4_1.0.0"


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def inject_analytics(base_html: str, *, surface: str) -> str:
    """Inject product-funnel instrumentation only; never changes betting data or decisions."""
    html = str(base_html)
    marker = f"SOCCER_PRODUCT_ANALYTICS_V4_{surface.upper()}"
    if marker in html:
        return html

    auth = supabase_auth_v4.public_auth_config()
    project_url = str(auth.get("project_url") or "").rstrip("/")
    config = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "surface": surface,
        "endpoint": f"{project_url}/functions/v1/track-product-event" if project_url else None,
        "publishable_key": auth.get("publishable_key"),
    }
    cfg = _safe_json(config)

    script = r'''
<script id="__MARKER__" type="application/json">__CONFIG__</script>
<script>
(() => {
  const cfg = JSON.parse(document.getElementById('__MARKER__').textContent);
  if (!cfg.endpoint) return;
  const anonKey = 'soccer_edge_anon_id';
  const sessionKey = 'soccer_edge_session_id';
  const attrKey = 'soccer_edge_attribution';
  const localeKey = 'soccer_edge_locale';
  const tokenKeyAnalytics = 'soccer_edge_access_token';
  const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}-4${Math.random().toString(16).slice(2,5)}-8${Math.random().toString(16).slice(2,5)}-${Math.random().toString(16).slice(2,14)}`);
  const ensure = (storage, key) => { let value = storage.getItem(key); if (!value) { value = uuid(); storage.setItem(key, value); } return value; };
  const anonId = ensure(localStorage, anonKey);
  const sessionId = ensure(sessionStorage, sessionKey);
  const params = new URLSearchParams(location.search);
  const current = {
    cohort: (params.get('cohort') || '').slice(0,64),
    utm_source: (params.get('utm_source') || '').slice(0,96),
    utm_medium: (params.get('utm_medium') || '').slice(0,96),
    utm_campaign: (params.get('utm_campaign') || '').slice(0,128),
    utm_content: (params.get('utm_content') || '').slice(0,128),
  };
  let attribution = {};
  try { attribution = JSON.parse(localStorage.getItem(attrKey) || '{}') || {}; } catch (_) {}
  if (Object.values(current).some(Boolean)) {
    attribution = {...attribution, ...Object.fromEntries(Object.entries(current).filter(([,v]) => v))};
    localStorage.setItem(attrKey, JSON.stringify(attribution));
  }
  const locale = () => {
    const saved = localStorage.getItem(localeKey);
    if (saved === 'en' || saved === 'es') return saved;
    return String(navigator.language || 'en').toLowerCase().startsWith('es') ? 'es' : 'en';
  };
  const token = () => sessionStorage.getItem(tokenKeyAnalytics) || '';
  const cleanProps = (props={}) => Object.fromEntries(Object.entries(props).slice(0,20).filter(([,v]) => ['string','number','boolean'].includes(typeof v) || v === null));

  async function track(eventName, properties={}) {
    const body = {
      event_id: uuid(), event_name: eventName, anonymous_id: anonId, session_id: sessionId,
      locale: locale(), cohort: attribution.cohort || null,
      utm_source: attribution.utm_source || null, utm_medium: attribution.utm_medium || null,
      utm_campaign: attribution.utm_campaign || null, utm_content: attribution.utm_content || null,
      path: location.pathname, properties: cleanProps(properties),
    };
    const headers = {'Content-Type':'application/json'};
    if (token()) headers.Authorization = `Bearer ${token()}`;
    try { await fetch(cfg.endpoint, {method:'POST', headers, body:JSON.stringify(body), keepalive:true}); } catch (_) {}
  }
  window.soccerEdgeAnalytics = {track, attribution: () => ({...attribution}), anonymousId: () => anonId, sessionId: () => sessionId};

  if (cfg.surface === 'landing') {
    track('landing_view');
    document.querySelectorAll('a[href="/app"]').forEach(el => el.addEventListener('click', () => track('explorer_cta', {placement: el.classList.contains('primary') ? 'primary' : 'secondary'})));
    ['enBtn','esBtn'].forEach(id => document.getElementById(id)?.addEventListener('click', () => setTimeout(() => track('language_change', {to: locale()}), 0)));
  }

  if (cfg.surface === 'app') {
    track('app_view');
    document.getElementById('signupBtn')?.addEventListener('click', () => track('signup_click'));
    document.getElementById('signinBtn')?.addEventListener('click', () => track('signin_click'));
    document.getElementById('upgradeBtn')?.addEventListener('click', () => track('pro_checkout_click'));
    document.getElementById('portalBtn')?.addEventListener('click', () => track('customer_portal_click'));
    ['langEn','langEs'].forEach(id => document.getElementById(id)?.addEventListener('click', () => setTimeout(() => track('language_change', {to: locale()}), 0)));
    if (new URLSearchParams(location.search).get('checkout') === 'cancel') track('checkout_canceled');

    if (typeof render === 'function') {
      const analyticsBaseRender = render;
      let lastAuthFingerprint = '';
      render = (data) => {
        analyticsBaseRender(data);
        if (data?.authenticated) {
          const fp = `${data?.user?.id || data?.user?.email || 'user'}:${data?.effective_plan || 'FREE'}`;
          if (fp !== lastAuthFingerprint) {
            lastAuthFingerprint = fp;
            track('authenticated_view', {plan: String(data?.effective_plan || 'FREE')});
          }
        }
      };
    }

    if (typeof edge === 'function') {
      edge = async (endpoint) => {
        const headers = {apikey: cfg.publishable_key || '', Authorization:`Bearer ${token()}`, 'Content-Type':'application/json'};
        const payload = {analytics:{anonymous_id:anonId,session_id:sessionId,locale:locale(),...attribution}};
        const r = await fetch(endpoint,{method:'POST',headers,body:JSON.stringify(payload)});
        const d = await r.json();
        if (!r.ok) throw new Error(d.error==='BILLING_NOT_CONFIGURED'?'Billing is staged but Stripe credentials are not connected yet.':d.error||d.detail||'Billing request failed');
        if (d.url) location.href=d.url;
      };
    }
  }
})();
</script>
'''.replaceAll("__MARKER__", marker).replace("__CONFIG__", cfg)
    return html.replace("</body>", script + "</body>", 1)
