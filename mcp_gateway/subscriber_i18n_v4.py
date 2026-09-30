from __future__ import annotations

import json

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_I18N_V4_1.0.0"
SUPPORTED_LOCALES = ("en", "es")

_TRANSLATIONS = {
    "en": {
        "title": "Soccer Edge — Model vs Market",
        "description": "Transparent soccer market intelligence: market prices, model probabilities, verified performance and closing-line evidence.",
        "subscriber_app": "SUBSCRIBER APP · V222",
        "hero_title": "Signals without the noise.",
        "hero_copy": "Verified slate, market readiness and model maturity are available on Explorer. Edge Pro unlocks strong signals, calibrated value, advanced markets, premium match detail and verified performance.",
        "upgrade": "Upgrade to Edge Pro",
        "portal": "Manage subscription",
        "logout": "Sign out",
        "account": "ACCOUNT",
        "account_title": "Sign in or create an account",
        "email": "Email",
        "password": "Password",
        "signin": "Sign in",
        "signup": "Create account",
        "today": "TODAY",
        "verified_slate": "Verified slate",
        "radar": "RADAR",
        "market_readiness": "Market readiness",
        "maturity": "MATURITY",
        "model_maturity": "Model maturity",
        "edge_pro": "EDGE PRO",
        "premium_intelligence": "Premium intelligence",
        "waiting_price": "Waiting for price",
        "waiting_xi": "Waiting for XI",
        "maturity_unavailable": "Maturity snapshot unavailable.",
        "pro_unlocked": "Unlocked for Edge Pro",
        "pro_active_empty": "Pro is active. No premium rows are available right now.",
        "pro_locked": "Edge Pro is locked. Strong signals: {strong} · Value plays: {value}. Sign in and activate Pro to unlock row-level details.",
        "no_rows": "No verified rows right now.",
        "session_expired": "Session expired. Sign in again.",
        "signed_in_as": "Signed in as {email} · {plan}",
        "credentials_required": "Email and password are required",
        "auth_failed": "Authentication failed",
        "signed_in": "Signed in.",
        "account_created": "Account created. Check your email if confirmation is required.",
        "signed_out": "Signed out.",
        "sign_in_first": "Sign in first.",
        "billing_not_configured": "Billing is staged but Stripe credentials are not connected yet.",
        "billing_failed": "Billing request failed",
        "checkout_success": "Checkout completed. Your Pro entitlement will appear after the signed webhook is processed.",
        "checkout_cancel": "Checkout canceled. No plan change was made.",
        "language": "Language",
    },
    "es": {
        "title": "Soccer Edge — Modelo vs Mercado",
        "description": "Inteligencia transparente del mercado de fútbol: precios, probabilidades del modelo, rendimiento verificable y evidencia contra la línea de cierre.",
        "subscriber_app": "APP DE SUSCRIPTOR · V222",
        "hero_title": "Señales sin ruido.",
        "hero_copy": "Explorer muestra la cartelera verificada, el estado del mercado y la maduración del modelo. Edge Pro desbloquea señales fuertes, valor calibrado, mercados avanzados, detalle premium y rendimiento verificable.",
        "upgrade": "Mejorar a Edge Pro",
        "portal": "Administrar suscripción",
        "logout": "Cerrar sesión",
        "account": "CUENTA",
        "account_title": "Inicia sesión o crea una cuenta",
        "email": "Correo electrónico",
        "password": "Contraseña",
        "signin": "Iniciar sesión",
        "signup": "Crear cuenta",
        "today": "HOY",
        "verified_slate": "Cartelera verificada",
        "radar": "RADAR",
        "market_readiness": "Estado del mercado",
        "maturity": "MADURACIÓN",
        "model_maturity": "Maduración del modelo",
        "edge_pro": "EDGE PRO",
        "premium_intelligence": "Inteligencia premium",
        "waiting_price": "Esperando precio",
        "waiting_xi": "Esperando XI",
        "maturity_unavailable": "La maduración no está disponible en este momento.",
        "pro_unlocked": "Desbloqueado con Edge Pro",
        "pro_active_empty": "Pro está activo. No hay filas premium disponibles en este momento.",
        "pro_locked": "Edge Pro está bloqueado. Señales fuertes: {strong} · Jugadas de valor: {value}. Inicia sesión y activa Pro para desbloquear el detalle por fila.",
        "no_rows": "No hay filas verificadas en este momento.",
        "session_expired": "La sesión expiró. Inicia sesión de nuevo.",
        "signed_in_as": "Sesión iniciada como {email} · {plan}",
        "credentials_required": "El correo y la contraseña son obligatorios",
        "auth_failed": "No se pudo iniciar sesión",
        "signed_in": "Sesión iniciada.",
        "account_created": "Cuenta creada. Revisa tu correo si se requiere confirmación.",
        "signed_out": "Sesión cerrada.",
        "sign_in_first": "Inicia sesión primero.",
        "billing_not_configured": "La facturación está preparada, pero las credenciales de Stripe todavía no están conectadas.",
        "billing_failed": "Falló la solicitud de facturación",
        "checkout_success": "Pago completado. Tu acceso Pro aparecerá después de procesar el webhook firmado.",
        "checkout_cancel": "Pago cancelado. No se realizó ningún cambio de plan.",
        "language": "Idioma",
    },
}

_STATUS_LABELS = {
    "en": {
        "READY": "Ready",
        "MATURING": "Maturing",
        "RESEARCH_HOLD": "Research hold",
        "WAIT_PRICE": "Price watch",
        "WAIT_XI": "Lineup watch",
        "NO_BET": "No bet",
        "ACTIVE": "Active",
        "TRIALING": "Trialing",
        "PAST_DUE": "Past due",
        "CANCELED": "Canceled",
    },
    "es": {
        "READY": "Listo",
        "MATURING": "Madurando",
        "RESEARCH_HOLD": "En investigación",
        "WAIT_PRICE": "Esperando precio",
        "WAIT_XI": "Esperando alineación",
        "NO_BET": "Sin apuesta",
        "ACTIVE": "Activo",
        "TRIALING": "En prueba",
        "PAST_DUE": "Pago vencido",
        "CANCELED": "Cancelado",
    },
}

_MARKET_LABELS = {
    "en": {
        "1X2": "1X2",
        "BTTS": "BTTS",
        "FT_TOTALS": "Full-time totals",
        "TEAM_TOTALS": "Team totals",
        "1H": "First half",
        "2H": "Second half",
        "CORNERS": "Corners",
        "CARDS": "Cards",
        "TEAM_CARDS": "Team cards",
        "PLAYER_PROPS": "Player props",
    },
    "es": {
        "1X2": "1X2",
        "BTTS": "Ambos anotan",
        "FT_TOTALS": "Totales del partido",
        "TEAM_TOTALS": "Totales de equipo",
        "1H": "Primera mitad",
        "2H": "Segunda mitad",
        "CORNERS": "Córners",
        "CARDS": "Tarjetas",
        "TEAM_CARDS": "Tarjetas de equipo",
        "PLAYER_PROPS": "Props de jugadores",
    },
}


def public_catalog() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "supported_locales": list(SUPPORTED_LOCALES),
        "translations": _TRANSLATIONS,
        "status_labels": _STATUS_LABELS,
        "market_labels": _MARKET_LABELS,
    }


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def inject_i18n(base_html: str) -> str:
    """Inject a client-only locale layer; canonical product payloads remain untouched."""
    html = str(base_html)
    if "SOCCER_SUBSCRIBER_I18N_V4" in html:
        return html

    html = html.replace(
        "<title>Soccer Edge</title>",
        '<title>Soccer Edge</title><meta id="metaDescription" name="description" content=""><meta id="ogTitle" property="og:title" content="Soccer Edge"><meta id="ogDescription" property="og:description" content="">',
        1,
    )
    html = html.replace(
        '<div class="badge" id="planBadge">FREE</div>',
        '<div class="top-actions"><div class="locale-switch" role="group" aria-label="Language"><button type="button" id="langEn" class="lang-btn">EN</button><button type="button" id="langEs" class="lang-btn">ES</button></div><div class="badge" id="planBadge">FREE</div></div>',
        1,
    )
    html = html.replace(
        "</style>",
        ".top-actions{display:flex;align-items:center;gap:10px}.locale-switch{display:flex;border:1px solid var(--line);border-radius:999px;padding:2px;background:#07131d}.lang-btn{border:0;background:transparent;color:var(--muted);font-size:10px;font-weight:900;padding:5px 8px;border-radius:999px;cursor:pointer}.lang-btn.active{background:#12344a;color:var(--text)}@media(max-width:520px){.top-actions{align-items:flex-end;flex-direction:column-reverse}}\n</style>",
        1,
    )

    # Give the existing sections stable presentation-only hooks. This does not
    # alter any canonical data identifier or API payload.
    replacements = {
        '<section class="hero">': '<section class="hero" id="heroCard">',
        '<section class="card full"><div class="eyebrow">ACCOUNT</div>': '<section class="card full" id="accountCard"><div class="eyebrow">ACCOUNT</div>',
        '<section class="card"><div class="eyebrow">TODAY</div>': '<section class="card" id="todayCard"><div class="eyebrow">TODAY</div>',
        '<section class="card"><div class="eyebrow">RADAR</div>': '<section class="card" id="radarCard"><div class="eyebrow">RADAR</div>',
        '<section class="card full"><div class="eyebrow">MATURITY</div>': '<section class="card full" id="maturityCard"><div class="eyebrow">MATURITY</div>',
        '<section class="card full"><div class="eyebrow">EDGE PRO</div>': '<section class="card full" id="proCard"><div class="eyebrow">EDGE PRO</div>',
    }
    for before, after in replacements.items():
        html = html.replace(before, after, 1)

    catalog = _safe_json(public_catalog())
    script = r'''
<script id="soccerEdgeI18nCatalog" type="application/json">__CATALOG__</script>
<script id="SOCCER_SUBSCRIBER_I18N_V4">
(() => {
  const catalog = JSON.parse(document.getElementById('soccerEdgeI18nCatalog').textContent);
  const localeKey = 'soccer_edge_locale';
  const detect = () => {
    const saved = localStorage.getItem(localeKey);
    if (catalog.supported_locales.includes(saved)) return saved;
    return String(navigator.language || 'en').toLowerCase().startsWith('es') ? 'es' : 'en';
  };
  let locale = detect();
  const tr = () => catalog.translations[locale] || catalog.translations.en;
  const byId = (id) => document.getElementById(id);
  const q = (selector) => document.querySelector(selector);
  const setText = (selector, value) => { const el = q(selector); if (el) el.textContent = value; };
  const format = (template, values) => Object.entries(values || {}).reduce((s,[k,v]) => s.replaceAll(`{${k}}`, String(v)), template);
  const marketLabel = (value) => (catalog.market_labels[locale] || {})[String(value || '').toUpperCase()] || value;
  const statusLabel = (value) => (catalog.status_labels[locale] || {})[String(value || '').toUpperCase()] || value;

  function translateDynamic() {
    const d = tr();
    const radarMeta = document.querySelectorAll('#radar .meta');
    if (radarMeta[0]) radarMeta[0].textContent = d.waiting_price;
    if (radarMeta[1]) radarMeta[1].textContent = d.waiting_xi;

    document.querySelectorAll('.badge').forEach(el => {
      const mapped = statusLabel(el.textContent.trim());
      if (mapped) el.textContent = mapped;
    });
    document.querySelectorAll('.mat span').forEach(el => {
      const parts = el.textContent.split(' · ');
      if (parts.length) {
        parts[0] = statusLabel(parts[0]);
        el.textContent = parts.join(' · ');
      }
    });
    document.querySelectorAll('.row .meta').forEach(el => {
      const parts = el.textContent.split(' · ');
      if (parts.length && catalog.market_labels[locale][parts[0]]) {
        parts[0] = marketLabel(parts[0]);
        el.textContent = parts.join(' · ');
      }
      if (el.textContent === 'Unlocked for Edge Pro' || el.textContent === 'Desbloqueado con Edge Pro') {
        el.textContent = d.pro_unlocked;
      }
    });

    const maturityLock = q('#maturity .lock');
    if (maturityLock && /Maturity snapshot unavailable|maduración no está disponible/i.test(maturityLock.textContent)) {
      maturityLock.textContent = d.maturity_unavailable;
    }
    const proLock = q('#pro .lock');
    if (proLock) {
      const text = proLock.textContent;
      if (/Pro is active|Pro está activo/i.test(text)) {
        proLock.textContent = d.pro_active_empty;
      } else {
        const strong = (text.match(/Strong signals:\s*(\d+)/i) || text.match(/Señales fuertes:\s*(\d+)/i) || [])[1] || '0';
        const value = (text.match(/Value plays:\s*(\d+)/i) || text.match(/Jugadas de valor:\s*(\d+)/i) || [])[1] || '0';
        proLock.textContent = format(d.pro_locked, {strong, value});
      }
    }
    const notice = byId('checkoutNotice');
    if (notice && notice.style.display === 'block') {
      const checkout = new URLSearchParams(location.search).get('checkout');
      if (checkout) notice.textContent = checkout === 'success' ? d.checkout_success : d.checkout_cancel;
    }
  }

  function applyLocale() {
    const d = tr();
    document.documentElement.lang = locale;
    document.title = d.title;
    const md = byId('metaDescription'); if (md) md.setAttribute('content', d.description);
    const ot = byId('ogTitle'); if (ot) ot.setAttribute('content', d.title);
    const od = byId('ogDescription'); if (od) od.setAttribute('content', d.description);

    setText('#heroCard .eyebrow', d.subscriber_app);
    setText('#heroCard h1', d.hero_title);
    setText('#heroCard p', d.hero_copy);
    setText('#upgradeBtn', d.upgrade);
    setText('#portalBtn', d.portal);
    setText('#logoutBtn', d.logout);
    setText('#accountCard .eyebrow', d.account);
    setText('#accountCard h2', d.account_title);
    setText('#signinBtn', d.signin);
    setText('#signupBtn', d.signup);
    setText('#todayCard .eyebrow', d.today);
    setText('#todayCard h2', d.verified_slate);
    setText('#radarCard .eyebrow', d.radar);
    setText('#radarCard h2', d.market_readiness);
    setText('#maturityCard .eyebrow', d.maturity);
    setText('#maturityCard h2', d.model_maturity);
    setText('#proCard .eyebrow', d.edge_pro);
    setText('#proCard h2', d.premium_intelligence);
    const email = byId('email'); if (email) email.placeholder = d.email;
    const password = byId('password'); if (password) password.placeholder = d.password;
    byId('langEn')?.classList.toggle('active', locale === 'en');
    byId('langEs')?.classList.toggle('active', locale === 'es');
    translateDynamic();
  }

  function translateMessage(text) {
    const d = tr();
    const raw = String(text || '');
    const exact = {
      'Session expired. Sign in again.': d.session_expired,
      'Email and password are required': d.credentials_required,
      'Authentication failed': d.auth_failed,
      'Signed in.': d.signed_in,
      'Account created. Check your email if confirmation is required.': d.account_created,
      'Signed out.': d.signed_out,
      'Sign in first.': d.sign_in_first,
      'Billing is staged but Stripe credentials are not connected yet.': d.billing_not_configured,
      'Billing request failed': d.billing_failed,
    };
    if (exact[raw]) return exact[raw];
    const signed = raw.match(/^Signed in as (.+) · (.+)$/);
    if (signed) return format(d.signed_in_as, {email: signed[1], plan: signed[2]});
    return raw;
  }

  if (typeof setStatus === 'function') {
    const baseSetStatus = setStatus;
    setStatus = (text) => baseSetStatus(translateMessage(text));
  }
  if (typeof render === 'function') {
    const baseRender = render;
    render = (data) => { baseRender(data); translateDynamic(); };
  }

  byId('langEn')?.addEventListener('click', () => { locale = 'en'; localStorage.setItem(localeKey, locale); applyLocale(); });
  byId('langEs')?.addEventListener('click', () => { locale = 'es'; localStorage.setItem(localeKey, locale); applyLocale(); });
  applyLocale();
})();
</script>
'''.replace("__CATALOG__", catalog)
    return html.replace("</body>", script + "</body>", 1)
