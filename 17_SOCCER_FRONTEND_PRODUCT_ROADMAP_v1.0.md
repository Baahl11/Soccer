# 17 — SOCCER FRONTEND PRODUCT ROADMAP v1.0

**Date:** 2026-10-07  
**Status:** GOVERNING FRONTEND ROADMAP  
**Parent:** SPORTS EDGE ENGINE / Soccer  
**Audit basis:** `16_SOCCER_FRONTEND_E2E_AUDIT_v1.0.md`

## Objective

Build a premium, trustworthy and monetizable Soccer Edge product where the first questions answered are:

```text
WHAT TO BET
WHAT TO LEAN
WHAT TO WATCH
WHY
AT WHAT EXACT PRICE
WITH WHAT DATA CONFIDENCE
```

The frontend must preserve **SPORT FIRST. MARKET SECOND.** It may present canonical engine output but may never invent picks, probabilities, odds, availability or supporting facts.

## Current execution status

```text
FE-0  Frontend stabilization            COMPLETE / MERGED
FE-1  Canonical Subscriber API v2       COMPLETE / MERGED
FE-2  Premium shell preview             COMPLETE / MERGED / LIVE ON /app-v2
FE-3  Picks / Leans / Today             NEXT ACTIVE BLOCK
FE-4  Match Intelligence                PENDING
FE-5  Performance & Trust               PENDING
FE-6  My Edge                           PENDING
FE-7  Auth + Subscriptions polish       PENDING
FE-8  Landing / Conversion              PENDING
FE-9  PWA / Notifications               PENDING
```

FE-0 removed the legacy multi-screen startup behavior by gating first paint, removing mock bootstrap data from the production shell, centralizing readiness events and eliminating the timer/MutationObserver startup stack from the active customer layers.

FE-1 introduced a read-only versioned customer contract under `/app/api/v2/*`. It keeps Raw Sport, Market-Shrunk, calibrated model, fair market and breakeven probabilities distinct. Only explicit persisted BET rows enter Picks and only explicit persisted LEAN rows enter Leans. Missing critical availability remains NOT VERIFIED.

FE-2 introduced a clean customer-first shell on `/app-v2` with one bootstrap controller and no design-time mock picks. The primary navigation is Today, Picks, Leans, Matches, Performance, My Edge and Account. Control Tower and Research Lab are intentionally excluded from the commercial primary navigation.

## Product hierarchy

Customer priority:

```text
1. BETS
2. LEANS
3. WATCHES
4. MATCH INTELLIGENCE
5. VERIFIED PERFORMANCE
6. FULL SLATE
7. ACCOUNT / SUBSCRIPTION
```

Advanced/secondary: Markets, Model Evidence and Maturity/Research.

Owner/Admin only: Control Tower, Scheduler/Provider Health, Research Lab, lifecycle/model gates and raw diagnostics.

## FE-3 — Today / Picks / Leans

**Priority:** P0  
**Status:** NEXT

Today must surface canonical BETS first, LEANS second, WATCH states third and the complete slate after them.

A BET card must show teams and crests, league and kickoff, BET + Tier when applicable, exact market, exact selection, exact line, exact price, bookmaker, market timestamp, Raw Sport Probability, Market-Shrunk Probability, Fair Market Probability, Probability Edge, EV when defensible, Availability Confidence, Data Tier, XI/GK status, core sporting reason, market-value reason, invalidation condition and model version.

A LEAN must use the same evidence contract but clearly state why it has not become a BET.

A WATCH must state the exact missing condition such as price, XI, goalkeeper, availability or another required verification.

## FE-4 — Match Intelligence

**Priority:** P1

Target tabs:

```text
Overview
Sport
Goals
Corners
Cards
Players
Availability
Market
Model
```

The Overview must visibly preserve:

```text
RAW SPORT PROJECTION
↓
MARKET-SHRUNK PROJECTION
↓
CURRENT EXACT MARKET
↓
VALUE / CLASSIFICATION
```

Unknown injuries, XI, goalkeeper, starter or weather facts must display NOT VERIFIED.

Only validated features may be shown as decision evidence.

## FE-5 — Performance & Trust

**Priority:** P1

Customer-facing verified performance must include graded BETS, W/L/P, ROI, average odds, True CLV where available, calibration/Brier where meaningful, sample size, date range and performance by market/tier/model version.

Always distinguish:

```text
BET-only realized performance
vs
research/OOS validation
```

No cherry-picked wins, hidden losses or research rows presented as realized bets.

## FE-6 — My Edge

**Priority:** P2

Authenticated personalization should support saved matches, saved BETS/LEANS, favorite leagues/teams, tracked markets, price alerts, XI/lineup alerts and kickoff reminders.

Authenticated state should be server persisted. Browser-local storage remains only an anonymous fallback.

## FE-7 — Authentication and Subscriptions

**Priority:** P2

Existing foundation includes Supabase Auth, FREE/PRO entitlements, Stripe Checkout, Customer Portal, Stripe webhook and RLS-protected subscription/billing tables.

Before paid launch:

- reconcile the current `billing_enabled` contract;
- explicitly approve pricing;
- test signup → checkout → webhook → PRO;
- test cancel / renewal / past_due;
- polish success/cancel return UX;
- verify entitlement propagation;
- enable leaked-password protection;
- add responsible-risk/legal copy;
- define jurisdiction policy.

Legacy pricing is not automatically governing.

## FE-8 — Landing / Conversion

**Priority:** P2

Positioning should center on:

```text
Model vs Market
Verified Picks
Transparent Leans
Exact Prices
Evidence Behind Every Decision
No Bet Is a Valid Answer
```

Do not market unverified fixed accuracy percentages.

## FE-9 — PWA / Notifications

**Priority:** P3

After stable web + subscriptions: PWA install, push notifications, lineup alerts, price-threshold alerts, BET/LEAN status changes and saved-match alerts.

Native mobile is not required for first commercial launch.

## Frontend CI gate

Frontend deployment is blocked unless active tests cover single boot/render ownership, no mock-content leakage, no duplicate critical IDs, exactly one active page, anonymous/FREE/PRO/OWNER states, premium redaction, mobile widths, missing-data states, auth/subscription regression, no client-created probabilities/prices/classifications and operator/customer route isolation.

## Parallel model and product lanes

```text
MODEL LANE
FM4 evidence → FM5 → FM6 → FM7

PRODUCT LANE
FE0 → FE1 → FE2 → FE3 → FE4 → FE5 → FE7 → launch readiness
```

Frontend work does not wait for FM5/FM6/FM7 evidence completion. When a model family is not production eligible, the frontend must show WATCH / RESEARCH / NOT VERIFIED instead of pretending it is ready.

## Governing firewall

No frontend work may change model weights, sport thresholds, market gates, historical predictions, provider budgets or production promotion rules. It may not create a BET or LEAN not emitted by canonical engine logic, infer missing injuries/XI/weather, synthesize odds/exact prices or convert research evidence into production evidence.

The frontend is a **truthful decision surface**, not a second model.


---

# 17. IMPLEMENTATION STATUS — FE0 TO FE9 — 2026-10-07

This section supersedes earlier implementation-status labels in this roadmap where they conflict.

## Current product branch

```text
soccer-edge-mcp-v1
current frontend release candidate through FE-9
```

## Phase status

```text
FE-0  Frontend stabilization                  IMPLEMENTED / DEPLOYED
FE-1  Canonical Subscriber API v2             IMPLEMENTED / DEPLOYED
FE-2  Premium visual shell                    IMPLEMENTED / DEPLOYED PREVIEW
FE-3  Today / Picks / Leans                   IMPLEMENTED / DEPLOYED PREVIEW
FE-4  Match Intelligence                      IMPLEMENTED / DEPLOYED PREVIEW
FE-5  Performance & Trust                     IMPLEMENTED / DEPLOYED PREVIEW
FE-6  My Edge                                 IMPLEMENTED / DEPLOYED PREVIEW
FE-7  Auth & Subscription lifecycle           IMPLEMENTED / GUARDED
FE-8  Landing / conversion                    IMPLEMENTED / DEPLOYED PREVIEW
FE-9  PWA installability                      IMPLEMENTED
FE-9  Push delivery                           BLOCKED / NOT ENABLED
```

## FE-0 stabilization result

The legacy visible boot-stack problem was addressed by:

- one boot gate;
- no design-time mock pick bootstrap on production first paint;
- one coordinated readiness lifecycle;
- removing duplicate startup fetches/timer loops from the active customer path;
- removing MutationObserver-driven application bootstrap from the active Today/visual hydration layers.

The legacy V230-V237 implementation remains in the repository for continuity, but the commercial target is now the V2 customer surface.

## FE-1 canonical customer contract

Current V2 resources:

```text
/app/api/v2/today
/app/api/v2/picks
/app/api/v2/leans
/app/api/v2/watches
/app/api/v2/match/{fixture_id}
/app/api/v2/performance
/app/api/v2/my-edge
/app/api/v2/account
```

Frontend policy:

```text
explicit persisted BET only → Picks
explicit persisted LEAN only → Leans
canonical wait state / WATCH → Watches

READY ≠ BET
STRONG ≠ BET
```

Raw Sport, Market-Shrunk, calibrated model, market fair probability,
breakeven probability, Probability Edge and EV remain distinct fields.

## FE-2 to FE-4 customer product

Current preview:

```text
/app-v2
```

Primary navigation:

```text
Today
Picks
Leans
Matches
Performance
My Edge
Account
```

Operator Control Tower and Research Lab are excluded from the primary
subscriber navigation.

Match Intelligence includes:

```text
Overview
Sport
Goals
Corners
Cards
Players
Availability
Market
Model
```

Missing persisted evidence remains NOT VERIFIED.

## FE-5 performance trust

Customer headline performance is:

```text
CANONICAL_BET_SETTLEMENT_ONLY
```

LEAN research and OOS/validation evidence are displayed separately and
cannot improve the BET headline.

Small-sample warnings remain visible.

No frontend process performs historical prediction reconstruction or
settlement backfill.

## FE-6 My Edge

Supabase table:

```text
public.subscriber_saved_items
```

is live with RLS enabled.

Users may save:

```text
MATCH
BET
LEAN
WATCH
```

Saved product state is explicitly forbidden as a betting-model input.

## FE-7 subscriptions

Current architecture has been validated against the connected live Stripe
account:

```text
product = Soccer Edge Pro

US
US$14.99 / month
active recurring monthly price

MX_LATAM
MX$249 / month
active recurring monthly price
```

These are verified **current implementation values**, not an approved final
public pricing strategy.

Architecture:

```text
FREE freemium base
→ explicit upgrade
→ Stripe-hosted Checkout
→ Stripe webhook
→ Supabase subscription_entitlements
→ PRO access

Customer self-management
→ Stripe Customer Portal
```

The browser sends only:

```text
billing_market = US | MX_LATAM
```

and never sends a Stripe Price ID.

Public checkout remains gated by:

```text
SOCCER_PUBLIC_BILLING_ENABLED
```

Default:

```text
false
```

Therefore code-ready does not mean paid launch is enabled.

## FE-8 landing

Current conversion preview:

```text
/landing-v2
```

It intentionally contains:

- no fabricated demo probabilities;
- no fixed accuracy claim;
- no public claim of unapproved subscription pricing;
- BET / LEAN / WATCH / PASS education;
- Sport First / Market Second explanation;
- responsible-risk copy;
- product-only analytics events.

## FE-9 PWA

Current PWA routes:

```text
/app.webmanifest
/sw.js
/pwa/icon.svg
```

Critical data policy:

```text
/app/api/v2/* = NETWORK ONLY
exact price = NEVER service-worker cached
decision payload = NEVER service-worker cached
```

An offline shell may load, but it explicitly refuses to show stale picks or
prices as current.

Push code remains dormant.

```text
push_subscription_enabled = false
outbound_push_delivery_enabled = false
```

Push cannot advance until:

1. explicit user consent UX exists;
2. push subscription persistence is implemented;
3. VAPID/server delivery infrastructure is configured;
4. notification types are product-approved;
5. notification data is prevented from entering model features.

## Current release-candidate boundary

The next release decision is **not another model or frontend feature patch**.

The next required step is visual/functional release-candidate QA of:

```text
/landing-v2
/app-v2
```

across desktop and mobile before replacing the current public root and
`/app` routes.

The route swap must not happen silently.

## Remaining commercial launch gates

Before public paid launch:

- approve final pricing strategy;
- explicitly enable public billing;
- confirm Stripe Checkout return URLs for the final production route;
- confirm Customer Portal return route;
- enable/check leaked-password protection in Supabase Auth;
- test real signup → checkout → webhook → PRO entitlement;
- test cancellation / renewal / past_due;
- perform responsive visual QA;
- approve legal/responsible-gambling copy and jurisdiction policy.



---

## FE-10 — Alertas de candidatos y notificaciones móviles (plan añadido 2026-11-06)

**Prioridad:** P1 para diseño de datos/eventos y alertas internas; P3 para entrega push/PWA; P4 para app nativa, solo tras validación de web y permisos.  
**Estado:** ESPECIFICADO / NO DESPLEGADO. No afirmar que el navegador o el teléfono ya reciben push.

**Objetivo del usuario:** recibir una notificación pop-up en su dispositivo cuando Soccer Edge identifique una oportunidad deportiva y un precio verificable de **-110 americano a +100 o mejor**. La regla de precio significa decimal **>= 21/11 (1.90909...)**, sin techo artificial: +110, +150 y superiores también cuentan. El precio por sí mismo nunca dispara una alerta de valor.

### Evento de candidato

El *trigger* debe ser emitido por el motor canónico **después** de SPORT FIRST y la verificación del precio. Requiere:

1. Mismo `fixture_id`, `market_family`, selección exacta, línea (si aplica), casa y fuente.
2. Cuota decimal >= 21/11; fuente Bet365 ID8 verificada como referencia, no afirmada ejecutable en otra casa.
3. Probabilidad RAW independiente de la cuota y proyección conservadora; edge frente a breakeven y EV estimado solo cuando el modelo/mercado concreto lo soporte.
4. `generated_at`, `captured_at`, `provider_updated_at` (si está disponible), expiración y `model_version`. No alertar precios caducados o ya iniciados.
5. Estado canónico, `availability_confidence`, confirmación XI/GK/lesiones relevante y lista de `blockers`. Nunca inferir que una alineación inexistente está confirmada.
6. La clasificación `BET` únicamente si el motor canónico ya la autorizó; `LEAN` y `WATCH` no se presentan como picks accionables.

### Modalidades y UX

- **ALERTA BET**: opt-in, solo cuando haya clasificación BET canónica, precio vigente y gate de disponibilidad completo. Incluir partido/mercado/línea/cuota/hora, razón principal, vencimiento y enlace a Match Center para **reconfirmar** precio antes de apostar manualmente. Nunca ejecutar apuesta ni abrir una casa con orden precargada.
- **RADAR DE CANDIDATOS**: opt-in separado para LEAN/WATCH con evidencia suficiente para seguimiento, etiquetado **INVESTIGACIÓN / NO BET**, indicando exactamente qué condición falta. Sin notificaciones agresivas ni garantías de ganar.
- Preferencias por ligas, mercados, rango de precio, hora de silencio, frecuencia máxima y canal (in-app, web push/PWA, app nativa futura). Permitir desactivar todo.
- Dedupe por `fixture_id|market|selection|line|bookmaker|quote_version|status`; evitar spam entre T-90/T-40/T-20, cambios de cuota menores, retiradas de precio o candidato invalidado.
- Alertas de `PRICE_MOVED`, `NO_LONGER_ELIGIBLE` y `CANCELLED` deben invalidar la recomendación anterior, nunca editarla silenciosamente.
- Entrega push requiere consentimiento explícito del usuario, permisos del navegador/dispositivo, persistencia de suscripción, VAPID/APNs/FCM según canal, backend de envío, expiración/TTL, notificación sin revelar datos sensibles en pantalla bloqueada y reglas legales / de edad aplicables.

### Contrato mínimo para futuros eventos

`candidate_id, fixture_id, market_family, selection, line, bookmaker_id, bookmaker_name, offered_price_decimal, quote_captured_at, quote_source_updated_at, classification, tier, raw_probability, shrunk_probability, breakeven_probability, estimated_ev, availability_confidence, model_version, status_version, expires_at, blockers, event_type, audience, created_at`.

El backend decide elegibilidad; el cliente no crea picks ni recalcula probabilidades. **No hay envío push ni alertas accionables hasta que existan productor canónico autorizado, servicio de entrega, consentimiento y pruebas E2E.**

### Aceptación y secuencia de entrega

1. **Render/estado actual:** verificar salud, frescura de datos, 200 y contrato para `/app-v2`, `/app/api/v2/today`, `/picks`, `/leans`; no publicar datos de octubre como actuales.
2. **Conexión de datos:** exponer vista de investigación/prospective con lectura controlada, manteniendo la lógica BET exclusivamente en el canónico; preservar distinción RAW, shrunk, fair, breakeven y EV de diagnóstico.
3. **Frontend:** tarjetas separadas `EDGE +100` y `EDGE REGULAR` filtrables por precio, estado y mercado, detalle Match Center e historial prospectivo; sin falsos BET ni cuotas obsoletas.
4. **QA:** probar navegación desktop/móvil, accesibilidad, FREE/PRO, manejo de errores, placeholders, tiempo de espera, disponibilidad de API, seguridad/RLS y discrepancias de datos. No mover rutas públicas sin verificación.
5. **Alertas de investigación:** registrar eventos idempotentes y preferencias; probar modo in-app/preview sin push real.
6. **Web Push/PWA (futuro):** completar consentimiento y entrega, pruebas de caducidad/dedupe/retirada, throttling, legal/jurisdicción; solo entonces habilitar.
7. **App nativa (si existe demanda):** reutilizar backend/notificaciones, sin duplicar lógica deportiva ni autorización de apuestas.

**Política comercial:** las alertas venden acceso a información y transparencia; no prometer ganancias, exigir atención a edad/jurisdicción y evitar mecánicas compulsivas. 


---

## FE-11 — Constructor multimerkado: goles, córners, hándicap y períodos

**Fecha:** 2026-11-06. **Origen:** captura Novibet facilitada por usuario. **Estado:** PERFIL DEPORTIVO DESCRIPTIVO IMPLEMENTADO; estimación conjunta OOS, boletos con precio verificable y producto comercial aún pendientes.

### Qué representa la captura

- **Necaxa–Tijuana:** misma partida, tres componentes: Over 6.5 córners totales, Over 1.5 goles totales y Tijuana +1 hándicap. El creador de apuestas muestra **cuota +100 para el conjunto**.
- **Atlético San Luis–FC Juárez:** Over 0.5 goles, Over 6.5 córners y San Luis +1 hándicap, combinados dentro del mismo encuentro a -300.
- Los dos constructores del mismo partido se reúnen en una apuesta de **dos boletos entre encuentros** a aproximadamente +167 en la captura. No confundir el precio de un componente individual con el precio conjunto del builder, ni multiplicar cuotas sin evidencia de una oferta.

Estos constructores son distintos de un mercado combinado ya cotizado (p.ej. Over 2.5 + BTTS), de un SGP personalizado con jugadores y de un parlay ordinario entre partidos.

### Orden SPORT FIRST → JOINT → MARKET

1. Verificar slate, torneo, kickoff, historia anterior al partido, disponibilidad, XI/GK, ritmo y calidad del rival.
2. Modelar por separado FT goles, team totals, 1H y 2H goles, corners 1H/2H/FT (ambos equipos), hándicap y márgenes de victoria/empate/derrota. Añadir tiros a puerta, presiones, centros, balón parado, jugadores y minutos cuando estén disponibles.
3. Construir una **distribución conjunta para selecciones del mismo partido**, no multiplicar marginales. Las decisiones de corners, goles y hándicap comparten estilo, ritmo y estado del marcador; verificar correlación OOS.
4. Solicitar **precio exacto de Bet Builder de la casa concreta**, selección, línea, lado, regla de liquidación, hora, disponibilidad. No inferir cuota ejecutable de la imagen ni de odds API-Football ID8.
5. Evaluar escenario de dos o más builders en un boleto interpartidos con **precio total confirmado**, sin asumir que el producto representa apuestas independientes.
6. Clasificar BET/LEAN/WATCH/PASS con el gate canónico, sin activar ninguna alerta BET sin OOS, cuotas y disponibilidad defendibles.

### Cobertura deportiva imprescindible

| Mercado | Evidencia deportiva previa a cuota | Dato que nunca debe inventarse |
|---|---|---|
| 1H y 2H goles | goles al descanso, FT coherente, xG y tiros por intervalo si fuente los da, formaciones, rival, local/visitante | p de cada tiempo sin modelo separado |
| FT goles, BTTS y TT | xG/xGA, chances, tiros totales y al arco, oposición, XI, calendario y localía | probabilidad independiente del resto del ticket |
| Córners 1H, 2H y FT | favor y contra por tiempo, centros, tiros bloqueados, juego por bandas, posesión avanzada, esquemas y rival | córners del rival cuando fuente devuelve null |
| Hándicap | distribución de diferencia de goles, probabilidad de cubrir, empatar o devolver según línea | asumir +1 asiático sin verificar Novibet |
| Jugadores | XI y minutos verificados, tiros por 90, rol, disciplina, rival y settlement exacto | cuota combinada de varias piernas no ofrecida |
| Contexto | recambios, cansancio, rivales, congestión, viajes y clima acreditados | motivaciones inventadas o muestras fuera de corte |

Cada cálculo debe conservar partido, liga, horario, n por local/visita, cortes temporales, periodización, fuente, porcentajes descriptivos y si está realmente calibrado. Ausente no es cero. Si hay pocos casos, mostrar muestra insuficiente, no un pick más fuerte.

### Código implementado ahora (solo SPORT FIRST)

- **multi_market_sport_profile.py:** recibe marcadores 90 minutos, descanso y estadísticas FT de ambos equipos; calcula perfiles descriptivos de goles 1H/2H/FT, córners a favor/en contra, patrones por localía y propensión temporal.
- Reproduce un ejemplo de triple componente: Over 1.5 FT goles + Over 6.5 FT córners + equipo +1. Cuenta coincidencias en **un mismo partido pasado**, y distingue push asiático de hándicap a tres vías. Es una frecuencia histórica, **NO** probabilidad futura de acierto.
- Restringe historia a FT anteriores al kickoff, rechaza tiempos incoherentes y córners cuyo periodo/identidad no estén documentados.
- Integración suplementaria en market_context, sin llamadas adicionales a la API. Devuelve calibración y elegibilidad BET falsas.

### Interfaz que falta desarrollar

**Match Center:** pestañas de 1H/2H, córners for/against, margen de victoria, estadísticas por rival con intervalos y n. **Builder de 2–4 piernas:** combina goles/córners/hándicaps (luego jugadores) con compatibilidad por fixture, settlement y dependencias, mostrando cuota conjunta solamente si está cotizada por el bookmaker. **Combinada de builders:** agrupa 2+ tickets de encuentros distintos, reconfirma el precio de cada uno y el precio del parlay total. **Historial público:** registra selección, precio, timestamp, modelo, resultado y liquidación por componente y ticket, incluso pérdidas.

**Secuencia:** histórico amplio y auditado → modelos de períodos/córners/hándicap calibrados por mercado → correlación OOS → puente oficial de cotización de Bet Builder → UI móvil y QA → alertas push opt-in. No promocionar mercados por ganar un ejemplo aislado.

**Restricciones:** no fabricar cuotas conjuntas multiplicando probabilidades o precios, no inferir que Tijuana +1 es asiático, no asignar cuota a un SGP de jugadores sin ticket verificable, no convertir WATCH a BET por estar +100, no mostrar selecciones incompatibles como recomendaciones simultáneas.
