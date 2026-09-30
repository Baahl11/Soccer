# Soccer Edge Commercial Roadmap

Canonical commercial/product roadmap layered on top of the existing Soccer Edge V4 evidence-first engine.

## Non-negotiable invariants

- Do not change model weights, thresholds, promotion gates, CLV semantics, scheduler logic, or provider budget as part of commercial work.
- Research/challenger surfaces remain `decision_weight=0` and `production_promotion_allowed=false` unless separately promoted by evidence.
- Subscriber UI must never manufacture missing price, edge, settlement, CLV, XI, or performance data.
- Free/Pro authorization is server-side. Client state alone can never unlock Pro.
- Billing must fail closed. Only verified billing lifecycle events may activate/revoke paid entitlements.
- Operator/internal surfaces stay separate from customer-facing product surfaces.

## Completed commercial track

### V214 — Commercial Dashboard Surface ✅
Subscriber-first presentation over persisted product views, with operator diagnostics retained.

### V215 — Commercial Product Shell ✅
Explorer/Free vs Edge Pro product contract, commercial shell, account/readiness framing.

### V216 — Performance & Track Record ✅
BET-only verified performance surface. Small samples remain explicitly labeled; no reconstructed ROI.

### V217 — Premium Match Detail ✅
Persisted price/probability/edge/provenance/blocker detail without invented fields.

### V218 — Auth + Accounts ✅
Dedicated Soccer Edge Supabase project connected to Render; JWT verification and account readiness.

### V219 — Subscription Entitlements ✅
Server-side Free/Pro entitlement resolver, RLS, fail-closed plan resolution, no client self-upgrade.

### V220 — Subscriber App + Billing Foundation ✅
- `/app` subscriber experience.
- `/app/data` Free/Pro filtered payload.
- Supabase billing tables with RLS.
- Checkout, Customer Portal, and signed Stripe webhook Edge Functions deployed fail-closed.
- Stripe account connected in LIVE mode.
- Stripe Product/Price still pending final pricing decision.

### V221 — Commercial Surface Hardening ✅
- `/dashboard` redirects to `/app`.
- `/product/views` returns entitlement-filtered subscriber data.
- Internal MCP product tools and V4 runtime remain untouched.
- Route ordering and Free redaction are regression-tested.

## Immediate roadmap

### V222 — ES/EN Internationalization
Goal: one product, two languages, one backend.

- Locale layer for `en` and `es`.
- Browser-language default plus manual EN/ES switch.
- Translate subscriber navigation, account, maturity states, market labels, blockers, performance copy, checkout copy, and alerts.
- Never translate internal canonical market/family identifiers in storage.
- Prepare locale-aware SEO metadata and share cards.

### V223 — Landing + Conversion System
Goal: turn traffic into free accounts before asking for payment.

- Bilingual landing page.
- Core positioning: `Model vs Market`, not generic “AI picks”.
- Explain evidence, calibration, CLV, and transparent track record.
- Free Explorer CTA -> account creation -> `/app`.
- Pro upgrade CTA only after user understands value.
- Responsible gambling/compliance footer and jurisdiction-aware messaging.

### V224 — Funnel Analytics
Goal: replace marketing guesses with Soccer Edge conversion data.

Measure:
- landing visit -> signup
- signup -> activated Explorer
- Explorer -> Pro checkout
- checkout -> paid Pro
- D1 / D7 / D30 retention
- churn
- feature usage by market family
- language/market cohort (MX Spanish, US Hispanic, US English)
- content source -> signup -> Pro attribution

No behavioral metric may alter betting-model decisions.

### V225 — Soccer Edge Content Factory
Goal: convert verified engine output into scalable organic acquisition content.

Pipeline:

`persisted signal -> content candidate -> ES/EN script -> render -> platform package -> attribution`

Outputs from one verified source row:
- 1080x1920 TikTok / Reel / YouTube Short
- X image/card
- X post/thread copy
- ES caption
- EN caption
- optional voice-over script

Core recurring formats:
1. Model vs Market
2. Why We Passed / NO BET
3. Line Movement / Price Watch
4. Verified Results & CLV transparency
5. Educational (+EV, fair probability, CLV, pricing)

Rules:
- Odds, probabilities, edge, settlements and CLV always come from persisted engine data.
- AI may generate narrative/layout/voice, never numerical betting facts.
- A losing day is never hidden from public verified-performance content.

### V226 — Pricing & Stripe Live Subscription
Goal: activate real recurring billing after pricing validation.

Initial market hypothesis to test, not final pricing:
- Mexico/LATAM: lower regional Pro price.
- United States: higher USD Pro price.
- Consider annual option after monthly conversion is measured.

Required before activation:
- Create Stripe `Soccer Edge Pro` Product.
- Create recurring regional Price(s).
- Configure Stripe secrets in Supabase.
- Register signed webhook endpoint.
- End-to-end Checkout -> webhook -> Supabase PRO -> cancel/past_due -> entitlement downgrade test.

### V227 — Beta Launch
Initial market order:
1. Mexico Spanish
2. US Hispanic
3. US English
4. broader LATAM

Do not split acquisition budget evenly on day one. Run independent cohorts and compare CAC, activation, Pro conversion, retention, and churn.

Target validation milestones:
- first 50 active beta users
- first 100 registered users
- first 10 voluntary paid users
- first 50 paid users
- first 100 paid users

Revenue/user-growth forecasts are planning scenarios only, never product promises.

### V228 — Growth Loops & Retention
After V224 provides real funnel data:
- favorites
- alerts
- watchlists
- weekly personalized recap
- referral system
- annual plan experiment
- win-back flows
- content-to-match deep links

## Market strategy

Build globally and bilingually from V222, but launch sequentially.

Recommended first wedge:
- Mexico + US Hispanic.

Then:
- US English.

Then:
- broader LATAM.

Reasoning:
- strong soccer affinity and Spanish content opportunity,
- lower-friction initial audience development,
- access to US Hispanic users with higher purchasing power,
- same engine/data can serve both languages without duplicating backend logic.

## Positioning

Primary brand concept:

> Model vs Market

Soccer Edge should be positioned as transparent soccer market intelligence, not as a “guaranteed picks” service.

English direction:
- `Stop betting blind.`
- Market prices.
- Model probabilities.
- Verified performance.
- Real closing-line evidence.

Spanish direction:
- `Deja de apostar a ciegas.`
- Precios de mercado.
- Probabilidades del modelo.
- Rendimiento verificable.
- Evidencia real contra el cierre.

## Current execution order

`V222 -> V223 -> V224 -> V225 -> V226 -> V227 -> V228`

Statistical maturation continues independently in the existing family order:

`1X2 -> BTTS -> FT Totals -> Team Totals -> 1H -> Corners -> 2H -> Cards -> Player Props`
