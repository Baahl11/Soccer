# Soccer Edge Commercial Roadmap

Canonical commercial/product roadmap layered on top of the existing Soccer Edge V4 evidence-first engine.

## Non-negotiable invariants

- Do not change model weights, thresholds, promotion gates, CLV semantics, scheduler logic, or provider budget as part of commercial work.
- Research/challenger surfaces remain `decision_weight=0` and `production_promotion_allowed=false` unless separately promoted by evidence.
- Subscriber UI must never manufacture missing price, edge, settlement, CLV, XI, or performance data.
- Free/Pro authorization is server-side. Client state alone can never unlock Pro.
- Billing must fail closed. Only verified billing lifecycle events may activate/revoke paid entitlements.
- Operator/internal surfaces stay separate from customer-facing product surfaces.
- Product/marketing analytics must never become betting-model input.

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

### V222 — ES/EN Internationalization ✅
- One subscriber product in English and Spanish.
- Browser-language default plus persistent manual EN/ES switch.
- Presentation labels for market/status/maturity/account/billing copy.
- Locale-aware title, description and Open Graph metadata.
- Canonical market/status identifiers remain unchanged in payload/storage.

### V223 — Landing + Conversion System ✅
- Public bilingual `/` landing page.
- `Model vs Market` positioning.
- Explorer/free-account CTAs into `/app`.
- Price awareness, evidence gating, verified performance and premium match detail explained.
- Responsible-gambling and legal-age/jurisdiction copy.
- No fabricated live odds or results on the marketing surface.

### V224 — Funnel Analytics ✅
- Privacy-conscious product event ledger in Supabase with RLS and client deny-all.
- Anonymous/session IDs without storing IP or browser fingerprint.
- Explicit UTM/cohort attribution; no inference of ethnicity or US-Hispanic identity.
- Landing, Explorer, signup/signin, language, checkout, portal and authenticated-view events.
- Server-side `checkout_created` attribution.
- Signed Stripe webhook is the only path that records `pro_activated` after ACTIVE/TRIALING subscription evidence.
- Analytics remains completely separate from betting-model decisions.

### V225 — Soccer Edge Content Factory ✅
Evidence-locked organic content generation is operational.

Pipeline:

`persisted signal -> content candidate -> ES/EN script -> Remotion render -> platform package -> GitHub artifact`

Validated outputs:
- 1080x1920 TikTok / Reel / YouTube Short in English.
- 1080x1920 TikTok / Reel / YouTube Short in Spanish.
- X image/card in both languages.
- X post copy, captions and voice-over scripts in both languages.
- Manifest containing the exact persisted source facts used by the render.

Production validation:
- First LIVE run exposed that an upstream `calibrated_edge_pp` field did not share the same semantics as the public Model-vs-Market probability gap.
- V225.1 now calculates the public gap deterministically as `(p_model_calibrated - p_market_fair) * 100` using persisted probabilities only.
- Regression protects the observed mismatch case.
- Focused Content Factory CI passed.
- Full V4 Runtime suite passed.
- Corrected Content Factory LIVE run rendered EN/ES videos, X cards and artifact successfully.

Rules:
- Odds, probabilities, settlements and CLV always come from persisted engine data.
- Public probability gap is a reproducible calculation from persisted calibrated-model and de-vig market probabilities.
- AI may generate narrative/layout/voice, never numerical betting facts.
- Research-only status remains visible and is never relabeled as a production pick.
- A losing day is never hidden from public verified-performance content.

## Immediate roadmap

### V226 — Pricing & Stripe Live Subscription
Goal: activate real recurring billing after pricing validation.

Market approach:
- Product stays bilingual/global.
- Initial acquisition wedge: Mexico + US Hispanic.
- US English follows after early conversion data.
- Regional pricing may differ while entitlements remain one `Edge Pro` product tier.

Required before activation:
- Validate current competitor pricing and willingness-to-pay bands.
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

Build globally and bilingually, but launch sequentially.

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

`V226 -> V227 -> V228`

Statistical maturation continues independently in the existing family order:

`1X2 -> BTTS -> FT Totals -> Team Totals -> 1H -> Corners -> 2H -> Cards -> Player Props`
