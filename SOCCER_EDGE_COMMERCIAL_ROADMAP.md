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

### V221 — Commercial Surface Hardening ✅
- `/dashboard` redirects to `/app`.
- `/product/views` returns entitlement-filtered subscriber data.
- Internal MCP product tools and V4 runtime remain untouched.
- Route ordering and Free redaction are regression-tested.

### V222 — ES/EN Internationalization ✅
- One subscriber product in English and Spanish.
- Browser-language default plus persistent manual EN/ES switch.
- Presentation labels for market/status/maturity/account/billing copy.
- Canonical market/status identifiers remain unchanged in payload/storage.

### V223 — Landing + Conversion System ✅
- Public bilingual landing/conversion surface.
- `Model vs Market` positioning.
- Explorer/free-account CTAs into `/app`.
- No fabricated live odds or results on the marketing surface.

### V224 — Funnel Analytics ✅
Privacy-conscious product analytics are isolated from betting-model decisions.

### V225 — Soccer Edge Content Factory ✅
Evidence-locked EN/ES short-video and X-card generation is operational. Public Model-vs-Market gap is calculated reproducibly from persisted calibrated-model and de-vig market probabilities.

### V233 — Frontend Completion Gate ✅
- Approved Soccer Edge analysis mockup is the product source of truth.
- `/app` now uses the mockup-first shell instead of the older incomplete subscriber UI.
- Today, Edge Feed, Match Center, Markets, Performance, My Edge, Control Tower and Research Lab are present in the production product.
- Auth/account and regional billing controls are integrated into the product shell.
- Edge Feed and Top Edge can select a fixture and open its Match Center.
- Missing premium values remain redacted instead of reconstructed.

### V234 — Functional Mockup Completion ✅
- Edge Feed league, market, kickoff, edge, confidence and status controls are functional.
- Strong Only, Ready Only, Next 3 Hours and Confirmed XI modes are functional.
- Match Center Goals, Corners, Cards, Players, Market and Model tabs read persisted fixture rows.
- Missing family rows render explicit N/V/empty states and are never synthesized.
- Production V4 Runtime remained green.

### V235 — Visual & Mobile Parity ✅
- Mobile navigation exposes Today, Feed, Match, Markets, Performance, My Edge, Tower, Lab and Account.
- Safe-area handling, horizontal tab/nav scrolling, responsive Match Center, table scrolling and account spacing are hardened.
- Owner/Admin presentation access receives the advanced product view without creating or mutating a paid subscription.
- `/app` externally verified HTTP 200 with V233 + V234 + V235 layers present.
- Production V4 Runtime remained green.

## Current gates before Beta

### Frontend visual acceptance — NEXT
Open the production `/app` on desktop and mobile and compare against the approved mockup. Any visual discrepancy is fixed before billing activation or beta launch.

### V226 — Pricing & Stripe Live Subscription — TECHNICALLY WIRED / ACTIVATION HOLD
Current LIVE pricing:
- United States: Edge Pro Founding Beta — US$14.99/month.
- Mexico/LATAM: Edge Pro Founding Beta — MX$249/month.

Already wired:
- Stripe `Soccer Edge Pro` Product.
- Two recurring regional Prices.
- Explicit server-side billing-market allowlist; browser never sends a Stripe Price ID.
- Signed Stripe webhook endpoint.
- Customer Portal with payment-method updates, invoice history and cancel-at-period-end.

Still required before activation:
- Configure `STRIPE_SECRET_KEY` in Supabase Edge Function secrets.
- Configure `STRIPE_WEBHOOK_SECRET` in Supabase Edge Function secrets.
- End-to-end Checkout -> webhook -> Supabase PRO -> cancel/past_due -> entitlement downgrade test.

### V227 — Beta Launch — BLOCKED
Do not launch until both gates are green:
1. frontend visual acceptance;
2. V226 end-to-end billing validation.

Initial market order after unlock:
1. Mexico Spanish
2. US Hispanic
3. US English
4. broader LATAM

Target validation milestones:
- first 50 active beta users
- first 100 registered users
- first 10 voluntary paid users
- first 50 paid users
- first 100 paid users

### V228 — Growth Loops & Retention
After real funnel data:
- favorites
- alerts
- watchlists
- weekly personalized recap
- referral system
- annual plan experiment
- win-back flows
- content-to-match deep links

## Positioning

Primary brand concept:

> Model vs Market

Soccer Edge should be positioned as transparent soccer market intelligence, not as a “guaranteed picks” service.

## Current execution order

`Frontend visual acceptance -> finish V226 billing validation -> V227 Beta Launch -> V228`

Statistical maturation continues independently in the existing family order:

`1X2 -> BTTS -> FT Totals -> Team Totals -> 1H -> Corners -> 2H -> Cards -> Player Props`
