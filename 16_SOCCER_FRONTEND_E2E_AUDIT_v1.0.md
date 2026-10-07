# 16 — SOCCER FRONTEND E2E AUDIT v1.0

**Date:** 2026-10-07  
**Status:** AUDIT COMPLETE — REMEDIATION NOT YET STARTED  
**Scope:** Customer-facing Soccer Edge frontend, live data contract, navigation/render lifecycle, match detail, picks/leans presentation, auth, subscriptions, billing readiness, responsive behavior, CI coverage and separation from operator tooling.

---

# 1. Executive verdict

The current Soccer Edge customer frontend contains valuable product work, including:

- an approved visual mockup encoded in the V230/V233/V236/V237 layers;
- FREE / PRO entitlement logic;
- Supabase authentication;
- Stripe Checkout / Customer Portal / webhook infrastructure;
- persisted-data adapters;
- match-level detail;
- mobile navigation;
- logos / fixture identity;
- market family tabs;
- performance and maturity surfaces;
- explicit missing-data handling.

However, the current `/app` is **not architecturally suitable as the final monetized frontend**.

The highest-severity problem is the render architecture itself.

The live customer page is composed through a nested chain of server-generated HTML plus successive string injections and runtime monkey-patches:

```text
subscriber_preview_v230
  ↓
subscriber_preview_live_v231
  ↓
subscriber_preview_performance_live_v231
  ↓
subscriber_preview_maturity_live_v232
  ↓
subscriber_product_v233
  ↓
subscriber_product_v234
  ↓
subscriber_product_v235
  ↓
subscriber_today_v236.install(...)
  ↓
subscriber_visual_v237.install(...)
  ↓
/app
```

This creates multiple independent JavaScript renderers, data fetches, timers and mutation observers operating on the same DOM.

That architecture explains the reported symptom:

> pages/screens appear to load over other pages or visually stack/recompose during startup.

This is a **P0 frontend architecture issue**, not merely a CSS issue.

No betting-model thresholds, gates, probabilities, weights, provider budget or canonical classification logic should be changed to fix it.

---

# 2. Live route inventory

Current customer routes are installed through `mcp_gateway/__init__.py`.

## Customer

```text
/                         → redirect /app
/app                      → subscriber_product_v235.app_page
/app/data                 → subscriber_product_v235.app_data
/app/match                → subscriber_product_v235.match_data
/app/fixture-identities   → subscriber_today_v236.fixture_identities
```

Legacy preview:

```text
/app-preview              → redirect /app
/app-preview/data
/app-preview/performance
/app-preview/maturity
```

## Operator / internal product surface

```text
/dashboard
/product/views
```

The operator dashboard and subscriber application should remain separate products.

**Recommendation:** customer-facing UI must never expose Control Tower / Research Lab with the same visual priority as Picks, Leans and Match Intelligence.

---

# 3. P0 — Render/bootstrap audit

## 3.1 The current first paint begins from a mock frontend

`subscriber_preview_v230.py` explicitly describes itself as:

```text
Mockup-first Soccer Edge frontend preview.
It uses representative mock data only so UX can be approved before wiring.
```

The V230 base HTML contains example rows such as Arsenal, Manchester City, Inter, Real Madrid, PSG, Liverpool, etc. Its own inline JavaScript immediately writes these mock rows and example market counts into the page.

Later V231+ scripts replace/neutralize them with persisted data.

For a real paid product, this must not happen.

### Required change

Production first paint must contain only:

- stable app chrome;
- neutral skeletons;
- explicit loading state;
- no mock teams;
- no mock picks;
- no fake confidence;
- no fake price;
- no fake market counts.

The production DOM should be populated only after the canonical persisted subscriber payload is available.

---

# 4. P0 — Multiple render controllers

The same document currently receives behavior from many different layers.

Examples found in the active chain:

## V231

- `neutralizeMocks()`
- `loadPublicTower()`
- `loadLive()`
- `renderToday()`
- `renderFeed()`
- `renderMatch()`
- `renderMarkets()`
- `renderTower()`
- `renderPerformance()`
- `renderResearch()`
- `renderMyEdge()`

## V233

Runs independently:

```text
syncAccount()
loadApp()
loadPreview()
setTimeout(wireRows/wireHero/wireTabs, 1000)
```

## V234

Adds another functional/filter script and additional delayed wiring.

## V235

Adds another navigation controller and customer/mobile rendering behavior.

## V236

Uses multiple `setTimeout` refreshes and a `MutationObserver`.

## V237

Runs:

```text
setTimeout(run, 250)
setTimeout(run, 900)
setTimeout(run, 1800)
MutationObserver(...)
```

and recomposes the hero/crests after earlier layers already rendered them.

### Consequence

The browser can transition through several visually distinct states:

```text
mockup
→ mock neutralized
→ anonymous app data
→ authenticated preview
→ enhanced filters
→ mobile slate
→ Today recomposition
→ crest/hero recomposition
```

This causes avoidable layout shift, flicker, duplicate work and the reported perception of screens loading on top of one another.

### Required change

There must be exactly **one frontend bootstrap controller**.

Target:

```text
BOOT
→ AUTH RESOLUTION
→ ONE CUSTOMER PAYLOAD
→ ONE ATOMIC RENDER
→ incremental data refresh without DOM reconstruction
```

No competing navigation controllers.

No mock-to-live replacement.

No render-time monkey-patching.

No MutationObserver used as the primary application state mechanism.

---

# 5. P0 — Missing atomic boot gate

An earlier patch concept referenced a full-page boot class such as `v232-booting`, but the currently active `subscriber_preview_maturity_live_v232.py` does not contain an active boot gate.

The final product should not depend on hiding multiple legacy render phases after they have already executed.

### Required production behavior

Before app data is ready:

```text
one shell
one skeleton
one loading message
no interactive stale/mock content
```

After data is ready:

```text
remove skeleton once
hydrate customer state once
show content
```

Failure state:

```text
explicit DATA TEMPORARILY UNAVAILABLE
last verified timestamp if available
retry
never fabricate values
```

---

# 6. Information architecture audit

Current approved IA in V233 contains:

- Today
- Edge Feed
- Matches
- Markets
- Performance
- My Edge
- Control Tower
- Research Lab

This is useful for an engineering/product prototype, but it is not the optimal retail subscriber hierarchy.

A paying user primarily wants:

1. What should I act on?
2. Why?
3. At what exact price / line?
4. How confident is the system in the underlying data?
5. What could invalidate the play?
6. What else is close but not yet actionable?
7. How has the system actually performed?

## Recommended subscriber IA

### Primary navigation

```text
Today
Picks
Leans
Matches
Performance
My Edge
Account
```

### Secondary / contextual

```text
Markets
Model Evidence
Research / Maturity
```

### Owner/Admin only

```text
Control Tower
Scheduler
Provider Health
Research Lab
Model Lifecycle
Raw diagnostics
```

Control Tower and Research Lab are valuable, but should not compete with Picks and Leans in the main commercial experience.

---

# 7. Picks / Leans product requirement

The final product should make the classification hierarchy explicit.

## BET

Show prominently:

- fixture;
- league / kickoff;
- market family;
- exact market;
- exact selection;
- exact threshold / line;
- exact price;
- bookmaker/source;
- market timestamp / freshness;
- BET classification;
- Tier S / A / B when applicable;
- proposed stake/units when the engine is authorized to supply it;
- Raw Sport Projection;
- Market-Shrunk Projection;
- fair market probability;
- Probability Edge;
- EV only when defensible;
- Availability Confidence;
- Data Tier;
- lineup / goalkeeper / injuries / weather verification state as relevant;
- primary sporting reasons;
- primary market/value reason;
- invalidation / blocker conditions;
- model version.

## LEAN

Show the same core evidence but make it visually clear why it is **not a BET**.

Examples:

- insufficient edge;
- price not good enough;
- availability confidence below BET gate;
- evidence still maturing;
- concentration / calibration limitation.

A LEAN must not visually resemble a BET.

## WATCH

Show:

- what is missing;
- what exact condition would make the event actionable;
- e.g. waiting for XI, price, line, verified starter, goalkeeper, sufficient market.

## PASS

Should be available for transparency, but not dominate the landing experience.

---

# 8. Data contract audit — current weakness

`subscriber_ui_contract_v231.adapt_market_row()` currently collapses several distinct sporting/model probability concepts into one presentation field.

The current priority order includes:

```text
p_model_calibrated
p_shrunk
p_raw
model_probability
probability
→ model.probability
```

This is no longer sufficient for the SPORTS EDGE ENGINE architecture.

The frontend must preserve the distinction:

```text
RAW SPORT PROJECTION
≠
MARKET-SHRUNK PROJECTION
≠
FAIR MARKET PROBABILITY
≠
BREAKEVEN PROBABILITY
```

A single generic "Model %" can hide the most important methodological separation in the project.

## Required Frontend Contract v2

At minimum, each candidate should expose explicit fields for:

```text
fixture
classification
market_family
market
selection
line
price
bookmaker
market_source
market_captured_at

raw_sport_probability
raw_sport_projection_version
market_shrunk_probability
fair_market_probability
breakeven_probability
probability_edge_pp
estimated_ev

availability_confidence
data_tier
lineup_status
injury_status
weather_status
starter_status
goalkeeper_status

sporting_reasons[]
market_reasons[]
blockers[]
invalidation_conditions[]

tier
stake_units
model_version
generated_at
freshness
```

Unavailable data must remain:

```text
NOT VERIFIED
```

or null with explicit presentation language.

Do not convert missing data to zero.

---

# 9. Current Today ranking audit

`subscriber_preview_data_v231.build_preview_payload()` currently constructs:

```text
feed = strong + value + slate
sort by edge descending
top_edge = first row with model probability + market probability + edge
```

This makes the hero primarily **edge-first**.

That is not the desired final customer policy.

## Required customer ranking

When production classifications exist:

```text
1. verified BET
2. verified LEAN
3. actionable WATCH / pending confirmation
4. full verified slate
5. PASS
```

Within BET/LEAN:

- classification strength;
- tier;
- availability confidence;
- data quality;
- edge / EV;
- freshness;
- kickoff urgency.

The market line must never create the sporting thesis.

---

# 10. Match Center audit

Good existing concepts:

- match identity;
- team crests;
- match probability;
- expected goals when persisted;
- score matrix when persisted;
- market tabs;
- model context;
- lineup state;
- data quality;
- bookmaker;
- provider update;
- model version.

Missing or under-modeled for the final subscriber product:

- explicit Raw vs Shrunk probability;
- exact candidate classification and tier;
- Availability Confidence;
- source provenance per critical fact;
- verified/unverified availability panel;
- injury list/status;
- weather status where material;
- starting XI / goalkeeper confirmation;
- reason tree for BET / LEAN / WATCH / PASS;
- exact price timestamp;
- price movement / close context when available;
- stake guidance only when model policy permits it;
- historical performance context for this market/model version;
- formation/personnel/style evidence when validated and authorized.

---

# 11. Visual/mockup audit

No standalone PNG/JPG mockup artifact was found in the active branch tree.

However, the approved mockup is clearly represented in code contracts:

```text
subscriber_product_v233.contract():
source_of_truth = PANEL_DE_ANALISIS_SOCCER_EDGE_MOCKUP
```

and:

```text
subscriber_visual_v237.contract():
visual_target = APPROVED_MOCKUP_TEAM_CREST_PARITY_STABLE
```

V237 explicitly attempts approved-mockup crest/faceoff parity.

Therefore:

- preserve the current dark premium visual direction;
- preserve the strong team-vs-team hero treatment;
- preserve real crests with safe fallback;
- preserve clean green/blue evidence accents;
- eliminate patch-layer visual instability.

If the original mockup screenshot is available, attach it to the Project before the visual rebuild so exact visual parity can be audited pixel-by-pixel.

---

# 12. Mobile/responsive audit

Good work exists:

- safe-area variables;
- mobile horizontal nav;
- responsive match hero;
- mobile slate;
- mobile account access;
- responsive table overflow;
- crest fallbacks.

Main risk:

responsive behavior is spread across multiple injected style sheets, each with its own media queries and specificity.

This makes regressions difficult to reason about and can cause layout conflicts.

## Required change

One design system.

One breakpoint system.

One navigation component.

One responsive source of truth.

---

# 13. Monetization audit

The monetization foundation is materially more advanced than the legacy roadmap implies.

## Present and working at infrastructure level

Supabase project:

```text
Soccer Edge
project_ref = pqoobjacewxigngzhbhp
status = ACTIVE_HEALTHY
```

Tables exist with RLS enabled:

- subscription_entitlements;
- billing_customers;
- billing_subscriptions;
- billing_events;
- product_analytics_events.

Active Edge Functions:

- create-checkout-session;
- create-customer-portal;
- stripe-webhook;
- track-product-event;
- billing-readiness.

Checkout and portal require JWT where appropriate.

Stripe webhook verifies Stripe signatures.

Current UI has FREE / PRO entitlement concepts.

## Current commercial UI

The V233 product currently displays:

```text
Edge Pro · Founding Beta
US$14.99 / month
MX$249 / month
```

These prices must be treated as **current implementation values, not permanent product strategy**, until explicitly approved in the governing commercial roadmap.

## Important inconsistency

`subscription_entitlements_v4.plan_contract()` currently reports:

```text
billing_enabled = false
```

while billing Edge Functions and checkout UI exist.

This should be reconciled before public monetization.

The customer UI should not present the subscription system as fully production-ready until:

- billing readiness is verified;
- webhook flow is E2E tested;
- cancel / renew / past_due transitions are tested;
- success/cancel return UX is polished;
- entitlement propagation is tested;
- legal copy / responsible gambling / jurisdiction behavior is approved.

---

# 14. Security audit

Supabase security advisor currently reports:

```text
Leaked Password Protection Disabled
```

This should be enabled before commercial launch.

Positive findings:

- subscription/billing tables use RLS;
- checkout verifies authenticated user;
- browser selects only a billing-market enum, not Stripe Price ID;
- Stripe Price IDs are allowlisted server-side;
- webhook validates Stripe signature;
- billing customer/subscription persistence is server managed.

---

# 15. Analytics audit

Product analytics infrastructure already exists.

`product_analytics_events` contains tracked events and currently has persisted rows.

The tracking function accepts a controlled allowlist including:

- landing_view;
- explorer_cta;
- app_view;
- language_change;
- signup_click;
- signin_click;
- authenticated_view;
- pro_checkout_click;
- checkout_created;
- checkout_canceled;
- customer_portal_click.

Before monetization, add product events around the core value path:

```text
pick_opened
lean_opened
match_detail_opened
market_detail_opened
pick_saved
lean_saved
alert_enabled
upgrade_paywall_viewed
performance_opened
```

No analytics event may feed betting-model features.

---

# 16. CI / regression audit

This is another high-priority weakness.

Legacy frontend-specific workflows exist, including:

- V233 Frontend Completion Gate;
- V234 Functional Mockup Completion;
- subscriber billing workflows;
- commercial dashboard workflows.

But the active generic runtime suite on the current engine branch only covers a subset of subscriber contract/data tests.

It does not currently provide a strong active gate for the entire V233→V237 rendered customer app.

## Required active frontend CI

Every frontend change must validate:

### HTML/runtime

- exactly one document shell;
- no duplicate critical IDs;
- exactly one active page after boot;
- no mock picks in production HTML;
- no mock market counts in production HTML;
- no hidden legacy page rendered above/below active page;
- one navigation controller;
- one bootstrap controller;
- one canonical customer-data request flow.

### Data integrity

- missing values remain missing / NOT VERIFIED;
- no frontend-created BET/LEAN classification;
- no frontend-created probabilities;
- no frontend-created price;
- no market data used to create the raw sporting projection.

### Entitlements

- anonymous;
- FREE authenticated;
- PRO;
- OWNER/ADMIN;
- premium-field redaction;
- match-detail access;
- subscription state transitions.

### Responsive

At minimum:

```text
360px
390px
430px
768px
1024px
1440px
```

### UX

- skeleton → data transition;
- empty state;
- 401;
- 403;
- 503;
- stale snapshot;
- no picks;
- many picks;
- long team names;
- missing crest;
- missing price;
- unverified XI.

---

# 17. Target architecture recommendation

## Immediate stabilization

Do not attempt to polish the current injection stack indefinitely.

First stop the visible stacking/flicker.

## Target commercial architecture

Recommended end state:

```text
Next.js + TypeScript customer frontend
        ↓
versioned read-only Frontend API contract
        ↓
Soccer Edge FastAPI/Render backend
        ↓
canonical persisted SPORTS EDGE state
```

Auth:

```text
Supabase Auth
```

Billing:

```text
Stripe Checkout / Portal
→ Supabase Edge Functions
→ subscription_entitlements
```

The model engine stays completely isolated from frontend rendering logic.

---

# 18. Migration strategy

## Phase FE-0 — Stability hotfix

Goal: fix the reported screens-over-screens problem without altering engine logic.

- production page no longer paints V230 mock picks;
- introduce one boot state;
- suppress intermediate legacy layers;
- remove competing startup renderers;
- preserve current approved look as much as possible;
- add loading/stacking regression tests.

## Phase FE-1 — Frontend Contract v2

Build the clean read-only subscriber API contract.

Primary objects:

- slate;
- picks;
- leans;
- watches;
- match_detail;
- performance;
- account/entitlement.

No visual rebuild should precede this contract.

## Phase FE-2 — Premium customer shell

Rebuild the approved mockup as components.

Primary landing:

```text
TODAY
→ BETS
→ LEANS
→ WATCHES
→ UPCOMING / FULL SLATE
```

## Phase FE-3 — Match Intelligence

Deep match detail with sport-first/model/market separation.

## Phase FE-4 — Trust / Performance

Verified record, CLV, calibration, sample size, model version.

## Phase FE-5 — Monetization

Finalize FREE/PRO offer, billing state, paywall, upgrade UX, account management.

## Phase FE-6 — PWA / notifications

Only after web UX and billing are stable.

---

# 19. Definition of frontend-ready

The frontend is not considered commercially ready until all are true:

1. no mock data can appear as live;
2. no stacked/intermediate screens on boot;
3. one bootstrap/render controller;
4. canonical picks and leans are the primary subscriber surface;
5. Raw Sport Projection is visually distinct from Market-Shrunk Projection;
6. exact price/line/source/timestamp are shown for actionable bets;
7. Availability Confidence and critical verification are visible;
8. missing facts display NOT VERIFIED;
9. FREE/PRO redaction is enforced server-side;
10. Match Detail is backed only by persisted/canonical data;
11. verified performance is available and transparent;
12. responsive layouts pass all target widths;
13. auth/billing E2E passes;
14. frontend CI runs on the active production branch;
15. no frontend change can mutate model logic, thresholds, gates or historical predictions.

---

# 20. Audit decision

**Do not continue adding V238/V239 presentation patches on top of the current HTML chain.**

The next engineering move should be:

```text
FE-0 stabilization
→ FE-1 canonical subscriber contract
→ componentized premium frontend
```

The existing mockup and product work should be preserved as design/product reference, not as the long-term rendering architecture.
