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

