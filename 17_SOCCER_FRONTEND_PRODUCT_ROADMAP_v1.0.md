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
