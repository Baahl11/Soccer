# SOCCER EDGE — SPORT models per family → exact quote → paid launch, v1.0

**Cutoff:** 2026-11-11, America/Mexico_City.  
**Status:** research model-code and security gates; **not calibrated**, no real executable quote, no provider redistribution rights, no commercial release.

## 3. Independent market-family SPORT development

All experimental outputs use model version `SOCCER_FAMILY_RESEARCH_v1.0`, strict pregame source as-of, immutable independent frozen forecast and `RESEARCH_ONLY` (no outcome/odds input).

| Family | Mathematical prototype | Minimum required *source input* | Evidence needed to promote |
|---|---|---|---|
| FT 1X2 | Two-side independent Poisson goal-rate probabilities | Venue/opp-adjusted rates, sample >=5, source timestamp | Official FT OOS + calibration and comparison versus prior baseline |
| FT totals | Independent count distribution, 1.5/2.5/3.5 half-lines | Venue/opp-adjusted scoring rates, sample >=5 | OOS per exact line; distinguish goals and actual shot-based xG |
| BTTS | Both scores >=1 from two-side goal model | Two independently measured scoring rates | OOS calibration of yes/no outcomes |
| Team totals | Independent home/away goal counts | Two-team goal rates, lines for each team | Team-specific settlement and exact line OOS |
| 1H 1X2, 1H totals | Period-specific separate count model | **Measured first-half** scoring rates (never infer from FT lambda) | Official half-1 and correct timestamp |
| 2H 1X2, 2H totals | Period-specific separate count model | **Measured second-half** scoring rates | Official half-2, goals excluding half-1 |
| FT corners | Research count baseline (NOT possession/field tilt model) | Explicit corners for/against, venue, opponent, sample >=5 | Official corner count OOS, count-model fit and edge versus prior |
| Team corners | Research team counts | Team corners for/against independently | Official corners by team and OOS |
| FT cards | Research count baseline (NOT a referee-adjusted model) | Both team disciplinary rates **plus referee verified** | Official cards and bookmaker-specific definition (yellow/second yellow/red) |
| Player shots | Research count baseline | Player ID, verified starting XI, expected minutes, >=8 matches, shot rates | Official player shots per book settlement, minutes and OOS |

**IMPORTANT:** The code explicitly marks every one of these as `MODELED_LIMITED` only *when all feature inputs for that family are supplied*. It never fabricates missing features. This is **not** proof all families have actual prospective input coverage. Research registry in owner Edge Radar is a code-readiness catalog, not 12 live-forecast successes. Thresholds here are provisional source minimums, not market maturity for BET. xG, npxG, PPDA, true field tilt and individual player impact are NOT VERIFIED unless an authorized source independently supplies them.

The family OOS grader `soccer_oos_family_validation_v1.py` uses immutable pregame probabilities, source-attested official FT postgame result, family-isolated Brier and no transfer of other sports or family weights. Provisional >=100 unique official FT fixture threshold is for **research sample QA only**, never automatically unlocks a market or price. Real historical OOS evidence for these twelve experimental family-specific models has **NOT** been backfilled or verified.

## 4. SPORT then MARKET — exact quotes before any BET

`soccer_quote_research_v1.py` checks same fixture, period, team/player ID, bookmaker, market ID, exact half-line, selection, complete opposite sides, producer/source timestamp, and at-most-ten-minute capture freshness. Missing fields lead to WATCH with explicit reason. Independent raw P, vig-normalized market fair P, market-shrunk research P, breakeven and EV are calculated **only after** sporting RAW has been frozen. Never combine opposite prices from another bookmaker, line or timestamp. Theoretical Tier B/A/S is **not** a BET classification.

Existing `soccer_model.evaluate_market()` previously had an automated **Tier-B BET** path using API-Football provider odds. This patch fail-closes `BET` and `LEAN` into **WATCH**, zero stake and reasons `DIRECT_EXECUTABLE_BOOKMAKER_QUOTE_NOT_VERIFIED` and `INDEPENDENT_FAMILY_OOS_AND_SETTLEMENT_GATE_NOT_PROMOTED`. A quote from an aggregator, even if associated with a bookmaker name, is not a customer-executable ticket. Source timestamp does not prove direct sportsbook execution or betslip availability.

Closing line: direct exact-instrument opening and closing screenshots/API attestation required, along with external, legally permissible bookmaker access. `compare_close_same_instrument()` never labels `true_clv` verified just because the user set a `source_type` field. Official results/graded evidence must be audited separately and never override the original frozen forecast.

**Blocked external dependency:** No independently confirmed direct sportsbook executable betslip connector/source currently exists in this integration. No prices, sportsbook quotes, market movement or CLV were fabricated or obtained. Do not procure gambling accounts or make purchases merely to satisfy a test. Launch quote dependency must be resolved with a lawful provider contract and a strictly documented integration, still subject to separate user approval.

## 5. Subscription, privacy and legal readiness — Stripe deferred

**User decision:** Stripe configuration and sandbox testing to occur **later from laptop**. This change does **not** operate a Stripe account, create charges, create accounts, change webhook secrets, create prices or enable subscriptions.

The app already has guarded Hosted Checkout and Customer Portal concepts plus webhook-authoritative entitlements. New `soccer_launch_safety_v1.py` adds nine explicit non-commercial/operational attestations: verified family OOS, executable quote rights, exact-instrument settlement/true CLV, third-party data licensing/redisplay, privacy & RLS, Stripe sandbox and idempotent webhook QA, territorial legal review, consented deduped alerts, and final product/pricing signoff. **SOCCER_PUBLIC_BILLING_ENABLED alone cannot turn on checkout**. All nine defaults are false. These env switches record operator assertions and do not independently establish legal rights or actual audit results.

Legal checklist (not legal advice):

- **México private data:** Updated Ley Federal de Protección de Datos Personales en Posesión de los Particulares (original act 2025-03-20, subsequent reforms 2025-11-14). Need privacy notice, data controller, access/rectification/cancellation/opposition procedures, retention/deletion and security review for app-user data. Official source: https://www.ordenjuridico.gob.mx/Documentos/Federal/html/wo125102.html (see latest consolidated update at https://www.te.gob.mx/legislacion/media/files/3cc41f81a66fca1.htm).
- **Gambling-related content:** Federal Juegos y Sorteos regulation and advertisement rules may apply depending on whether the product offers information versus operates/promotes licensed wagering. Formal jurisdiction-specific counsel needed before betting-content or bookmaker advertisements. Source: https://juegosysorteos.segob.gob.mx/marco-juridico ; https://www.ordenjuridico.gob.mx/Documentos/Federal/pdf/wo88470.pdf .
- **Third-party stats/league rights:** API access does NOT automatically grant sublicensing, display or sale of underlying data. Highlightly and other vendors need signed/explicit terms per family and intended customer usage; currently NOT VERIFIED and only owner-side research metadata is surfaced.
- **Opt-in alerts:** No send workflow activated. `allow_research_alert()` requires user opt-in, fresh persisted data, verified fixture identity, dedupe and source display rights; returns research-only eligibility, never an actual send or betting signal.
- **Account QA:** test FREE/PRO/owner, route access, superseded token and revoked user handling, Supabase RLS on favorites, data deletion/access requests, mobile UX, subscription cancellation and webhook race/idempotence. Legal and security QA remain pending.

### Laptop-only Stripe handoff (not executed)

1. In Stripe **sandbox/test mode**, choose the correct connected company/account; verify test products, subscription prices/currencies and webhook subscriptions. Never paste secret keys into GitHub issues, chats or code.
2. Inspect Supabase Edge Functions for `create-checkout-session`, `create-customer-portal` and Stripe webhook idempotency, tax/country settings and customer matching. Use server-side Price IDs only.
3. Test test-card checkout success, cancel, failed payment, refund, trial if supported, portal cancellation, webhook retries, duplicate/out-of-order events and RLS entitlement changes. Confirm commercial entitlements react **only** to authenticated webhook persistence.
4. Keep public billing disabled until data licensing, OOS/CLV, jurisdiction and other approvals are documented. An approved checkout test is not authorization to sell a betting product.

## Sequenced remaining execution

**Priority P0:** Full eligible slate cheap sport-first screening + per-fixture status/coverage, budget scheduler with API-Football / Highlightly caching and no needless quotes.  
**P1:** Timed as-of history and actual official OOS samples by family, calibration confidence, error tracking, freeze provenance/version, no hindsight.  
**P2:** Direct bookmaker executable quotes and independent FT/CLV audit, legal market licensing; until then WATCH.  
**P3:** Approved redistribution licenses, privacy/RLS mobile account smoke and alert opt-in, Stripe sandbox from laptop, jurisdiction-specific counsel.  
**P4:** Manual product go/no-go and pricing approval after gates, not prior.

**No new provider requests, purchases, auto-recurring tasks, subscriptions, or public billing activation from this change.**

This is a technical gate/architecture completion, **not** a completed launch or validated business. The research family model version differs from existing Soccer Edge Engine v1.0 live baseline; existing historical predictions remain unchanged.
