# SOCCER EDGE ENGINE — Commercialization launch gates v1.0

**Owner:** SPORTS EDGE ENGINE / Soccer. **As-of:** 2026-11-10 America/Mexico_City.
**Decision:** **RESEARCH ENGINE EVIDENCE VALIDATED IN PART; PAID PICKS / SGP / +110 ARE NO-GO.**
This is a product-execution checklist, not permission to take bets or a statement
of legal authorization. Always use the *live* generated
`soccer_edge_state/analysis/soccer_commercialization_readiness_v1.json`
and current workflow logs for up-to-date counters.

## What exists (actual outcomes, not interpretations)

1. **SPORT FIRST outside Render:** GitHub runner uses the existing canonical
   `GOAL_RATE_POISSON_LIMITED_v0.1`, live official API-Football season splits,
   bounded official requests and immutable `captured_at_utc < kickoff` records.
   It explicitly lacks XI/injury/form/xG confirmation, has availability
   confidence 0.50 and **cannot** be called a fully modeled Tier A/B prediction.
2. **Postgame scoring:** independent official regulation FT results are linked
   only to source-frozen pregame snapshots. There are genuine small-sample OOS
   rows for research 1X2, BTTS and Over2.5, **not** calibrated production ROI.
3. **PostgreSQL:** signed main-only GitHub OIDC imports only identical source
   predictions whose original Git commit also predates kickoff. Dedicated,
   immutable `soccer_isolated_sport_frozen_research_v1` table. No row is
   written to canonical betting/CLV event tables.
4. **MARKET SECOND:** separate live research workflow tests whether
   API-Football provides bookmaker-ID-8 **proxy** quotes with exact instrument
   (market, selection, line) during pre-kickoff; genuine repeated visits and
   later `provider_update` are checked. Price provenance is *API-Football*,
   not a direct executable sportsbook betslip. Missing price stays missing.
5. **Canonical engine:** market maturity framework still 7 RESEARCH /
   3 SHADOW, 0 production promotion. Canonical CLV and settlement are
   distinct from isolated Poisson research OOS and proxy-price movement.

## Hard no-go gates before any paid prediction product

| Gate | Required evidence | Status on 2026-11-10 |
| --- | --- | --- |
| Full sport-model coverage | Prospective full deep dives with trustworthy availability and feature timestamps | **BLOCKED**; 5/59 due Tier A/B deep dives on latest inspected tick |
| Model validation by family | OOS, calibration, settlement sample and stable performance per sport-specific market rule | **BLOCKED**; isolated FT OOS remains tiny; canonical promotion 0 |
| Entry **and** close prices | Exact bookmaker, family, selection, threshold, true time and provider update, independently captured twice prekickoff | **BLOCKED**; canonical CLV 789 historical, no increase in last daily report |
| Execution quality | Actual fresh book price, vig/devig and timestamp; executable betslip confirmation (not proxy) | **NOT VERIFIED** |
| SGP / same-game builder | Joint quote for the entire ticket, correlation/OOS and explicit leg availability | **BLOCKED**; prior SGP capture returned 0 quoted tickets |
| EDGE +100/+110 | Verified current line/price, sport-first raw model, market-shrunk model, odds expiry and opt-in notifications | **BLOCKED**; most recent Bet365 scanner 0 fresh eligible quotes |
| Legal / data licensing | Jurisdiction-by-jurisdiction review, source reuse and display rights, terms, advertising and responsible-gambling advice | **NOT VERIFIED** |
| Subscription and privacy | Authentication, access control, billing, refunds, data handling, age/jurisdiction, security and audited entitlements | **NOT VERIFIED** |
| Operational reliability | Daily no-stall monitor, repeated pregame captures, bounded provider usage, no OOM and reproducible incident rollback | **IN PROGRESS** |

**Hard rule:** none of these can be waived by a green CI or many simulated
picks. No implied A/S/B tier, BET, staking or claims of proven profitability
until the sport-specific promotion evidence actually passes.

## Commercial sequence (start work in order, not with selling picks)

### Stage A — product discovery and private demonstration

- Publish honest "research-only, not wagering recommendations" materials,
  including observed-versus-unverified sources, model limitations, OOS sample
  sizes and real data freshness; prepare neutral screenshots and a walkthrough.
- Interview prospective customers about features, countries, sports/markets,
  dashboard and notification utility. Maintain waitlist *only after* privacy,
  consent, terms and relevant marketing permissions are reviewed.
- Produce an operational cost baseline: source API quota, Render memory, GitHub
  compute, storage and support. Report observed metrics rather than invented
  revenue or conversion forecasts.
- **Gate:** security/product review + legal/data license validation before
  any external sign-up or sale. A research demo does not need a BET signal.

### Stage B — private, research-only beta

- Verified source freshness on the product surface, explicit RESEARCH /
  SHADOW / WATCH and reasons, user access/privacy tested, no payable action.
- Show independent FT OOS and quote-availability ratios separately by 1X2,
  FT goals, BTTS, team totals, 1H/2H, corners, cards and players.
- Measure latency, completeness, uptime, price-attempt conversion, OOS
  calibration, drift and incident count without cross-sport calibration.
- Do **not** label "Bet365 executable" based solely on provider ID 8.

### Stage C — limited commercial subscription (GO only when allowed)

- Need a lawful/approved plan for target jurisdiction(s), data license, product
  terms, payments and age/identity safeguards before collecting subscription
  fees. Product and legal approvals remain **NOT VERIFIED** until evidenced.
- Prediction/market family must have independently audited OOS, calibration,
  settlement, price evidence, reproducible model version and human
  market-promotion approval. Hold BET until this is met.
- Only enable user-visible BET/tier or paid edge alerts for the *specific*
  validated market, not soccer in general. Zero recommendations is valid.
- SGP and +110 are separate additional launch gates, **not** shortcuts to +100.

## Concrete engineering sequence / acceptance

1. **Source intake (done):** real Git pre-kickoff commit verified, immutable
   research-only PG, 0 fabricated events/CLV.
2. **Prospective quote visits (implemented; observations pending):** max 2
   official bookmaker-ID-8 proxy API calls/run; visits after model, before
   kickoff; two provider timestamps must genuinely advance. Persist missing
   quoted markets as missing, no price reconstruction.
3. **Sport→price→official FT reconciliation (audited separately):** per market
   count of actual frozen sport rows, valid entry market, repeat provider
   update, official FT match, canonical settlement gate, and how many
   observations fail each step. Never count research price movement as CLV.
4. **Full model quality:** restore full Tier A/B deep-dive throughput with
   memory-safe isolated job, confirmed XI/GK and availability. Research-only
   Poisson baseline is not an acceptable substitute.
5. **Operational status/go-no-go report (automated):** daily generated JSON
   from actual state; failures and zero growth are explicit, not greenwashed.
6. **Security, product, finance and legal check:** owner records written signoff
   for geo/data rights, UI subscription security, terms and independent
   performance claims before any commercialization is switched on.

## Daily scorecard

Source of truth:
`soccer-edge-state/soccer_edge_state/analysis/soccer_commercialization_readiness_v1.json`.

For each market require the five numbers **frozen raw predictions**, **official
FT OOS**, **entry quotes**, **later fresh provider updates**, **valid canonical
CLV & settled results**, with timestamps. SGP adds separately priced joint
ticket; +110 adds fresh price and candidate expiry. Unavailable = 0 or
**NOT VERIFIED**, never estimated.

**Next decision:** technical prelaunch is no longer dependent on manually
running one enormous Render job, but **commercial paid-bet readiness remains
NO-GO** until full-model coverage, real two-visit price evidence, per-family
settlement and applicable legal/product controls pass.
