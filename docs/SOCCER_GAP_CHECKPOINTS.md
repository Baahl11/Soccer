# SOCCER EDGE ENGINE — GAP CHECKPOINT TRACKER

Last updated: 2026-09-17 CDMX

Purpose: one authoritative point-by-point roadmap. Engineering continues without waiting for a natural tick after every module. Natural validation remains mandatory before a module is called fully resolved.

## Checkpoint dimensions

- DATA: required verified inputs exist for the stated use.
- MODEL: explicit SPORT-FIRST probability/projection exists.
- LIVE: attached to relevant fixtures during normal scheduler execution.
- PRODUCTION: permitted to affect canonical BET/LEAN classification.
- NATURAL: confirmed in persisted natural state after deployment.

Statuses: ✅ PASS · 🟢 IMPROVED · 🟡 LIVE RESEARCH · 🟠 OFFLINE/BUILT · 🔵 CALIBRATING · 🔴 GAP · ⚫ EXTERNAL BLOCK · ⏳ NATURAL CHECK.

## Current master map

| # | Module | DATA | MODEL | LIVE | PRODUCTION | NATURAL | Checkpoint | Exactly what remains |
|---|---|---|---|---|---|---|---|---|
| 1 | FT Goals | ✅ | ✅ | ✅ | ✅ Tier-B constrained | ⏳ partial | 🟢 v3.14 | Persisted ladder detail; challenger OOS; xG/GK/set-piece/rest/weather enrichment remains separate. |
| 2 | Team Totals | ✅ | ✅ Poisson 0.5/1.5/2.5 | 🟡 v3.15 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥100 OOS market review, ≥200 actionable review, price/CLV history, cross-league calibration, integer/quarter settlement. |
| 3 | Correct Score | ✅ | ✅ exact-score distribution | 🟡 v3.16 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥300 OOS market review, ≥500 actionable review, price/CLV history, sparse-outcome shrinkage. |
| 4 | BTTS | ✅ | ✅ | 🟡 v3.17 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥150/300 OOS gates, true CLV, shrinkage, probability-bucket/league/data-tier calibration. |
| 5 | 1X2 | ✅ | ✅ canonical + challengers | 🟡 v3.18 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Matched same-fixture OOS, true CLV, final versioned model selection and shrinkage. |
| 6 | Double Chance | ✅ | ✅ 1X/X2/12 | 🟡 v3.19 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Parent 1X2 approval, OOS/CLV, selection/league calibration and shrinkage. |
| 7 | DNB | ✅ | ✅ conditional no-draw + push-aware EV | 🟡 v3.20 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Parent 1X2 approval, non-draw OOS/CLV, settlement compatibility and shrinkage. |
| 8 | Asian Handicap | ✅ | ✅ margin + integer/half/quarter settlement | 🟡 v3.21 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥250/500 observed-line OOS, true CLV, line/league calibration, stronger parent margin model. |
| 9 | 1H Goals | ✅ HT history + observed lines | ✅ period Poisson | 🟡 v3.22 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Registry/natural persistence, ≥100/200 OOS, exact-line CLV, line/league calibration. |
| 10 | 2H Goals pregame | ✅ FT+HT history + observed lines | ✅ independent 2H period Poisson | 🟡 v3.23 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Registry/natural persistence, ≥100/200 OOS, exact-line CLV/calibration; never relabel as halftime-conditioned. |
| 11 | 2H live / halftime | 🟢 verified HT score/state; red cards/shots/SOT pending | ✅ pregame 2H baseline × walk-forward HT-state multiplier | 🟡 v3.28 dedicated HT path | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥200 OOS review/≥400 actionable review, verify red-card state + HT shots/SOT, attach real live 2H price/CLV, state-bucket/league calibration. |
| 12 | Corners FT | ✅ finalized corner history | ✅ league/team Poisson + formation shadow | 🟡 v3.24 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Natural persistence, exact-line OOS/CLV, ≥150 OOS + ≥100 formation-adjusted, stronger territory inputs. |
| 13 | Team Corners | ✅ team corner history/lambdas | ✅ team-specific corner Poisson | 🟡 v3.25 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | OOS/CLV, exact-line calibration by team role/league, stronger territory/game-state features. |
| 14 | Cards total | ✅ yellows/fouls + optional referee | ✅ discipline + referee yellow-card model | 🟡 v3.26 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Referee-history enrichment, explicit sportsbook scoring rules, ≥200 OOS/≥100 referee-adjusted, true CLV. |
| 15 | Team Cards | ✅ team yellow history | ✅ team yellow-card Poisson | 🟡 v3.27 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Team-line OOS/CLV/calibration, sportsbook settlement mapping; reds remain separate. |
| 16 | Red Cards | ✅ recorded separately | 🔴 | 🔴 | 🔴 | — | 🔴 NEXT | Separate low-frequency model, team/referee/game-state features, book-specific settlement; never merge blindly with yellow count. |
| 17 | Referee | 🟢 assignment captured | 🟠 optional cards effect | 🟠 partial | 🔴 | partial | 🟠 | Historical referee enrichment + current assignment verification + OOS lift. |
| 18 | Formations | ✅ | ✅ research intelligence | 🟠 partial | 🔴 | partial | 🟠 | Attach confirmed formation pair automatically to live research models; quantify only after OOS lift. |
| 19 | Coaches | ✅ identity from lineup | 🔴 regime model | 🟠 identity only | 🔴 | partial | 🟠 | Coach tenure/regime persistence, before/after effects, rotation/substitution behavior. |
| 20 | XI | ✅ | ✅ verification logic | ✅ | ✅ availability gate | ✅ historical | 🟢 IMPROVED | Better persistence/re-run/source quality; no identity probability model required. |
| 21 | Goalkeeper | ✅ starter capture | 🔴 impact model | 🟠 | 🔴 | partial | 🟠 | Shot-stopping/saves/concession impact with verified starter. |
| 22 | Injuries / suspensions | 🟢 provider support | 🔴 player-impact model | 🟠 partial | ✅ availability block only | partial | 🟠 | Quantify impact by player/role/minutes + stronger official verification. |
| 23 | Player trends | ✅ selective capture | 🟠 descriptive L5/L10/L20 | 🟡 | 🔴 | partial | 🟡 DESCRIPTIVE | Convert to probabilistic role/minutes-adjusted models. |
| 24 | Shots props | 🟢 capture | 🔴 | 🔴 | 🔴 | — | 🔴 | Minutes/role + shots/90 + opponent/formation distribution + exact threshold probability. |
| 25 | SOT props | 🟢 capture | 🔴 | 🔴 | 🔴 | — | 🔴 | P(1+/2+/3+) with minutes, role, opponent suppression and starting status. |
| 26 | Goalscorer | 🟠 goals/minutes | 🔴 | 🔴 | 🔴 | — | 🔴 | Player xG/share or defensible proxy, minutes, penalties, opponent/GK, calibration. |
| 27 | Assists | 🟠 assists/key passes | 🔴 | 🔴 | 🔴 | — | 🔴 | Chance creation/xA-quality, minutes and teammate finishing model. |
| 28 | GK Saves | 🟠 identity + some stats | 🔴 | 🔴 | 🔴 | — | 🔴 | Opponent SOT projection × save expectation; exact save-line probabilities. |
| 29 | Player Cards | 🟠 partial context | 🔴 | 🔴 | 🔴 | — | 🔴 | Position/role/fouls/opponent/referee/minutes model. |
| 30 | xG/xGA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Stable legal advanced-data source. Never infer xG from goals. |
| 31 | npxG/npxGA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Same source + penalty exclusion. |
| 32 | PPDA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Provider/source + normalized definition. |
| 33 | Field Tilt | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Provider/source + consistent territorial definition. |
| 34 | Box Entries | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Provider/source integration. |
| 35 | Big Chances | 🟠 provider-dependent | 🔴 | 🔴 | 🔴 | — | ⚫/🟠 | Consistent live/historical source before weighting. |
| 36 | Set Pieces | 🟠 corners/fouls | 🔴 | 🔴 | 🔴 | — | 🔴 | Attack/defense set-piece rates, aerial mismatch, ideally set-piece xG. |
| 37 | Tactical Style | 🟢 formation/trends | 🟠 descriptive | 🟡 partial | 🔴 | partial | Formal press/block/transition/width/possession classifier + OOS feature lift. |
| 38 | Rest | ✅ fixtures | 🟠 derivable | 🔴 | 🔴 | — | 🟠 EASY | Days since last match live + material thresholds. |
| 39 | Congestion | ✅ fixtures | 🟠 derivable | 🔴 | 🔴 | — | 🟠 EASY | 7/14/21-day windows + rotation pressure. |
| 40 | Travel | 🟠 venue/country | 🔴 | 🔴 | 🔴 | — | 🔴 | Distance/timezone/altitude/logistics only when material. |
| 41 | Weather | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Venue/kickoff forecast + materiality logic. |
| 42 | Competition Context | ✅ | 🟠 structured | 🟡 partial | 🔴 | partial | Table/aggregate/qualification math; verified objective context only. |
| 43 | Galaxy Multi | ✅ | ✅ rolling pool v0.5 | ✅ | 🔴 final without quote | partial | 🟢 IMPROVED | Feed only production-valid new families; preserve same-book/freshness/edge/final-quote gates. |
| 44 | Galaxy SGP correlation | 🟠 components | 🔴 joint model | 🔴 | 🔴 | — | 🔴 | Explicit same-game joint/correlation model; never multiply same-game marginals. |
| 45 | Exact SGP quote | ⚫ | — | 🔴 | required | — | ⚫ EXTERNAL | Sportsbook/provider actual combined quote. |
| 46 | Calibration lifecycle | ✅ ledger/results | ✅ validators growing | 🟡 | ✅ existing FT only | partial | 🟢 IMPROVED | Standardize Brier/log-loss/ROI/CLV/OOS gates and promotion-review reports per family/version/archetype; never auto-change weights. |

## Engineering checkpoints completed or materially improved

#1 FT Goals v3.14 · #2 Team Totals v3.15 · #3 Correct Score v3.16 · #4 BTTS v3.17 · #5 1X2 v3.18 · #6 Double Chance v3.19 · #7 DNB v3.20 · #8 Asian Handicap v3.21 · #9 1H Goals v3.22 · #10 2H pregame v3.23 · #12 Corners FT v3.24 · #13 Team Corners v3.25 · #14 Cards total v3.26 · #15 Team Cards v3.27 · #11 2H halftime v3.28.

### #11 2H live / halftime — v3.28

- Dedicated `HT` scheduler stage is added by wrapper only; base scheduler code is not rewritten.
- HT event uses the current fixture row already fetched for the daily slate, so v3.28 adds zero API-Football requests.
- Verified halftime score/lead state/HT-goal bucket feed a separate state multiplier learned walk-forward from finalized fixtures.
- The pregame 2H period model remains the baseline but is never relabeled as halftime-conditioned.
- Missing red-card state, halftime shots/SOT and live 2H price are explicit blockers.
- `actionable=false`, `decision_weight=0`; no BET/LEAN/Galaxy promotion.
- Existing offline validator remains the calibration basis and requires the conditioned challenger to beat the pregame 2H baseline.

## Engineering rule

Natural validation is asynchronous and non-blocking. A module is not called fully RESOLVED until its persisted natural state and calibration/promotion gates are satisfied.

## Next engineering target

#16 Red Cards — build a separate low-frequency research model. Red cards remain separate from yellow-card totals and require explicit bookmaker settlement rules before any market comparison can become actionable.
