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
| 11 | 2H live / halftime | 🟢 HT score/state verified; red/shots/SOT pending | ✅ pregame 2H baseline × HT-state multiplier | 🟡 v3.28 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥200/400 OOS, red-card + HT shots/SOT verification, live 2H price/CLV, state/league calibration. |
| 12 | Corners FT | ✅ finalized corner history | ✅ league/team Poisson + formation shadow | 🟡 v3.24 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Natural persistence, exact-line OOS/CLV, ≥150 OOS + ≥100 formation-adjusted, stronger territory inputs. |
| 13 | Team Corners | ✅ team corner history/lambdas | ✅ team-specific corner Poisson | 🟡 v3.25 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | OOS/CLV, team-role/league calibration, stronger territory/game-state features. |
| 14 | Cards total | ✅ yellows/fouls + optional referee | ✅ discipline + referee yellow-card model | 🟡 v3.26 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Referee-history enrichment, sportsbook scoring rules, ≥200 OOS/≥100 referee-adjusted, true CLV. |
| 15 | Team Cards | ✅ team yellow history | ✅ team yellow-card Poisson | 🟡 v3.27 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Team-line OOS/CLV/calibration, settlement mapping; reds separate. |
| 16 | Red Cards | ✅ separate postgame reds | ✅ Empirical-Bayes any-red YES/NO + optional referee | 🟡 v3.29 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Registry natural persistence; ≥500 OOS market review/≥1000 actionable; ≥200 referee-adjusted; explicit red-card YES/NO price/CLV and settlement. |
| 17 | Referee | 🟢 current API assignment + yellow/red registries | ✅ feature profile/gates | 🟡 v3.30 feature-only | 🔴 standalone | ⏳ | 🟡 LIVE RESEARCH FEATURE | Add historical fouls/penalty rate, independent official assignment verification, demonstrate OOS lift in cards/red models. |
| 18 | Formations | ✅ confirmed XI formation pair + historical formation report | ✅ matchup profile + OOS residual feature gate | 🟡 v3.31 feature-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH FEATURE | Natural persistence; ≥100 aggregate OOS with Brier+log-loss lift; matchup n≥8 before feature candidacy; still zero decision weight until versioned promotion. |
| 19 | Coaches | ✅ current coach from confirmed XI + historical regime registry | ✅ descriptive regime/tenure/change context | 🟡 v3.32 context-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH CONTEXT | Natural registry persistence; substitution/rotation behavior; OOS coach-feature lift; before/after differences remain non-causal. |
| 20 | XI | ✅ confirmed starters/GK/formation | ✅ verification + persistent XI fingerprint/change detector | ✅ v3.33 | ✅ existing availability gate only | ⏳ new detector | 🟢 IMPROVED v3.33 | Natural persistence of fingerprints; recheck behavior after material change; player-specific impact model still separate #22/#23. Existing availability gate unchanged. |
| 21 | Goalkeeper | ✅ starter capture | 🔴 impact model | 🟠 identity only | 🔴 | partial | 🟠 NEXT | Audit verified GK saves/SOT faced/goals conceded data; build shot-stopping/concession-impact research model only from supported fields. No xG/PSxG fabrication. |
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

## Engineering checkpoints materially completed/improved

#1 v3.14 FT Goals · #2 v3.15 Team Totals · #3 v3.16 Correct Score · #4 v3.17 BTTS · #5 v3.18 1X2 · #6 v3.19 Double Chance · #7 v3.20 DNB · #8 v3.21 Asian Handicap · #9 v3.22 1H Goals · #10 v3.23 2H pregame · #12 v3.24 Corners FT · #13 v3.25 Team Corners · #14 v3.26 Cards total · #15 v3.27 Team Cards · #11 v3.28 2H halftime · #16 v3.29 Red Cards · #17 v3.30 Referee · #18 v3.31 Formations · #19 v3.32 Coaches · #20 v3.33 XI.

### #18 Formations — v3.31
- Requires confirmed both-XI formations.
- Crosses the exact current formation pair with historical matchup summaries.
- A feature candidate is only flagged if aggregate OOS residual challenger has ≥100 evaluations and improves both Brier and log-loss, plus the exact matchup has n≥8.
- Descriptive formation history never upgrades a BET; `decision_weight=0`.

### #19 Coaches — v3.32
- Historical coach regimes are built from verified lineup coach identities plus finalized scores.
- Regime profile includes tenure/matches, GF/GA, O2.5 and BTTS; current-vs-previous differences are explicitly descriptive, not causal.
- New/unseen live coach is marked as a new regime instead of extrapolating old stats.
- Coach context has zero decision weight until a separate OOS feature-lift study exists.

### #20 XI — v3.33
- Persists per-fixture XI fingerprints across pregame captures.
- Detects starter, formation, GK and coach changes.
- Material changes are flagged for recheck, but no player impact is invented.
- Existing availability-confidence logic is unchanged.

## Engineering rule
Natural validation is asynchronous and non-blocking. A module is not called fully RESOLVED until persisted natural state and calibration/promotion gates are satisfied.

## Next engineering target
#21 Goalkeeper — audit what verified GK match/player statistics are actually persisted, then build only the impact model supported by those fields.