# SOCCER EDGE ENGINE — GAP CHECKPOINT TRACKER

Last updated: 2026-09-17 CDMX

Purpose: one authoritative point-by-point roadmap. Engineering may continue without waiting for a natural tick after every module. Natural validation remains mandatory before a module is called fully resolved.

## Checkpoint dimensions

- DATA: required verified inputs exist for the stated use.
- MODEL: explicit SPORT-FIRST probability/projection exists.
- LIVE: attached to upcoming fixtures during normal scheduler execution.
- PRODUCTION: permitted to affect canonical BET/LEAN classification.
- NATURAL VALIDATION: confirmed in persisted natural state after deployment.

Statuses: ✅ PASS · 🟢 IMPROVED · 🟡 LIVE RESEARCH · 🟠 OFFLINE/BUILT · 🔵 CALIBRATING · 🔴 GAP · ⚫ EXTERNAL BLOCK · ⏳ NATURAL CHECK.

## Current master map

| # | Module | DATA | MODEL | LIVE | PRODUCTION | NATURAL | Checkpoint | Exactly what remains |
|---|---|---|---|---|---|---|---|---|
| 1 | FT Goals | ✅ | ✅ | ✅ | ✅ Tier-B constrained | ⏳ partial | 🟢 v3.14 | Persisted intelligence/ladder confirmation; challenger OOS; advanced xG/GK/set-piece/rest/weather inputs remain separate gaps. |
| 2 | Team Totals | ✅ | ✅ Poisson 0.5/1.5/2.5 | 🟡 v3.15 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥100 OOS market review, ≥200 actionable review, price history/CLV, cross-league calibration, integer/quarter settlement support. |
| 3 | Correct Score | ✅ | ✅ exact-score distribution | 🟡 v3.16 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥300 OOS market review, ≥500 actionable review, correct-score price/CLV history, sparse-outcome shrinkage, cross-league calibration. |
| 4 | BTTS | ✅ | ✅ | 🟡 v3.17 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥150 OOS market review, ≥300 actionable review, price/true-CLV history, shrinkage, bucket/league/data-tier calibration. |
| 5 | 1X2 | ✅ | ✅ canonical + challengers | 🟡 v3.18 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | ≥200 matched challenger OOS, ≥300 same-fixture head-to-head, ≥400 actionable review, true CLV, final versioned model selection and shrinkage. |
| 6 | Double Chance | ✅ | ✅ 1X/X2/12 derivation | 🟡 v3.19 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Parent 1X2 approval; ≥200/400 OOS gates; historical DC prices/CLV; selection/league calibration and shrinkage. |
| 7 | DNB | ✅ parent 1X2 + DNB prices | ✅ conditional no-draw + push-aware EV | 🟡 v3.20 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Parent 1X2 approval; ≥200 non-draw OOS market review / ≥400 actionable; DNB price history/true CLV; settlement compatibility and shrinkage. |
| 8 | Asian Handicap | ✅ lambdas + retained AH market | ✅ margin distribution + integer/half/quarter settlement | 🟡 v3.21 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Natural persistence; ≥250 observed-line OOS market review / ≥500 actionable; price/true-CLV history; line-bucket/league calibration; settlement-aware shrinkage; stronger parent margin model. |
| 9 | 1H Goals | ✅ HT results + observed 1H lines | ✅ hierarchical period Poisson | 🟡 v3.22 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Initial period registry persistence; natural attachment; ≥100 OOS market review / ≥200 actionable; exact-line price/CLV history; line/league calibration; XI/availability gates if promoted. |
| 10 | 2H Goals pregame | ✅ FT+HT results + observed 2H lines | ✅ independent pregame 2H hierarchical Poisson | 🟡 v3.23 | 🔴 | ⏳ | 🟡 LIVE RESEARCH | Initial registry persistence; natural attachment; ≥100/200 OOS gates; exact-line price/CLV history; calibration. Must never be represented as halftime-conditioned. |
| 11 | 2H live / halftime | 🟠 halftime state partially available | 🔴 | 🔴 | 🔴 | — | 🔴 NEXT | Separate HT-conditioned model using score, cards, shots/SOT and game state; independent calibration; do not reuse pregame 2H probabilities. |
| 12 | Corners FT | ✅ postgame corners | ✅ offline walk-forward baseline + formation challenger | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach live; exact observed lines; stronger territorial features; 150 OOS + 100 formation-adjusted + CLV gate. |
| 13 | Team Corners | ✅ team corner history | 🟠 | 🔴 | 🔴 | — | 🔴 | Explicit home/away team-corner distribution, observed-line pricing, OOS/CLV. |
| 14 | Cards total | ✅ yellows/fouls | ✅ offline baseline + optional referee scale | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach live, referee enrichment, sportsbook scoring rules, 200 OOS/100 referee-adjusted + CLV. |
| 15 | Team Cards | ✅ team yellow history | 🟠 components | 🔴 | 🔴 | — | 🔴 | Per-team card lambdas + exact team-card pricing/calibration. |
| 16 | Red Cards | ✅ recorded separately | 🔴 | 🔴 | 🔴 | — | 🔴 | Separate low-frequency model and book-specific settlement; never merge blindly with yellow count. |
| 17 | Referee | 🟢 assignment captured | 🟠 optional cards effect | 🟠 partial | 🔴 | partial | 🟠 | Historical referee enrichment and current assignment verification; OOS lift. |
| 18 | Formations | ✅ | ✅ research intelligence | 🟠 partial | 🔴 | partial | 🟠 | Attach confirmed formation pair automatically to live research models; quantify only after OOS lift. |
| 19 | Coaches | ✅ identity from lineup | 🔴 regime model | 🟠 identity only | 🔴 | partial | 🟠 | Coach tenure/regime persistence, before/after effects, rotation/substitution behavior. |
| 20 | XI | ✅ | ✅ verification logic | ✅ | ✅ availability gate | ✅ historical | 🟢 IMPROVED | Better persistence/re-run/source quality; no identity probability model required. |
| 21 | Goalkeeper | ✅ starter capture | 🔴 impact model | 🟠 | 🔴 | partial | 🟠 | Shot-stopping/saves/concession impact with verified starter. |
| 22 | Injuries / suspensions | 🟢 provider support | 🔴 player-impact model | 🟠 partial | ✅ availability block only | partial | 🟠 | Quantify impact by player/role/minutes + stronger official material-absence verification. |
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
| 38 | Rest | ✅ fixtures | 🟠 feature derivable | 🔴 | 🔴 | — | 🟠 EASY | Days since last match live + material thresholds. |
| 39 | Congestion | ✅ fixtures | 🟠 feature derivable | 🔴 | 🔴 | — | 🟠 EASY | 7/14/21-day windows + rotation pressure. |
| 40 | Travel | 🟠 venue/country | 🔴 | 🔴 | 🔴 | — | 🔴 | Distance/timezone/altitude/logistics only when material. |
| 41 | Weather | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Venue/kickoff forecast + materiality logic. |
| 42 | Competition Context | ✅ | 🟠 structured | 🟡 partial | 🔴 | partial | Table/aggregate/qualification math; verified objective context only. |
| 43 | Galaxy Multi | ✅ | ✅ rolling pool v0.5 | ✅ | 🔴 final without quote | partial | 🟢 IMPROVED | Feed only production-valid new families; preserve same-book/freshness/edge/final-quote gates. |
| 44 | Galaxy SGP correlation | 🟠 components | 🔴 joint model | 🔴 | 🔴 | — | 🔴 | Explicit same-game joint/correlation model; never multiply same-game marginals. |
| 45 | Exact SGP quote | ⚫ | — | 🔴 | required | — | ⚫ EXTERNAL | Sportsbook/provider actual combined quote. |
| 46 | Calibration lifecycle | ✅ ledger/results | ✅ validators growing | 🟡 | ✅ existing FT only | partial | 🟢 IMPROVED | Standardize Brier/log-loss/ROI/CLV/OOS gates and promotion-review reports per family/version/archetype; never auto-change weights. |

## Completed engineering checkpoints #1–#10

### #1 FT Goals — v3.14
Passes SPORT-FIRST lambdas/score matrix/exact observed total ladder; no-vig/shrinkage/edge/EV; relative-strength shadow zero weight. Remaining: natural persistence detail, challenger OOS and advanced-input gaps.

### #2 Team Totals — v3.15
Passes team-specific Poisson O/U 0.5/1.5/2.5, observed markets only, fair/no-vig/EV diagnostics, research-only validator. Remaining: OOS/CLV/cross-league and integer/quarter settlement.

### #3 Correct Score — v3.16
Passes exact-score probabilities and observed exact-score pricing with NLL/multiclass-Brier/top-k validation. Remaining: larger OOS, price/CLV history, sparse-outcome shrinkage.

### #4 BTTS — v3.17
Passes independent YES/NO probability, exact market mapping and Brier/log-loss validator. Remaining: OOS/CLV/shrinkage and parent FT improvements.

### #5 1X2 — v3.18
Passes canonical H/D/A live research, relative-strength shadow, offline Dixon-Coles/prior challengers and model-selection report. No candidate may be selected across mismatched cohorts; same-fixture head-to-head is required. Remaining: OOS/CLV/final versioned selection/shrinkage.

### #6 Double Chance — v3.19
Passes 1X/X2/12 derivation and real DC mapping. DC selections overlap, so their three prices are never normalized directly; when possible fair DC is derived from same-book complete 1X2 no-vig. Remaining: parent approval + OOS/CLV/shrinkage.

### #7 DNB — v3.20
Passes `P(win|no draw)`, explicit draw push, real DNB mapping, paired conditional no-vig and push-aware EV `P(win)*price + P(draw) - 1`. Remaining: parent 1X2 approval, non-draw OOS/CLV/shrinkage/settlement verification.

### #8 Asian Handicap — v3.21
Passes score-margin model and exact settlement for integer/half/quarter lines. Quarter lines split into adjacent half-lines; model exposes win/push/loss fractions, fair price and settlement-aware EV. It deliberately does not invent a binary probability/no-vig transform for push/quarter markets. Remaining: observed-line OOS/CLV/calibration and stronger parent margin model.

### #9 1H Goals — v3.22
Passes a period-specific live hierarchical Poisson model sourced from a finalized-history registry, not FT probabilities. Prices exact observed 1H totals with integer/half/quarter settlement math. The registry is cached from `soccer-edge-state` and adds zero API-Football requests. Remaining: initial registry/natural persistence, OOS exact-line market calibration and true CLV.

### #10 2H Goals pregame — v3.23
Passes an independent period-specific pregame 2H model using finalized 2H rates and exact observed 2H totals. Explicitly tagged `PREGAME_ONLY_NOT_HALFTIME_CONDITIONED`. Remaining: registry/natural persistence, OOS/CLV/calibration. A live halftime model remains a separate #11 gap.

## Engineering rule

Natural validation is asynchronous and non-blocking for the next engineering point. A module is not called fully RESOLVED until its persisted natural state and stated calibration/promotion gates are satisfied.

## Next engineering target

#11 2H live / halftime — build a genuinely halftime-conditioned research model from verified HT score/game-state/red-card/shots/SOT context. It must never reuse or relabel the pregame 2H model.
