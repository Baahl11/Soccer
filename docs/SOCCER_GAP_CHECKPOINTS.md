# SOCCER EDGE ENGINE — GAP CHECKPOINT TRACKER

Last updated: 2026-09-17 CDMX

Purpose: keep one authoritative point-by-point roadmap so implementation can continue without waiting for one natural tick after every module.

## Checkpoint rules

Each module is tracked across five dimensions:

- DATA: required source/input exists and is verified enough for the stated use.
- MODEL: explicit SPORT-FIRST probability/projection model exists.
- LIVE: model/data is attached to upcoming fixtures during normal scheduler execution.
- PRODUCTION: allowed to affect canonical BET/LEAN classification.
- NATURAL VALIDATION: confirmed in persisted natural state after deployment.

Statuses:

- ✅ PASS = implemented and meets current criterion.
- 🟢 IMPROVED = materially better; still has stated remaining work.
- 🟡 LIVE RESEARCH = live probability/model output, decision weight 0 / not production-approved.
- 🟠 OFFLINE / BUILT = code/model exists but is not attached live.
- 🔵 CALIBRATING = live/built and accumulating OOS / Brier / log-loss / CLV before promotion.
- 🔴 GAP = model or integration does not yet exist.
- ⚫ EXTERNAL BLOCK = requires a source/provider/market capability not yet integrated.
- ⏳ NATURAL CHECK = implementation/deploy done but persisted natural-state evidence is still pending or incomplete.

A module may advance to the next engineering task before NATURAL VALIDATION is complete. Natural validation must still be recorded before calling the module fully RESOLVED.

## Current master map

| # | Module | DATA | MODEL | LIVE | PRODUCTION | NATURAL VALIDATION | Current checkpoint | Exactly what remains |
|---|---|---|---|---|---|---|---|---|
| 1 | FT Goals | ✅ | ✅ | ✅ | ✅ Tier-B constrained | ⏳ partial | 🟢 IMPROVED v3.14 | Natural state is v3.14.0, provider-call safety passed. Need persisted `ft_goals_intelligence` evidence / ladder visibility; continue accumulating OOS and challenger comparison. Advanced inputs remain separate gaps. |
| 2 | Team Totals | ✅ canonical home/away lambdas + retained team-total markets | ✅ Poisson O/U 0.5/1.5/2.5 research model | 🟡 v3.15 bridge | 🔴 | ⏳ | 🟡 LIVE RESEARCH v3.15 | Need natural persistence confirmation, ≥100 OOS for market comparison, ≥200 for actionable review, historical team-total prices/CLV and cross-league calibration. Integer/quarter lines remain unsupported until explicit settlement math exists. No Galaxy/production promotion yet. |
| 3 | Correct Score | ✅ canonical home/away lambdas + retained correct-score markets | ✅ independent-Poisson exact-score distribution + OOS validator | 🟡 v3.16 bridge | 🔴 | ⏳ | 🟡 LIVE RESEARCH v3.16 | Need natural persistence confirmation, ≥300 OOS for meaningful market comparison, ≥500 for actionable review, verified correct-score price/CLV history, validated shrinkage for sparse outcomes and cross-league stability. No Galaxy/production promotion yet. |
| 4 | BTTS | ✅ canonical score-matrix probability + observed YES/NO markets | ✅ explicit BTTS probability + OOS validator | 🟡 v3.17 bridge | 🔴 | ⏳ | 🟡 LIVE RESEARCH v3.17 | Need natural persistence confirmation, ≥150 OOS for market comparison, ≥300 for actionable review, verified BTTS price history/true CLV, validated shrinkage and stable calibration by probability bucket/league/data tier. No Galaxy/production promotion yet. |
| 5 | 1X2 | ✅ canonical H/D/A + observed three-way markets | ✅ canonical + live relative-strength shadow + offline Dixon-Coles/prior challengers + selection report | 🟡 v3.18 bridge | 🔴 | ⏳ | 🟡 LIVE RESEARCH v3.18 | Need natural persistence confirmation; ≥200 matched-method OOS before challenger review, ≥300 same-fixture head-to-head for shortlisted models, ≥400 before actionable review; verified historical 1X2 prices/true CLV; final model selection and market-shrinkage calibration. No Galaxy/production promotion yet. |
| 6 | Double Chance | ✅ | 🟠 mathematically derivable from validated 1X2 | 🟡 research | 🔴 | partial | 🟠 NEXT | Derive 1X/X2/12 strictly from the accepted 1X2 probability surface; exact observed-market mapping/no-vig/validation; stay research-only while 1X2 is not production-approved. |
| 7 | DNB | ✅ | 🔴 | 🔴 | 🔴 | — | 🔴 | Build conditional no-draw probability from accepted 1X2/score matrix and map exact prices. |
| 8 | Asian Handicap | ✅ score matrix foundation | 🔴 | 🔴 | 🔴 | — | 🔴 | Build margin distribution pricing including quarter lines/push/half-win mechanics; OOS calibration. |
| 9 | 1H Goals | ✅ historical HT scores | ✅ walk-forward Poisson v0.1 | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach upcoming-match predictions live; map exact observed 1H lines; then 100 OOS market comparison / 200 actionable review. |
| 10 | 2H Goals pregame | ✅ historical 2H targets | ✅ walk-forward model | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach live pregame; exact-line mapping/calibration. Never confuse with halftime-conditioned live model. |
| 11 | 2H live / halftime | 🟠 some halftime stats | 🔴 | 🔴 | 🔴 | — | 🔴 | Build separate HT-conditioned model using score/game-state/red cards/shots etc.; no reuse of pregame 2H model. |
| 12 | Corners FT | ✅ postgame corners | ✅ walk-forward baseline + formation challenger | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach live; map exact 8.5/9.5/10.5 etc.; add stronger territorial features when available; 150 OOS + 100 formation-adjusted + CLV gate. |
| 13 | Team Corners | ✅ team corner history | 🟠 lambdas can be extended | 🔴 | 🔴 | — | 🔴 | Explicit home/away team-corner probability distribution + exact-line market mapping/calibration. |
| 14 | Cards total | ✅ postgame yellows/fouls | ✅ baseline + optional referee scale | 🟠 | 🔴 | — | 🟠 OFFLINE | Attach live; referee-history enrichment; exact sportsbook scoring-rule mapping; 200 OOS/100 referee-adjusted + CLV. |
| 15 | Team Cards | ✅ team yellow history | 🟠 baseline components exist | 🔴 | 🔴 | — | 🔴 | Produce per-team card lambdas and exact team-card line probabilities; calibrate. |
| 16 | Red Cards | ✅ recorded separately | 🔴 | 🔴 | 🔴 | — | 🔴 | Separate low-frequency model; do not mix with yellow-card count without book-specific scoring rule. |
| 17 | Referee | 🟢 fixture assignment captured forward | 🟠 cards baseline can use referee | 🟠 partial | 🔴 | partial | 🟠 | Historical referee enrichment is sparse pre-v3.9; build forward or integrate trustworthy history; assignment verification at match time. |
| 18 | Formations | ✅ lineups/formation capture | ✅ formation intelligence research | 🟠 partial | 🔴 | partial | 🟠 | Automatically attach current confirmed formation pair to live research models; quantify only after OOS lift. |
| 19 | Coaches | ✅ coach from lineup | 🔴 regime model | 🟠 identity only | 🔴 | partial | 🟠 | Persist coach tenure/regime changes; before/after tactical/goal effects; substitution/rotation behavior model. |
| 20 | XI | ✅ | ✅ verification logic | ✅ | ✅ as availability gate | ✅ historical | 🟢 IMPROVED | Improve persistence/re-run on changes and source quality; no new probability model required for identity itself. |
| 21 | Goalkeeper | ✅ starter capture | 🔴 impact model | 🟠 confirmed/not-confirmed | 🔴 | partial | 🟠 | Build shot-stopping / saves / concession-impact model with verified starter and historical performance. |
| 22 | Injuries / suspensions | 🟢 provider support | 🔴 player-impact model | 🟠 verification partial | ✅ as availability block only | partial | 🟠 | Quantify impact by player/role/minutes; stronger official verification for material absences. |
| 23 | Player trends | ✅ selective capture | 🟠 descriptive L5/L10/L20 | 🟡 research | 🔴 | partial | 🟡 DESCRIPTIVE | Convert to probabilistic role/minutes-adjusted models; current decision weight stays 0. |
| 24 | Shots props | 🟢 player shots capture | 🔴 | 🔴 | 🔴 | — | 🔴 | Minutes/role + shots/90 + opponent/formation distribution; exact threshold probability. |
| 25 | SOT props | 🟢 player SOT capture | 🔴 | 🔴 | 🔴 | — | 🔴 | P(1+/2+/3+ SOT) with minutes, role, opponent shot suppression and starting status. |
| 26 | Goalscorer | 🟠 goals/minutes available | 🔴 | 🔴 | 🔴 | — | 🔴 | Player xG/share or defensible proxy, minutes, penalty role, opponent/GK; calibration. |
| 27 | Assists | 🟠 assists/key passes | 🔴 | 🔴 | 🔴 | — | 🔴 | Chance creation/xA-quality input + minutes + teammate finishing model. |
| 28 | GK Saves | 🟠 GK identity + some player stats | 🔴 | 🔴 | 🔴 | — | 🔴 | Opponent SOT projection × starter save expectation; exact saves-line probabilities. |
| 29 | Player Cards | 🟠 cards/fouls context partial | 🔴 | 🔴 | 🔴 | — | 🔴 | Position/role/fouls/opponent/referee/minutes model; exact card prop mapping. |
| 30 | xG/xGA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Integrate stable legal advanced-data source; never infer xG from goals. |
| 31 | npxG/npxGA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Same source requirement as xG plus penalty exclusion. |
| 32 | PPDA | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Integrate provider and normalize definitions across competitions. |
| 33 | Field Tilt | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Provider/source integration and consistent territorial definition. |
| 34 | Box Entries | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Provider/source integration. |
| 35 | Big Chances | 🟠 provider-dependent | 🔴 feature model | 🔴 | 🔴 | — | ⚫/🟠 | Find consistent live/historical source before weighting. |
| 36 | Set Pieces | 🟠 corners/fouls exist | 🔴 | 🔴 | 🔴 | — | 🔴 | Set-piece attack/defense rates, aerial mismatch, ideally set-piece xG; then effect on goals/corners. |
| 37 | Tactical Style | 🟢 formation/trend signals | 🟠 descriptive | 🟡 partial | 🔴 | partial | Formal classifier for press/block/transition/width/possession; OOS feature lift before decision weight. |
| 38 | Rest | ✅ fixtures | 🔴 feature only | 🔴 | 🔴 | — | 🟠 EASY | Calculate days since last match and expose live; validate material thresholds. |
| 39 | Congestion | ✅ fixtures | 🔴 feature only | 🔴 | 🔴 | — | 🟠 EASY | Matches/minutes in 7/14/21-day windows + rotation pressure. |
| 40 | Travel | 🟠 venue/country | 🔴 | 🔴 | 🔴 | — | 🔴 | Distance/timezone/altitude/logistics when material; avoid noise for local fixtures. |
| 41 | Weather | ⚫ | 🔴 | 🔴 | 🔴 | — | ⚫ EXTERNAL | Live forecast by venue/kickoff: wind/rain/temp/humidity/severe; materiality logic. |
| 42 | Competition Context | ✅ league/round/fixtures | 🟠 structured only | 🟡 partial | 🔴 | partial | Table/aggregate/qualification math; verified objective incentives only; never invent motivation. |
| 43 | Galaxy Multi | ✅ market-backed FT/accepted legs | ✅ rolling leg pool v0.5 | ✅ | 🔴 final without quote | partial | 🟢 IMPROVED | Feed more production-valid families; preserve same-book, one-leg-per-fixture, freshness, edge and final-quote gates. |
| 44 | Galaxy SGP correlation | 🟠 component models | 🔴 joint correlation model | 🔴 | 🔴 | — | 🔴 | Explicit same-game joint model; never multiply same-game marginals. |
| 45 | Exact SGP quote | ⚫ provider/book dependent | — | 🔴 | required | — | ⚫ EXTERNAL | Integrate sportsbook/provider that exposes actual combined quote; component product remains reference only. |
| 46 | Calibration lifecycle | ✅ ledger/results | ✅ validators partly built | 🟡 | ✅ for existing FT only | partial | 🟢 IMPROVED | Standardize Brier/log-loss/ROI/CLV/OOS gates per family/version/archetype and automatic promotion-review reports, without auto-changing weights. |

## #1 FT Goals checkpoint — v3.14

### Already passes
- SPORT FIRST raw projection exists before market inspection.
- Home/away/total goal lambdas exist.
- Score matrix and O1.5/O2.5/O3.5 probabilities exist.
- Canonical market evaluation uses exact observed lines and prices, no-vig fair probability when both sides are present, market shrinkage, probability edge and EV.
- Availability/data-tier/lineup gates remain intact.
- Relative-strength challenger remains research-only with decision weight 0.
- v3.14 adds FT Goals Intelligence / observed-line ladder without changing canonical weights, thresholds, stake or provider-request policy.

### Still missing / not allowed to claim resolved
- Persisted natural-state evidence for the new `ft_goals_intelligence` object still needs explicit confirmation.
- Relative-strength challenger still needs enough OOS observations before promotion.
- Advanced xG/GK/set-piece/rest/weather inputs remain separate gaps.

## #2 Team Totals checkpoint — v3.15

### Already passes
- Uses canonical home/away lambdas and explicit Home/Away O/U 0.5/1.5/2.5 probabilities.
- Exact observed team-total markets only; fair price/no-vig/edge/EV research diagnostics.
- Integer/quarter lines rejected until settlement math exists.
- `actionable=false`, `decision_weight=0`; no BET/LEAN/Galaxy.
- Dedicated OOS validator integrated.

### Still missing / not allowed to claim resolved
- Natural persistence confirmation.
- ≥100 OOS market review / ≥200 actionable review.
- Historical prices/CLV and cross-league calibration.
- Integer/quarter-line settlement support.

## #3 Correct Score checkpoint — v3.16

### Already passes
- Exact-score probabilities derived only from canonical home/away lambdas.
- Correct Score/Exact Score odds retained without extra provider requests.
- Exact observed score selections only; grouped buckets separated.
- Research probability/fair price/price/edge/EV diagnostics.
- `actionable=false`, `decision_weight=0`; no BET/LEAN/Galaxy.
- Dedicated NLL/multiclass-Brier/top-k validator integrated.

### Still missing / not allowed to claim resolved
- Natural persistence confirmation.
- ≥300 OOS market review / ≥500 actionable review.
- Historical correct-score prices/CLV and sparse-outcome shrinkage.
- Cross-league calibration.

## #4 BTTS checkpoint — v3.17

### Already passes
- Uses the existing canonical score-matrix `raw_btts_yes_prob`; market prices never create the sporting probability.
- Exposes P(YES), P(NO) and independent model fair prices.
- Compares only to observed BTTS YES/NO markets.
- Computes no-vig fair market probability only when both YES and NO are present for the same bookmaker market.
- Research rows expose model probability, fair price, breakeven, no-vig market probability, raw edge and raw EV.
- `actionable=false`, `decision_weight=0`, classification `RESEARCH_ONLY`; cannot produce canonical BET/LEAN or enter Galaxy.
- Adds zero provider requests and changes no canonical weights, thresholds, stakes or BET logic.
- Dedicated historical validator measures Brier, log-loss, observed-vs-predicted rate, probability buckets, competition and data tier.
- History pipeline runs the BTTS validator automatically.

### Still missing / not allowed to claim resolved
- Natural v3.17 persistence confirmation is pending and non-blocking.
- Need at least 150 OOS fixtures before meaningful market-comparison review and 300 before actionable promotion review.
- Need verified historical BTTS prices and true CLV evidence.
- Need validated market shrinkage and stable calibration across probability buckets, leagues and data tiers.
- BTTS remains downstream of the parent FT-goals model, so xG/GK/set-piece/rest/weather limitations remain inherited.

## #5 1X2 checkpoint — v3.18

### Already passes
- Uses canonical SPORT-FIRST H/D/A probabilities already created before market inspection; sportsbook odds never create the sporting projection.
- Exposes canonical H/D/A probabilities and fair prices live as a dedicated research surface.
- Maps only observed three-way 1X2 markets and computes no-vig probabilities only when HOME/DRAW/AWAY are all available for the same bookmaker market.
- Relative-strength challenger is exposed live only when its existing shadow projection is present; `decision_weight=0` and it cannot replace canonical output.
- Dixon-Coles and class-prior calibration challengers remain offline OOS validators and are explicitly not applied live.
- Consolidated model-selection report evaluates each challenger first against its own matched baseline and refuses to rank models across different fixture cohorts from summary metrics alone.
- Model-selection report requires same-fixture head-to-head before any challenger can be selected for promotion review.
- `actionable=false`, `decision_weight=0`; no canonical BET/LEAN/Galaxy promotion.
- Adds zero provider requests and changes no canonical weights, thresholds, stakes or BET logic.

### Still missing / not allowed to claim resolved
- Natural v3.18 persistence confirmation is pending and non-blocking.
- Need at least 200 OOS inside each matched challenger report before it can enter formal review.
- Need at least 300 same-fixture head-to-head observations for shortlisted challengers and 400 OOS before actionable review.
- Need verified historical 1X2 prices and true CLV evidence.
- Need a versioned final model-selection decision plus validated market shrinkage and stable H/D/A calibration by probability bucket/competition.
- Until those gates pass, Double Chance/DNB/Asian Handicap derived from 1X2 must also remain research-only.

### Engineering rule going forward

Do not block the next module on the natural-validation check. Record the check as pending and continue. When later natural state supplies evidence, update this file asynchronously.

## Next engineering target

#6 Double Chance — derive 1X/X2/12 strictly from the canonical 1X2 surface, map only exact observed Double Chance prices, validate calibration, and preserve research-only status until the parent 1X2 model is production-approved.
