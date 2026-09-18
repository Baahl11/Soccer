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
| 18 | Formations | ✅ confirmed XI formation pair + historical formation report | ✅ matchup profile + OOS residual feature gate | 🟡 v3.31 feature-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH FEATURE | Natural persistence; ≥100 aggregate OOS with Brier+log-loss lift; matchup n≥8 before feature candidacy; zero decision weight until versioned promotion. |
| 19 | Coaches | ✅ current coach from confirmed XI + historical regime registry | ✅ descriptive regime/tenure/change context | 🟡 v3.32 context-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH CONTEXT | Natural registry persistence; substitution/rotation behavior; OOS coach-feature lift; before/after differences remain non-causal. |
| 20 | XI | ✅ confirmed starters/GK/formation | ✅ verification + persistent XI fingerprint/change detector | ✅ v3.33 | ✅ existing availability gate only | ⏳ new detector | 🟢 IMPROVED v3.33 | Natural persistence of fingerprints; recheck behavior after material change; player-specific impact remains separate. |
| 21 | Goalkeeper | 🟢 confirmed starter + saves/conceded fields retained + low-priority finalized player capture | ✅ descriptive GK profile; true impact model intentionally NOT claimed | 🟡 v3.34 context-only | 🔴 | ⏳ | 🟡 DATA/PROFILE IMPROVED v3.34 | Accumulate finalized GK samples; verify minutes/substitutions; OOS feature-lift test; shot-quality/PSxG remains external. No λ adjustment until validated. |
| 22 | Injuries / suspensions | 🟢 provider report + historical role/minutes exposure + confirmed-XI conflict check | ✅ research-only role/materiality framework; true player→team impact intentionally NOT claimed | 🟡 v3.35 context/recheck-only | ✅ existing availability gate only; research recheck flag adds no probability weight | ⏳ | 🟡 LIVE RESEARCH CONTEXT v3.35 | Natural persistence; independent official/team verification; OOS player→team impact evidence. Replacement quality remains NOT_MODELED. |
| 23 | Player trends | ✅ persisted L5/L10/L20 + exact observed-minute exposure + role registry | ✅ v3.36 shrunk role/minutes + Gamma-Poisson per-90 count-rate profiles | 🟡 v3.36 research-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH MODEL v3.36 | Accumulate natural player samples; ≥500 OOS player-games for rate review and ≥1500 before any prop dependency; validate minutes MAE/dispersion by position/competition. No market-line probabilities here. |
| 24 | Shots props | ✅ confirmed starter + persisted shots/minutes + opponent shots-allowed trend | ✅ v3.37 Gamma-Poisson/Negative-Binomial predictive distribution; O/U 0.5–5.5 exact half-line probabilities | 🟡 v3.37 research-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH MODEL v3.37 | Natural persistence; ≥500 OOS count/line review, ≥1000 market review, ≥2000 actionable review; attach observed sportsbook line/price + CLV. Formation numeric factor stays 1.0 until shots-specific residual OOS lift. |
| 25 | SOT props | ✅ confirmed starter + persisted SOT/minutes + opponent SOT-allowed trend | ✅ v3.38 Gamma-Poisson/Negative-Binomial predictive distribution; O/U 0.5–3.5 exact half-line probabilities | 🟡 v3.38 research-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH MODEL v3.38 | Natural persistence; ≥500 OOS count/line review, ≥1000 market review, ≥2000 actionable review; attach observed sportsbook SOT line/price + CLV. Formation numeric factor stays 1.0 until SOT-specific residual OOS lift. |
| 26 | Goalscorer | ✅ confirmed starter + persisted goals/minutes + opponent goals-allowed trend | ✅ v3.39 Gamma-Poisson/Negative-Binomial anytime + 2+ goal probabilities | 🟡 v3.39 research-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH MODEL v3.39 | Natural persistence; ≥1000 OOS anytime review, ≥2000 market review, ≥4000 actionable review; verified penalty role + sportsbook scorer price/CLV required. GK/formation modifiers remain 1.0 until defensible OOS evidence. |
| 27 | Assists | ✅ confirmed starter + persisted assists/minutes + team/rival goal environment | ✅ v3.40 Gamma-Poisson/Negative-Binomial 1+/2+ assist probabilities | 🟡 v3.40 research-only | 🔴 | ⏳ | 🟡 LIVE RESEARCH MODEL v3.40 | Natural persistence; ≥1000 OOS assist review, ≥2000 market review, ≥4000 actionable review; xA/chance-quality and observed assist price/CLV remain required before promotion. |
| 28 | GK Saves | 🟠 capture/persistence path fixed; confirmed GK + team/opponent SOT available, but 0 finalized GK saves+conceded samples currently persisted | ✅ v3.41 model built: projected opponent SOT × shrunk save-result proxy; Poisson O/U 1.5–5.5 | 🟠 dormant/data-blocked | 🔴 | ⏳ data accumulation | 🟠 BUILT / DATA-BLOCKED v3.41 | Accumulate natural finalized GK samples first; then ≥300 OOS save-line review, ≥750 market review, ≥1500 actionable review; observed save price/CLV required. PSxG/shot-quality remains external. |
| 29 | Player Cards | 🟠 capture path fixed; yellow/red fields retained forward, but 0 finalized yellow-card player samples currently persisted | ✅ v3.42 Gamma-Poisson/Negative-Binomial player-booked + 2+ yellow probabilities with shrunk team-discipline environment | 🟠 dormant/data-blocked | 🔴 | ⏳ data accumulation | 🟠 BUILT / DATA-BLOCKED v3.42 | Accumulate natural finalized player-yellow samples; ≥1000 OOS booked review, ≥2000 market review, ≥4000 actionable review; verified sportsbook player-card price + explicit card-scoring rule + CLV required. Referee numeric effect remains disabled until player-card-specific OOS lift. |
| 30 | xG/xGA | 🟠 API-Football finalized /fixtures/statistics exposes expected_goals; zero-extra-call POSTGAME capture path now persists it, but Soccer Edge historical sample starts accumulating only from v3.43.2 | ✅ v3.43.2 historical xGF + opponent xGA shrinkage architecture with same-fixture leakage guard | 🟠 capture live / projection dormant until ≥5 team samples | 🔴 | ⏳ | 🟠 BUILT / DATA-ACCUMULATING v3.43.2 | Accumulate natural finalized xG samples, validate registry persistence, then ≥500 OOS feature review, ≥1000 λ challenger, ≥2000 production review. Realized same-fixture xG is never used pregame; internal lambdas are never xG. |
| 31 | npxG/npxGA | ⚫ Approved-source audit complete: neither current API-Football nor GalaxyParlay contract exposes verified npxG | 🔴 native/shot-level non-penalty xG model unavailable | 🟠 v3.44 source/penalty-exclusion guard attached and data-blocked | 🔴 | ⏳ | 🟠 LIVE GUARD / APPROVED-SOURCE BLOCKED v3.44 | Require native npxG or verified shot-level/penalty xG provenance from API-Football/GalaxyParlay. Never subtract a fixed penalty-xG constant or penalty goal count. Then OOS/calibration gates before any weight. |
| 32 | PPDA | ⚫ Approved-source audit complete: current API-Football/Galaxy contracts do not expose normalized PPDA inputs/definition | 🔴 true PPDA model unavailable; total-pass/tackle proxy prohibited | 🟠 v3.45 source/definition guard attached and data-blocked | 🔴 | ⏳ | 🟠 LIVE GUARD / APPROVED-SOURCE BLOCKED v3.45 | Require approved source with auditable spatial zone + defensive-action denominator. Legacy ppda placeholder is not data. No proxy relabeling; then OOS lift/calibration before weighting. |
| 33 | Field Tilt | ⚫ Approved-source audit complete: API-Football exposes total possession but not the territorial inputs/definition required for Field Tilt; Galaxy contract exposes no Field Tilt | 🔴 true Field Tilt unavailable; possession relabel prohibited | 🟠 v3.46 source/definition guard attached and data-blocked | 🔴 | ⏳ | 🟠 LIVE GUARD / APPROVED-SOURCE BLOCKED v3.46 | Require approved territorial touch/pass/possession definition and consistent historical/live source. Total possession is not Field Tilt; OOS lift/calibration required before weighting. |
| 34 | Box Entries | ⚫ Approved-source audit complete: API-Football may expose shots inside box but not explicit box-entry events/metric; Galaxy contract exposes no Box Entries | 🔴 true box-entry model unavailable; shots/crosses/corners relabel prohibited | 🟠 v3.47 source/definition guard attached and data-blocked | 🔴 | ⏳ | 🟠 LIVE GUARD / APPROVED-SOURCE BLOCKED v3.47 | Require approved provider-defined penalty-area entry event/metric plus counting rules and historical/live consistency. Then OOS lift/calibration before weighting. |
| 35 | Big Chances | 🟠 v3.48 zero-extra-call POSTGAME capture accepts only an explicit API-Football Big Chances field when present; no verified Soccer Edge sample persisted yet | 🔴 no predictive Big Chances model until consistent history exists | 🟠 provider-dependent capture live / pregame guard data-blocked | 🔴 | ⏳ | 🟠 CAPTURE BUILT / DATA-ACCUMULATING v3.48 | Observe natural provider coverage, persist verified samples, then build historical rates/model and OOS lift/calibration. Shots/xG/goals are never proxies for Big Chances. |
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

#1 v3.14 FT Goals · #2 v3.15 Team Totals · #3 v3.16 Correct Score · #4 v3.17 BTTS · #5 v3.18 1X2 · #6 v3.19 Double Chance · #7 v3.20 DNB · #8 v3.21 Asian Handicap · #9 v3.22 1H Goals · #10 v3.23 2H pregame · #12 v3.24 Corners FT · #13 v3.25 Team Corners · #14 v3.26 Cards total · #15 v3.27 Team Cards · #11 v3.28 2H halftime · #16 v3.29 Red Cards · #17 v3.30 Referee · #18 v3.31 Formations · #19 v3.32 Coaches · #20 v3.33 XI · #21 v3.34 Goalkeeper data/profile · #22 v3.35 Injuries/suspensions · #23 v3.36 Player trends · #24 v3.37 Shots props · #25 v3.38 SOT props · #26 v3.39 Goalscorer · #27 v3.40 Assists · #28 v3.41 GK Saves (built/data-blocked) · #29 v3.42 Player Cards (built/data-blocked) · #30 v3.43.2 API-Football historical xG/xGA (built/data-accumulating) · #31 v3.44 npxG/npxGA approved-source guard · #32 v3.45 PPDA approved-source guard · #33 v3.46 Field Tilt approved-source guard · #34 v3.47 Box Entries approved-source guard · #35 v3.48 Big Chances provider-dependent capture guard.

### #21 Goalkeeper — v3.34
- `fixtures/players` compact now retains provider fields `goals.saves` and `goals.conceded` when supplied.
- Player-trend history now produces GK L5/L10/L20 averages and a `save_result_proxy = saves/(saves+goals_conceded)` explicitly labeled NOT PSxG / not shot-quality adjusted / not guaranteed complete SOT faced.
- A goalkeeper profile registry is built from persisted player history; confirmed starting GK IDs are matched live to those profiles.
- Canonical goal lambda adjustment remains exactly 0.0; `decision_weight=0`.
- Up to two finalized `/fixtures/players` captures may be added postgame only after all normal work, only in NORMAL budget mode and only while keeping at least three calls free. Otherwise capture is deferred.
- True shot-stopping impact remains unresolved until sufficient finalized samples plus OOS feature-lift evidence exist. xG/PSxG remains an external-data gap.

### #22 Injuries / suspensions — v3.35
- Current provider injury/suspension reports are profiled as verified provider context; unsupported or missing payloads remain NOT_VERIFIED.
- Historical L10 minutes/sample exposure is used only as descriptive role/materiality context; it is explicitly NOT player impact.
- Reported players are cross-checked against confirmed starting XI; a report-vs-XI conflict raises a recheck flag instead of silently changing probabilities.
- Replacement quality is explicitly NOT_MODELED and quantified player-to-team goal impact remains null.
- Independent official/team-source verification is still not automated and remains a documented gap.
- `actionable=false`, `decision_weight=0`; canonical model weights, thresholds, bet logic and availability confidence are unchanged.
- v3.35 adds zero provider requests and preserves the normal scheduler/budget path.

### #23 Player trends — v3.36
- Player trend history schema is now 1.5.0 and retains exact observed exposure totals: minutes, shots, SOT, goals, assists and player yellow/red cards, plus per-90 rates for L5/L10/L20.
- A dedicated research registry combines the existing shrunk role/minutes model with exposure-normalized Gamma-Poisson count-rate shrinkage.
- Position priors are used only with sufficient exposure; otherwise the model falls back to global priors.
- Confirmed starting XI sets live start probability to 1.0; historical start/minutes outputs remain diagnostic context.
- Expected counts for a confirmed starter are rate × expected minutes research expectations only. They are NOT sportsbook threshold probabilities.
- Shots/SOT/goals/assists line probabilities remain separate future modules (#24–#27); v3.36 cannot create a prop bet or Galaxy leg.
- `actionable=false`, `decision_weight=0`; canonical model weights, thresholds, bet logic, tier and stake remain unchanged.
- v3.36 adds zero provider requests during the live tick; registry generation is asynchronous from persisted history.

### #24 Shots props — v3.37
- Confirmed starters use the v3.36 shrunk shots-rate posterior plus expected starter minutes; no unconfirmed player is treated as a starter.
- The posterior Gamma rate is integrated over future minutes to produce a Negative-Binomial predictive shot-count distribution.
- Exact fair research probabilities are emitted for O/U 0.5, 1.5, 2.5, 3.5, 4.5 and 5.5 shots.
- Opponent adjustment uses only persisted L10 shots allowed relative to the global shots baseline, shrunk toward neutral with 12 pseudo-matches and clipped to 0.75–1.25.
- Team trend history now retains opponent shots/SOT explicitly; missing or thin opponent data forces a neutral 1.0 modifier.
- Confirmed formation context is attached, but its numerical modifier remains exactly 1.0 until a shots-specific residual walk-forward test shows OOS lift. Raw formation averages cannot alter a player probability.
- No sportsbook player-prop line/price is currently attached, so edge/EV are null and the module cannot create BET/LEAN/Galaxy legs.
- `actionable=false`, `decision_weight=0`; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged.
- v3.37 adds zero provider requests during the live tick.

### #25 SOT props — v3.38
- Confirmed starters use the v3.36 shrunk role/minutes profile plus the player's persisted SOT Gamma-Poisson rate.
- Opponent adjustment uses finalized team avg_opponent_sot versus global avg_team_sot, shrunk toward neutral and clipped to limit overreaction.
- Predictive Negative-Binomial probabilities are emitted for SOT half-lines 0.5/1.5/2.5/3.5 with no-vig fair decimal prices.
- Formation is retained as context only; numeric multiplier remains exactly 1.0 until SOT-specific OOS residual lift exists.
- No sportsbook player-SOT price is currently attached, so EV/CLV and BET/LEAN/Galaxy eligibility remain blocked.
- Structural validator checks probability bounds, monotonicity and over/under complementarity; this is not a substitute for OOS performance validation.
- actionable=false, decision_weight=0; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged.
- v3.38 adds zero provider requests during the live tick.

### #26 Goalscorer — v3.39
- Confirmed starters use the v3.36 shrunk role/minutes profile plus the player's persisted goal Gamma-Poisson rate.
- Opponent adjustment uses last-10 goals allowed relative to the new global team-goals baseline, shrunk toward neutral and clipped.
- Predictive Negative-Binomial output includes anytime goal probability, P(2+ goals), expected goals and no-vig fair anytime decimal price.
- Current penalty-taker role is not verified, so penalty factor remains exactly 1.0.
- Confirmed goalkeeper profile is attached as context, but GK factor remains exactly 1.0 because PSxG/shot-quality impact is unavailable and unvalidated.
- Formation is context-only with multiplier 1.0 until goalscorer-specific OOS lift exists.
- No sportsbook scorer price is attached, so EV/CLV and BET/LEAN/Galaxy eligibility remain blocked.
- actionable=false, decision_weight=0; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged.
- v3.39 adds zero provider requests during the live tick.

### #27 Assists — v3.40
- Confirmed starters use the v3.36 shrunk role/minutes profile plus the player's persisted assist Gamma-Poisson rate.
- Scoring-environment adjustment combines recent team goals-for and opponent goals-against relative to the global goals baseline, with neutral shrinkage and clipping.
- Predictive Negative-Binomial output includes P(1+ assist), P(2+ assists), expected assists and no-vig fair 1+ assist price.
- xA and calibrated chance-quality/key-pass conversion are unavailable, so their numeric factor remains exactly 1.0 and the gap stays explicit.
- No sportsbook assist price is attached, so EV/CLV and BET/LEAN/Galaxy eligibility remain blocked.
- actionable=false, decision_weight=0; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged.
- v3.40 adds zero provider requests during the live tick.

### #28 GK Saves — v3.41
- Model code is built, but deployment status is DATA-BLOCKED: the current state has 101 goalkeeper profiles and 0 with finalized saves+goals-conceded counts.
- Root cause found: `postgame_player_stats` existed in the worker payload but was omitted by the Scheduler compact persistence contract; the workflow is now fixed so future natural POSTGAME captures are retained.
- Goalkeeper profile schema retains exact saves and goals-conceded counts once those samples arrive.
- Opponent SOT projection combines opponent recent SOT attack with goalkeeper-team recent SOT allowed, shrunk toward the global SOT baseline and clipped.
- Once a GK has data, save probability is saves/(saves+goals_conceded) shrunk toward a global proxy; expected saves then feed Poisson half-lines 1.5–5.5.
- This is explicitly NOT PSxG and not shot-quality adjusted; no goalkeeper quality claim is made from the proxy.
- Structural validation reports BLOCKED_INSUFFICIENT_FINALIZED_GK_DATA when zero samples exist; it only reports PASS after real GK samples can actually be checked, and FAIL only for a mathematical inconsistency.
- No sportsbook save price is attached, so EV/CLV and BET/LEAN/Galaxy eligibility remain blocked.
- actionable=false, decision_weight=0; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged.

### #29 Player Cards — v3.42
- `fixtures/players` compact now retains provider player `cards.yellow` and `cards.red` separately in both research and low-priority finalized POSTGAME captures.
- Player-trend L5/L10/L20 history now carries exact yellow/red counts and yellow-card hit/rate fields; the probabilistic registry adds a shrunk `yellow_cards` Gamma-Poisson rate only.
- Red cards are deliberately excluded from the player-card probability target. No sportsbook card-points equivalence is assumed.
- Confirmed starters use expected starter minutes × shrunk player yellow-card rate; a venue-role team discipline factor is shrunk toward neutral and clipped to 0.75–1.25.
- Predictive Negative-Binomial output is prepared for player-booked (1+ yellow) and 2+ yellow outcomes with no-vig fair research prices.
- Referee identity may be carried as context, but its numeric multiplier remains exactly 1.0 until player-card-specific OOS evidence supports a residual effect.
- Current natural state is DATA-BLOCKED: the history validator found 0 profiles with finalized player-yellow-card exposure. This is expected because the fields were not persisted before v3.42.
- Structural validation therefore reports `BLOCKED_INSUFFICIENT_FINALIZED_PLAYER_CARD_DATA`, not PASS; future natural POSTGAME samples unlock probability validation automatically.
- No sportsbook player-card price or explicit bookmaker card-scoring rule is attached, so EV/CLV and BET/LEAN/Galaxy eligibility remain blocked.
- actionable=false, decision_weight=0; canonical model weights, thresholds, tier, stake and bet eligibility remain unchanged; v3.42 adds zero live provider requests.

### #30 xG/xGA — v3.43.2
- Correction completed: Sportmonks is not part of the approved architecture and its importer/registry path was removed.
- The existing GalaxyParlay codebase confirms the approved provider path: API-Football /fixtures/statistics can return expected_goals, and Galaxy already has get_fixture_xg() to extract it.
- Galaxy's own leakage test explicitly blocks that realized xG for unfinished fixtures and only permits the API-Football actual-xG path after a fixture is finished. Soccer Edge now follows the same rule.
- Soccer Edge POSTGAME already requests /fixtures/statistics; v3.43.2 extracts expected_goals from that existing response, adding zero provider requests.
- The scheduler now persists a compact postgame_xg_observation with fixture/team IDs and home/away xG only when both provider values are present.
- The daily history workflow builds team L5/L10/L20 xGF/xGA from those finalized observations. xGA is opponent verified xG from the same completed fixture.
- Pregame research projection combines shrunk own historical xGF with opponent historical xGA relative to the global historical xG baseline; minimum five samples per side are required.
- Same-fixture realized xG leakage is prohibited. Internal Soccer Edge/Galaxy goal lambdas, goals, shots and scorelines are never relabeled as provider xG.
- Historical data is currently accumulating because earlier Soccer Edge compact history did not persist the required postgame xG observation.
- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.
- Promotion still requires ≥500 OOS feature-lift fixtures, ≥1000 lambda-challenger fixtures, ≥2000 production-review fixtures plus league/sample calibration.
### #31 npxG/npxGA — v3.44
- A live zero-call source/definition guard is built on top of corrected v3.43.1 and uses only API-Football + GalaxyParlay.
- Current approved contracts do not expose native verified npxG, so DATA remains externally blocked and no numeric npxG model is claimed.
- API-Football penalty goal/missed-penalty events are useful event facts but are not sufficient to identify the xG mass of each penalty or to reconstruct shot-level non-penalty xG.
- Fixed penalty-xG subtraction is prohibited; subtracting a constant such as 0.76/0.78 per penalty would manufacture a provider-independent metric and is not accepted.
- Subtracting penalty goal counts from xG is also prohibited because goals are outcomes, not expected-goal mass.
- A future implementation may activate only if API-Football or GalaxyParlay exposes native npxG or verified shot-level/penalty xG provenance with an auditable definition.
- npxGA is defined only as opponent verified npxG from the same fixture.
- v3.44 attaches this blocked state to normal eligible scheduler events and match-intelligence areas with zero additional provider requests.
- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.
- Natural persistence must confirm the guard after deployment; predictive/OOS gates begin only after real approved-source npxG data exists.

### #32 PPDA — v3.45
- API-Football + GalaxyParlay remain the only approved data sources for this module.
- Current API-Football fixture statistics expose possession, passes, tackles/fouls and related match facts, but not a normalized PPDA metric or the spatial-zone event data needed to reconstruct PPDA faithfully.
- GalaxyParlay currently does not expose a persisted PPDA field in its Sports Edge contract.
- Legacy tactical_analysis code contains a defensive read of stats['ppda'] when supplied by a caller; no approved ingestion path populates that field, so it is explicitly treated as a placeholder rather than verified data.
- v3.45 attaches a zero-call guard to eligible live events and match-intelligence areas so PPDA stays visibly NOT MODELED instead of silently becoming 0 or a synthetic proxy.
- Whole-match passes divided by tackles/fouls/defensive actions is explicitly prohibited from being labeled PPDA because the required provider zone/action definition is missing.
- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.
- Activation requires an approved source with auditable PPDA definition, historical/live consistency and then OOS feature-lift/calibration evidence.
### #33 Field Tilt — v3.46\n- API-Football + GalaxyParlay remain the only approved data sources for this module.\n- API-Football provides whole-match ball possession, but that is not a territorial Field Tilt measure and cannot be relabeled as one.\n- The current GalaxyParlay Sports Edge contract does not expose a persisted Field Tilt metric or the required territorial touch/pass inputs.\n- v3.46 attaches a zero-call guard to eligible live events and match-intelligence areas so the gap remains explicit.\n- Activation requires an auditable territorial-zone definition, a consistent denominator (for example touches/passes/possession within that provider-defined zone), and historical/live compatibility from an approved source.\n- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.\n### #34 Box Entries — v3.47\n- API-Football + GalaxyParlay remain the only approved data sources for this module.\n- API-Football can expose shots inside the penalty area in fixture statistics, but shots inside the box are a subset of outcomes after entry and are not an entry count.\n- The current GalaxyParlay Sports Edge contract does not expose Box Entries.\n- v3.47 attaches a zero-call guard to eligible live events and match-intelligence areas; shots inside box, crosses, corners and possession cannot be relabeled as Box Entries.\n- Activation requires an approved provider-defined penalty-area entry event/metric, explicit repeat-entry counting rules and historical/live compatibility.\n- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.\n### #35 Big Chances — v3.48\n- API-Football + GalaxyParlay remain the approved architecture; no third provider was added.\n- Big Chances is treated as provider-dependent rather than assumed present across competitions.\n- The existing POSTGAME /fixtures/statistics response is inspected for an explicit normalized Big Chances field; this adds zero API calls.\n- Only an explicit provider field is persisted. Shots, shots inside box, xG, goals, corners and key passes cannot be reconstructed or relabeled as Big Chances.\n- If both teams expose the same accepted provider stat type, a compact postgame_big_chances_observation is persisted; otherwise status is API_FOOTBALL_BIG_CHANCES_NOT_AVAILABLE.\n- Pregame remains data-blocked and model-free until consistent historical samples demonstrate actual coverage.\n- actionable=false, decision_weight=0; canonical probabilities, weights, thresholds, tier, stake and bet eligibility remain unchanged.\n## Engineering rule
Natural validation is asynchronous and non-blocking. A module is not called fully RESOLVED until persisted natural state and calibration/promotion gates are satisfied.

## Next engineering target
#33 Field Tilt — audit API-Football + GalaxyParlay for an explicit, consistent territorial-possession definition before implementing any numeric feature.