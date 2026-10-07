# SOCCER EDGE ENGINE — FORMATION MATCHUP ENGINE IMPLEMENTATION PLAN v1.0

**File:** `13_SOCCER_FORMATION_MATCHUP_ENGINE_IMPLEMENTATION_PLAN_v1.0.md`  
**Status:** APPROVED FOR IMPLEMENTATION / RESEARCH-ONLY UNTIL GATES PASS  
**Sport:** Soccer only  
**Project:** SPORTS EDGE ENGINE  
**Created:** 2026-10-06  
**Primary objective:** Convert formation-vs-formation interaction into a reproducible SPORT-FIRST feature family for Corners, Team Corners, Goals, Shots and Shots on Target without allowing betting prices to create the sporting projection.

---

# 1. PURPOSE

The SOCCER EDGE ENGINE must explicitly learn how one verified formation behaves against another verified formation.

The target question is not merely:

> What does a 4-3-3 usually produce?

It is:

> When Formation X faces Formation Y, what does each side produce and concede, after controlling as much as possible for team strength, league, venue and prior information?

Primary outputs include side-specific corners, shots, shots on target and goals, plus match totals and verified tactical statistics. The formation feature must improve the sporting model independently of market prices.

---

# 2. GOVERNING PRINCIPLES

## 2.1 SPORT FIRST. MARKET SECOND.

1. verify fixture;
2. verify lineups/formations;
3. collect sporting inputs;
4. build RAW SPORT PROJECTION;
5. apply formation/style/player matchup features;
6. create sporting shortlist;
7. only then inspect current market;
8. calculate market-shrunk projection and edge;
9. classify BET / LEAN / WATCH / PASS.

Betting prices must never determine the formation adjustment.

## 2.2 NO DATA LEAKAGE

A formation may influence a pregame/OOS prediction only if it was available at or before the prediction point.

- use the latest valid pre-kickoff formation snapshot;
- never backfill a post-kickoff formation into a historical pregame prediction;
- never infer a formation that was not verified;
- unverified formation = NOT VERIFIED.

## 2.3 FORMATION IS NOT STYLE

A nominal formation is geometry, not the full tactical identity. Keep separate:

- nominal formation;
- team strength;
- venue role;
- style/tactical indicators;
- player/role availability;
- coach/context;
- formation-vs-formation historical interaction.

## 2.4 RESEARCH BEFORE PRODUCTION

Formation intelligence may not upgrade a market to BET merely because a descriptive average looks strong. Production eligibility requires walk-forward/OOS evidence, sample sufficiency, calibration and market validation.

---

# 3. CURRENT STATE — BASELINE SNAPSHOT

Current canonical research state at plan creation:

- 299 fixtures with both formations confirmed and a final result;
- 217 fixtures with formations plus postgame corners;
- 81 unique formation-vs-formation matchups;
- 8 formation matchups with at least 8 observations;
- 282 rows with verified postgame corners in the corners baseline;
- 244 corners walk-forward evaluations;
- 39 formation-adjusted corners evaluations;
- current formation-adjusted minimum gate: 100.

Existing components:

- `mcp_gateway/analyze_formation_intelligence.py`
- `mcp_gateway/analyze_formation_intelligence_v2.py`
- `mcp_gateway/analyze_corners_baseline.py`
- `.github/workflows/soccer-edge-formation-intelligence.yml`
- `soccer_edge_state/analysis/formation_intelligence.json`
- `soccer_edge_state/analysis/corners_baseline.json`
- `soccer_edge_state/analysis/v4_022_corners_oos_validation.json`

Existing strengths:

- exact home formation vs away formation is stored;
- formation selection is restricted to pre-kickoff evidence;
- descriptive matchup outputs already include goals, corners, shots and SOT totals;
- the corners challenger is walk-forward;
- formation residual history is built only from earlier matches;
- corners formation effects are shrunk toward no effect;
- corners promotion is currently disabled.

Primary limitation: research is stronger for aggregate match totals than side-specific production.

---

# 4. TARGET ARCHITECTURE

Create `FORMATION MATCHUP ENGINE v1`.

## Layer A — Verified Formation Capture

Canonical directional key:

`HOME_FORMATION vs AWAY_FORMATION`

Direction matters.

## Layer B — Side-Specific Postgame Outcomes

Minimum schema:

```text
fixture_id
kickoff_local
league_id
season
home_team_id
away_team_id
home_formation
away_formation
matchup_key
home_goals
away_goals
home_1h_goals
away_1h_goals
home_2h_goals
away_2h_goals
home_corners
away_corners
total_corners
home_shots
away_shots
total_shots
home_sot
away_sot
total_sot
home_blocked_shots
away_blocked_shots
home_shots_inside_box
away_shots_inside_box
home_shots_outside_box
away_shots_outside_box
home_possession
away_possession
home_fouls
away_fouls
home_yellow_cards
away_yellow_cards
btts
over_2_5
```

Missing is not zero. Optional xG/npxG/field tilt/PPDA are allowed only when independently verified.

## Layer C — Descriptive Matchup Matrix

For every directional formation pair calculate per-side and total averages, rates and metric completeness. Descriptive values remain research context only.

## Layer D — Walk-Forward Formation Residuals

For corners, compare against independent sporting baselines:

```text
baseline_home_corners
baseline_away_corners
baseline_total_corners
```

Then calculate prior-only home, away and total formation residuals. Starting rule:

- minimum prior same-matchup observations: 8;
- strong shrinkage toward 1.00;
- clipped extreme multipliers;
- side-specific and total effects separate;
- no market input.

For goals/shots/SOT, create residual research only after an independent non-market baseline exists for the same target.

## Layer E — Tactical/Style Overlay

After the pair engine is stable, add only verified contextual fields such as back-three/back-four structure, wing-backs, winger count, pivot/attacking-mid roles, coach continuity and personnel changes. Do not infer unsupported tactical roles.

---

# 5. FORMATION MATCHUP OUTPUTS

Eventually expose for serious candidates:

```text
home_formation
away_formation
formation_matchup_key
formation_matchup_sample_n
formation_matchup_confidence
formation_expected_home_corners
formation_expected_away_corners
formation_expected_total_corners
formation_expected_home_shots
formation_expected_away_shots
formation_expected_total_shots
formation_expected_home_sot
formation_expected_away_sot
formation_expected_total_sot
formation_expected_home_goals
formation_expected_away_goals
formation_expected_total_goals
```

Sporting diagnostics may include HIGH/LOW corners, home/away corners, shot-volume, SOT and goal-environment flags. These are not betting recommendations.

---

# 6. CORNERS / TEAM CORNERS PRIORITY

Compare three models:

### Baseline

```text
league/locality baseline
× team corner attack
× opponent corner concession
```

### Challenger A — Existing

```text
baseline total corners
× formation matchup total residual
```

### Challenger B — New side-specific

```text
baseline home corners × home-side formation matchup residual
+
baseline away corners × away-side formation matchup residual
```

Track Home Corners MAE, Away Corners MAE, Total Corners MAE, relevant Brier/log-loss and eventually True CLV separately for FT Corners and Team Corners.

---

# 7. SAMPLE AND PROMOTION GATES

Formation research gate:

```text
same matchup prior n >= 8
```

Corners model gates remain:

```text
OOS evaluations >= 150
formation-adjusted evaluations >= 100
```

Also require lower error than baseline, stable multi-league lift, no concentration in one matchup, verified True CLV evidence and zero leakage violations.

Project minimum remains 100 graded bets before material betting-model recalibration unless an implementation/data bug is found.

---

# 8. IMPLEMENTATION PHASES

## FM-0 — Governance + Snapshot
Plan, state snapshot, no model/threshold changes.

## FM-1 — Side-Specific Formation Dataset
Build `mcp_gateway/formation_matchup_engine_v1.py` and output `soccer_edge_state/analysis/formation_matchup_engine_v1.json`.

Acceptance: pre-kickoff formations only, side-specific goals/corners/shots/SOT, completeness, directional summaries, tests, no odds, no production decision weight.

## FM-2 — Side-Specific Corners Challenger
Extend corners research with home/away residuals, side-specific shrunk lambdas and MAE diagnostics. Keep existing challenger for comparison and production disabled.

## FM-3 — Shots / SOT / Goals Residual Research
Independent non-market baselines, walk-forward residuals and OOS diagnostics before any activation.

## FM-4 — Style + Personnel Overlay
Verified formation geometry/player-role/coach/style context with documented derivation and ablation testing.

## FM-5 — Sporting Projection Integration
Only after gates pass, expose baseline, formation adjustment and raw projection with formation separately. No market information allowed.

## FM-6 — Market Validation
Only after raw sporting projection: exact threshold, price, source, timestamp, close, CLV and settlement. FT Corners and Team Corners evaluated independently.

## FM-7 — Promotion Decision
Statuses: RESEARCH_ONLY, OOS_REVIEW_ELIGIBLE, SHADOW_VALIDATION, PRODUCTION_REVIEW_ELIGIBLE, PRODUCTION_ENABLED, REJECTED_NO_STABLE_LIFT.

---

# 9. SCHEDULER DESIGN

Historical formation aggregation must stay outside the live hot tick.

Daily/offline job rebuilds the matrix and walk-forward reports. Pregame live tick performs only lightweight lookup of current verified formations and pre-materialized features. Postgame persists verified results/tactical stats without rewriting pregame forecasts.

---

# 10. DATA QUALITY / COMPLETENESS

Every metric carries its own observed sample count. Missing is never treated as zero. Sparse metrics require stricter confidence.

---

# 11. ANTI-OVERFITTING RULES

Never tune from one extreme game, tiny n, the same sample used for evaluation, or current market direction. Use chronology, shrinkage, minimum samples, league stability, ablation, OOS evaluation and versioned reports.

---

# 12. OBSERVABILITY

Persist a `formation_matchup_health` block with report version, verified-pair counts, unique matchups, n>=8 count, metric coverage, OOS/adjusted counts, leakage violations, production status and blockers.

---

# 13. TEST PLAN

Required:

1. formation normalization;
2. directional keys;
3. latest pre-kickoff formation selection;
4. post-kickoff-only formation exclusion;
5. missing != zero;
6. side totals reconcile;
7. metric-specific sample counts;
8. no future information in challenger;
9. minimum matchup history gate;
10. shrinkage toward 1;
11. multiplier clipping;
12. no odds input;
13. promotion false by default;
14. backward compatibility;
15. scheduled report persistence.

---

# 14. ROLLOUT / ROLLBACK

Rollout FM-1 → FM-2 → accumulate OOS → FM-3/4 → raw sport integration only after gates → shadow market validation → production review.

Every formation component must be removable without changing base team model, market capture, settlement, CLV or other market families.

---

# 15. DEFINITION OF DONE

Technical implementation is complete when the side-specific dataset is automatic, current verified formations map to a historical profile, Corners and Team Corners have side-specific challenger outputs, Goals/Shots/SOT have baseline-vs-challenger research, overlays are traceable, features are versioned, raw formation logic is market-independent, historical scans stay outside live runtime, postgame feeds learning without hindsight rewrite, and health/blockers persist.

Production readiness still requires independent OOS, calibration, True CLV and settlement gates.

---

# 16. ROADMAP POSITION

Priority:

1. FM-1 Formation Matchup dataset;
2. FM-2 side-specific Corners / Team Corners challenger;
3. continue 1X2 + Team Totals outcome/settlement evaluation;
4. continue FT Totals + BTTS True CLV maturation;
5. FM-3 Shots/SOT/Goals research;
6. FM-4 style/personnel ablation;
7. production review only after gates.

The broader roadmap can continue while formation samples accumulate.

---

# 17. NON-NEGOTIABLE RULE

**Formation matchup intelligence exists to improve our understanding of the football.**

Never: “The market says Over, therefore this formation matchup is an Over matchup.”

Correct: “Verified football evidence indicates that this specific tactical matchup changes expected production. Now, and only now, compare that sporting projection with the market.”


---

# 18. IMPLEMENTATION STATUS — 2026-10-07

## FM-0 — COMPLETE

- Governance and implementation plan materialized.
- Production promotion remains disabled.
- No market input enters formation research.

## FM-1 — COMPLETE

Materialized:

- `mcp_gateway/formation_matchup_engine_v1.py`
- `soccer_edge_state/analysis/formation_matchup_engine_v1.json`

Current canonical state:

- 299 fixtures with verified pre-kickoff formation pair and final result;
- 81 directional formation matchups;
- 8 matchups with n >= 8;
- side-specific goals, corners, shots and SOT materialized;
- odds consumed = false;
- decision weight = 0.

## FM-2 — COMPLETE AS RESEARCH CHALLENGER

The side-specific Corners/Team Corners challenger is integrated into V4-022 and health.

Current evidence:

- 39 formation-adjusted side-specific evaluations;
- Home Corners MAE improves;
- Total Corners MAE improves;
- Away Corners MAE does not yet improve;
- minimum review gate remains 100.

Status:

`RESEARCH_HOLD`

No production promotion.

## FM-3 — IMPLEMENTED / RESEARCH_HOLD

Materialized:

- `mcp_gateway/formation_matchup_fm3_oos_v1.py`
- `soccer_edge_state/analysis/formation_matchup_fm3_oos_v1.json`

Current OOS evidence:

### GOALS

- observed rows: 299;
- baseline evaluations: 298;
- formation-adjusted evaluations: 79;
- Home/Away/Total MAE do not improve vs baseline.

### SHOTS

- observed rows: 217;
- baseline evaluations: 216;
- formation-adjusted evaluations: 49;
- Home/Away/Total MAE do not improve;
- eligible sample is highly concentrated in two formation matchups.

### SOT

- observed rows: 219;
- baseline evaluations: 218;
- formation-adjusted evaluations: 49;
- Home/Away/Total MAE do not improve;
- eligible sample is highly concentrated in two formation matchups.

Conclusion:

Nominal formation-vs-formation is **not currently validated as an additive production feature for Shots, SOT or Goals**.

Do not promote or tune toward these results.

## FM-4 — IMPLEMENTED / BLOCKED BY PRIOR HISTORY DEPTH

The prior-only style/personnel ablation is materialized as:

- `mcp_gateway/formation_matchup_fm4_style_ablation_v1.py`
- `soccer_edge_state/analysis/formation_matchup_fm4_style_ablation_v1.json`

Verified diagnosis:

- 520 unique teams appear in the 299 source fixtures;
- all 520 appear somewhere in the tactical-history store;
- therefore team-ID mapping is not the blocker;
- however the project history store begins only immediately before the Formation Matchup cohort;
- only a very small number of source fixtures have any prior tactical observations;
- zero source fixtures currently have both teams with >=3 complete prior observations across every required style field.

Current style fields:

- possession;
- shot accuracy;
- box share;
- blocked-shot share;
- fouls;
- yellow cards.

The system must **not**:

- lower the minimum prior-match requirement merely to create samples;
- use post-target matches;
- use current-match postgame statistics in its own prediction;
- infer missing player roles or coach continuity.

FM-4 now exposes an explicit historical-depth blocker and coverage window.

Status:

`RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION`

Production enabled = false.  
Decision weight = 0.

## Historical backfill policy

A tactical-history backfill may be added only as a separate offline process with:

- bounded provider-request budget;
- exact fixture IDs and timestamps;
- provider-supported match statistics only;
- no synthetic tactical values;
- no rewriting of original pregame predictions;
- strict cutoff before each evaluated fixture;
- persisted provenance and retrieval timestamps.

Backfilled data may expand the research sample, but it may not retroactively alter an already-issued historical prediction.

## Roadmap continuation rule

FM-3 and FM-4 blockers do **not** justify delaying the broader Soccer roadmap.

Continue:

1. accumulate Corners/Team Corners formation samples toward the 100-adjusted gate;
2. keep 1X2 + Team Totals result/settlement evaluation active;
3. keep FT Totals + BTTS True CLV maturation active;
4. accumulate deeper verified tactical and personnel history;
5. rerun FM-3/FM-4 automatically as sample depth increases;
6. integrate a formation/style feature into RAW SPORT PROJECTION only if OOS lift is eventually demonstrated.

Zero production weight is the correct state until those gates pass.


---

# 19. ROADMAP UPDATE — 2026-10-07 POST P1F / FM-4 BACKFILL ATTEMPT

The broader Soccer roadmap is now tracked in:

`14_SOCCER_EDGE_ENGINE_ROADMAP_STATUS_v1.0.md`

## 19.1 FM-4 guarded historical backfill infrastructure — IMPLEMENTED

Materialized:

- `mcp_gateway/formation_tactical_history_backfill_v1.py`;
- protected Render route `/internal/fm4-tactical-history-backfill-v1/run`;
- `.github/workflows/fm4-tactical-history-backfill.yml`.

Safety contract:

- finalized historical fixtures only;
- fixtures must precede the Formation Matchup target cohort;
- exact provider `fixtures/statistics` observations only;
- max 8 statistics requests per run;
- daily provider reserve guard;
- retrieval provenance persisted;
- no synthetic tactical values;
- no retroactive prediction rewrite;
- no retroactive market, CLV or bet creation;
- production decision weight = 0.

## 19.2 First FM-4 backfill run — NO HISTORICAL CANDIDATES FOUND LOCALLY

First workflow execution observed:

```text
target_team_count = 520
candidate_pool_rows = 0
selected_fixture_count = 0
attempted = 0
captured = 0
provider_requests_added = 0
```

Interpretation:

The next FM-4 bottleneck is not statistics quota.

The local Postgres fixture/result store does not currently expose eligible finalized pre-cohort fixtures for the 520 target teams under the backfill query.

Do **not** weaken the prior-history gate.

Next required FM-4 task:

```text
HISTORICAL FIXTURE DISCOVERY / INGESTION
↓
VERIFIED PRE-COHORT FIXTURE IDS
↓
GUARDED FIXTURES/STATISTICS BACKFILL
↓
FM-4 PRIOR-ONLY STYLE REBUILD
```

## 19.3 Offline rebuild workflow dependency issue

The same first workflow later failed at the FM-4 rebuild step because the GitHub runner did not contain the `httpx` dependency loaded by the package bootstrap.

This is an implementation-environment issue.

It is **not** evidence against the Formation Matchup model and did not alter production state.

Required correction:

- install/use the required lightweight runtime dependency in the offline job, or;
- isolate the research module from unrelated commercial package imports.

Prefer isolation when practical.

## 19.4 FM-4 remains correctly blocked

Current status remains:

`RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION`

Current critical blocker:

`STYLE_HISTORY_DEPTH_BOTH_TEAMS_N_GE_3_0_LT_100`

Production enabled = false.  
Decision weight = 0.

## 19.5 FM-5 remains blocked

Do not integrate Formation/Style into RAW SPORT PROJECTION until:

1. verified prior historical depth exists;
2. style/personnel ablation has a meaningful OOS sample;
3. challenger improves the appropriate independent baseline;
4. lift is not concentrated in one league/matchup;
5. no leakage exists.

Nominal formation results alone are not sufficient.

