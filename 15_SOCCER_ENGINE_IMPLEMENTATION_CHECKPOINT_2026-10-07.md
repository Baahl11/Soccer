# SOCCER EDGE ENGINE — IMPLEMENTATION CHECKPOINT — 2026-10-07

**File:** `15_SOCCER_ENGINE_IMPLEMENTATION_CHECKPOINT_2026-10-07.md`  
**Status:** ACTIVE RESUME POINT / DO NOT RECONSTRUCT FROM CHAT  
**Purpose:** Preserve the exact technical state, current blockers, branch heads, active background workflows, and next permitted engineering actions so work can resume after any chat/UI interruption.

---

# 1. GOVERNING RULE

**SPORT FIRST. MARKET SECOND.**

The betting market may evaluate a completed sporting projection, but may not create the sporting thesis.

The current order remains:

```text
FULL SLATE
→ VERIFIED SPORTING DATA
→ RAW SPORT PROJECTION
→ SPORTING SHORTLIST
→ MARKET
→ MARKET-SHRUNK PROJECTION
→ EDGE / EV
→ BET / LEAN / WATCH / PASS
```

No change in this checkpoint alters that order.

---

# 2. DOCUMENT AUTHORITY

Use these as current Soccer authority:

```text
00_PROJECT_CHARTER.md
01_SHARED_DATA_SOURCE_POLICY.md
02_SCHEDULER_AND_REFRESH_PROTOCOL.md
03_LEDGER_SCHEMA.md
04_POSTGAME_LEARNING_PROTOCOL.md
11_SOCCER_DATA_API_PROTOCOL_v1.0.md
12_SOCCER_EDGE_ENGINE_MASTER_v1.0.md
13_SOCCER_FORMATION_MATCHUP_ENGINE_IMPLEMENTATION_PLAN_v1.0.md
14_SOCCER_EDGE_ENGINE_ROADMAP_STATUS_v1.0.md
15_SOCCER_ENGINE_IMPLEMENTATION_CHECKPOINT_2026-10-07.md
```

Older repository roadmaps that claim fixed accuracy, completion percentages, or production readiness are historical only unless explicitly reconciled.

---

# 3. BRANCH HEADS AT CHECKPOINT

```text
main
a264c537370a749f5ff9540d7138174773706981
P1 persist prospective evaluation progress in health

soccer-edge-mcp-v1
a9c0cbbf35e2d51a3289442b240853d82dc83025
FM-4 expose guarded prior-personnel backfill route

soccer-edge-state
9f21eef1854665ece5dea3a3ae37cf3742f763eb
Soccer Edge tick 2026-10-07
```

These hashes are a resume reference, not a request to reset branches.

---

# 4. LIVE PRIMARY CLV INFRASTRUCTURE — COMPLETE

Current live runtime version persisted in state:

```text
4.38.8-normalized-primary-clv-anchor-fallback
```

Primary CLV loader health:

```text
source = POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V4_NORMALIZED
normalized_storage = true
legacy_json_expansion_used = false
fallback_used = false
legacy_mode = EMERGENCY_FALLBACK_ONLY
```

Current live tick snapshot at checkpoint:

```text
candidate_count = 3
candidate_family_counts:
  1X2 = 3
  BTTS = 3
  FT_TOTALS = 1
```

Normalized relational anchor storage is the normal hot path.

Legacy JSON expansion is permitted only as an emergency loader fallback.

Do not reintroduce legacy JSON expansion as the normal runtime path.

---

# 5. MARKET FAMILY STATUS

## 5.1 1X2

```text
status = CALIBRATION_REVIEW_ELIGIBLE
True CLV rows = 50
unique fixtures = 50
minimum = 50
blockers = none
```

Prospective evaluation is now append-only and active, but the new prospective cohort currently contains zero selections.

Do not count historical research rows as the new 100 graded-bet cohort.

## 5.2 Team Totals

```text
status = OOS_REVIEW_ELIGIBLE
True CLV rows = 343
unique fixtures = 44
minimum rows = 50
blockers = none
```

Multiple exact Team Total lines in one fixture remain correlated.

Production promotion remains disabled.

## 5.3 FT Totals

```text
status = RESEARCH_HOLD
True CLV = 6 / 50
remaining = 44
maturation = NATURAL_MATURATION_REQUIRED
```

Do not weaken strict close.

## 5.4 BTTS

```text
status = RESEARCH_HOLD
True CLV = 17 / 50
remaining = 33
maturation = NATURAL_MATURATION_REQUIRED
```

Do not manufacture True CLV.

## 5.5 Corners

```text
FT_CORNERS formation-adjusted = 39 / 100
TEAM_CORNERS side-specific formation-adjusted = 39 / 100
home-side MAE improves = true
away-side MAE improves = false
both-side MAE improves = false
True CLV = 0
```

No promotion.

## 5.6 1H / 2H / Cards / Player Props

All remain RESEARCH_HOLD.

They are below the current core priorities.

---

# 6. FORMATION MATCHUP STATUS

## FM-1 — COMPLETE

```text
verified pre-kickoff formation-pair fixtures = 299
directional matchup types = 81
matchups with n >= 8 = 8
```

Side-specific Goals / Corners / Shots / SOT are materialized.

Odds consumed = false.  
Decision weight = 0.

## FM-2 — RESEARCH CHALLENGER ACTIVE

Corners / Team Corners formation challenger is integrated into V4-022.

Current blocker:

```text
formation-adjusted 39 < 100
```

Away-side MAE also still fails improvement.

## FM-3 — IMPLEMENTED / RESEARCH_HOLD

Nominal formation alone has not demonstrated OOS lift for Goals, Shots or SOT.

Do not promote.

## FM-4 — IMPLEMENTED / RESEARCH_HOLD

Current style fields:

```text
possession
shot_accuracy
box_share
blocked_share
fouls
yellow_cards
```

Current style-history depth:

```text
source fixtures = 299
source unique teams = 520
source-team overlap with tactical history = 100%

both teams any prior style history = 12
both teams complete n>=1 = 12
both teams complete n>=2 = 3
both teams complete n>=3 = 0

style-eligible source rows = 0
```

Therefore:

```text
STYLE_HISTORY_DEPTH_BOTH_TEAMS_N_GE_3_0_LT_100
```

remains the critical blocker.

Do not lower the n>=3 prior-history gate.

---

# 7. FM-4 GUARDED TACTICAL HISTORY BACKFILL — ACTIVE

Backfill model:

```text
SOCCER_FM4_TACTICAL_HISTORY_BACKFILL_V1.2.0
```

Workflow:

```text
.github/workflows/fm4-tactical-history-backfill.yml
```

Current workflow cadence:

```text
every 30 minutes
```

Safety contract:

- finalized historical fixtures only;
- strict cutoff before Formation Matchup cohort;
- verified provider fixture statistics only;
- no synthetic tactical values;
- max 8 historical fixture discovery calls per run;
- max 8 statistics calls per run;
- daily reserve guard;
- retrieval provenance persisted;
- known incomplete-stat fixtures are not repeatedly queried;
- successful-statistics leagues may be used only as a tie-breaker;
- no retroactive prediction rewrite;
- no retroactive odds / CLV / bet creation;
- production decision weight = 0.

Historical fixture discovery is operational.

The prior `candidate_pool_rows = 0` blocker has been resolved.

---

# 8. LATEST FM-4 BACKFILL STATE

Latest canonical backfill report at checkpoint:

```text
target_team_count = 520
provided_team_season_count = 520
candidate_pool_rows = 56
selected_fixture_count = 8

statistics attempted = 8
captured = 5
incomplete = 3
errors = 0

materialized_history_fixture_count = 17

fixture discovery provider calls = 0
statistics provider calls = 8
total provider calls = 8

daily provider remaining = 5004
```

Successful statistics-capture leagues currently include:

```text
league 2  = 1
league 39 = 6
league 40 = 1
league 48 = 4
```

Interpretation:

The discovery/ingestion problem is solved.

The remaining FM-4 problem is genuine prior-history depth and provider-stat completeness.

The backfill must continue accumulating verified prior history.

---

# 9. PROSPECTIVE EVALUATION REGISTRY — ACTIVE

New append-only evaluation workflow exists:

```text
.github/workflows/prospective-evaluation-registry.yml
```

Model:

```text
SOCCER_PROSPECTIVE_EVALUATION_REGISTRY_V1.0.0
```

Current cohort start:

```text
2026-10-07T14:30:00+00:00
```

Current state:

```text
canonical selection rows = 0
canonical selection unique fixtures = 0
settlement rows = 0
graded-bet sample rows = 0
graded-bet target = 100
material recalibration allowed = false
```

Policy safeguards:

- append-only selections;
- outcomes never used for selection;
- historical predictions not rewritten;
- one selection per fixture per track;
- stage = T-10;
- Data Tier A/B only;
- Availability Confidence >= 0.85;
- both XI and both goalkeepers confirmed;
- Sporting Shortlist must already be true;
- market price does not create raw sporting projection;
- production promotion disabled.

### 1X2 prospective counting rule

Only runtime Tier B BET selections count toward the 100 graded-bet sample.

Advanced-cap WATCH shadow rows do not automatically count.

### Team Totals prospective rule

Team Totals prospective research may select an exact positive raw-edge market row for research, but:

```text
counts_toward_100_graded_bets = false
```

until a validated Team Totals production decision rule exists.

Do not silently change that.

---

# 10. CURRENT SCHEDULER HEALTH

Current canonical scheduler state continues to expose:

- Primary CLV normalized loader health;
- per-family CLV maturation;
- Formation Matchup health;
- FM-3 blockers;
- FM-4 style/personnel blockers;
- prospective evaluation progress.

No current evidence supports changing model weights or market gates.

---

# 11. DO NOT REDO

The following work is already complete and should not be repeated unless a regression is detected:

- OOM/historical hot-path cleanup;
- Team Totals historical provenance removal from live tick;
- FT Totals historical coverage removal from live tick;
- Primary CLV loader diagnostics moved offline;
- market cache N+1 batching;
- exact per-fixture cache freshness;
- Primary CLV normalized relational store;
- legacy-vs-normalized equivalence validation;
- legacy loader fallback-only policy;
- 1X2 canonical outcome loop;
- Team Totals canonical outcome loop;
- V4-017/V4-019 research-only outcome context;
- FM-1 formation dataset;
- FM-2 side-specific Corners challenger;
- FM-3 Goals/Shots/SOT walk-forward research;
- FM-4 style/personnel research scaffold;
- FM-4 historical fixture discovery;
- FM-4 guarded tactical backfill;
- prospective append-only evaluation registry;
- legacy roadmap documents marked non-governing.

---

# 12. RESUME HERE

If work is interrupted, resume in this order:

## Immediate active process

Let FM-4 guarded tactical backfill continue accumulating verified pre-cohort history.

Do not manually manufacture sample depth.

## Next engineering review trigger

When FM-4 reaches materially higher prior-history density, inspect:

```text
rows_with_both_min_field_n_ge_3
style_eligible_source_rows
per-target style ablation sample
league/matchup concentration
```

The first major review threshold remains:

```text
both-team complete prior style n>=3 >= 100 source fixtures
```

## In parallel

Continue:

```text
1X2 prospective graded-bet accumulation
Team Totals prospective/OOS evaluation
FT Totals True CLV maturation
BTTS True CLV maturation
Corners formation-adjusted sample accumulation
```

## After FM-4 has sufficient history

Run style/personnel ablation.

Only if OOS lift is demonstrated:

```text
FM-4 research
→ FM-5 RAW SPORT PROJECTION integration
→ FM-6 exact market validation
→ FM-7 production review
```

No odds are allowed to create the FM-5 raw sporting feature.

---

# 13. CURRENT MAIN BLOCKERS

```text
FM-4:
  both-team complete prior style n>=3 = 0 / 100

FT Totals:
  True CLV = 6 / 50

BTTS:
  True CLV = 17 / 50

FT Corners:
  formation-adjusted = 39 / 100
  True CLV = 0 / 50

Team Corners:
  side-specific formation-adjusted = 39 / 100
  Away MAE not better than baseline
  True CLV = 0 / 50

Prospective graded-bet sample:
  0 / 100
```

These are sample/evidence blockers, not invitations to lower thresholds.

---

# 14. NEXT ALLOWED ACTIONS

Allowed:

- accumulate verified data;
- improve data completeness;
- improve provider discovery efficiency;
- fix implementation/data bugs;
- strengthen observability;
- perform prior-only OOS research;
- grade immutable prospective selections;
- document every material change.

Not allowed merely to accelerate progress:

- lower OOS sample gates;
- lower the 100 graded-bet recalibration minimum;
- count correlated rows as independent bets;
- turn WATCH research rows into historical BETs;
- use future lineups/statistics in historical pregame features;
- weaken strict True CLV close chronology;
- use odds to generate the raw sporting thesis.

---

# 15. CHECKPOINT SUMMARY

The system is not waiting on a single engineering repair anymore.

The main active constraint is **evidence accumulation**.

The highest-value football-specific process currently running is FM-4 tactical-history accumulation so that formation can be evaluated together with real prior style/personnel context rather than nominal shape alone.

The highest-value evaluation process currently running is the new prospective append-only registry so future calibration uses decisions frozen before outcomes.

**Resume from this file, not from chat history.**


---

# 16. CODE-COMPLETE ADDENDUM — 2026-10-07

**This section supersedes earlier numerical snapshots in this file wherever they conflict.**  
Earlier sections remain useful as the chronological engineering record.

## Current branch heads

```text
main
de728e8e395167cd6bc3b2564dbe191b115fe77d
Surface FM4-FM7 lifecycle health in scheduler

soccer-edge-mcp-v1
24156202f82e363db0f139b59347e3c2e6d8ab50
Run formation lifecycle health contract in runtime CI

soccer-edge-state
f6a0bd87c444e4e78a1936379287257d2eed4245
Update Phase 19 promotion framework report
```

These are resume references only. Do not reset branches to these hashes.

## Formation software lifecycle status

The formation engineering chain is now materially implemented through the production-review boundary:

```text
FM-4 STYLE/PERSONNEL ABLATION
  FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.1.0

FM-4 PERSONNEL OUTCOME ABLATION
  FORMATION_PERSONNEL_OUTCOME_ABLATION_V1.0.0

FM-5 READINESS GATE
  FORMATION_FM5_READINESS_GATE_V1.0.0

FM-5 RAW SPORT PROJECTION CONTRACT
  FORMATION_FM5_RAW_SPORT_PROJECTION_V1.0.0

FM-6 EXACT MARKET VALIDATION CONTRACT
  FORMATION_FM6_EXACT_MARKET_VALIDATION_V1.0.0

FM-7 PROMOTION REVIEW CONTRACT
  FORMATION_FM7_PROMOTION_REVIEW_V1.0.0

FORMATION LIFECYCLE HEALTH
  FORMATION_LIFECYCLE_HEALTH_V1.0.0
```

Canonical lifecycle state currently reports:

```text
status = SOFTWARE_CONTRACTS_COMPLETE_EVIDENCE_ACCUMULATING
software_contracts_complete = true
production_promotion_allowed = false
model_weights_changed = false
thresholds_changed = false
canonical_bet_logic_changed = false
market_prices_consumed_to_create_sporting_projection = false
historical_predictions_rewritten = false
```

This means **software completion is no longer the same thing as evidence readiness**.

FM-5, FM-6 and FM-7 exist as guarded contracts, but cannot acquire production weight merely because the code exists.

## FM-4 current evidence

Latest canonical FM-4 state:

```text
model_version = FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.1.0
status = RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION

tactical_history_fixtures_loaded = 2441
tactical_history_unique_teams = 4034
source_unique_teams = 520
source_team_overlap_pct = 100%

both teams complete prior style n>=1 = 12
both teams complete prior style n>=2 = 3
both teams complete prior style n>=3 = 2
style_eligible_source_rows = 2

required both-team prior style n>=3 = 100
```

Current personnel state:

```text
current both-XI rows = 18
both teams with prior confirmed XI = 0
both teams with previous-coach comparison = 0
both teams with last-3 core-return feature = 0
personnel outcome ablation ready = false
```

The personnel outcome ablation is now implemented rather than a placeholder. It remains blocked by genuine prior-personnel history.

## Corners current evidence

Latest FM-5 readiness input reports:

```text
FT Corners formation-adjusted evaluations = 44 / 100
Team Corners formation-adjusted evaluations = 44 / 100

FT Corners MAE improves = true
required line Brier/log-loss improvement = true

Team Corners home MAE improves = true
Team Corners away MAE improves = true
Team Corners both-side MAE improves = true
Team Corners total MAE improves = true

stable FT-Corners lift leagues = 0 / 2
league/venue stability review ready = false
```

Therefore Corners remains research-only despite improved directional performance.

## FM-5 current gate

```text
status = FM5_BLOCKED_EVIDENCE_GATES
ready_components = []
integration_review_allowed = false
automatic_integration_allowed = false
production_enabled = false
decision_weight = 0
```

Primary blockers:

```text
FM4_STYLE_HISTORY_2_LT_100
FM4_GOALS_STYLE_OOS_NOT_READY
FM4_SHOTS_STYLE_OOS_NOT_READY
FM4_SOT_STYLE_OOS_NOT_READY
FM4_PERSONNEL_SAMPLE_NOT_READY
FM4_PERSONNEL_OUTCOME_ABLATION_NOT_READY
FM2_FT_CORNERS_SPORT_OOS_NOT_READY
FM2_TEAM_CORNERS_SPORT_OOS_NOT_READY
```

## FM-6 / FM-7 current state

FM-6 is coded to compare only an already-created FM-5 sporting projection against an exact market instrument.

It requires:

- exact market family;
- exact selection;
- exact line where applicable;
- exact decimal price;
- bookmaker;
- source;
- market capture timestamp;
- market snapshot after the raw sporting feature timestamp;
- market snapshot strictly before kickoff.

FM-6 cannot create a sporting thesis and currently has zero production weight.

FM-7 is coded as a production-review gate, not an auto-promotion mechanism.

It requires, among other evidence:

- an FM-5 component that passed sporting OOS gates;
- FM-6 exact-market validation;
- clean strict-close chronology;
- True CLV gate;
- calibration gate;
- multi-league stability;
- concentration gate;
- stable OOS lift;
- at least 100 prospective graded bets;
- material recalibration permission;
- append-only selections with no outcome leakage;
- manual approval;
- rollback model pointer.

Automatic activation remains prohibited.

## Prospective sample

```text
canonical selection rows = 0
graded-bet rows = 0 / 100
material recalibration allowed = false
```

No historical research rows are being relabeled to accelerate this sample.

## CI / regression state

Recent successful validation includes:

```text
Soccer Edge V4 Runtime Tests:
  520 passed

Formation Intelligence:
  latest repaired run = success

Contract Regression:
  FM5 / FM6 / FM7 / CLV / prospective / formation contracts = passing

Scheduler:
  state-push retry hardening = active
  latest validated scheduler runs = success

Formation Lifecycle Health:
  scheduled materialization = success
```

The regression suite verifies zero production weight and no market leakage for the new formation stages.

## Remaining operational deployment blocker

The FM-4 personnel history source code and protected route exist in `soccer-edge-mcp-v1`, but the current Render runtime observed by the workflow is still deployed from:

```text
render_git_commit = fda8a25dc81731d2867746d7433ce5905e7fb00e
```

The protected personnel backfill route was added later, so the live service currently returns:

```text
HTTP 404
/internal/fm4-personnel-history-backfill-v1/run
```

As a result:

```text
fm4_personnel_history_backfill_report.json = NOT MATERIALIZED
```

This is now a **deployment/runtime synchronization blocker**, not a missing model implementation.

Do not weaken the personnel gate to work around it.

## Current definition of “done”

Engineering can be considered code-complete when:

1. FM4-FM7 contracts remain green in CI;
2. lifecycle health remains materialized;
3. scheduler health exposes the lifecycle state;
4. Render is synchronized to the engine revision containing the protected personnel route;
5. the personnel backfill workflow executes successfully and materializes its canonical state artifact.

Evidence completion is separate and will continue naturally after code completion.



---

# 17. RENDER SYNCHRONIZATION RESOLVED — 2026-10-07

This section supersedes the deployment blocker described in Section 16.

The confirmed Render workspace is:

```text
Baahl
```

Service:

```text
soccer-edge-api
branch = soccer-edge-mcp-v1
```

A manual deploy was triggered after Render failed to auto-deploy later branch commits.

Successful deployed engine commit:

```text
2869ca085fcdad826280247653319ec4ce2572bd
```

Render deployment result:

```text
deploy = dep-db36jonlk1mc739jpr3g
status = succeeded
```

The live health endpoint subsequently reported:

```text
render_git_commit = 2869ca085fcdad826280247653319ec4ce2572bd
status = ok
```

The protected personnel route is now deployed and operational:

```text
POST /internal/fm4-personnel-history-backfill-v1/run
HTTP 200
```

## First successful post-deploy personnel backfill

The guarded FM-4 personnel backfill was re-run end-to-end after deployment synchronization.

Result:

```text
attempted = 8
captured = 7
incomplete = 1
errors = 0
provider_requests_added = 8
materialized_personnel_history_fixture_count = 7
daily_remaining_after_run = 4694
production_promotion_allowed = false
decision_weight = 0
```

Canonical personnel artifact:

```text
soccer_edge_state/analysis/fm4_personnel_history_backfill_report.json
status = FM4_PERSONNEL_HISTORY_BACKFILL_COMPLETE
model_version = SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0
```

The stale Render deployment blocker is therefore **resolved**.

## Current personnel evidence after first successful backfill

FM-4 now reports:

```text
personnel_history_fixtures_loaded = 49
current both-XI target rows = 18
both teams with prior confirmed XI = 0
both teams with previous-coach comparable history = 0
both teams with last-3 core return rate = 0
personnel outcome ablation ready = false
```

Interpretation:

The remaining personnel blocker is now genuine historical depth/overlap, not deployment.

Continue guarded prior-personnel accumulation. Do not lower the 100-row gate.

## Lifecycle state after deployment repair

```text
status = SOFTWARE_CONTRACTS_COMPLETE_EVIDENCE_ACCUMULATING
personnel_backfill_artifact_materialized = true
personnel_backfill_status = FM4_PERSONNEL_HISTORY_BACKFILL_COMPLETE
personnel_backfill_last_run_captured = 7
personnel_history_fixture_count = 7
FM5 = FM5_BLOCKED_EVIDENCE_GATES
FM6 = BLOCKED_WAITING_FM5_SPORTING_COMPONENT
FM7 = BLOCKED_UPSTREAM_FM5
production_promotion_allowed = false
```

There is no remaining known formation-engine code blocker in the active FM4-FM7 path.

The remaining blockers are evidence accumulation and normal gated evaluation.
