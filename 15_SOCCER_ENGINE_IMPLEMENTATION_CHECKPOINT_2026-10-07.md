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
