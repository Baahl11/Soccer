# SOCCER EDGE ENGINE — ROADMAP STATUS v1.0

**File:** `14_SOCCER_EDGE_ENGINE_ROADMAP_STATUS_v1.0.md`  
**Status:** ACTIVE / CANONICAL IMPLEMENTATION STATUS  
**Sport:** Soccer only  
**Project:** SPORTS EDGE ENGINE  
**Updated:** 2026-10-07  
**Governing principle:** **SPORT FIRST. MARKET SECOND.**

---

# 1. PURPOSE

This file records the current implementation state of SOCCER EDGE ENGINE and the ordered next work.

It does **not** replace:

- `00_PROJECT_CHARTER.md`;
- `01_SHARED_DATA_SOURCE_POLICY.md`;
- `02_SCHEDULER_AND_REFRESH_PROTOCOL.md`;
- `03_LEDGER_SCHEMA.md`;
- `04_POSTGAME_LEARNING_PROTOCOL.md`;
- `11_SOCCER_DATA_API_PROTOCOL_v1.0.md`;
- `12_SOCCER_EDGE_ENGINE_MASTER_v1.0.md`;
- `13_SOCCER_FORMATION_MATCHUP_ENGINE_IMPLEMENTATION_PLAN_v1.0.md`.

If this status document conflicts with a governing Project file, the governing file wins.

## Active resume checkpoint

For exact implementation state, branch heads, live blockers, and the current resume point, read:

`15_SOCCER_ENGINE_IMPLEMENTATION_CHECKPOINT_2026-10-07.md`

That checkpoint is the preferred recovery source after chat/UI interruption. Do not reconstruct current state from legacy roadmap documents or memory when the checkpoint is available.

---

# 2. NON-NEGOTIABLE MODEL ORDER

The current engine remains:

```text
FULL SLATE
  ↓
SPORTING DATA
  ↓
RAW SPORT PROJECTION
  ↓
SPORT-SPECIFIC SHORTLISTS
  ↓
MARKET
  ↓
MARKET-SHRUNK PROJECTION
  ↓
EDGE / EV
  ↓
BET / LEAN / WATCH / PASS
```

Raw sporting projections must be based on soccer variables such as:

- team and player quality;
- opponent-adjusted strength;
- xG/npxG when verified;
- shots / SOT / big chances;
- field tilt / territory;
- PPDA / press interaction;
- transition matchup;
- box access;
- set pieces;
- goalkeeper;
- XI / availability;
- formation and tactical fit;
- rest / travel / congestion;
- coaching/context;
- weather/referee only when material and verified.

Betting prices may not create the sporting thesis.

---

# 3. CURRENT SYSTEM STATE

## 3.1 Scheduler / live pipeline — OPERATIONAL

Canonical stages remain:

```text
MORNING / DAILY DISCOVERY
T-90
T-60
T-40 PRIMARY REFRESH
T-30
T-20
T-10
CLOSE
POSTGAME
```

Heavy historical validation is kept outside the live hot tick.

The live runtime has been stabilized after prior OOM / historical-query problems.

---

## 3.2 Primary CLV anchor normalization — COMPLETE

The normalized relational Primary CLV anchor store is now the normal live path.

Current canonical normalization state:

- status: `EQUIVALENT`;
- current normalized anchor rows:
  - 1X2 = 1,885;
  - BTTS = 2,694;
  - FT_TOTALS = 5,009;
- total normalized anchors = 9,588;
- provider requests added = 0;
- strict-close semantics changed = false;
- models changed = false;
- thresholds changed = false;
- gates changed = false;
- canonical bet logic changed = false.

Latest equivalence comparison:

- legacy candidate events = 6;
- normalized candidate events = 6;
- legacy signal keys = 14;
- normalized signal keys = 14;
- missing from normalized = 0;
- extra in normalized = 0.

Important:

**9,588 normalized anchors are NOT 9,588 True CLV observations.**

Legacy JSON expansion is no longer the normal hot path.

Legacy loader policy:

```text
EMERGENCY FALLBACK ONLY
```

Current persisted loader health has shown:

```text
normalized_storage = true
legacy_json_expansion_used = false
fallback_used = false
```

---

# 4. MARKET-FAMILY STATUS

## 4.1 1X2 — CALIBRATION_REVIEW_ELIGIBLE

Canonical V4-017 state:

- True CLV rows = 50;
- True CLV unique fixtures = 50;
- minimum CLV gate = 50;
- blockers = none;
- status = `CALIBRATION_REVIEW_ELIGIBLE`.

Canonical outcome research:

- exact 1X2 rows = 50;
- settled = 43;
- settled unique fixtures = 43.

The outcome report is research context only.

Its Brier/log-loss fields score de-vig **market fair probability**, not Soccer model probability.

Production promotion remains separate from this research artifact.

### Next for 1X2

1. continue immutable result / settlement capture;
2. accumulate a genuine predeclared independent graded-bet sample;
3. review Soccer-model calibration separately from market benchmark;
4. do not materially recalibrate until Project sample rules are satisfied;
5. preserve segmentation by model version / league / Data Tier / Availability Confidence.

---

## 4.2 TEAM TOTALS — OOS_REVIEW_ELIGIBLE

Canonical V4-019 state:

- True CLV rows = 343;
- unique True CLV fixtures = 44;
- required True CLV rows = 50;
- required unique fixtures = 20;
- blockers = none;
- status = `OOS_REVIEW_ELIGIBLE`;
- manual review required = true;
- production promotion allowed = false.

Canonical exact-signal outcome research currently contains:

- 335 deduplicated exact Team Total observations;
- 43 unique fixtures;
- 202 settled observations;
- 24 settled unique fixtures.

Multiple Team Total lines inside the same fixture are correlated and must not be treated as independent bets.

### Next for Team Totals

1. improve settlement coverage / fixture diversity;
2. evaluate role × line calibration;
3. preserve exact Asian/half-goal settlement semantics;
4. build a predeclared independent shadow-bet sample;
5. keep production disabled until review gates explicitly pass.

---

## 4.3 FT TOTALS — RESEARCH_HOLD / NATURAL MATURATION

Canonical V4-016:

- True CLV = 6 / 50;
- unique fixtures = 6;
- blocker = `FT_TOTALS_TRUE_CLV_6_LT_50`;
- avg probability CLV currently positive, but sample is far too small.

Do not loosen strict-close chronology.

### Next

Continue natural exact-market close capture and maturation.

---

## 4.4 BTTS — RESEARCH_HOLD / NATURAL MATURATION

Canonical V4-018:

- True CLV = 17 / 50;
- unique fixtures = 17;
- blocker = `BTTS_TRUE_CLV_17_LT_50`.

Do not manufacture CLV.

### Next

Continue natural exact-market close capture and maturation.

---

## 4.5 1H — RESEARCH_HOLD

Current blockers include:

- challenger Brier not better than baseline;
- challenger log-loss not better than baseline;
- True CLV 0 / 50;
- source promotion gate disabled.

### Priority

Below 1X2, Team Totals, FT Totals, BTTS and formation/corners work.

---

## 4.6 2H — RESEARCH_HOLD

Current blockers include:

- halftime-conditioned challenger does not beat baseline;
- True CLV 0 / 50;
- source promotion gate disabled.

### Priority

Below the core markets.

---

## 4.7 FT CORNERS / TEAM CORNERS — RESEARCH_HOLD

Canonical V4-022:

- formation-adjusted evaluations = 39 / 100;
- True CLV = 0;
- FT Corners league lift not yet stable;
- Team Corners parent FT model not review-ready;
- Team Corners league/venue stability not ready.

Side-specific formation challenger:

- Home Corners MAE improves;
- Total Corners MAE improves;
- Away Corners MAE does not yet improve;
- both-side improvement gate therefore fails.

### Next

1. accumulate formation-adjusted observations toward 100;
2. require Home + Away side-specific improvement;
3. require league/venue stability;
4. then mature exact Corners / Team Corners True CLV;
5. do not promote from descriptive matchup averages alone.

---

## 4.8 CARDS — RESEARCH_HOLD

Current major blockers include:

- referee-adjusted yellow-card sample below required level;
- red-card OOS sample below market/actionable review levels;
- True CLV = 0;
- observed match/player card price history missing;
- player-card OOS incomplete.

### Priority

Research only until core market families are further matured.

---

## 4.9 PLAYER PROPS — RESEARCH_HOLD

Families include:

- Shots;
- SOT;
- Goalscorer;
- Assists;
- Cards;
- GK Saves.

Current state:

- True CLV = 0 for each family;
- exact observed market history is incomplete;
- confirmed-XI market overlap is incomplete;
- OOS validation incomplete;
- goalkeeper valid profiles = 22 / 100.

### Priority

Do not distract from core Soccer-market validation yet.

---

# 5. FORMATION MATCHUP ROADMAP

## FM-0 — COMPLETE

Governance / no-market-leakage rules materialized.

## FM-1 — COMPLETE

Canonical Formation Matchup Engine materialized.

Current research state:

- 299 verified pre-kickoff formation-pair fixtures;
- 81 directional matchup types;
- 8 matchups with n >= 8;
- side-specific goals / corners / shots / SOT available;
- odds consumed = false;
- decision weight = 0.

## FM-2 — COMPLETE AS RESEARCH CHALLENGER / RESEARCH_HOLD

Integrated into Corners / Team Corners V4-022.

Current blocker:

```text
39 formation-adjusted evaluations < 100
```

Plus Away-side MAE does not yet beat baseline.

## FM-3 — IMPLEMENTED / RESEARCH_HOLD

Current OOS evidence:

### Goals
- observed = 299;
- baseline evaluations = 298;
- formation-adjusted = 79;
- Home / Away / Total MAE do not improve.

### Shots
- observed = 217;
- baseline evaluations = 216;
- formation-adjusted = 49;
- no Home / Away / Total MAE lift.

### SOT
- observed = 219;
- baseline evaluations = 218;
- formation-adjusted = 49;
- no Home / Away / Total MAE lift.

Conclusion:

**Nominal formation alone has not demonstrated additive OOS lift for Goals, Shots or SOT.**

Do not promote it.

## FM-4 — IMPLEMENTED / RESEARCH_HOLD

Current prior-style diagnosis:

- source fixtures = 299;
- unique source teams = 520;
- source-team overlap with tactical history = 100%;
- minimum prior style matches per team = 3;
- source fixtures with both teams meeting complete n>=3 style history = 0;
- production enabled = false;
- decision weight = 0.

Therefore the blocker is historical depth, not team-ID overlap.

### FM-4 tactical backfill infrastructure

Implemented:

- guarded offline tactical-history backfill module;
- protected runtime route;
- bounded provider budget;
- max 8 statistics fixtures per run;
- finalized historical fixtures only;
- strict cutoff before target cohort;
- provenance / retrieval time;
- no retroactive prediction rewrite;
- no retroactive market / CLV / bet creation;
- automatic FM-4 rebuild workflow.

### Historical fixture discovery — COMPLETE

Provider-backed historical fixture discovery is now operational and bound to verified cohort seasons.

Current backfill state:

- target teams = 520;
- local eligible historical candidate pool = 66;
- materialized tactical-history fixtures = 12;
- latest guarded batch attempted = 8;
- latest batch captured = 2;
- latest batch incomplete provider-stat rows = 6;
- provider errors = 0;
- latest daily provider remaining = 5,099.

The earlier `candidate_pool_rows = 0` blocker is therefore resolved.

The offline `httpx` / package-bootstrap dependency issue is also resolved; recent FM-4 workflows complete successfully.

### Current FM-4 depth

Current prior-style coverage:

- source fixtures with both teams having any prior style history = 11;
- source fixtures with both teams having complete n>=1 style history = 11;
- source fixtures with both teams having complete n>=2 style history = 1;
- source fixtures with both teams having complete n>=3 style history = 0;
- style-eligible source rows = 0.

So FM-4 remains blocked by **actual prior-history depth**, not discovery, IDs, workflow dependencies, or provider quota.

### Backfill yield optimization v1.2

Implemented and unit-tested:

- do not re-request historical fixtures already proven to return incomplete statistics;
- persist incomplete-stat attempts as research diagnostics;
- use successful-statistics leagues only as a tie-breaker when candidate undercoverage is otherwise equal;
- retain the 8 statistics-call limit per run;
- retain the provider reserve guard;
- no model/market/CLV/bet state is altered.

Production remains disabled.

## FM-5 — BLOCKED

Sporting Projection Integration is not allowed until a formation/style challenger demonstrates OOS lift and sample sufficiency.

## FM-6 — BLOCKED

Market validation of the formation/style feature must occur only after FM-5 raw sporting projection exists.

## FM-7 — BLOCKED

No production review until OOS / CLV / settlement / calibration gates pass.

---

# 6. CURRENT PRIORITY ORDER

The current roadmap priority is:

## P0 — FM-4 HISTORICAL DEPTH ACCUMULATION

Discovery and workflow repair are complete.

Current work:

1. continue guarded pre-cohort tactical backfill every 30 minutes;
2. skip known no-stat historical fixtures instead of wasting repeat provider calls;
3. prioritize undercovered teams first;
4. retain the 8-statistics-request batch cap and daily reserve guard;
5. materialize every successful capture with provenance;
6. rerun FM-4 after every batch;
7. continue until a meaningful prior-style sample exists;
8. do not reduce the n>=3 prior-team-style gate.

Current target for review:

```text
source fixtures with BOTH teams complete prior style n>=3 >= 100
```

This remains the highest-value immediate Sporting-Layer infrastructure task.

---

## P1 — 1X2 + TEAM TOTALS EVALUATION

Run continuously in parallel:

- immutable outcome grading;
- settlement coverage;
- model calibration review;
- True CLV;
- ROI as secondary/noisy evidence;
- league/Data Tier/Availability segmentation;
- independent graded-bet sample accumulation.

Do not interpret research shadow rows as 100 independent bets.

---

## P2 — FT TOTALS + BTTS TRUE CLV MATURATION

Continue strict exact-market close capture.

Current gates:

```text
FT_TOTALS 6 / 50
BTTS     17 / 50
```

No semantic relaxation.

---

## P3 — CORNERS / TEAM CORNERS FORMATION SAMPLE

Continue toward:

```text
formation-adjusted n >= 100
```

Then require:

- side-specific Home + Away lift;
- total lift;
- league stability;
- venue stability;
- OOS;
- True CLV;
- settlement.

---

## P4 — FM-4 STYLE + PERSONNEL ABLATION

After real historical depth exists:

1. rerun prior-only style profiles;
2. measure style ablation for Goals / Shots / SOT;
3. add personnel / coach continuity only when prior data are sufficient;
4. require OOS improvement;
5. keep market prices completely outside the feature.

---

## P5 — FM-5 RAW SPORT PROJECTION INTEGRATION

Only if FM-2/FM-4 features demonstrate stable OOS lift:

Expose separately:

```text
baseline sport projection
formation adjustment
style/personnel adjustment
final RAW SPORT PROJECTION
```

No market input in any of those components.

---

## P6 — FORMATION MARKET VALIDATION / SHADOW

After FM-5:

- exact market;
- exact threshold;
- exact price;
- bookmaker/source;
- timestamp;
- closing quote;
- CLV;
- settlement;
- calibration.

FT Corners and Team Corners remain separate families.

---

## P7 — SECONDARY MARKETS

Only after core markets are healthier:

1. rework 1H challenger;
2. rework 2H challenger;
3. Cards;
4. Player Props.

Do not expand breadth faster than evidence quality.

---

# 7. CALIBRATION RULE

The Project rule remains:

```text
100 graded bets
```

before material methodology recalibration, except for a proven:

- implementation bug;
- data bug;
- source mapping error;
- formula error.

Calibration is market-specific.

Examples:

- Corners do not recalibrate 1X2;
- Team Totals do not automatically recalibrate BTTS;
- 1H does not recalibrate full-game totals.

---

# 8. WHAT COUNTS AS SUCCESS

The engine is not judged by number of picks.

Success means:

1. stronger sporting understanding;
2. raw Soccer projections independent of odds;
3. stable OOS prediction quality;
4. verified availability handling;
5. disciplined price comparison;
6. positive/process-consistent CLV after adequate sample;
7. calibrated probabilities;
8. reproducible immutable decisions;
9. controlled portfolio exposure;
10. no hindsight rewriting.

Zero bets remains a valid output.

---

# 9. DOCUMENTATION AUTHORITY

Current governing / active Soccer documentation:

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
```

Older repository documentation that claims fixed accuracy, completion percentages, or production readiness is historical reference only unless explicitly reconciled with these files.

---

# 10. IMMEDIATE NEXT ACTION

The next engineering action is:

```text
FM-4 guarded tactical-history accumulation
→ increase both-team prior-style n>=3 coverage
→ rerun style/personnel ablation
→ require OOS lift before FM-5
```

while these continue automatically in parallel:

```text
1X2 outcome/calibration accumulation
Team Totals OOS/settlement accumulation
FT Totals True CLV maturation
BTTS True CLV maturation
Corners formation sample accumulation
```

After FM-4 obtains real prior history, rerun style/personnel ablation.

Only demonstrated OOS lift may advance to FM-5 raw sporting projection integration.

---

**SPORT FIRST. MARKET SECOND.**


---

# 11. CODE-COMPLETE ROADMAP UPDATE — 2026-10-07

This section supersedes earlier formation/corners counts and implementation labels in this file where they conflict.

## Engineering status

The formation roadmap is now implemented through FM-7 as guarded research/review contracts:

```text
FM-4  STYLE + PERSONNEL OUTCOME ABLATION      IMPLEMENTED / RESEARCH_HOLD
FM-5  READINESS + RAW SPORT PROJECTION        IMPLEMENTED / EVIDENCE_BLOCKED
FM-6  EXACT MARKET VALIDATION                 IMPLEMENTED / UPSTREAM_BLOCKED
FM-7  PRODUCTION REVIEW + ACTIVATION PLAN     IMPLEMENTED / UPSTREAM_BLOCKED
```

The canonical lifecycle artifact is:

```text
soccer_edge_state/analysis/formation_lifecycle_health_v1.json
```

Current lifecycle status:

```text
SOFTWARE_CONTRACTS_COMPLETE_EVIDENCE_ACCUMULATING
```

This does **not** mean production-ready.

## Current FM-4 evidence

```text
both-team complete prior style n>=3 = 2 / 100
style-eligible source rows = 2
tactical history fixtures loaded = 2441

current both-XI rows = 18
both-team prior confirmed-XI rows = 0
personnel GOALS outcome ablation eligible rows = 0 / 100
```

FM-4 remains research-only.

## Current Corners evidence

```text
FT Corners formation-adjusted = 44 / 100
Team Corners formation-adjusted = 44 / 100

Home MAE improves = true
Away MAE improves = true
Both-side MAE improves = true
Total MAE improves = true

stable FT-Corners lift leagues = 0 / 2
league/venue stability ready = false
```

The previous Away-side MAE blocker has improved, but sample/stability gates still block advancement.

## Current FM-5 readiness

```text
ready_components = []
integration_review_allowed = false
production_enabled = false
decision_weight = 0
```

No formation/style/personnel component may enter the raw sporting projection until its OOS evidence gate passes.

## Current FM-6 rule

FM-6 code exists, but it can only evaluate an exact market **after** a valid FM-5 sporting candidate exists.

It cannot use price, bookmaker, odds, CLV or breakeven data to construct the raw sporting projection.

## Current FM-7 rule

FM-7 code exists, but production review requires all evidence gates plus the Project minimum prospective sample.

Current prospective graded sample:

```text
0 / 100
```

Automatic activation is prohibited.

## Remaining code-completion deployment task

The protected FM-4 personnel-history route exists in source but has not yet reached the deployed Render runtime.

Until Render is synchronized, the personnel backfill workflow receives HTTP 404 and the canonical personnel history report cannot materialize.

Do not work around this deployment issue by lowering personnel-history requirements.



---

# 12. RENDER/PERSONNEL DEPLOYMENT RESOLVED — 2026-10-07

The deployment blocker documented in the previous section is resolved.

Render workspace:

```text
Baahl
```

Service:

```text
soccer-edge-api
```

Live deployed engine commit:

```text
2869ca085fcdad826280247653319ec4ce2572bd
```

The protected FM-4 personnel-history route is now available and the post-deploy workflow completed successfully.

First successful batch:

```text
attempted = 8
captured = 7
incomplete = 1
errors = 0
provider requests = 8
materialized canonical personnel history fixtures = 7
```

Current FM-4 personnel evidence remains below review gates:

```text
both-team prior confirmed XI rows = 0 / 100
both-team previous-coach comparable rows = 0 / 100
both-team last-3 core-return rows = 0 / 100
personnel GOALS outcome ablation = 0 / 100
```

Therefore the roadmap is now:

```text
CODE IMPLEMENTATION COMPLETE THROUGH FM-7
+
RENDER ROUTE SYNCHRONIZED
+
PERSONNEL BACKFILL OPERATIONAL
→ continue evidence accumulation
→ no production promotion
```

No threshold was reduced, no historical prediction was rewritten, no retroactive bet/market was created, and the personnel component retains zero decision weight.
