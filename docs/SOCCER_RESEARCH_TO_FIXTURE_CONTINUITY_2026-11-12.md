# SOCCER EDGE ENGINE — Existing Research to Every Fixture, Continuity Checkpoint
**As of:** 2026-11-12 (UTC)  
**Project:** SPORTS EDGE ENGINE — Soccer only  
**Purpose:** Keep incremental corrections on the existing data architecture; this is an operational checkpoint, not a replacement model, new betting method or deployment proof.

## Goal
Use work already running (official fixtures, cached team season splits, prior matches, FE11 period/corner/handicap history, player source/OOS, market evidence and frozen RAW models) inside the correct **fixture-level** file, with identity, league, season, venue, source availability and data timestamps intact. All eligible games should be screened or explicitly deferred; lack of evidence must not be converted into a prediction.

**Order:** SOURCE SLATE → SPORT EVIDENCE → ORIGINAL RAW SPORT PROJECTION → shortlist → independent MARKET research → price verification → conservative shrunk projections → OOS/grading. If a prerequisite fails, the state remains WATCH/RESEARCH/NOT VERIFIED (no BET).

## Verified facts (point-in-time; not cumulative or worldwide)
- [Independent prospective capture](../.github/scripts/soccer_isolated_pregame_sport_first.py) and `soccer-edge-state/soccer_edge_state/analysis/isolated_sport_first_pregame_receipt_v1.json`: 2026-11-12T20:52:41Z, **253** eligible across **106** competitions, **18** fixture attempts, **9** new frozen sport models; **32** API-Football requests in that cycle.
- Original configured scanner FE11 leagues: `{2,3,17,18,39,61,78,135,140,262,848}`. Their overlap with that **two-date** source report was **0**. This is not evidence that world football had zero games.
- The same source report contains **50** fixtures in four independently observed competition IDs: UEFA Nations League **5** (10), Friendlies **10** (19), Africa Cup of Nations Qualification **36** (12), CONCACAF Nations League **536** (9). These are data identities, **not** approvals that 50 have adequate model inputs.
- [Earlier scanner run](https://github.com/Baahl11/Soccer/actions/runs/43289523750): `eligible_games=0` in configured scope and **0** exportable FE11 profiles. It finished green because no matching fixtures can be a valid zero, not because profiles were persisted.
- [Earlier FE11 universal run](https://github.com/Baahl11/Soccer/actions/runs/43293107007): provider fixtures seen **365**, approved fixture rows **0**; existing FE11 inventory had **49** matched official finals and verified FT-corner histories. The 49 historical rows do not prove current profile readiness.
- Previous [PR #188](https://github.com/Baahl11/Soccer/pull/188) updated **owner match detail** to display earlier valid, same-fixture PostgreSQL feature evidence from `evidence_sections`, with provenance and pre-kickoff filter. Its merge into `soccer-edge-mcp-v1` was proven; Render deployment of this code is **NOT VERIFIED** here.
- [PR #189](https://github.com/Baahl11/Soccer/pull/189) merged to **soccer-edge-mcp-v1**, broadens only existing FE11 official research fixture scope for the four source-observed competitions. Offline lifecycle and registry QA passed. Live Render deployment is **NOT VERIFIED** here.
- [PR #190](https://github.com/Baahl11/Soccer/pull/190) merged to **main**, broadens the scanner's configured research league scope and adds `scripts/soccer_scanner_scope_audit.py`. Scanner full unit tests and multimarket-profile QA passed. A post-merge scanner run with positive actual profiles is **NOT YET VERIFIED** here.

## Existing code to reuse (do NOT rebuild)
| Stage | Owner/branch | Existing component |
|---|---|---|
| Fixture registry and full-slate accounting | `soccer-edge-mcp-v1` | `mcp_gateway/soccer_sport_queue_v1.py`, `mcp_gateway/fe11_registry_next_day_v1.py` |
| Lightweight prospective sport models | `main` | `.github/scripts/soccer_isolated_pregame_sport_first.py` and existing PG import workflow |
| Season and recent historical research | `soccer-edge-mcp-v1` | `mcp_gateway/fixture_owner_team_context_v1.py`, `fixture_owner_recent_context_v1.py`, `soccer_verified_sport_read_v1.py` |
| FE11 all-eligible approved-source lifecycle / FT corners | `soccer-edge-mcp-v1` | `fe11_universal_worker_v1.py`, `fe11_universal_lifecycle_v1.py`, `fe11_universal_persistence_v1.py` |
| Periods, FT corners, Asian +1 vs three-way +1 examples | `main` and gateway | `tools/sports_edge_scanner_v05/sports_edge/multi_market_sport_profile.py`, `multimarket_export.py`, `mcp_gateway/multimarket_history_bridge_v1.py` |
| Exact model/market and player research | Both | Existing market registry/validation and FE11 player-prop modules; no unverified XI/minutes, price or bookmaker claim |
| Per-fixture readout | `soccer-edge-mcp-v1` | `mcp_gateway/subscriber_contract_v2.py`, `subscriber_frontend_v2.py`, `/app/api/v2/match/{fixture_id}` |

## Sequential acceptance gates — complete these in order
1. **DEPLOYMENT:** verify Render is executing the commit containing #188 and #189; do not infer deploy from GitHub merge or CI alone. If no authorized workspace/service is selected, deployment remains NOT VERIFIED.
2. **LIVE FE11 CAPTURE:** on next official existing schedule/authorized run, inspect `source_date_audit` and `approved_league_fixtures` per date. Confirm provider counts, actual eligible source rows and DB persistence. Absence of matches on a day is acceptable if the source confirms it. Preserve <=12 total calls/tick and original quota reserve.
3. **LIVE SCANNER:** inspect the **new** `SCOPE_OVERLAP`/zero/stale audit (two-date comparison, not a one-date scanner forecast); then the scheduled scanner's same-day `eligible_games`, `sports_screened`, `historical_market_deep_dives`, and `FE11 transport reports`. Verify no API cap increase or BET promotion.
4. **EXACT MATCH JOIN:** where scanner sends a report, verify the backend import ACK counts (`validated_reports`, `newly_persisted`, explicit reject reasons), `soccer_multimarket_history_profiles` identity and source timestamp, and `/app/api/v2/match/{fixture_id}` owner-only presentation. Do not count exported-but-rejected profiles as persisted. No post-kickoff observation can become a pregame forecast.
5. **MISSING-MARKET COVERAGE:** for goals/xG, corners, cards, shots/SOT and players, reconcile available historical snapshots, required source/split/sample, capture time and original model status separately. Missing market-family evidence must show `DATA_INSUFFICIENT` / `NOT VERIFIED`, never 0 or a guessed probability. Existing postgame player/backfill schedules can remain disabled for memory safety; do not restart them without cgroup/capacity proof.
6. **INDEPENDENT OOS & VALUE:** retain immutable prediction before kickoff and official regulation FT joins; enforce independent per-market calibration, availability confidence, exact priced selection/line/bookmaker/timestamp/rights. No verified executable price or production OOS -> WATCH/RESEARCH, **not BET**. No weight changes based on this integration.

## Change control and non-goals
No new paid provider subscription. Do not fork an alternative soccer model, train on market odds, cross-contaminate other sports, enable experimental player backfills without resource testing, or claim coverage is 100%. Never rewrite frozen forecasts or transfer a completed fixture's later-observed features into a historical pregame snapshot.

**Meaning of “done”:** every eligible fixture has either a real source-backed sporting dossier and independent states for research families, or an explicit reason/blocker; all displayed data can be traced to source and as-of time, and publisher/BET gates remain independent. The three merged PRs are **necessary corrections**, not a proof that this entire objective has been reached.
