# Soccer Edge Maturation Incident Audit — 2026-10-03

## Scope
End-to-end audit of persisted maturation flow:
Postgres refresh events / market snapshots -> signal ledger materialization -> canonical Phase17 True CLV -> V4 family validations -> Control Tower.

## Confirmed root cause A — signal ledger materialization frozen
- Canonical state branch: `soccer-edge-state`.
- `signal_ledger_summary.json` report refreshed at 2026-10-04T03:11:21Z.
- Fresh live evidence exists: `last_evidence_at_utc=2026-10-04T02:31:34.988101Z`, event_count=16, database_persisted=true.
- But `last_materialized_ledger_row_at_utc=2026-10-01T16:55:37.111597Z`.
- Rows remain 21,314. Therefore DB ingestion is alive while canonical ledger materialization is stale.

### Failure mechanism
Scheduled workflow `.github/workflows/v215-signal-ledger-postgres-materialization.yml` runs from `main`.
GitHub OIDC therefore emits `ref=refs/heads/main`.
The runtime v215 route required `ref=refs/heads/soccer-edge-mcp-v1`, even though checkout of that branch cannot change the OIDC triggering ref.
Result: HTTP 401 before any delta page is returned:
`Only soccer-edge-mcp-v1 v215 materializer is allowed`.

Observed continuously in scheduled v215 runs from 2026-10-01 onward. Earlier initial failures included HTTP 502; after the endpoint/auth wiring existed, the persistent blocker became 401.

## Fix v219
Commit: `f5c7a099a74705e4853deb0d8fddaf80281bb335` on `soccer-edge-mcp-v1`.
Change is isolated to v215 route authorization:
- repository identity remains required;
- exact v215 workflow_ref remains required;
- accepted ref now matches the actual scheduled workflow trigger: `refs/heads/main`.
No model, threshold, gate, provider budget, strict-close rule, historical probability, or canonical BET logic change.

## Confirmed root cause B / independent maturation behavior
Canonical Phase17 is NOT frozen. Successful refresh at 2026-10-03T19:02Z read current Postgres and persisted:
- comparable True CLV total: 89
- 1X2: 48/50
- BTTS: 17/50
- FT Totals: 6/50
- HOME_TT: 9 rows / 4 unique fixtures
- AWAY_TT: 9 rows / 4 unique fixtures
The unchanged primary-family counts are therefore not solely a stale dashboard artifact.

Phase17 current funnel:
- 2,000 pipeline market rows loaded
- 1,858 derivative market rows loaded
- FT Totals mapped 938; priced 655; True CLV 6
- FT Totals failures: 300 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT + 349 NO_LATER_PROVIDER_UPDATE
- 1H: 9 mapped/priced; 9 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT
- 2H: 9 mapped/priced; 9 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT
- FT Corners: 9 mapped/priced; 9 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT
- Team Corners: 8 mapped/priced; 8 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT

Interpretation: repairing v215 is required to restore ledger/OOS freshness, but it will not by itself manufacture True CLV. Phase17 is independently enforcing strict later-snapshot/provider-update semantics correctly.

## Stale artifacts found
`ft_totals_maturation_audit_v4.json` is stale (generated 2026-09-27) and must not be used as current evidence.
Some family validation artifacts can lag canonical Phase17; canonical CLV report + fresh validation workflow outputs should be preferred for current maturation state.

## Operational rule for future audits
Before declaring maturation "not moving":
1. Check `last_evidence_at_utc`.
2. Check `last_materialized_ledger_row_at_utc`.
3. Check latest v215 workflow conclusion/log.
4. Check latest canonical Phase17 run and family_counts.
5. Check V4 Market Validation Refresh completion.
6. Compare Control Tower against those canonical sources.
7. Treat stale audit artifacts separately from live evidence.
Never recommend changing gates/thresholds/budget/strict-close merely because counters are flat.

## Next validation
After v219 is live:
- scheduled v215 must return HTTP 200;
- delta cursor must advance beyond 2026-10-01T16:55:37.111597Z;
- merge and persist steps must execute;
- `last_materialized_ledger_row_at_utc` and ledger row count must advance;
- then family validations/Control Tower should be re-read.


## Checkpoint v219-v221 — 2026-10-04
- v215 canonical materialization repaired end-to-end. Final successful run: 37178111076 / job 111364895511. Canonical ledger advanced to 23,758 rows, last_materialized_ledger_row_at_utc=2026-10-04T03:47:09.053906+00:00, provider_requests_added=0.
- v219 auth: f5c7a099a74705e4853deb0d8fddaf80281bb335. v219.1 bounded delta pages: dceda5642fb70a1d810aca5f0422e689766c9a65. v219.2 delta contract: 5e5afa3d6b2ab18130fe48bce7e6f316020f4472. v219.3 summary guard: 23d1cfd80d805bb6dae4bd79f1d666428830e792.
- Post-repair Phase17 run 37178605858 succeeded; canonical primary counts remained 1X2=48, BTTS=17, FT_TOTALS=6. This proves ledger repair did not fabricate True CLV.
- v220 canonical temporal audit run 37180733390 succeeded. Pre-history current cohort produced only 4 1X2 True CLV; canonical historical merge returned 1X2=48. Current 1X2 skip reasons: INVALID_ENTRY_PRICE=17; NO_LATER_PREKICKOFF_MARKET_SNAPSHOT=230; NO_LATER_PROVIDER_UPDATE=231. provider_requests_added=0.
- v220.3 commit 31cdad45fd1c8baf3fc29210f7ddbc8f39dd029e fixes skip_reason_family_counts orientation and persists the v220 artifact in the canonical Phase17 commit.
- Scheduler audit found primary CLV maturation is bounded to a 55-minute lookahead, a global backlog LIMIT 80 shared by 1X2/FT_TOTALS/BTTS, and max 8 maturation provider calls per tick. These are candidate starvation mechanisms; no limits/budget were changed without measurement.
- v221 initial telemetry commit 91ec64b376fd773728d0ce4ee691a3fb029a438d had incomplete runtime wiring and failed V4 tests; it was superseded before becoming the accepted checkpoint.
- v221.1 commit 6d7c712095810b977ec8b3696523dfae4acb8e9e fixes the wiring and exposes selected family mix + global limit saturation alongside existing evaluated/refresh/not-matured/budget-exhausted counters. It adds zero provider calls and changes no model, threshold, gate, strict-close rule, canonical BET logic, or provider budget.
- Next decision must use live v221.1 telemetry from a normal scheduled tick. Do not spend a manual provider tick merely for diagnostics. If global LIMIT starvation is proven, fix family fairness/backlog selection before increasing provider budget. If budget exhaustion dominates, evaluate bounded reallocation/reuse before any cap increase.


## v222 preventive capture fix — 2026-10-04
- Pre-fix normal scheduler evidence at 2026-10-04T05:17:37Z: 61 total API calls, 11 price-resolver calls, 7,169 daily remaining, primary maturation candidates=0, primary maturation calls=0, budget_exhausted=0. The blocker is therefore before provider spend in that tick.
- Phase17 historical audit simultaneously reports 1X2 NO_LATER_PREKICKOFF_MARKET_SNAPSHOT=230 and NO_LATER_PROVIDER_UPDATE=231. Past fixtures cannot be repaired retroactively under strict-close; engineering must prevent future missed close observations.
- Root mechanism: primary maturation loader admitted upcoming fixtures only inside 55 minutes, leaving too few scheduled opportunities to obtain a genuinely later provider update before kickoff.
- v221.3 54493ae74332dcdcafdd3594d49f7f62afbf5821 passed V4 Runtime Tests and preserves BACKLOG_V2 source compatibility while exposing saturation telemetry.
- v222 2efceb04062ee6f2ef3d55073f8617f21aa722e7 widens ONLY primary CLV maturation eligibility from 55 to 90 minutes. Max primary maturation calls remains 8/tick; global provider cap and strict provider_update semantics remain unchanged; no model/threshold/gate/canonical BET changes.
- Validation: normal scheduled ticks should expose primary 1X2 candidates earlier while staying inside existing caps. Subsequent Phase17 cohorts should gradually show more genuine current 1X2 True CLV. Historical 230/231 counts will not disappear retroactively.


## v223 / v223.1 — primary CLV preventive capture + provider chronology proof (2026-10-04)

- v223 runtime commit `348b4137079573acc15c0bbb4aeb175f8d32e28f` widened only `SOCCER_PRIMARY_CLV_MATURATION_LOOKAHEAD_MINUTES` from 90 to 180. Primary max remains 8 calls/tick; global provider cap, strict-close, models, thresholds, gates and canonical BET logic are unchanged.
- Render deploy `dep-db0vp22d0e5s73de9hb0` reached LIVE at 2026-10-04T07:13:43.615095Z. Runtime CI `37185114559` succeeded.
- Manual v223 validation `37185177971` succeeded. Canonical state at 2026-10-04T07:20:15.262075Z showed 4 primary candidates: 1X2=1, BTTS=1, FT_TOTALS=2; 4 primary provider calls; budget_exhausted=0; refreshed=0; unchanged_provider_updates=4. This proves the 180-minute window removed the zero-candidate scheduler starvation without increasing budget.
- v223.1 runtime commit `5d1ef483f44411d2d1a8c1cbfa2117d7be6c3a8e` added bounded diagnostic provenance only (max 8 unchanged examples): fixture, family, signal_generated_at, provider_update, resolution status, provider requests. It does not alter maturation decisions.
- Render deploy `dep-db106dpsrm7s739es4u0` reached LIVE at 2026-10-04T07:42:28.864705Z. Runtime CI `37186564399` succeeded.
- Manual v223.1 validation `37186650644` succeeded. Canonical state at 2026-10-04T07:43:58.678850Z showed 6 candidates (1X2=1, BTTS=1, FT_TOTALS=4), 6 provider calls, budget_exhausted=0, refreshed=0, unchanged_provider_updates=6.
- Provenance proves strict chronology is working rather than falsely rejecting a later update. Examples: fixture 1643004 1X2 signal 2026-10-04T06:06:37.137651Z vs provider_update 2026-10-04T04:12:19Z; fixture 1633999 BTTS signal 06:06:37.137651Z vs provider_update 06:00:37Z; fixture 1634001 FT_TOTALS signal 03:47:09.053906Z vs provider_update 00:00:57Z. All observed provider timestamps were older than their signal anchors.
- Conclusion: current blocker is genuine provider chronology / timing of first valid priced signal, not provider budget, family displacement, timestamp parsing, captured_at substitution, or strict-close comparison. Do not widen the window again or increase provider calls solely to force maturation. v223.1 should remain as preventive scheduled capture; a family matures only after API-Football supplies a real provider_update strictly later than the persisted signal and before kickoff.


## v223.2–v226.17 — maturation continuity and Team Totals reconciliation (2026-10-04)
- v223.2 Phase17 sparse-checkout persistence was repaired by main commit `cbc59cc839f81c97d8dfa6312b5432494570626a`; run `37187055579` completed successfully with provider_requests_added=0.
- v224 BTTS audit run `37205396022` completed successfully: 17/50 canonical BTTS True CLV, research hold remains natural maturation rather than selector starvation.
- v225 FT Totals audit run `37205563227` completed successfully: 6/50 canonical FT_TOTALS True CLV; strict later provider chronology remains the blocker. Post-v225 Phase17 run `37205597084` completed successfully and preserved strict-close semantics.
- v226 reconciliation proved capture persistence is healthy: 1,369 strict Team Totals capture fixtures, 1,358 with persisted exact observed Team Total markets, 8 unsupported-only, 3 intelligence rows with neither exact nor unsupported, and 0 fixtures without Team Totals intelligence. Therefore the old `capture_without_modeled_signal=1315` funnel field is NOT evidence that paid Team Totals data was discarded; it reflects the bounded current Phase17 modeled-signal slice versus the much larger persisted capture history.
- Phase17 run `37208522715` completed SUCCESS end-to-end, including Postgres CLV build, v220 audit, v226 reconciliation, strict history merge and state persistence. Reconciliation artifact generated at 2026-10-04T14:48:19.120939Z.
- Runtime v226.15 `ae325b86bbf10b3ccafd47f1e8e50bf01b5cf4c5` changed reconciliation to indexed bounded probes; runtime CI `37209215768` passed.
- v226.16 `b8dcb8e7eb9b166f25ba26d44a16054bdbe51e24` upgraded canonical history merge to v1.2.0 so a bounded current Postgres refresh cannot silently delete previously validated strict True CLV evidence. v226.17 `0175f3d9fb34e48c7e1e419391d5782a132e109b` added regression tests. Runtime CI passed and Render deploy `dep-db169c7avr4c73afjt60` is LIVE on v226.17.
- Canonical post-merge CLV now contains 100 comparable strict rows: 1X2=48, BTTS=17, FT_TOTALS=6, HOME_TT=14, AWAY_TT=15. Team Totals cover 4 HOME fixtures and 5 AWAY fixtures. The merge preserved 86 previously canonical strict rows and added 0 permissive historical rows; provider_requests_added remains 0.
- Important semantic distinction: `team_totals_maturation_funnel.true_clv_unique_fixtures` and `modeled_signal_unique_fixtures` describe the bounded current Postgres build before canonical history preservation. Canonical maturation status must use post-merge `family_counts` / `unique_fixtures_by_family`; the funnel must not be interpreted as loss of persisted paid data.
- No provider budget, thresholds, gates, models, strict-close chronology, or canonical production BET logic were relaxed in this sequence.
