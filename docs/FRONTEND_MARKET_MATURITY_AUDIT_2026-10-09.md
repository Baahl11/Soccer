# FE Market Maturity — Frontend-to-Research Audit (2026-10-09)

## Verified route topology
- `/app` and `/app-v2`: current customer-facing V2 app; contains BET-only Performance and a limited OOS validation table, but previously lacked a market maturation route/page.
- `/app-preview/maturity`: authenticated, PRO/OWNER-gated V232 family-level research endpoint.
- `/app-v3-react`: React/Vite preview. Performance nav remains disabled and previously had no maturity view.
- Operator Control Tower: uses `maturity_snapshot_v4`, persisted-state watchdogs, evidence freshness and report baseline. Do **not** conflate this operator plane with customer BET performance.

## Delivered in this change
1. `GET /app/api/v2/maturity` is a read-only protected alias of the existing maturity endpoint (same entitlement check; no public raw data leak).
2. `subscriber_maturity_v232` retains its nine family aggregates and adds 21 explicit market inventory rows:
   1X2, BTTS, FT Totals, Home/Away Team Totals, 1H, 2H, FT/Team Corners, Yellow/Red Cards, six player props (Shots, Shots on Target, Goalscorer, Assists, Player Cards, Goalkeeper Saves), Double Chance, Draw No Bet, Asian Handicap and Correct Score.
3. Every market row displays model/OOS evidence when independently sourced, mapped count, observed priced count, strict True CLV, parent research phase, next blocking gate and source file. Unknown child-model samples and unknown strict closes are **NOT VERIFIED**, never inherited from a parent report.
4. Customer V2 navigation and React V3 preview navigation render market maturity separately from realized BET performance.
5. New regression suite requires no synthetic CLV, no parent-to-child OOS sample cloning, no implicit production promotion, route/auth coverage, Python compilation, render-time JavaScript syntax and React build.

## Still *not* solved by a frontend patch
- Raw evidence freshness for reports and fixture-level artifact lineage must be verified live after a real workflow run. Loading a JSON report does not prove it is fresh.
- True CLV requires a later **real same-book strict pre-kickoff close**. No close must remain zero only if the canonical report confirms zero; absent report must remain NOT VERIFIED.
- Full calibration metrics (Brier, log loss, ECE, reliability buckets), actual model version history, per-league/line/venue OOS slices and lookahead leakage checks need dedicated back-end data contracts for each family, not generic UI assumptions.
- Derived markets (Double Chance, DNB, Asian Handicap, Correct Score) are listed transparently but currently do **not** have family-level research reports wired into this V232 source; their independent model evidence is NOT VERIFIED.
- Cards and player-prop per-market report fields may be partial or absent; no aggregation is interpreted as an independent 0.
- No market in this change receives BET eligibility, calibration weight changes, stake changes or automatic production promotion.
- Paid entitlements, exact prices, live deployment status and rendered mobile UX require post-deploy browser E2E confirmation.

## Governing decision rule
```
LIVE SPORT EVIDENCE
  -> PROSPECTIVE FIXTURE SNAPSHOT
  -> OUT-OF-SAMPLE MATURATION (independent sample, temporal provenance)
  -> MARKET PRICING + STRICT TRUE CLV
  -> CALIBRATION / BASELINE / BLOCKER REVIEW
  -> EXPLICIT MANUAL MODEL ELIGIBILITY (never triggered by frontend)
  -> CANONICAL BET / LEAN / WATCH / PASS
  -> CUSTOMER UI
```

Maturation is the **evidence governance system**, not a replacement for Sport First / Market Second. The subscriber UI is a read-only display of verified state. The current model gate and historical ledger remain authoritative and unchanged.

## Release acceptance
1. Validate Python + rendered JS + React TypeScript/Vite through `FE Market Maturity Contract` CI on the pull request.
2. Merge only after passing CI and verifying no route conflicts.
3. Confirm the source branch and state variables on the actual backend deployment.
4. Smoke-test `/app/api/v2/maturity`: anonymous 401, FREE 403, PRO/OWNER 200, missing state transparent.
5. Browser-validate `/app` and `/app-v3-react` on desktop and mobile, including restricted/partial/missing reports.
6. Ensure the frontend build for V3 is refreshed and deployed. The source change alone does **not** update the committed `web_v3/dist`.
7. Reconcile observed market rows against canonical artifacts and then expand per-market calibration/freshness. Do not assert scientific maturity from UI completeness.


## Sparse-evidence integrity hardening (Issue #70 follow-up)

- **Canonical zero vs missing:** numeric zero is shown only if that specific market key explicitly stores zero in the canonical CLV report. A CLV report for another market does not prove zero for an absent key, including at parent family level.
- **No phantom artifact source:** a configured report filename is not displayed as a verified source when the report could not be loaded; source, parent phase and report status are unavailable.
- **No inherited child clearance:** Home/Away Team Totals, Team Corners, Yellow/Red Cards and six player props never borrow a parent's OOS sample or aggregate CLV target as proof of child-level maturity.
- **Explicit independent blockers:** absent report, OOS, priced history and strict True CLV each generate a market-specific blocker, alongside source-report blockers.
- **Customer truth:** customer V2 and React V3 show these blockers and the production block. Parent-stage columns are labeled as parent stages. Timestamp/fixture-level artifact freshness remains NOT VERIFIED without evidence.
- **Transport status:** zero retrieved scientific artifacts is UNAVAILABLE, even if the static 21-row market inventory can render.

This is a read-only source/presentation correction. It does not approve research models for production, change decision thresholds, re-price bets, alter historical rows, or deploy the UI.

**Outstanding before release:** validate actual persisted state, test anonymous/FREE/PRO/OWNER against deployed auth, exercise mobile/desktop browsers, resolve four excluded pre-existing FE2 copy assertions independently, add real per-market calibration and timestamp/fixture provenance. React V3's BET Performance view remains disabled and is not represented as completed.


## Reconciliation with actual persisted scientific reports — 2026-10-09

Read-only source: repository `Baahl11/Soccer`, branch `soccer-edge-state`, files under `soccer_edge_state/analysis/`. The PR CI now checks out those ten files and verifies live source-to-contract parity on every relevant PR change. That is real persisted research evidence, **not** an authenticated production API or verified production browser session.

### Canonical strict True CLV snapshot
| Canonical market key | Observations | Notes |
| --- | ---: | --- |
| 1X2 | 53 | Family-level minimum 50 met; report says CALIBRATION_REVIEW_ELIGIBLE, not approved for BET |
| BTTS | 20 | RESEARCH_HOLD, minimum 50 |
| FT_TOTALS | 8 | RESEARCH_HOLD, missing relevant line coverage |
| HOME_TT | 248 | Team Totals per-side CLV observations; parent model review not auto-promotion |
| AWAY_TT | 266 | Team Totals per-side CLV observations; 71 unique away-TT fixtures |
| **TOTAL** | **595** | Same event can contribute multiple market rows; not 595 independent fixtures |

No individual canonical CLV key is present for 1H, 2H, FT/Team Corners, Yellow/Red Cards, or six Player Props. A missing key is **NOT VERIFIED**, not an invented zero. Separate research-family files report some zero observations; those are shown in `report_true_clv_rows` **separately** from canonical CLV.

### Model-source matching
| Market | Canonical model evidence | Distinction |
| --- | --- | --- |
| 1X2 | 670 temperature-scaled multiclass OOS rows | Do not substitute 870 calibration-sample rows |
| BTTS | 870 canonical OOS calibration rows | Separate Brier/log loss/ECE validation sample n=500; never silently merge |
| FT Totals | 228 model-settled decisions | Mixed validation cohort is **not** independent verified OOS |
| Home/Away Team Totals | 3,000 OOS probability rows each role | Shares 500 evaluated fixtures; rows are not 3,000 unique matches |
| 1H | 220 challenger calibration rows | Challenger does not beat baseline on Brier/log loss |
| 2H | 604 walk-forward evaluated | Challenger does not beat baseline |
| FT Corners | 44 formation-adjusted evaluations | Minimum 100 and unstable league lift |
| Team Corners | 1,464 evaluated team-corners rows | Shared 244 evaluated fixtures; league/venue stability pending |
| Yellow Cards | 254 OOS events | Referee-adjusted sample is 0, minimum 100 |
| Red Cards | 184 OOS events | Market-review minimum 500 |
| Six Player Props | 0 player-game OOS rows each | Genuine explicit zero from lower-case `prop_families`; validation incomplete |

### Source repository commit provenance (not report generation times)
The last Git commit touching each JSON file on `soccer-edge-state`, checked 2026-10-09 UTC:
- Strict True CLV: **2026-10-09 18:39 UTC**, commit `ffd8eb6`.
- 1X2, BTTS, FT Totals: **2026-10-07 18:42 UTC**, commit `8d2844c`.
- Team Totals, Player Props: **2026-10-09 18:40 UTC**, commit `d8d6461`.
- 1H, 2H: **2026-10-02 18:39 UTC**, commit `2d85293`.
- Corners: **2026-10-09 19:01 UTC**, commit `57c20b9`.
- Cards: **2026-10-06 07:05 UTC**, commit `5f3de94`.

These commit timestamps establish that files existed in the GitHub state branch; **they do not prove when model evaluation ran, which fixtures were assessed, whether the underlying source is current, or that the deployed server is consuming this exact state branch**. Until a source-specific generation timestamp / lineage contract exists, the customer API must show `SOURCE_GENERATED_AT_NOT_VERIFIED` and production stays blocked.

### Product-safety and release status
- Code and actual persisted state have been reconciled in CI. Brier/log loss/ECE fields are shown only in their correct source scope; missing calibration remains NOT VERIFIED.
- React V3 browser QA uses a mocked authenticated success payload **derived from the real state branch**, plus anonymous/forbidden/unavailable responses. It is not real deployed account E2E.
- The local analysis environment could not directly reach `https://soccer-edge-api.onrender.com/`, but a **read-only GitHub Actions probe succeeded** on 2026-10-09: `/health` HTTP 200 with deployed commit `258646340247f4dce120e630416c1935f9ed04fc` (`soccer-edge-mcp-v1` baseline); `/app` HTTP 200; `/app-v3-react` HTTP 200; anonymous `/app/api/v2/maturity` HTTP **404**, confirming PR #68 is **not deployed**. This does not verify paid entitlements or production maturity UI. Render service configuration inspection still requires confirmation of the workspace; production artifact freshness, role-specific authorization and deployed `web_v3/dist` maturity changes remain NOT VERIFIED.
- Four pre-existing FE2 copy assertions remain excluded by the current contract workflow, so a green workflow is **not** an all-suite pass.
- React V3 BET Performance remains disabled. **Issue #70 cannot be closed** until verified production data/authorization/browser testing, complete frontend market maturity requirements and remaining scientific provenance are addressed.
- Do not merge/deploy, change model weights/gates, or grant BET eligibility from the display alone.


### Verified CI browser and public-state results
- GitHub Actions `FE Market Maturity Contract` run **37989456698**: completed **SUCCESS**.
- CI checks out current `soccer-edge-state` reports and verifies the 21-row source contract; Python route permissions (401/403/PRO), rendered JavaScript and Vite build pass.
- Chromium browser QA using a **mock response with real-state rows**, run at desktop 1440x900 and mobile 390x844: 21 visible market rows, independently scrollable mobile table, no full-document horizontal overflow; 401, 403 and 503 states hide research rows. Screenshot artifacts are archived under that Actions run.
- Public deployed endpoint probe: health **200 / ok**, deployed SHA **258646340247f4dce120e630416c1935f9ed04fc**; `/app` and `/app-v3-react` **200**, maturity API **404** for anonymous request because the PR branch is not deployed.
- This verifies the running public backend commit and shell route availability; it does **not** verify a production maturity page, a 401 entitlement guard on the undeployed route, real PRO/OWNER accounts, or source branch configuration inside Render.

**Release decision remains NO-GO.** Do not close Issue #70 or merge/deploy PR #68 based on CI alone.
