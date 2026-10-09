# Soccer Edge — complete market-family audit (2026-10-09)

## Scope and rules
Source: canonical soccer-edge-state research artifacts and their GitHub commit dates, plus inspected GitHub Actions workflow definitions. This is an audit, not a new betting card. Sport-first research and real, strictly later pre-kickoff provider quotes are required for market CLV. Never promote or rewrite a historical prediction on the basis of this memo. The thresholds here are the thresholds from the model's respective reports.

**Key distinction:** fresh runtime and CLV are not the same as fresh sporting OOS, fresh settlement or production approval. A market can have high OOS sample and zero valid True CLV; a commercial settlement can remain unchanged legitimately if no new BET/LEAN decisions exist.

## Market-by-market current evidence

| Family | Raw sporting research / OOS | Canonical True CLV | Status / hard hold |
| --- | --- | --- | --- |
| 1X2 | Current model 870 historical OOS; matched multiclass calibration 670 OOS rows; temperature challenger improves Brier and log-loss | 53 / 50 unique fixtures | CALIBRATION_REVIEW_ELIGIBLE, manual review only; draw buckets sparse, commercial settled decisions only 10; model-selection challengers not formally approved |
| TEAM_TOTALS | 500 fixtures / 6000 derived role-line probability rows, required 0.5/1.5/2.5 present | 514 observed quote comparisons, from 73 fixtures | OOS_REVIEW_ELIGIBLE, manual review only; account for within-fixture correlation |
| BTTS | 870 current-model OOS, challenger calibrator improves Brier/log-loss; legacy 500-calibration ECE 0.12049 | 20 / 50 fixtures | RESEARCH_HOLD; 30 more strict CLV rows and bucket reliability review |
| FT_TOTALS | 228 model outcomes, but supported market line samples 1.5=35, 2.5=92, 3.5=68, other Asian lines absent | 8 / 50 fixtures | RESEARCH_HOLD; pricing/line diversity and strict close gaps |
| 1H | 220 walk-forward OOS; challenger Brier 0.2484 vs baseline 0.2459, challenger log-loss 0.6907 vs 0.6853 | 0 / 50 | RESEARCH_HOLD; challenger worse, 5 of 6 required lines missing, gate disabled |
| 2H | 604 walk-forward OOS; conditional challenger Brier 0.254011 vs 0.253033, log-loss 0.701232 vs 0.699253; MAE worse | 0 / 50 | RESEARCH_HOLD; model underperforms baseline, missing live red-card and dedicated HT market proof |
| FT_CORNERS | 266 OOS, only 44 eligible adjusted / 100 minimum; 222 excluded, of which 162 have fewer than 8 verified previous matchup examples | 0 / 50 | RESEARCH_HOLD; no verified market close, no stable league lift |
| TEAM_CORNERS | Same 44 side-adjusted OOS / 100; home and away MAE direction improves but sample inadequate | 0 / 50 | RESEARCH_HOLD; parent FT corners, venue stability and strict market history |
| Match yellow cards | 254 OOS / 200 minimum | 0 / 50 | RESEARCH_HOLD; 0/100 referee-adjusted OOS, no verified compatible sportsbook scoring/price history |
| Match red cards | 184 OOS, 500 review and 1000 actionable minimum | 0 / 50 | RESEARCH_HOLD; 0 referee-adjusted OOS, rare event calibration and line history unavailable |
| Player shots | 63 observed player-games / 500 | 0 / 50 | RESEARCH_HOLD; XI-matched exact lines and later strictly valid closes absent |
| Player SOT | 61 / 500 | 0 / 50 | RESEARCH_HOLD; no mapped exact line and strict close |
| Anytime scorer | 61 / 1000 | 0 / 50 | RESEARCH_HOLD; price/confirmed XI/model probability alignment |
| Player assists | 61 / 1000 | 0 / 50 | RESEARCH_HOLD; limited modeled sample and exact price history |
| Player cards | 43 / 1000 | 0 / 50 | RESEARCH_HOLD; card scoring compatibility and market overlap |
| Goalkeeper saves | 35 / 300 | 0 / 50 | RESEARCH_HOLD; XI and goalkeeper model line availability |

The props OOS counts above are player-game rows, not unique football matches. Across props, the current separate OOS report has 324 player-game rows across 21 fixtures, so multiple players/lines within a match are correlated.

### Non-primary/derived research (not active BET approvals)
- Double Chance: 500 historical fixtures, 1500 three-selection binary research rows. Parent 1X2 production approval and dedicated priced True CLV are missing.
- Draw No Bet: 500 completed, 384 non-draw evaluated fixtures; actionable minimum 400 non-draw. Parent 1X2 and push-aware market validation remain open.
- Asian Handicap: 500 fixtures, 5500 line-settlement research rows. No verified exact handicaps/price/strict closing history to authorize.
- Correct Score: 500 fixtures; top-1 exact-score accuracy 0.112 and research multiclass negative log likelihood 3.292713. Market-priced OOS/strict CLV not verified.
- FT Goals relative-strength shadow: 245 chronological OOS, challenger **worse** than canonical Brier/log loss on 1.5/2.5/3.5 and higher goals MAE. Keep canonical model; do not promote shadow.

### Canonical strict CLV failure funnel
The materialized canonical CLV report (Oct 9) has 595 comparable rows across 1X2=53, BTTS=20, FT_TOTALS=8, HOME_TT=248, AWAY_TT=266. Team totals are 514 rows in 73 unique fixtures. Each counted True CLV row requires an actually later provider quotation before kickoff, not a duplicated unchanged quote. Recent primary CLV audit shows the major loss modes for BTTS and FT Totals are NO_LATER_PROVIDER_UPDATE and NO_LATER_PREKICKOFF_MARKET_SNAPSHOT. Do not widen the definition or backfill after kickoff.

### Player Props execution vs persistence — verified software defect
Inspected player-props-clv-validation.yml: schedule daily 09:40 UTC, authorization to call protected Render endpoint and validate/log JSON, but **the workflow terminated without a step to persist the newly returned report or its strict tracking rows**. Persisted player_props_clv_v4_report.json was last committed on Sep 27 even though player_props_oos_v4_report.json advanced Oct 9. Historical report shows 451 model-based player signal rows and zero verified later strict closes; close skip reason for all 451 is NO_LATER_STRICT_PLAYER_PROP_CLOSE.

Remediation in this PR: persist validated response safely in soccer-edge-state, preserve exact returned row-level evidence, never synthesize a quote, attach refresh timestamp. This fixes report durability but cannot force the bookmaker to publish a later valid quote.

### Market taxonomy audit
research_derivative_market_audit.json (Oct 9) scanned 5000 candidate rows from 59 fixtures, classified 4853, left 147 unclassified; observed market names include Shots.1x2 (87) and Player to Score or Assist (60). 21 fixtures have pre-kickoff confirmed XI and 17 player-aligned XI coverage. Mapping must be verified by exact market semantics and provider ID before operational use; no fuzzy relabeling for settlement.

### Commercial outcome evidence and stale dates
market_performance_summary.json (latest commit Sep 24) records 29 commercial/research-classification decisions (not a current live pick report); individual market families have fewer than 20 settled examples. The Oct 3 Postgres settlement source reported 12789 refresh events, 9035 pre-kickoff, **0 BET/LEAN classified events** and 0 fresh actionable settlements. Therefore an unchanged commercial result is **not proof of a broken scheduler**. Still, review the scheduled settlement job separately for actual failure/lag before relying on its absence of changes.
Canonical calibration report latest commit Sep 29, 1H/2H validator reports Oct 2, Phase14 Cards Oct 6, latest Phase15 Oct 9, and current main CLV Oct 9. Some are expected to remain identical if research data do not change; the new monitor warns on old artifacts, but does not assume every unchanged scientific report is a failed job.

## Change set
- Daily Player Props CLV workflow now persists its bounded zero-or-more **actual** True CLV rows and a timestamped research summary to the state branch. No false CLV from diagnostic market snapshots.
- Read-only all-market watchdog every six hours: every family status, CLV sample, source-vs-derived parity, props diagnostics age, settlement zero-actionable case, taxonomy row accounting, and production-safety flags.
- Deduplicated incident, Actions-readable summary and machine-readable artifact retained 30 days.
- Unittests protect zero-bet outcomes, zero CLV research holds, true-CLV count divergence, stale prop report, taxonomy count mismatch and disabled production promotion.

## Outstanding engineering tasks, ordered by impact
1. **Observe the first persisted Player Props CLV daily run** and investigate genuine missing strictly later close. If no real later quote, leave market WATCH.
2. **Diagnose 1H/2H model error** by same-fixture and league/stage segmentation before considering feature recalibration; no betting promotion.
3. Improve **primary later-provider-update capture** in 1X2, FT Totals and BTTS with source-based timing/ID auditing; do not simulate provider updates or relax True CLV.
4. For cards, verify referee assignment and exact bookmaker scoring systems, then capture later strict prices and OOS referee cohorts prospectively.
5. For player props, improve exact player ID/XI/line agreement and genuinely contemporaneous market snapshots; investigate late price quote collection and compare vendor coverage before seeking new paid providers.
6. Fix historically eligible corners coverage source (postgame tactical observations vs proper pre-kickoff XI/formation snapshots), independently of market pricing.
7. Bring archived audit reports up to date only when their upstream evidence has actually changed; do not allow report generation time to masquerade as new betting evidence.
8. Assess 1X2 and team totals for **manual** model/versioned review only after coverage, correlated sampling and competition-specific calibration checks.

## Closeout / invariants
No historical probabilities recalculated; no retroactive bets, True CLV, XI or lineups; no provider limit change; no model weights transferred from football/baseball; no silent promotion. A mature market may still be PASS if the price offers no edge. Critical alert does not create bets.

Operational rollout is complete only after PR merge, passing automated tests, a successful real scheduled Player Props CLV materialization, and confirmation of alerts/family report count consistency.
