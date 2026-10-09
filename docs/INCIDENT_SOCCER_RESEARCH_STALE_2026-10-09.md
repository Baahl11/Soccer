# Soccer Edge — SE-2026-10-09 research freshness incident

STATUS: Operational hardening proposed. Root cause of the Sep 22 observation cutoff NOT VERIFIED; scientific gate remains 44/100. No historical rewrites.

## Evidence confirmed October 9, 2026

- Canonical live health on October 9: database_persisted=true; the main runtime and ledger are advancing.
- Historical files dated October 6 through 9 are several MB each, NOT EMPTY. Partial fetch operations cannot prove absence of data.
- The last change to the corners baseline report was October 7 at 15:34 UTC, even as the Oct 9 history advanced.
- Corners baseline: 266 OOS evaluations, 44 eligible formation-adjusted evaluations, 222 excluded. 162 excluded for verified matchup history below 8 prior examples; 60 absent verified matchup; 0 historically recoverable. Last OOS evaluated kickoff Sep 22.
- FM4 verified style source latest Sep 22, 22:30 UTC. Both-team >=3 prior style fixtures: 2/100; 18 current both confirmed XI; no both-team prior XI. Research decision weight remains zero.
- A separate Postgres snapshot reconciliation checked 60 missing OOS formations and found 0 recoverable verified pre-kickoff formations.
- Backfill of five historical final results ran Sep 22. That manual success does not prove continuously refreshed postgame tactical/corners inputs. FM4 tactical and personnel backfills sometimes legitimately capture 0 new rows.

## Actual architectural problems

1. GREEN RUNTIME IS NOT GREEN RESEARCH. Prior health status did not independently monitor the source-to-OOS freshness gap.
2. Heavy formation research allowed 10 minutes, ran once daily and checked out the full state, with no independent alert after a failed/degraded run.
3. Team Corners Validation Refresh had no scheduled trigger and depended on an already persisted corners report.
4. V4 Market Validation Refresh regenerated reports from possibly stale corners/team input and did not depend on the team-corners workflow.
5. Low formation density is a different scientific obstacle. Rerunning code cannot make 44 genuine chronological evaluations into 100.
6. Specific reason no new eligible corners/formation inputs exist in report after Sep 22 is UNVERIFIED. A collector gap, postgame tactical stats not materialized, timestamp filter or other source limitation must be differentiated with fixture-level evidence.

## Hardening in this change

- Formation Intelligence: 08:15 and 20:15 UTC; 45-minute timeout; sparse state checkout; serialized runs and latest-verified-source summaries.
- Team Corners Validation Refresh: runs on successful formation refresh, with 10:15/22:15 UTC fallback; serialized.
- V4 Market Validation Refresh: also runs on successful team-corners refresh; serialized.
- Six-hourly Soccer Research Freshness Watchdog: read-only artifact, conspicuous failed Actions job, deduplicated GitHub issue on critical gaps.
- Tests: 44/100 alone is NOT an incident; outdated report and OOS history, malformed inputs, inconsistent counters or unauthorized production promotion ARE actionable.

## Independent freshness gates

| Signal | Critical trigger |
| --- | --- |
| Live health | Older than 6 hours / absent |
| Canonical evidence | Older than 12 hours / absent |
| Research baseline file commit | Older than 48 hours / unverified |
| Latest daily historical file | Older than 48 hours / unknown |
| Most recent corners OOS kickoff | More than 7 days old while ledger is fresh |
| FM4 source latest verified tactical observation | More than 7 days old while ledger is fresh |
| Report counters/production status | Missing, inconsistent or unauthorized change |

Research artifact age >30h but under 48h is a warning. A coverage gap is an INVESTIGATION trigger, not proof of a bug if the eligible cohort is legitimately sparse.

## Required evidence before declaring complete

1. Count each day's history events, finalized fixture IDs, result.tactical_stats corners, both pre-kickoff formation timestamps and valid source fields for Sep 23 through Oct 9.
2. Join the same fixture IDs against Postgres canonical evidence and identify exactly which stage loses evidence: API source, final/result persistence, tactical stats, confirmed lineups, timestamp chronology, research history loader or minimum prior-matchup requirement.
3. Only reconcile already verified pre-kickoff source observations. Do not manufacture formations, use post-kickoff formations in OOS, rewrite historical betting recommendations or relax the 8-prior-matchup minimum.
4. Fix the responsible runtime collector/normalizer ONLY after a fixture-level reproduction, with tests proving chronology and data provenance.
5. Run full baseline to Team Corners to V4 lineage, and reconcile fixture counts and last-observed dates. Legitimately flat sample count is acceptable with documented excluded-reason counts.
6. Audit strict True CLV separately; provider close updates after kickoff do not count. 0 corners True CLV is still 0.
7. Close incident after source-to-report reconciliation plus two consecutive passing watchdog checks. Never close on a green generic dashboard alone.

## Operations

Inspect GitHub Actions workflow Soccer Research Freshness Watchdog, its 30-day report artifact, and any open issue titled Soccer research freshness critical. Alerts are deduplicated. If the downstream research is stale but live runtime is healthy, investigate the research source without pausing the live model. If research exceeds its 45-minute budget, inspect logs and optimize I/O instead of indefinitely expanding timeout.

Rollback: revert this pull request; no historical data rollback needed because this change only adds scheduling and diagnostics.

Branch: fix/soccer-research-freshness-20261009. Model version, production weights, history, odds, thresholds, quota policies and True CLV chronology unchanged.
