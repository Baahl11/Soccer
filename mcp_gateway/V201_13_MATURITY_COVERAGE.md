# v201.13 — Maturity coverage + dashboard truth

This checkpoint closes two research-observability gaps without changing the production decision engine.

## Dashboard maturity truth

`maturity_snapshot_v4.py` reads the persisted validation artifacts from `STATE_REPO` / `STATE_BRANCH` and exposes the actual counters used by the Control Tower. The product layer remains read-only and caches the state snapshot; it does not call API-Football.

Expected gates currently include 1X2 true CLV, BTTS true CLV, Team Totals true CLV, 1H true CLV, Corners formation-adjusted evaluations, and Player Props true CLV per family.

## Due-league maturity floor

`fair_scheduler.py` keeps the existing weighted scheduler and applies a post-plan diversity correction. When a due eligible league is missing from the planned set and capacity already contains a duplicate league, a duplicate non-urgent slot may be exchanged for a fixture from the missing league. Same-category swaps are preferred. Urgent actionable T-40/T-30/T-20/T-10/CLOSE slots are never displaced.

## Invariants

- provider request cap unchanged
- provider_requests_added = 0 for the scheduler correction and dashboard reader
- canonical bet logic unchanged
- model weights unchanged
- thresholds and gates unchanged
- decision weights unchanged
- production promotion remains manual/disabled for research families
- no synthetic true-CLV rows
