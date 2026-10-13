# Soccer Edge: API pause and GitHub Actions cost-control mode
Activated: 2026-10-13. Scope: `Baahl11/Soccer` **default branch `main` only**.

## Why
The owner reports temporarily unavailable soccer provider API and requests avoiding further needless usage. GitHub billing shows **$40.41 gross** attributed to Soccer this month, but gross is **not** net spend: actual SKU/discount/net invoice must be checked in Settings > Billing and licensing > Usage. This public repository uses standard GitHub-hosted Actions, which GitHub documents as free for public repositories; billing attribution is NOT VERIFIED.

Historic evidence: approximately 7,448 GitHub Actions run records (repository lifetime), including scheduler every 10 minutes, hourly prospective materialization, frequent FM4 backfills and other scheduled validation. This pause prevents new timer-triggered runs; it cannot cancel already queued/active runs, paid GitHub services other than Actions, or Render service charges.

## Temporary change
Remove `on.schedule` from all **28** GitHub Actions workflows under `.github/workflows` that define a schedule; leave each existing `workflow_dispatch`, `push`, `pull_request`, `workflow_run` etc. intact. This preserves manual use and CI on demand. Note: a workflow file modified on `main` may run once due to its own `push` trigger, and workflow_run chains can still activate; do not assume zero runs.
No provider quota, model weight, market projection, signal, bet, threshold, database row, state-branch or historical prediction is modified.

High-frequency schedules paused: `soccer-edge-scheduler` (every 10 min), both FM4 history backfills (twice/hour each), prospective evaluation registry (hourly), FM5 readiness, formation lifecycle, v181/v215 signal materializations, and the other 20 scheduled audits and watchguards.

## Re-enable after API subscription
1. Verify provider access, API quota/billing, Render stability and budget limit **first**.
2. Restore required workflows' `on.schedule` gradually from the commit immediately before this pause (do not blindly resume all 28), starting with the daily scheduler/needed capture and then selected CLV/evaluation jobs.
3. Restore staggering, use `concurrency` cancellation/admission guards, cap runtime, monitor usage by SKU, and avoid overlapping memory-heavy endpoints.
4. Run one manually and verify persistence and provider quota; only then add cron for that workflow.
5. Keep model promotion, betting and historical rewrites blocked without scientific gates.

## Financial hard stop (owner UI)
Go to GitHub Billing > Budgets and alerts > New budget > Product-level **GitHub Actions**, scope Soccer if available, choose a low dollar limit and enable **Stop usage when budget limit is reached**. Also enable alerts. The hard stop affects only applicable metered usage; first-cycle budgets may not account for costs incurred before creation. For a complete diagnosis export detailed usage CSV with SKU, Gross amount, Discount amount, Net amount and repository. Without this breakdown **$40.41 cannot be attributed to a paid SKU**.

## Rollback
This pause is intentionally one commit on `main`, so a normal **revert of that single commit** restores the cron definitions and original schedules. Reverting will restart all 28 automatic schedules and should be done only after billing controls are in place.
