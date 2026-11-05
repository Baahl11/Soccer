# SOCCER EDGE — Bet365 market unblock audit (2026-11-05)

**Scope:** Soccer only. Bet365 bookmaker ID 8 is the exclusive odds reference for the scanner.  
**Source:** existing immutable/readonly analysis files under the `soccer-edge-state` branch.  
**Result:** research triage completed for 12 families; **zero markets promoted to BET**.  
**Important:** original market-validation files do not reliably identify the book for their legacy `true_clv` count. Those counts are *not* certified Bet365 CLV and cannot satisfy a Bet365-only gate. Most reports have no reliable generation timestamp. Project Supabase databases could not be read during this audit, so no live ledger verification was possible.

## A. First — 1X2 and team totals (review-ready, not approved)

### 1X2
- Canonical report: `v4_017_1x2_calibration_validation.json`.
- `CALIBRATION_REVIEW_ELIGIBLE`; 670 multiclass walk-forward OOS rows (14 folds).
- Multiclass Brier **0.67180017 baseline → 0.65499971 challenger**, log-loss **1.11185733 → 1.08266382**.
- 53 reported historical CLV observations; sportsbook attribution **NOT VERIFIED**.
- The research outcome-context subset has **negative** hypothetical per-fixture ROI; model score improvement cannot be taken as evidence of profitable Bet365 wagers.
- Next action: independently audit returns per unique fixture and exact market selection, identify verified Bet365 entry and closing prices, document timestamp/provenance, then perform manual production review. The historical report expressly forbids auto-promotion.

### Team totals
- Canonical report: `v4_019_team_totals_oos_validation.json`.
- `OOS_REVIEW_ELIGIBLE`; 500 OOS fixtures and 6,000 probability rows across multiple related thresholds.
- 703 reported historical CLV observations across 110 fixtures, not certified as Bet365-only.
- Research subset with priced settled observations has **202 rows in 24 unique fixtures**, with reported aggregate hypothetical ROI **negative**. A favorable Under split must not be retrospectively selected as a strategy without fresh holdout validation.
- Next action: audit exact 0.5/1.5/2.5 lines separately by home/away and Over/Under. Recheck source-specific CLV and promotion policy; keep the existing model unpromoted.

## B. Second — BTTS and FT goals (closing evidence bottleneck)

### BTTS
- Canonical report: `v4_018_btts_calibration_validation.json`.
- Research calibrator has 870 OOS rows and reported improvements in Brier and log-loss.
- Legacy `true_clv.rows=20` vs report minimum 50, **NOT confirmed Bet365-only**.
- Next: observe genuine Bet365 ID 8 snapshots before kickoff and closest trustworthy pre-kickoff close for the **same fixture, market, selection and threshold**; validate by league and probability bucket.

### Full-time goals totals
- Canonical report: `v4_016_ft_totals_production_validation.json`.
- 228 settled model cases; *not* equivalent to 228 independently verified OOS forecasts.
- Legacy `true_clv.rows=8` vs report minimum 50, **NOT confirmed Bet365-only**.
- No recorded pricing for several supported Asian lines (2.0, 2.25, 2.75, 3.0, 3.25).
- Next: capture exact priced thresholds and proper Asian push/half-win settlement; continue out-of-time model validation.

## C. Third — 1H, 2H and corners (model weakness plus market history)

### 1H goals
- Canonical report: `v4_020_1h_oos_validation.json`.
- OOS N=220. Baseline Brier **0.2459** versus challenger **0.2484**; baseline log-loss **0.6853** versus challenger **0.6907**.
- Challenger **does not beat baseline**. Legacy CLV=0.
- Priority: improve first-half-specific features and validate chronologically before seeking BET approval.

### 2H goals
- Canonical report: `v4_021_2h_oos_validation.json`.
- Walk-forward N=604. Baseline Brier **0.253033** versus challenger **0.254011**; log-loss **0.699253** versus **0.701232**.
- Challenger **does not beat baseline**. Legacy CLV=0.
- Priority: halftime score and red-card context, then strict walk-forward testing; never reuse FT calibration for 2H.

### FT and team corners
- Canonical report: `v4_022_corners_oos_validation.json`.
- Only 44 formation-adjusted cases against report minimum 100; cross-league and league/venue stability fail; legacy true CLV=0.
- Team corners has 244 evaluated fixtures but no sufficiently stable league/venue segment.
- Priority: credible historical corner counts, formation-adjusted OOS, stable league-venue segments, then exact Bet365 corner line snapshots.

## D. Fourth — cards, players, fouls and styles (data architecture first)

- **Cards:** 254 yellow-card OOS rows, no referee-adjusted rows, and 0 reported CLV; exact Bet365 scoring rules and referee assignment needed. Red-card OOS N=184 is below report's review minimum.
- **Player props:** prop-specific OOS and minutes/XI-aligned price ledger incomplete; zero verified historical CLV in the canonical report. Keep player-specific markets unpromoted.
- **Fouls:** no dedicated report; dedicated statistical and OOS model required.
- **Formation/style:** insufficient comparable past manager/team/XI observations to validate incremental signal.

## New daily audit

The scanner now builds `market_unblock_YYYY-MM-DD.json` and `market_unblock_YYYY-MM-DD.md` **without any additional API-Football calls**. Each contains the fixed research order, canonical report/source status, sample size **with its unit**, verified baseline comparison when explicitly available, historical CLV counts separated from **Bet365-only verified** CLV, blockers and one concrete next experiment.

The Actions job summary shows review-ready and baseline-failing families; tests prove legacy CLV cannot become verified Bet365 CLV and no research source can promote a wager.

## What is not claimed complete

This task **completed the diagnostic and tracking implementation**, not new model training, verified Bet365 historical closing capture, live database recovery, or production certification. No weights, line prices, historic forecasts or bet classifications were changed. A new model can only be promoted after sport-specific OOS validation, availability verification, Bet365 exact price evidence, shrinkage and documented risk review.

**Policy:** SPORT FIRST. MARKET SECOND. Bet365 only for market references. Research does not grant trading eligibility.
