# V209 Research Note — Market Baseline, Point-in-Time Ledger, and Research Challengers

Date: 2026-09-29

## Why this exists

Soccer Edge does not need to wait for every market family to mature before improving its research architecture. Production promotion must still wait for sufficient true-CLV, OOS, calibration, and stability evidence, but several research-only controls can be added now without changing any live decision weight, threshold, gate, stake, tier, provider budget, or strict-close rule.

## External evidence reviewed

| Evidence | Main takeaway for Soccer Edge |
| --- | --- |
| *The Incremental Value of Player Information in Football Match Prediction* (SSRN 7295578, 2026) | On the cited holdout, adding recent player information did not reliably beat a recalibrated closing-market baseline. Player/XI features should therefore be judged by incremental value over a strong de-vig market baseline, not only by fit against outcomes. |
| *Machine learning for sports betting: should model selection be based on accuracy or calibration?* (arXiv:2303.06021) | Calibration quality is a first-class model-selection criterion for betting. Keep Brier/log-loss and add RPS/reliability diagnostics for 1X2 rather than optimizing accuracy alone. |
| *Bayesian weighted discrete-time dynamic models for association football prediction* (arXiv:2508.05891, 2025) | Dynamic attack/defence strengths are a useful challenger when team strength shifts after transfers, coaching changes, or regime breaks. This belongs in research-only challenger mode first. |
| *AI-driven Forecasting and Execution in Soccer Prediction Markets...* (SSRN 7103799, 2026) | Historical decisions should remain point-in-time frozen. Reapplying today's calibrator to an old decision can create leakage. Soccer Edge should audit the exact calibrator/model artifact identity used at decision time. |
| *Testing semi-strong market efficiency in Brazilian football...* (SSRN 7412249, 2026) | Good calibration alone does not establish betting edge versus the closing market; sizing can be destructive when the model has no genuine market edge. Kelly/risk activation stays blocked until CLV/OOS/calibration/stability gates pass. |
| *A market-calibrated accelerated failure time model for in-play football forecasting* (arXiv:2605.16066, 2026) | Market calibration can be highly informative in-play. This is a later Phase-21/Live direction after Soccer Edge's 2H/live feed is mature enough. |
| *Can Simple Models Predict Football — and Beat the Odds? Lessons from the German Bundesliga* (SSRN 5381388) | Closing odds are a strong benchmark, while residual model signals may still exist. This supports measuring model-minus-market residuals explicitly rather than treating market probability only as a price field. |

## Implement-now research work

### 1. Frozen point-in-time calibration audit

Audit the decision/signal ledger for exact point-in-time provenance:

- model version;
- calibrator type/version or immutable artifact identity;
- raw probability;
- calibrated probability actually used at that timestamp;
- de-vig market probability available at that timestamp;
- stage and provider-update timestamp.

Historical rows must never be silently recalibrated with a newer artifact during evaluation.

### 2. Market Residual Challenger

Add a research-only benchmark/challenger for supported markets:

`market_residual = p_model_calibrated - p_market_devig`

Then test whether existing feature families (xG, XI, injuries, formation, team strength, form, venue, etc.) explain persistent OOS residual information after the market baseline. Required diagnostics should include Brier, log-loss, calibration/reliability buckets, CLV direction, and for 1X2 RPS where practical.

This challenger must have `decision_weight = 0.0` and cannot create a production BET.

### 3. Dynamic Strength Challenger

Build a research-only dynamic team attack/defence strength challenger that can adapt faster around regime changes. It must run side-by-side with the current baseline and earn promotion only through aligned OOS evidence.

### 4. Player-feature incremental-value audit

For Player Props/XI work, evaluate each family against a market-only baseline. Keep a feature only if it adds reproducible OOS information after price/de-vig context. Granularity alone is not evidence of edge.

## Implement later

Market-calibrated live/in-play forecasting belongs after 2H/live collection has reliable, timestamped prices and live state. Do not expand the current provider budget merely to accelerate this research.

## Current roadmap order

1. Finish v209 provider-budget phase correctness and verify it live.
2. Continue natural maturation in priority order: 1X2 -> BTTS -> FT Totals -> Team Totals -> 1H -> Corners -> 2H -> Cards/Player Props.
3. Run the frozen-calibration audit in parallel because it is zero-call/read-only.
4. Introduce Market Residual Challenger as research-only.
5. Introduce Dynamic Strength Challenger as research-only.
6. Evaluate player/XI features by incremental value over market baseline.
7. Keep risk/Kelly, promotion, and in-play production gates locked until their existing evidence requirements pass.

## Safety invariants

- `decision_weight = 0.0` for all new challengers until formal promotion.
- `production_promotion_allowed = false` for this research work.
- no model-weight replacement in production.
- no threshold lowering.
- no gate relaxation.
- no stake/tier increase.
- no strict-close semantic change.
- no synthetic CLV or fabricated market prices.
- no increase to the global provider request budget.
- all historical comparisons must remain point-in-time and leakage-safe.
