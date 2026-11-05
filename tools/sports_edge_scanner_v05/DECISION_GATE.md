# Soccer Decision Gate v1.0 — verified route to BET

The scanner used to set `actionable_bets = 0` after scanning, regardless of the market. This patch removes that fixed output and delegates its count to `adjudicate_daily()`.

## End-to-end flow

1. Screen the entire eligible soccer slate **without odds**.
2. Preserve raw soccer projections and the current canonical model's independent outputs; never infer OOS approval from a research calibrator.
3. Collect exact market and price metadata only after the sporting shortlist.
4. Read the canonical analysis for audited blockers and display WATCH for unpromoted models.
5. If, **and only if**, a cryptographically authenticated decision envelope is present, evaluate each single through the original `sports_edge.core.evaluate()` and then assess offered two-leg parlay tickets.
6. Generate `decision_gate_YYYY-MM-DD.json` and derive `actionable_bets` and `parlay_actionable_bets` from the actual evaluator results. No fixed zero.

## Trusted decision envelope

Environment variables:

- `SOCCER_TRUSTED_DECISION_INPUT`: path to an upstream producer's JSON export (not currently configured).
- `SOCCER_DECISION_FEED_HMAC_KEY`: private, high-entropy HMAC key shared only with the authorized upstream producer (not configured by this change).

The payload must be a current `SOCCER_TRUSTED_DECISION_FEED_v1.0` JSON envelope with `trusted_producer=SOCCER_EDGE_CANONICAL_SIGNED_EXPORT`, `offers`, `models`, `parlay_offers`, `joint_evidence`, and `signature_hmac_sha256` over the canonical JSON body (sorted keys, UTF-8, separators `(',', ':')`, excluding the signature). The signature authenticates the upstream producer's envelope; it does **not** independently prove the vendor or sportsbook data are true. The producer must verify original identities, times, prices and data rights.

The model row must supply exact fixture/market/line identity; RAW, market-shrunk and lower-bound probabilities; source model/version/artifact; validated OOS sample and explicit production approval; sport-first lineage; confirmed match status; XI/GK and material availability. No automated promotion of research-only model output occurs.

The quote must be a fresh, exact Caliente/Playdoit offer with an original publication timestamp and source reference; current API-Football odds do not satisfy this requirement. For parlays, the bookmaker must actually offer the exact two-leg ticket, and a separate OOS joint-risk report must support the dependence and lower-bound probability. `joint_p_lower` is **not** inferred by multiplying model estimates.

## Test truth vs live truth

- Synthetic integration tests explicitly exercise a valid scenario and produce **2 BET singles and 1 BET parlay** to demonstrate the software path is reachable. These are **not real games, models or prices**.
- The current canonical production data exposes research calibration whose promotion is disabled, and API-Football odds are not an authenticated Caliente/Playdoit ticket. Those conditions remain WATCH in live scans; the JSON report describes exact blockers.
- No wagers are ever sent or automatically placed; the decision classifier's BET is an *eligibility* result, subject to a final bankroll review.

The scanner should not be described as capable of automatically selecting a genuine +100 parlay until the real approved model and trusted sportsbook offer producers both work end to end.