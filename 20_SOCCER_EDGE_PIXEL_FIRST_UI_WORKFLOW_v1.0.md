# 20_SOCCER_EDGE_PIXEL_FIRST_UI_WORKFLOW_v1.0

## Status
ACTIVE FRONTEND WORKFLOW

## Why this replaces the previous approach
The incremental live-data-first V3 iterations produced a technically valid interface but repeatedly drifted away from the approved Soccer Edge mockup. The main failure mode was attempting to solve visual design, sparse-data states, production data binding, and responsive behavior at the same time.

That workflow is retired for the Match Center rebuild.

## New rule: GOLDEN MASTER FIRST
Before binding live matchup data, build one visually complete reference Match Center that is judged only on design fidelity.

The golden master may use illustrative sample values, but it must be explicitly isolated under a design-lab route and labeled as sample data. It must never be presented as a live prediction or customer result.

Approved route:

`/design-lab/match-center`

## Phase A — Visual replication
Build the Match Center to closely reproduce the approved product language:

- strong team identity and matchup hero;
- premium dark sports-terminal visual system;
- meaningful typographic hierarchy;
- large high-signal numbers;
- compact analytical visualizations;
- asymmetric composition instead of repeated equal cards;
- Match Result Probability;
- Expected Goals;
- Sport Profile;
- Score Matrix heatmap;
- Goal Distribution;
- Model vs Market / Edge Gap;
- disciplined spacing and alignment;
- clear desktop and mobile composition.

No backend constraints should distort this design phase.

## Phase B — Screenshot acceptance
Compare screenshots against the approved mockup.

Do not move to live-data binding until the user accepts the design family.

Acceptance is based on:

- visual hierarchy;
- density;
- typography;
- spacing;
- chart quality;
- proportions;
- product polish;
- resemblance to the approved mockup.

## Phase C — Component extraction
Once approved, convert the golden master into reusable frontend components.

Examples:

- MatchHero
- QualityStrip
- ResultProbability
- ExpectedGoals
- SportProfile
- ScoreMatrix
- GoalDistribution
- EdgeGap
- AvailabilityPanel

## Phase D — Live data binding
Replace sample values with the real Soccer Edge customer contract.

Rules remain unchanged:

- SPORT FIRST;
- MARKET SECOND;
- no invented xG;
- no invented probabilities;
- no invented availability;
- missing data stays explicitly unavailable;
- market data never creates the raw sporting thesis.

## Phase E — Missing-data variants
Only after the rich-data golden master is approved should we design low-data states.

Missing-data design must preserve the same visual system without filling gaps with fabricated metrics.

## Production promotion rule
`/app-v3` or any successor must not replace the current primary subscriber surface until:

1. golden master accepted;
2. live-data component binding complete;
3. mobile screenshot QA passes;
4. no clipping / overflow;
5. match navigation works;
6. current Soccer model logic remains unchanged.

## Model firewall
This workflow changes frontend presentation only.

It must not change:

- model weights;
- BET/LEAN/WATCH/PASS thresholds;
- shrinkage;
- availability confidence methodology;
- provider request policy;
- persisted historical predictions.

## Current implementation
The first isolated golden-master screen is implemented at:

`/design-lab/match-center`

It intentionally uses illustrative Arsenal vs Brighton values based on the approved visual reference and is labeled DESIGN LAB / SAMPLE DATA.

This route exists to solve the visual problem before reconnecting production data.
