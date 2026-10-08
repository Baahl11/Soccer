# 18_SOCCER_MATCH_CENTER_UI_BLUEPRINT_v1.0

## SOCCER EDGE ENGINE
### Match Center UI Blueprint + Visual Reference Documentation

**Version:** 1.0  
**Status:** IMPLEMENTATION SOURCE OF TRUTH  
**Scope:** Soccer subscriber product / Match Center  
**Primary route family:** `/app`, `/app/match/{fixture_id}`  
**Design philosophy:** SPORT FIRST. MARKET SECOND.

---

# 1. PURPOSE

This document is the implementation source of truth for the Soccer Edge Match Center.

It defines the reusable UI/UX system that every matchup must inherit so the product becomes visually premium, auditable, mobile-safe, and consistent with the Soccer Edge methodology.

The Match Center must help a subscriber answer, quickly:

1. What do we know about the football match itself?
2. What sporting evidence is verified?
3. What remains missing or uncertain?
4. Is there a meaningful sporting angle?
5. What is the availability state?
6. Only after that: does the market create value?
7. Why is the engine classifying the match as BET / LEAN / WATCH / PASS?

This is not merely a styling document. It is a product contract between data, model logic, and presentation.

---

# 2. CORE PRODUCT PRINCIPLE

## SPORT FIRST. MARKET SECOND.

Every Match Center must follow this order:

1. **Match identity**
2. **Sporting evidence**
3. **Sporting gaps**
4. **Sport projection / match intelligence**
5. **Availability**
6. **Market context**
7. **Decision state**
8. **Advanced audit evidence**

The page must never lead with market count, odds breadth, or bookmaker coverage when sporting evidence is shallow.

A match with 80+ markets but weak sporting evidence must be presented as:

- `MARKET DATA ONLY`
- `INSUFFICIENT SPORTING EVIDENCE`
- or the appropriate non-actionable state

It must not appear “rich” merely because many betting markets exist.

---

# 3. NON-NEGOTIABLE DATA RULES

## 3.1 No fabrication

Never invent or infer unsupported:

- xG / xGA
- npxG / npxGA
- PPDA
- field tilt
- box entries
- shots or shots on target
- injuries
- lineups
- goalkeeper status
- weather
- trends
- possession
- market prices
- bookmaker source
- model probability
- market fair probability
- expected value

If the source does not verify it, show:

- `NOT VERIFIED`
- `NOT CAPTURED`
- `NOT AVAILABLE FROM PROVIDER`

## 3.2 Missing data is not zero

`null`, missing or unverified values must never become:

- 0
- 0.0%
- healthy
- confirmed
- positive

unless the underlying source explicitly reports zero.

## 3.3 Market breadth is not analytical depth

The number of market snapshots or bookmaker lines must never be used as a proxy for sporting understanding.

## 3.4 Visuals must be evidence-backed

Charts only render meaningful values when the required inputs are verified.

No decorative/fake charts.

---

# 4. VISUAL REFERENCES

The following references define the intended visual direction.

## 4.1 Original premium dashboard mockup

**Project image asset:** `1000164422.jpg`

The reference contains four primary product surfaces:

1. **Today**
2. **Edge Feed**
3. **Match Center**
4. **Control Tower**

### Important visual characteristics

- deep navy / blue-black background
- thin teal borders
- green verified/positive accents
- amber WATCH/pending accents
- blue informational accents
- compact cards
- strong number hierarchy
- visually dense but highly organized
- charts embedded directly in decision context
- clean sidebar / navigation hierarchy
- sportsbook-intelligence feel without sportsbook clutter

### Match Center reference concepts

The reference Match Center includes:

- matchup identity
- data quality block
- lineup state
- market freshness
- result probability
- expected goals
- edge gap
- sport profile
- score matrix
- over/under visualization
- goal distribution

These are design targets only.

**Numbers shown in the mockup are illustrative and must never be copied into production.**

---

## 4.2 Current-product screenshot used during redesign

**Project image asset:** `1000167301.jpg`

This captured the real mobile Match Center before the premium redesign.

It exposed these UX problems:

- excessive text
- raw technical data presented before interpretation
- poor visual hierarchy
- tables wider than the phone
- too many `NOT VERIFIED` rows without context
- market breadth visually overpowering sporting depth
- weak separation between customer UI and internal audit data

This image should be retained as a “before” reference.

---

## 4.3 Generated Match Center target concept

**Reference image generated during product design:**  
`análisis_soccer_edge_arsenal_vs_brighton.png`

This generated concept is the primary visual target for the matchup experience.

It contains three mobile surfaces:

### Screen A — Overview

- matchup header with crests
- league / kickoff
- high-level data state
- sport-first read
- goal-rate comparison bars
- input coverage gauge
- form momentum chart
- key takeaways

### Screen B — Sport

- Match Result Probability
- Home / Draw / Away visual probability distribution
- Expected Goals comparison
- score matrix / heatmap
- goal distribution
- Team Sport Profile
- Verified Sporting Inputs

### Screen C — Markets / Availability

- explicit `Market context (secondary layer)`
- market opportunity rows
- Model vs Market vs Edge
- Market confidence
- freshness
- recent market activity
- availability and team news

### Key design rule extracted from this reference

**The page must communicate the sporting story visually before asking the user to interpret market data.**

The image is a UI reference only; all production numbers must come from verified pipeline data.

---

# 5. MATCH CENTER PRODUCT STANDARD

Every eligible matchup must resolve to the same reusable Match Center system.

Primary route:

`/app/match/{fixture_id}`

The design must adapt to high-data, medium-data and low-data fixtures without breaking its information hierarchy.

---

# 6. MATCH CENTER PAGE ARCHITECTURE

## 6.1 Header

Always attempt to show:

- competition
- country
- home team
- away team
- team crests
- kickoff
- timezone
- venue if verified
- current decision / evidence state
- freshness

All subscriber kickoff times must be displayed in:

`America/Mexico_City`

---

## 6.2 High-level summary strip

Recommended primary metrics:

- Sport Coverage
- Data Tier
- Availability Confidence
- Freshness

Do **not** put market count among the primary four metrics.

Market breadth is secondary.

---

## 6.3 Sport-first read

A short human-readable interpretation.

Examples:

- sporting evidence is strong enough for deeper review
- sporting evidence is partial
- market data exists but sporting evidence is insufficient
- lineup / availability remains unresolved
- current evidence does not support a sporting conclusion

This component must not create a BET or LEAN by itself.

---

# 7. TAB ORDER

Required default order:

1. **Overview**
2. **Sport**
3. **Availability**
4. **Markets**
5. **Advanced**

Markets must remain after Sport and Availability.

---

# 8. OVERVIEW TAB

Goal: explain the match in less than 10 seconds.

## 8.1 Verified Sporting Inputs

Every input counted in Sport Coverage must be visibly listed.

If the UI says:

`5 visible inputs`

the user must be able to see exactly those 5 inputs.

Potential entries:

- team form
- goals for / match
- goals against / match
- clean sheets
- failed to score
- verified goal-rate baseline
- formation
- XI status
- goalkeeper status
- venue
- rest days
- verified advanced metrics

Each entry should show where possible:

- label
- value
- source
- freshness
- sample size

Technical identifiers such as league ID or season do not count as subscriber-facing sporting inputs.

---

## 8.2 Sporting Evidence Card

Show the most useful football-specific values first.

Example ordering:

1. recent form
2. GF / match
3. GA / match
4. goal-rate baseline
5. clean sheets
6. failed to score
7. formation
8. XI / goalkeeper state
9. venue / rest context

---

## 8.3 Sporting Gaps

Clearly list important absent inputs.

Examples:

- xG / xGA NOT VERIFIED
- Recent form NOT CAPTURED
- XI NOT VERIFIED
- Goalkeeper NOT VERIFIED
- Injuries NOT VERIFIED
- Weather NOT VERIFIED

This is part of the product’s value: showing the user exactly what the model does not know.

---

## 8.4 Key Takeaways

When supported by data, create a compact visual list of 2–4 facts.

Examples:

- strong home scoring profile
- low away scoring rate
- defensive mismatch
- recent form divergence
- meaningful rest differential
- lineup uncertainty

These must be deterministic summaries of verified data, not AI-flavored narrative filler.

---

# 9. SPORT TAB

This is the primary analytical tab.

## 9.1 Team Form & Scoring

For each team, where available:

- form
- played
- W-D-L
- GF / match
- GA / match
- clean sheets
- failed to score

Preferred visual:

- comparison cards
- compact bars
- mini trend chart where true recent-match sequence exists

---

## 9.2 Goal-rate Comparison

Display verified:

- Home goal-rate baseline
- Away goal-rate baseline
- Total goal-rate baseline

Preferred visual:

- horizontal comparison bars
- team colors / neutral total line

Do not imply these are xG unless the input is truly xG.

---

## 9.3 Match Result Probability

Only when verified model probabilities exist.

Display:

- Home %
- Draw %
- Away %

Preferred visual:

- three probability tiles
- one stacked probability bar

If absent:

`NOT VERIFIED`

No synthetic probabilities.

---

## 9.4 Expected Goals

Only when verified xG or a formally defined model expectation is available.

Display:

- Home
- Away
- Total

The label must state the actual metric source.

Do not rename raw goal-rate baselines to xG.

---

## 9.5 Score Matrix

Only render when a persisted score distribution exists.

Preferred visual:

- compact heatmap
- top outcome highlighted
- probability shown

No fake matrix.

---

## 9.6 Goal Distribution

Only render when the model stores the distribution.

Potential buckets:

- 0
- 1
- 2
- 3
- 4
- 5+

Preferred visual:

- compact grouped bars

---

## 9.7 Sport Profile

Possible verified categories:

- Attack strength
- Defensive strength
- Recent form
- Home advantage
- Rest
- Opponent quality
- Lineup strength
- Goalkeeper confidence

Important:

Do not show GOOD / NEUTRAL / WEAK unless the classification is backed by a defined rule or verified metric.

---

# 10. AVAILABILITY TAB

## 10.1 Required concepts

- Availability Confidence
- Starting XI
- Goalkeepers
- Injuries
- Suspensions if available
- Weather if material
- latest lineup snapshot
- latest availability snapshot
- latest refresh

## 10.2 Visual states

Green:
- explicitly verified

Amber:
- partial / pending

Gray:
- not verified

Red:
- verified material blocker / absence

Missing data must never appear green.

---

# 11. MARKETS TAB

Markets are the second analytical layer.

The top of the tab should explicitly communicate:

**Market context — secondary layer**

Suggested copy:

> Markets can confirm or challenge the sporting read. Market availability alone is not evidence of value.

---

## 11.1 Selected Market

When verified:

- market
- selection
- line
- price
- bookmaker
- captured timestamp
- breakeven probability
- market fair probability
- model probability
- probability edge
- EV where defensible

---

## 11.2 Key Market Opportunities

Preferred columns:

- Market
- Model
- Market
- Edge
- Price
- Confidence
- Status

Possible states:

- BET
- LEAN
- WATCH
- WAIT PRICE
- WAIT XI
- RESEARCH
- PASS

Never display a market opportunity solely because the bookmaker offers the market.

---

## 11.3 Market Activity

Use:

- recent market snapshots
- price direction
- freshness
- bookmaker
- timestamp

Preferred visual:

- compact rows
- small line/spark trend only when multiple comparable timestamps exist

---

# 12. ADVANCED TAB

This tab houses the audit layer.

It should not dominate the customer experience.

Use collapsible accordions for:

- feature snapshots
- model runs
- market snapshots
- refresh events
- lineup snapshots
- availability snapshots
- raw current-snapshot candidate table
- schema/version metadata

---

# 13. VISUAL COMPONENT LIBRARY

Reusable components should include:

- `MatchHeader`
- `StatusBadge`
- `FreshnessBadge`
- `SportFirstRead`
- `VerifiedInputList`
- `SportingGapsCard`
- `KeyTakeaways`
- `GoalRateBars`
- `FormMomentumChart`
- `InputCoverageGauge`
- `ProbabilityStack`
- `ExpectedGoalsComparison`
- `ScoreMatrix`
- `GoalDistributionChart`
- `SportProfileComparison`
- `AvailabilityChecklist`
- `MarketEdgeTable`
- `MarketActivityList`
- `EvidenceAccordion`

The goal is one design system reused across every fixture, not handcrafted match pages.

---

# 14. CHART RULES

Recommended:

- horizontal bars
- stacked probability bars
- compact line charts
- mini sparklines
- donut / gauge for data coverage only
- heatmaps
- histograms
- paired comparison bars

Avoid:

- decorative charts
- unsourced radar charts
- artificial curves
- visually impressive but analytically meaningless graphics

Every chart must answer a question.

---

# 15. MOBILE-FIRST RULES

Mobile is the primary constraint.

Requirements:

- no core horizontal overflow
- tabs can horizontally scroll
- charts fit phone width
- raw tables live only in Advanced
- long market names truncate gracefully
- sections stack vertically
- sticky bottom navigation must not cover content
- chart labels remain legible
- the main story appears before excessive scrolling

Desktop can expand into 2- or 3-column cards.

---

# 16. MATCH DATA STATES

Every fixture must gracefully fit one of these broad product states.

## A. SPORT DATA AVAILABLE

Verified sporting evidence exists.

Market may or may not be available.

## B. PARTIAL SPORT DATA

Some useful sporting inputs exist but important gaps remain.

## C. MARKET DATA ONLY

Bookmaker/market data exists but meaningful sporting evidence is not verified.

This must be visually treated as a warning / limited research state.

## D. FIXTURE ONLY / INSUFFICIENT DATA

Only matchup identity is known.

## E. BET / LEAN / WATCH / PASS

Canonical decision states remain separate from the data-coverage state.

---

# 17. TIME AND SLATE BEHAVIOR

Subscriber timezone:

`America/Mexico_City`

Upcoming Match Center discovery rules:

- completed fixtures do not remain in active Matches
- already-started fixtures do not remain in the pregame slate
- cancelled / postponed / suspended / abandoned fixtures are excluded from normal pregame analysis
- once the current Mexico-Central day has no remaining eligible pregame fixtures, Matches rolls forward to the next persisted future slate

Historical Results is the correct surface for completed games.

---

# 18. SPORT-FIRST PIPELINE EXPECTATION

Target flow:

```
FULL SLATE
    ↓
SPORT DISCOVERY
    ↓
CHEAP SPORT SCREEN
    ↓
SPORTING SHORTLIST
    ↓
DEEP SPORT / AVAILABILITY
    ↓
MARKET INSPECTION
    ↓
BET / LEAN / WATCH / PASS
```

Coverage tier may block automated BET eligibility.

Coverage tier must **not** automatically erase all sporting research.

---

# 19. CURRENT SPORT INPUT TARGETS

Persist and surface, when verified:

- team form
- played sample
- W-D-L
- goals for / match
- goals against / match
- clean sheets
- failed to score
- recent match count
- verified goal-rate baselines
- formations
- XI confirmation
- goalkeeper confirmation
- injury report status
- venue
- city
- weather
- rest

Future expansion when valid sources exist:

- xG / xGA
- npxG / npxGA
- PPDA
- field tilt
- box entries
- final-third entries
- opponent-adjusted form
- chance quality
- defensive pressure
- schedule difficulty

No future metric should be introduced without a verified source and a documented definition.

---

# 20. PRODUCT COPY RULES

Tone:

- concise
- analytical
- premium
- high-trust
- non-hype

Good:

- “Sporting evidence is partial.”
- “Market data exists, but sporting evidence is insufficient.”
- “XI remains unverified.”
- “Market availability alone is not an edge.”

Avoid:

- “Easy win”
- “Lock”
- “Free money”
- “AI likes this”
- vague autogenerated sports prose

---

# 21. VISUAL IMPLEMENTATION PHASES

## Phase 1 — Match Center shell

- header
- tabs
- responsive hierarchy
- mobile-safe cards
- verified inputs
- sporting gaps
- sport-first read

## Phase 2 — Sporting visuals

- goal-rate bars
- team form comparison
- scoring profile
- input coverage
- probability visual
- xG visual when verified
- score matrix when verified
- goal distribution when verified

## Phase 3 — Availability

- XI
- goalkeeper
- injuries
- weather
- confidence
- freshness

## Phase 4 — Market layer

- market opportunity table
- model vs market comparison
- price freshness
- market movement
- blockers

## Phase 5 — Advanced audit

- snapshots
- raw tables
- model runs
- feature history
- evidence audit

---

# 22. ACCEPTANCE CRITERIA

A Match Center iteration does not pass QA unless:

1. Sport appears before Markets.
2. Every Sport Coverage count is visibly auditable.
3. Missing data never becomes zero.
4. High market count never masquerades as high sporting coverage.
5. The key football story is understandable within seconds.
6. Charts only render from verified data.
7. Mobile has no core horizontal overflow.
8. Raw technical tables are secondary.
9. BET / LEAN / WATCH remain canonical engine states.
10. Low-data matches still look intentionally designed.
11. High-data matches become highly visual.
12. The product looks like a professional sports-intelligence platform, not an internal dashboard or generic AI template.

---

# 23. QA FIXTURE ARCHETYPES

Every major Match Center release must be tested against:

## High-data fixture

Expected:
- multiple sport visualizations
- availability
- model outputs
- market context

## Medium-data fixture

Expected:
- partial sport visuals
- explicit gaps
- useful human review

## Market-rich / sport-poor fixture

Expected:
- clear MARKET DATA ONLY / limited sporting coverage
- no false impression of edge

## Fixture-only

Expected:
- professional empty state
- no fabricated graphics

---

# 24. DESIGN REFERENCE CHECKLIST

Before approving any new widget, compare it against the reference images.

Ask:

- Does it look like the same product family?
- Is the information hierarchy as clear as the premium mockup?
- Does the chart help the user interpret the matchup?
- Is Sport still visually dominant?
- Is Market visibly secondary?
- Would a paying subscriber understand why this information matters?
- Does this feel intentionally designed rather than automatically generated?

If the answer is no, redesign before shipping.

---

# 25. SOURCE-OF-TRUTH STATUS

This document is the active Match Center UI/UX blueprint until explicitly versioned or superseded.

Implementation work should consult this file before modifying:

- matchup layout
- sporting cards
- chart behavior
- market presentation
- availability presentation
- mobile Match Center hierarchy
- Match Center copy

Material product architecture changes require:

- an update to this document, or
- a new version such as `18_SOCCER_MATCH_CENTER_UI_BLUEPRINT_v1.1.md`.

Do not silently drift from this blueprint.

---

# 26. TARGET OUTCOME

Every matchup should ultimately feel like a compact professional football intelligence report:

**understand the football → understand uncertainty → understand availability → inspect market → understand the decision.**

That is the Match Center standard for Soccer Edge.
