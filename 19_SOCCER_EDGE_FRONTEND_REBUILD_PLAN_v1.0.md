# 19_SOCCER_EDGE_FRONTEND_REBUILD_PLAN_v1.0

## SOCCER EDGE ENGINE
### Premium Frontend Rebuild Plan — Match Center + Subscriber Product

**Version:** 1.0  
**Status:** IMPLEMENTATION PLAN / SOURCE OF TRUTH  
**Scope:** Soccer subscriber frontend  
**Primary goal:** Rebuild the customer UI so it can closely reproduce the premium dashboard language of the approved mockup while preserving SPORT FIRST / MARKET SECOND and strict anti-hallucination rules.

---

# 1. WHY THIS REBUILD EXISTS

The current subscriber frontend proved the product flow, but it is not the correct long-term rendering architecture for the visual target.

The present implementation is primarily:

- server-generated HTML;
- a large monolithic Python-rendered template;
- CSS added incrementally;
- data blocks stacked vertically;
- limited custom visualization primitives.

This makes it possible to display data, but difficult to reproduce the premium mockup faithfully.

The approved visual target is not simply “more cards.” It is a **component-driven sports intelligence interface** with compact grid composition, charts, heatmaps, probability visuals, small multiples, status systems, and distinct mobile/desktop layouts.

Therefore:

> **Do not keep treating the existing subscriber template as the final visual architecture.**

The backend and persisted evidence remain valuable. The customer-facing rendering layer should be rebuilt as a dedicated frontend application.

---

# 2. AUTHORITATIVE VISUAL REFERENCES

The rebuild must continuously reference these design assets.

## 2.1 Premium 4-panel product mockup

**Reference asset:** `1000164422.jpg`

Contains the desired family of surfaces:

1. Today
2. Edge Feed
3. Match Center
4. Control Tower

Key visual characteristics:

- deep navy / blue-black canvas;
- thin blue-teal borders;
- compact modules;
- strong typographic hierarchy;
- large high-signal numbers;
- integrated charts rather than raw rows;
- green positive/verified states;
- amber pending/watch states;
- blue informational states;
- small data-quality indicators;
- dense but controlled information layout;
- premium sports-intelligence feel.

## 2.2 Generated mobile Match Center concept

**Reference asset:** `análisis_soccer_edge_arsenal_vs_brighton.png`

This is the primary mobile composition reference.

It demonstrates the intended use of:

- matchup hero;
- probability cards;
- expected-goals comparison;
- score matrix;
- goal distribution;
- sport profile;
- verified-input indicators;
- market context as a secondary layer.

All numbers in the reference are illustrative only.

## 2.3 Current production screenshot / gap reference

**Reference asset:** `1000167309.jpg`

This screenshot represents the current gap between implementation and target.

Observed problems:

- too much vertical stacking;
- large single-purpose cards;
- insufficient chart density;
- customer viewport consumed by explanation rather than intelligence;
- narrow/mobile clipping risk;
- debug-like data disclosure still visually prominent;
- current layout reads like styled documentation, not a premium dashboard.

The rebuild should be judged by whether this gap materially closes.

---

# 3. TECHNOLOGY DECISION

## 3.1 Selected frontend stack

The recommended implementation stack is:

### Core

- **React**
- **TypeScript**
- **Vite**

### Styling

- CSS variables for design tokens
- component-scoped CSS or Tailwind CSS

### Data fetching/state

- **TanStack Query** for API state, caching, loading and refetch behavior

### Visualization

Primary:

- **custom SVG React components**

Supporting utility when useful:

- **Visx** for scales, shapes, axes and heatmap primitives

Optional later:

- Framer Motion for subtle transitions
- Radix UI primitives for accessible tabs, accordions, popovers and dialogs

## 3.2 Why React + Vite instead of continuing Python-rendered HTML

The visual target requires reusable interactive components and exact control over responsive composition.

React gives us:

- reusable chart widgets;
- deterministic component states;
- easier mobile/desktop variants;
- isolated visual QA;
- much cleaner data-to-UI mapping;
- easier reuse across Today, Edge Feed, Match Center and Control Tower.

Vite keeps the frontend lightweight and avoids introducing a full SSR platform unnecessarily.

## 3.3 Why not make Next.js mandatory now

Next.js is capable, but the current backend already owns the data/API layer.

The immediate product need is a premium SPA, not SEO or server-side rendering.

Using Vite allows us to:

- keep the existing Python backend;
- build a static React bundle;
- serve the compiled frontend from the existing service;
- avoid creating a second production runtime during the migration.

Next.js can be reconsidered later if public SEO, server components or multi-surface web delivery becomes important.

---

# 4. DEPLOYMENT ARCHITECTURE

## 4.1 Keep the backend

The existing Python service remains responsible for:

- auth integration;
- subscriber contracts;
- fixture registry;
- model outputs;
- feature snapshots;
- market snapshots;
- availability snapshots;
- evidence persistence;
- sport-first decision logic.

The frontend must **not** recalculate canonical model decisions.

## 4.2 New frontend directory

Recommended repository layout:

```text
/web
  /src
    /app
    /components
    /charts
    /features
    /pages
    /services
    /styles
    /types
  package.json
  vite.config.ts
  tsconfig.json
```

Production build output:

```text
/web/dist
```

The Python server can serve this build as static assets.

## 4.3 Side-by-side migration route

Do not replace the current app immediately.

Recommended rollout:

```text
/app       -> current stable subscriber UI
/app-v3    -> React premium rebuild during QA
```

After the React product passes acceptance tests:

```text
/app       -> React premium product
/app-legacy -> old interface temporarily, if needed
```

This prevents UI work from destabilizing the existing customer surface.

---

# 5. API CONTRACT PRINCIPLE

The React UI should not have to understand raw persistence tables.

The backend should provide a clean customer contract.

Recommended conceptual response:

```ts
interface MatchCenterContract {
  fixture: MatchIdentity;
  status: MatchStatus;
  sportCoverage: SportCoverage;
  sport: SportIntelligence;
  availability: AvailabilityState;
  market: MarketContext;
  decision: DecisionSummary;
  evidence: EvidenceMeta;
}
```

The browser should receive display-ready domain objects, not database internals.

---

# 6. MATCH CENTER DATA CONTRACT

## 6.1 Fixture

```ts
fixture: {
  fixtureId: number;
  league: string;
  country?: string;
  kickoff: string;
  displayTimezone: "America/Mexico_City";
  venue?: string;
  home: {
    teamId: number;
    name: string;
    logo?: string;
  };
  away: {
    teamId: number;
    name: string;
    logo?: string;
  };
}
```

## 6.2 Sport coverage

```ts
sportCoverage: {
  status: "SPORT_DATA_AVAILABLE" | "PARTIAL_SPORT_DATA" | "MARKET_DATA_ONLY" | "INSUFFICIENT_DATA";
  verifiedVisibleInputs: number;
  coreVerified: number;
  coreTotal: number;
  percentage: number;
  dataTier?: string;
  freshness?: string;
}
```

Market snapshot count must not increase sport coverage.

## 6.3 Team form and scoring

```ts
sport.teamPerformance: {
  home: {
    sampleN?: number;
    form?: string;
    wins?: number;
    draws?: number;
    losses?: number;
    goalsForPerMatch?: number;
    goalsAgainstPerMatch?: number;
    cleanSheets?: number;
    failedToScore?: number;
  };
  away: { ... };
}
```

### Validity rule

If a team-stat sample is zero, the UI must not show rates such as `0.00 GF/match` as verified evidence.

Zero sample means insufficient evidence, not a verified zero rate.

## 6.4 Model probabilities

```ts
sport.resultProbability?: {
  home: number;
  draw: number;
  away: number;
  source: string;
}
```

Render only when persisted and verified.

## 6.5 Expected goals

```ts
sport.expectedGoals?: {
  home: number;
  away: number;
  total?: number;
  source: string;
  metricName: string;
}
```

Do not label goal-rate baselines as xG.

## 6.6 Score matrix

```ts
sport.scoreMatrix?: Array<{
  homeGoals: number;
  awayGoals: number;
  probability: number;
}>;
```

## 6.7 Sport profile

```ts
sport.profile?: Array<{
  key: string;
  label: string;
  score: number;
  definitionVersion: string;
}>;
```

Do not invent GOOD/NEUTRAL/WEAK labels without documented scoring rules.

---

# 7. MATCH CENTER COMPONENT ARCHITECTURE

## 7.1 Page shell

```text
MatchCenterPage
├── MatchHero
├── MatchStatusStrip
├── MatchTabs
├── OverviewTab
├── SportTab
├── AvailabilityTab
├── MarketsTab
└── AdvancedTab
```

---

# 8. MATCH HERO

The hero must resemble the approved mockup rather than the generic application header.

Required:

- Soccer Edge branding mark;
- back navigation;
- competition;
- Mexico-Central kickoff;
- team crests;
- team names;
- HOME / AWAY labels;
- current data/decision status.

Mobile height should remain compact.

The user should reach meaningful charts quickly.

---

# 9. OVERVIEW TAB — TARGET COMPOSITION

The Overview tab should feel like an executive intelligence dashboard.

Recommended mobile order:

```text
[ Sport-first Read / primary insight ]

[ Goal / Scoring Comparison Visual ]

[ Coverage Gauge ] [ Form Momentum ]

[ Key Sporting Takeaways ]

[ Compact Availability Signal ]

[ Expandable Evidence / Gaps ]
```

Do not use the first viewport for long technical explanations.

---

# 10. REQUIRED CHART COMPONENTS

These should be built as real reusable React/SVG components.

## 10.1 Coverage Donut

Component:

`<CoverageDonut />`

Inputs:

- verified
- total
- percentage

Purpose:

Show how much of the defined core sport packet is currently verified.

Must not incorporate markets.

---

## 10.2 Goal Rate Comparison

Component:

`<GoalRateComparison />`

Inputs:

- home rate
- away rate
- combined rate
- metric label
- sample size where relevant

Visual:

- compact horizontal bars
- clear values
- home / away distinction

If source is raw goal-rate baseline:

Label exactly as goal-rate baseline.

If source is verified xG:

Label as expected goals / xG according to source definition.

---

## 10.3 Form Momentum Chart

Component:

`<FormMomentum />`

Inputs:

- verified W/D/L sequence
- optional match dates/opponents when available

Visual:

- W/D/L status dots
- compact SVG trend

The trend is a visualization of verified results, not a new predictive metric.

---

## 10.4 Result Probability

Component:

`<ResultProbability />`

Inputs:

- Home probability
- Draw probability
- Away probability

Visual:

- 3 large probability blocks
- stacked horizontal probability bar

This should closely reproduce the visual hierarchy in the mockup.

---

## 10.5 Expected Goals Card

Component:

`<ExpectedGoalsComparison />`

Render only when a verified source exists.

Visual:

- Home xG large number
- Away xG large number
- comparison line/bar

Missing state must remain deliberate and compact.

---

## 10.6 Score Matrix Heatmap

Component:

`<ScoreMatrixHeatmap />`

This is one of the main visual differences between the current UI and the target.

Implementation:

- SVG or CSS grid;
- cells represent exact score probabilities;
- intensity derived from probability;
- highest-probability score highlighted;
- axes: Home goals / Away goals.

This must look like a real matrix, not a collection of score chips.

---

## 10.7 Goal Distribution

Component:

`<GoalDistribution />`

Visual:

- paired bars for Home/Away;
- goal buckets such as 0,1,2,3,4+;
- compact legend.

Preferred source:

- persisted model goal distribution.

Fallback source:

- deterministic aggregation of a verified persisted score matrix.

Do not generate a distribution from unrelated market odds.

---

## 10.8 Sport Profile

Component:

`<SportProfile />`

Visual:

- compact category rows;
- progress/strength bar;
- label;
- score/state.

Possible future categories:

- Attack
- Defense
- Recent form
- Territory
- Rest
- Opponent quality
- Lineup strength
- Goalkeeper confidence

Every category must have a documented calculation.

---

## 10.9 Edge Gap Visual

Component:

`<EdgeGap />`

Market tab only.

Shows:

- model probability
- market fair probability
- probability edge

This belongs **after** the sport layer.

---

## 10.10 Market Sparkline

Component:

`<MarketMovementSparkline />`

Only render when multiple comparable timestamped price snapshots exist.

No line chart from a single point.

---

# 11. SPORT TAB — TARGET COMPOSITION

High-data fixture target:

```text
[ Match Result Probability ]
[ Expected Goals ]

[ Score Matrix ] [ Goal Distribution ]

[ Sport Profile ]

[ Team Form & Scoring ]

[ Formation / Squad Context — expandable ]
```

This is the section that should visually resemble the Match Center panel in the approved mockup most strongly.

---

# 12. AVAILABILITY TAB

Build a visual status panel instead of a key/value list.

Recommended component:

`<AvailabilityGrid />`

Items:

- XI confirmation
- GK confirmation
- injuries
- suspensions if available
- weather
- availability confidence
- last refresh

Visual states:

- green = explicitly verified positive
- amber = unresolved / partial
- red = verified material problem
- gray = NOT VERIFIED

No data must never look green.

---

# 13. MARKETS TAB

The Markets tab should use visual comparison but remain secondary.

Recommended composition:

```text
[ Market Layer explanation ]

[ Best verified opportunity ]

[ Model vs Market / Edge Gap ]

[ Opportunity table ]

[ Price movement ]

[ Market evidence / snapshots — expandable ]
```

The frontend must never imply that high market count equals strong analysis.

---

# 14. ADVANCED TAB

Advanced remains audit-oriented.

Use accordions and data tables for:

- feature snapshots
- model runs
- refresh events
- lineup snapshots
- availability snapshots
- market snapshots
- raw persisted evidence
- model/schema version

This data should not dominate Overview or Sport.

---

# 15. DESIGN SYSTEM TOKENS

Define tokens once.

Example categories:

```css
--bg-canvas
--bg-panel
--bg-panel-elevated
--border-subtle
--border-active
--text-primary
--text-secondary
--text-muted
--accent-blue
--accent-teal
--accent-green
--accent-amber
--accent-red
--radius-sm
--radius-md
--radius-lg
--space-1 ... --space-8
```

The exact values should be tuned visually against the mockup.

Do not scatter arbitrary colors and spacing values through components.

---

# 16. TYPOGRAPHY

Use a compact sans-serif product hierarchy.

Required levels:

- product title
- page title
- match/team name
- section title
- metric number
- metric label
- helper text
- metadata

The premium mockup depends heavily on size contrast.

Do not render every label at roughly the same visual weight.

---

# 17. RESPONSIVE STRATEGY

## 17.1 Mobile is not desktop stacked vertically

Build mobile compositions intentionally.

Examples:

- two compact charts can remain 2-up;
- a large table becomes a ranked card list;
- detailed evidence becomes accordion content;
- score matrix shrinks with fewer visible buckets rather than overflowing;
- market table can switch to card rows.

## 17.2 Desktop

Desktop may use:

- left navigation rail;
- 12-column content grid;
- 2–3 analytical cards per row;
- full market table;
- larger score matrix;
- visible secondary context.

## 17.3 Breakpoints

Define explicit layouts rather than only relying on fluid wrapping.

Suggested conceptual breakpoints:

- phone
- tablet
- desktop
- wide desktop

---

# 18. DATA-ABSENCE DESIGN

One of the strongest product requirements is making low-data games still look intentional.

Do not show huge empty rectangles.

Use compact missing states.

Example:

```text
Expected Goals
NOT VERIFIED
Provider does not currently supply a verified xG source for this fixture.
```

A missing widget should occupy less space than a fully populated widget.

---

# 19. ANTI-HALLUCINATION UI REQUIREMENTS

The frontend must reinforce backend truthfulness.

Never infer:

- injuries from an empty unsupported endpoint;
- no injuries from zero results when coverage is unavailable;
- xG from goals;
- probabilities from market lines unless explicitly part of the market layer;
- confirmed XI from lineup presence unless confirmation rules pass;
- healthy status from missing availability data.

The UI is not allowed to “fill visual holes” with guessed content.

---

# 20. MARKET FIREWALL

The React frontend must preserve the model firewall.

No chart should silently use market information to populate a sporting widget.

Examples:

- Result Probability in Sport must use the sport/model projection source.
- Market fair probability belongs in Markets.
- Edge Gap combines the two only after both are independently available.

---

# 21. MIGRATION PHASES

## Phase 0 — Freeze requirements

Source documents:

- `18_SOCCER_MATCH_CENTER_UI_BLUEPRINT_v1.0.md`
- `19_SOCCER_EDGE_FRONTEND_REBUILD_PLAN_v1.0.md`

Visual references:

- `1000164422.jpg`
- `análisis_soccer_edge_arsenal_vs_brighton.png`
- `1000167309.jpg`

No major Match Center UI decision should ignore these references.

---

## Phase 1 — React shell

Build:

- Vite/React/TypeScript project
- app shell
- route handling
- API client
- auth token integration
- mobile bottom nav
- desktop side navigation
- theme tokens

Success criterion:

The app loads the existing backend data without changing model logic.

---

## Phase 2 — Match Center visual foundation

Build:

- MatchHero
- MatchStatusStrip
- MatchTabs
- CoverageDonut
- GoalRateComparison
- FormMomentum
- compact missing states

Success criterion:

The first mobile viewport visually resembles the approved design family.

---

## Phase 3 — Core Sport visuals

Build:

- ResultProbability
- ExpectedGoalsComparison
- ScoreMatrixHeatmap
- GoalDistribution
- SportProfile
- TeamPerformanceComparison

Success criterion:

A high-data match clearly resembles the mockup Match Center in structure and information density.

---

## Phase 4 — Availability

Build:

- AvailabilityGrid
- lineup state
- GK state
- injury state
- weather state
- confidence / freshness

---

## Phase 5 — Market layer

Build:

- MarketOpportunityCard
- EdgeGap
- OpportunityTable
- MarketMovementSparkline
- blocker states

Markets stay after sporting intelligence.

---

## Phase 6 — Today / Edge Feed

Once Match Center establishes the design system, reuse it to rebuild:

- Today dashboard
- Edge Feed
- Picks
- Leans
- Results

Do not separately invent a new design language for those pages.

---

## Phase 7 — Control Tower

Admin-only surface can reuse the same shell while exposing:

- Render health
- Postgres health
- scheduler health
- provider budget
- latest tick
- pipeline stages
- errors
- model maturity
- CLV/performance

This surface is separate from subscriber Match Center.

---

# 22. VISUAL QA PROCESS

UI QA must be screenshot-based.

For every major iteration compare:

1. production screenshot;
2. target mockup;
3. generated target concept.

Check:

- information density;
- chart presence;
- vertical space usage;
- hierarchy;
- typography;
- card proportions;
- mobile clipping;
- color balance;
- visual dominance of Sport over Markets.

Do not accept “the same data is present” as visual completion.

---

# 23. REPLICA ACCEPTANCE TEST

The rebuild is successful when a user looking at a screenshot can immediately recognize the approved mockup as the same product family.

Specifically:

1. Match Center no longer looks like stacked documentation.
2. The first viewport contains real analytical visuals.
3. Score Matrix is a heatmap, not a list of chips.
4. Probability is visually dominant when available.
5. Form is visual, not a text string only.
6. Goal distribution is charted.
7. Sport Profile is visual.
8. Missing-data blocks are compact.
9. Raw evidence is secondary.
10. Mobile has no horizontal clipping.
11. Desktop uses dashboard composition rather than a single vertical column.
12. Markets remain visibly secondary to Sport.

---

# 24. IMPLEMENTATION GUARDRAILS

During frontend rebuild:

- do not change Soccer Edge model weights;
- do not change BET/LEAN/WATCH/PASS thresholds merely for UI needs;
- do not reverse-engineer missing sport data from market prices;
- do not increase provider requests just to fill charts without documenting budget impact;
- do not remove auditability;
- do not silently change feature definitions;
- do not rewrite historical model outputs.

Frontend work consumes the existing truth; it does not manufacture truth.

---

# 25. CURRENT BACKEND GAP TO ADDRESS IN PARALLEL

The frontend can only become as rich as verified backend sport data permits.

The data roadmap should continue to improve:

- team statistics persistence;
- recent form;
- home/away splits;
- scoring and conceding trends;
- lineup persistence;
- verified availability;
- opponent-adjusted context;
- verified advanced metrics where legitimate sources exist.

But the frontend must already be capable of rendering these inputs once available.

---

# 26. FINAL ARCHITECTURE PRINCIPLE

The target product is:

```text
BACKEND
fixture + sport evidence + model + availability + market + audit
                    ↓
CUSTOMER API CONTRACT
clean verified domain objects
                    ↓
REACT MATCH CENTER
real visual components and charts
                    ↓
SPORT FIRST USER EXPERIENCE
understand football → uncertainty → availability → market → decision
```

---

# 27. SOURCE-OF-TRUTH STATUS

This file governs the frontend rebuild strategy until versioned or superseded.

Consult it together with:

- `18_SOCCER_MATCH_CENTER_UI_BLUEPRINT_v1.0.md`
- Soccer Edge master methodology
- data/API protocol
- scheduler/refresh protocol

Material changes to frontend architecture should update this file or produce a new version.

---

# 28. IMMEDIATE NEXT ACTION

Start Phase 1 as a side-by-side React implementation under `/app-v3`.

The first development milestone is not “finish the whole app.”

It is:

> **Reproduce one Match Center mobile matchup closely enough to the approved mockup that the visual architecture is proven.**

Use real backend data for the selected fixture.

Once that component system is accepted, generalize it to every matchup and then reuse it across the rest of Soccer Edge.
