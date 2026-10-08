import { hasStoredSession, refreshSession, storedAccessToken } from "./auth";
import type {
  MatchCenterViewModel,
  ScoreMatrix,
  SectionState,
  TeamScoringProfile,
  VerificationState,
} from "./model";

type Json = Record<string, unknown>;
const API_BASE = "/app/api/v2";
const SCORE_LABELS = ["0", "1", "2", "3", "4+"];

export interface SlateItem {
  fixtureId: number;
  kickoff: string;
  kickoffRaw: string;
  league: string;
  home: string;
  away: string;
  status: string;
  reason: string;
  sportEvidenceCount: number;
  marketEvidenceCount: number;
  persistedEvidenceCount: number;
  analysisRows: number;
  homeLogoUrl: string | null;
  awayLogoUrl: string | null;
}

function record(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {};
}

function rows(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter((x): x is Json => !!x && typeof x === "object" && !Array.isArray(x)) : [];
}

function numberValue(value: unknown): number | null {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function percentValue(value: unknown): number | null {
  const n = numberValue(value);
  if (n === null) return null;
  return Math.abs(n) <= 1 ? n * 100 : n;
}

function scoreValue(value: unknown): number | null {
  const n = numberValue(value);
  if (n === null) return null;
  return Math.max(0, Math.min(100, Math.abs(n) <= 1 ? n * 100 : n));
}

function text(value: unknown, fallback = "NOT VERIFIED"): string {
  const s = String(value ?? "").trim();
  return s || fallback;
}

function initials(name: string): string {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((x) => x[0]).join("").toUpperCase().slice(0, 3) || "?";
}

function kickoffLabel(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (!raw) return "TBD";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return raw;
  return new Intl.DateTimeFormat("es-MX", {
    timeZone: "America/Mexico_City",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function section(state: VerificationState, note?: string): SectionState {
  return note ? { state, note } : { state };
}

async function jsonFetch(path: string, token?: string): Promise<{ response: Response; data: Json }> {
  const headers: HeadersInit = token ? { Authorization: "Bearer " + token } : {};
  const response = await fetch(path, { headers, cache: "no-store" });
  const data = record(await response.json().catch(() => ({})));
  return { response, data };
}

function fixtureFromSlateRow(row: Json): Json {
  return record(row.fixture);
}

function fixtureIdFromLocation(): number | null {
  const query = new URLSearchParams(window.location.search).get("fixture_id");
  const pathMatch = window.location.pathname.match(/\/app-v3-react\/match\/(\d+)/);
  const raw = query || pathMatch?.[1] || "";
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function lockedView(fixture: Json, state: VerificationState, note: string): MatchCenterViewModel {
  const home = text(fixture.home_team, "Home");
  const away = text(fixture.away_team, "Away");
  const common = section(state, note);
  return {
    sample: false,
    live: true,
    fixtureId: numberValue(fixture.fixture_id),
    league: text(fixture.league ?? fixture.country, "Competition"),
    kickoff: kickoffLabel(fixture.kickoff),
    home: {
      name: home,
      shortName: initials(home),
      side: "home",
      logoUrl: typeof fixture.home_team_logo === "string" ? fixture.home_team_logo : null,
    },
    away: {
      name: away,
      shortName: initials(away),
      side: "away",
      logoUrl: typeof fixture.away_team_logo === "string" ? fixture.away_team_logo : null,
    },
    dataQuality: "NOT VERIFIED",
    confidence: null,
    lineup: "NOT VERIFIED",
    market: "NOT VERIFIED",
    probability: null,
    xg: null,
    edge: null,
    sportProfile: [],
    scoreMatrix: null,
    scoringProfiles: [],
    over25: null,
    decision: { classification: null, displayBucket: state, tier: null, reason: note },
    sections: {
      probability: common,
      xg: common,
      edge: common,
      sportProfile: common,
      scoreMatrix: common,
      scoringProfiles: common,
      over25: common,
    },
    missingSections: [],
    disclosure: state + " · LIVE FIXTURE IDENTITY ONLY",
  };
}

function parseScoreMatrix(raw: unknown): ScoreMatrix | null {
  const entries = rows(raw);
  if (!entries.length) return null;
  const values: Array<Array<number | null>> = Array.from({ length: 5 }, () => Array<number | null>(5).fill(null));
  let hot: [number, number] | null = null;
  let best = -1;
  let mostLikely: { score: string; probability: number } | null = null;

  for (const item of entries) {
    const score = String(item.score ?? "").replace(":", "-").trim();
    const parts = score.split("-");
    if (parts.length !== 2) continue;
    const hg = Number(parts[0]);
    const ag = Number(parts[1]);
    if (!Number.isFinite(hg) || !Number.isFinite(ag)) continue;
    const r = Math.max(0, Math.min(4, Math.trunc(ag)));
    const c = Math.max(0, Math.min(4, Math.trunc(hg)));
    const p = percentValue(item.probability);
    if (p === null) continue;
    values[r][c] = Number(p.toFixed(1));
    if (p > best) {
      best = p;
      hot = [r, c];
      mostLikely = { score, probability: Number(p.toFixed(1)) };
    }
  }
  return values.some((row) => row.some((x) => x !== null))
    ? { labels: SCORE_LABELS, values, hot, mostLikely }
    : null;
}

function parseSportProfile(raw: unknown): MatchCenterViewModel["sportProfile"] {
  return rows(raw).flatMap((item) => {
    const score = scoreValue(item.score);
    if (score === null) return [];
    return [{
      label: text(item.label, "Verified input"),
      score,
      status: score >= 65 ? "GOOD" as const : "NEUTRAL" as const,
    }];
  });
}

function chooseEdge(payload: Json): MatchCenterViewModel["edge"] {
  const candidate = record(payload.selected_candidate);
  const projections = record(candidate.projections);
  const ladder = record(payload.projection_ladder);
  const selectedMarket = record(candidate.market);
  const price = record(selectedMarket.price);

  const options: Array<[string, unknown]> = [
    ["CALIBRATED MODEL", ladder.calibrated_model_probability ?? projections.calibrated_model_probability],
    ["MARKET-SHRUNK", ladder.market_shrunk_probability ?? projections.market_shrunk_probability],
    ["RAW SPORT", ladder.raw_sport_probability ?? projections.raw_sport_probability],
  ];
  const picked = options.map(([kind, value]) => [kind, percentValue(value)] as const).find(([, value]) => value !== null);
  const market = percentValue(ladder.fair_market_probability ?? projections.fair_market_probability);
  if (!picked || market === null) return null;
  const model = picked[1] as number;
  const explicitGap = numberValue(ladder.probability_edge_pp ?? projections.probability_edge_pp);
  return {
    model: Number(model.toFixed(1)),
    market: Number(market.toFixed(1)),
    gap: Number((explicitGap ?? (model - market)).toFixed(1)),
    fairPrice: numberValue(selectedMarket.fair_price),
    marketPrice: numberValue(price.value),
    modelKind: picked[0],
  };
}

function chooseOver25(payload: Json): MatchCenterViewModel["over25"] {
  const marketContext = record(payload.market_context);
  const groups = record(marketContext.groups);
  const candidates = rows(groups.GOALS);
  for (const candidate of candidates) {
    const market = record(candidate.market);
    const selection = String(market.selection ?? market.name ?? "").toUpperCase();
    const line = numberValue(market.line);
    if (line !== 2.5 || !selection.includes("OVER")) continue;
    const projections = record(candidate.projections);
    const model = percentValue(projections.raw_sport_probability);
    const fair = percentValue(projections.fair_market_probability);
    if (model === null || fair === null) continue;
    const explicit = numberValue(projections.probability_edge_pp);
    return {
      model: Number(model.toFixed(1)),
      market: Number(fair.toFixed(1)),
      edge: Number((explicit ?? (model - fair)).toFixed(1)),
      modelKind: "RAW SPORT",
    };
  }
  return null;
}

function adaptMatch(payload: Json): MatchCenterViewModel {
  const fixture = record(payload.fixture);
  const sport = record(payload.sport_context);
  const outcome = record(sport.outcome_probabilities);
  const expected = record(sport.expected_goals);
  const availability = record(payload.availability);
  const modelContext = record(payload.model_context);
  const selected = record(payload.selected_candidate);
  const freshness = record(selected.freshness);
  const review = record(payload.analyst_review);
  const decision = record(payload.decision_summary);

  const homeName = text(fixture.home_team, "Home");
  const awayName = text(fixture.away_team, "Away");
  const ph = percentValue(outcome.home);
  const pd = percentValue(outcome.draw);
  const pa = percentValue(outcome.away);
  const probability = ph !== null && pd !== null && pa !== null
    ? { home: Number(ph.toFixed(1)), draw: Number(pd.toFixed(1)), away: Number(pa.toFixed(1)) }
    : null;

  const xgHome = numberValue(expected.home);
  const xgAway = numberValue(expected.away);
  const xg = xgHome !== null || xgAway !== null ? { home: xgHome, away: xgAway } : null;

  const confidenceRaw = numberValue(availability.confidence ?? modelContext.confidence);
  const confidence = confidenceRaw === null ? null : Number((Math.abs(confidenceRaw) <= 1 ? confidenceRaw * 100 : confidenceRaw).toFixed(0));
  const matrix = parseScoreMatrix(sport.score_matrix);
  const profile = parseSportProfile(sport.sport_profile);
  const edge = chooseEdge(payload);
  const over25 = chooseOver25(payload);
  const missingSections = Array.isArray(review.missing_sections) ? review.missing_sections.map(String) : [];

  const stateFor = (present: boolean, missingKey: string, marketOnly = false): SectionState => {
    if (present) return section("VERIFIED");
    if (marketOnly) return section("MARKET_DATA_ONLY", "Market evidence exists, but the sporting input required for this visual is not verified.");
    return section(
      missingSections.includes(missingKey) ? "NOT_VERIFIED" : "INSUFFICIENT_DATA",
      "This visual is hidden rather than filled with assumed values."
    );
  };

  const edgeState = edge
    ? section("VERIFIED")
    : section(
        record(payload.market_context).candidate_count ? "MARKET_DATA_ONLY" : "NOT_VERIFIED",
        "No verified model-versus-fair-market comparison is available for this fixture."
      );

  const marketFresh = freshness.market_fresh === true ? "FRESH" : "NOT VERIFIED";
  const dataTier = text(availability.data_tier ?? modelContext.data_quality);
  const lineup = text(availability.starting_xi_status ?? availability.lineup_status ?? modelContext.lineup);

  return {
    sample: false,
    live: true,
    fixtureId: numberValue(fixture.fixture_id),
    league: text(fixture.league ?? fixture.country, "Competition"),
    kickoff: kickoffLabel(fixture.kickoff),
    home: {
      name: homeName,
      shortName: initials(homeName),
      side: "home",
      logoUrl: typeof fixture.home_team_logo === "string" ? fixture.home_team_logo : null,
    },
    away: {
      name: awayName,
      shortName: initials(awayName),
      side: "away",
      logoUrl: typeof fixture.away_team_logo === "string" ? fixture.away_team_logo : null,
    },
    dataQuality: dataTier,
    confidence,
    lineup,
    market: marketFresh,
    probability,
    xg,
    edge,
    sportProfile: profile,
    scoreMatrix: matrix,
    scoringProfiles: [],
    over25,
    decision: {
      classification: decision.classification ? String(decision.classification) : null,
      displayBucket: decision.display_bucket ? String(decision.display_bucket) : null,
      tier: decision.tier ? String(decision.tier) : null,
      reason: decision.reason_display ? String(decision.reason_display) : null,
    },
    sections: {
      probability: stateFor(!!probability, "OUTCOME_PROBABILITIES"),
      xg: stateFor(!!xg, "EXPECTED_GOALS"),
      edge: edgeState,
      sportProfile: stateFor(profile.length > 0, "SPORT_PROFILE"),
      scoreMatrix: stateFor(!!matrix, "SCORE_MATRIX"),
      scoringProfiles: section("NOT_VERIFIED", "Team GF × GA heatmaps need a persisted verified source. No sample values are shown in live mode."),
      over25: over25 ? section("VERIFIED") : section("NOT_VERIFIED", "No verified Over 2.5 row with both RAW SPORT and fair-market probability is available."),
    },
    missingSections,
    disclosure: "LIVE PERSISTED DATA · UNKNOWN = NOT VERIFIED · PROVIDER REQUESTS ADDED: 0",
  };
}

async function loadTodayContract(): Promise<{ data: Json; token: string }> {
  let token = storedAccessToken();
  let result = await jsonFetch(API_BASE + "/today", token || undefined);

  if (result.response.status === 401 && hasStoredSession()) {
    token = await refreshSession();
    result = await jsonFetch(API_BASE + "/today", token || undefined);
  }
  if (result.response.status === 401) {
    token = "";
    result = await jsonFetch(API_BASE + "/today");
  }
  if (!result.response.ok) {
    throw new Error(text(result.data.error, "TODAY_DATA_UNAVAILABLE"));
  }
  return { data: result.data, token };
}

function slateRowsFromToday(today: Json): Json[] {
  return rows(record(today.slate).rows);
}

function slateScore(row: Json): number {
  const coverage = record(row.coverage);
  const sport = numberValue(coverage.sport_evidence_count) ?? 0;
  const analysis = numberValue(coverage.analysis_rows) ?? 0;
  const persisted = numberValue(coverage.persisted_evidence_count) ?? 0;
  const market = numberValue(coverage.market_evidence_count) ?? 0;
  // SPORT FIRST: verified sporting evidence dominates market-only evidence.
  return sport * 100000 + analysis * 1000 + persisted * 10 + market;
}

export async function loadSlate(): Promise<SlateItem[]> {
  const { data } = await loadTodayContract();
  return slateRowsFromToday(data).flatMap((row) => {
    const fixture = fixtureFromSlateRow(row);
    const coverage = record(row.coverage);
    const state = record(row.state);
    const fixtureId = numberValue(fixture.fixture_id);
    if (!fixtureId) return [];
    const kickoffRaw = String(fixture.kickoff ?? "");
    return [{
      fixtureId,
      kickoff: kickoffLabel(kickoffRaw),
      kickoffRaw,
      league: text(fixture.league ?? fixture.country, "Competition"),
      home: text(fixture.home_team, "Home"),
      away: text(fixture.away_team, "Away"),
      status: text(state.display_status ?? coverage.coverage_status, "INSUFFICIENT DATA"),
      reason: text(state.reason_display ?? coverage.decision_reason, "No persisted deep-analysis evidence is available yet."),
      sportEvidenceCount: numberValue(coverage.sport_evidence_count) ?? 0,
      marketEvidenceCount: numberValue(coverage.market_evidence_count) ?? 0,
      persistedEvidenceCount: numberValue(coverage.persisted_evidence_count) ?? 0,
      analysisRows: numberValue(coverage.analysis_rows) ?? 0,
      homeLogoUrl: typeof fixture.home_team_logo === "string" ? fixture.home_team_logo : null,
      awayLogoUrl: typeof fixture.away_team_logo === "string" ? fixture.away_team_logo : null,
    }];
  });
}

export async function loadMatchCenter(): Promise<MatchCenterViewModel> {
  const params = new URLSearchParams(window.location.search);
  if (params.get("sample") === "1") {
    const { sampleMatch } = await import("./sample");
    return sampleMatch;
  }

  const todayLoaded = await loadTodayContract();
  let token = todayLoaded.token;
  const slateRows = slateRowsFromToday(todayLoaded.data);
  const requestedId = fixtureIdFromLocation();
  let selectedRow = requestedId
    ? slateRows.find((row) => numberValue(fixtureFromSlateRow(row).fixture_id) === requestedId)
    : [...slateRows].sort((a, b) => slateScore(b) - slateScore(a))[0];

  if (!selectedRow && requestedId) {
    selectedRow = { fixture: { fixture_id: requestedId } };
  }
  if (!selectedRow) {
    throw new Error("NO_ELIGIBLE_FIXTURE");
  }

  const fixture = fixtureFromSlateRow(selectedRow);
  const fixtureId = requestedId ?? numberValue(fixture.fixture_id);
  if (!fixtureId) {
    return lockedView(fixture, "INSUFFICIENT_DATA", "Eligible fixture identity is incomplete.");
  }

  let detail = await jsonFetch(API_BASE + "/match/" + fixtureId, token || undefined);
  if (detail.response.status === 401 && hasStoredSession()) {
    token = await refreshSession();
    detail = await jsonFetch(API_BASE + "/match/" + fixtureId, token || undefined);
  }
  if (detail.response.status === 403) {
    return lockedView(fixture, "PREMIUM_REQUIRED", token
      ? "Your current plan does not unlock persisted Match Intelligence."
      : "Sign in with an Edge Pro account to unlock persisted Match Intelligence.");
  }
  if (detail.response.status === 401) {
    return lockedView(fixture, "UNAVAILABLE", "Your session expired and could not be refreshed. Sign in again.");
  }
  if (!detail.response.ok) {
    return lockedView(
      fixture,
      detail.response.status === 404 ? "INSUFFICIENT_DATA" : "UNAVAILABLE",
      text(detail.data.error, "MATCH_DATA_UNAVAILABLE")
    );
  }
  return adaptMatch(detail.data);
}
