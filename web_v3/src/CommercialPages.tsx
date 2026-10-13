import { useEffect, useState } from "react";
import { refreshSession, storedAccessToken } from "./auth";

/** Read-only commercial screens. Persisted evidence is NOT a live wager. */
export type CommercialSection = "picks" | "leans" | "performance" | "results";
type Json = Record<string, unknown>;
type State =
  | { status: "loading" }
  | { status: "locked"; detail: string }
  | { status: "unavailable"; detail: string }
  | { status: "ready"; data: Json };

const ENDPOINT: Record<CommercialSection, string> = {
  picks: "/app/api/v2/picks",
  leans: "/app/api/v2/leans",
  performance: "/app/api/v2/performance",
  results: "/app/api/v2/performance",
};
const TITLE: Record<CommercialSection, string> = {
  picks: "Persisted picks",
  leans: "Research leans",
  performance: "BET-only performance",
  results: "Historical results",
};

function object(value: unknown): Json {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Json : {};
}
function list(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter((item): item is Json => item !== null && typeof item === "object" && !Array.isArray(item)) : [];
}
function string(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function count(value: unknown): string {
  const n = numeric(value);
  return n === null ? "NOT VERIFIED" : n.toLocaleString("en-US");
}
function localDate(value: unknown): string {
  const raw = string(value);
  if (!raw) return "NOT VERIFIED";
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? new Intl.DateTimeFormat("es-MX", {
    timeZone: "America/Mexico_City", day: "2-digit", month: "short",
    year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(new Date(ms)) : "NOT VERIFIED";
}
function sourceAge(value: unknown): string {
  const raw = string(value);
  if (!raw) return "SOURCE TIME NOT VERIFIED";
  const ms = Date.parse(raw);
  if (!Number.isFinite(ms)) return "SOURCE TIME NOT VERIFIED";
  return Date.now() - ms > 2 * 60 * 60 * 1000
    ? "ARCHIVED SNAPSHOT · NOT A CURRENT QUOTE"
    : "RECENT PERSISTED SNAPSHOT · PRICE FRESHNESS NOT VERIFIED";
}
function percentage(value: unknown): string {
  const ratio = numeric(value);
  return ratio === null || ratio < 0 || ratio > 1 ? "NOT VERIFIED" : (ratio * 100).toFixed(1) + "%";
}
function units(value: unknown): string {
  const n = numeric(value);
  return n === null ? "NOT VERIFIED" : (n >= 0 ? "+" : "") + n.toFixed(2) + " units";
}

async function fetchWithAccess(endpoint: string, token: string): Promise<Response> {
  return fetch(endpoint, {
    method: "GET", cache: "no-store",
    headers: token ? { Authorization: "Bearer " + token } : {},
  });
}
async function readOnlyResource(section: CommercialSection): Promise<State> {
  try {
    let token = storedAccessToken();
    let response = await fetchWithAccess(ENDPOINT[section], token);
    if (response.status === 401 && token) {
      token = await refreshSession();
      response = await fetchWithAccess(ENDPOINT[section], token);
    }
    if (response.status === 401) return { status: "locked", detail: "SIGN IN REQUIRED" };
    if (response.status === 403) return { status: "locked", detail: "EDGE PRO OR OWNER REQUIRED" };
    if (!response.ok) return { status: "unavailable", detail: "SOURCE NOT VERIFIED · HTTP " + response.status };
    const payload = object(await response.json().catch(() => ({})));
    if (!Object.keys(payload).length) return { status: "unavailable", detail: "EMPTY OR INVALID SOURCE" };
    if (section === "picks" || section === "leans") {
      const group = object(payload[section]);
      if (!Array.isArray(group.rows) || !numeric(group.total) && group.total !== 0) {
        return { status: "unavailable", detail: "CANONICAL CANDIDATE CONTRACT NOT VERIFIED" };
      }
    } else {
      if (!payload.performance_policy || !payload.bet_track_record) {
        return { status: "unavailable", detail: "CANONICAL SETTLEMENT CONTRACT NOT VERIFIED" };
      }
      const policy = object(payload.performance_policy);
      if (policy.headline_scope !== "CANONICAL_BET_SETTLEMENT_ONLY" ||
        policy.leans_in_headline !== false) {
        return { status: "unavailable", detail: "BET-ONLY HEADLINE POLICY NOT VERIFIED" };
      }
    }
    return { status: "ready", data: payload };
  } catch {
    return { status: "unavailable", detail: "READ-ONLY SOURCE UNAVAILABLE" };
  }
}

function DataMarker({ value }: { value: unknown }) {
  return <span className="commercial-source">Captured: {localDate(value)}</span>;
}
function CandidateRow({ item, expected }: { item: Json; expected: "BET" | "LEAN" }) {
  const fixture = object(item.fixture);
  const market = object(item.market);
  const decision = object(item.decision);
  const price = object(market.price);
  const classification = string(decision.classification);
  if (classification !== expected) return null; // never promote WATCH into BET
  const home = string(fixture.home_team), away = string(fixture.away_team);
  const fixtureId = numeric(fixture.fixture_id);
  return <article className="commercial-candidate">
    <div className="commercial-candidate-top">
      <strong>{home && away ? home + " vs " + away : "FIXTURE IDENTITY NOT VERIFIED"}</strong>
      <span className="commercial-classification">{expected} · STORED DECISION</span>
    </div>
    <small>{string(fixture.league) || "Competition NOT VERIFIED"} · {localDate(fixture.kickoff)}</small>
    <div className="commercial-candidate-details">
      <span>Market <b>{string(market.name) || string(market.family) || "NOT VERIFIED"}</b></span>
      <span>Selection <b>{string(market.selection) || "NOT VERIFIED"}</b></span>
      <span>Price <b>{numeric(price.value)?.toFixed(2) || "NOT VERIFIED"}</b></span>
      <span>Bookmaker <b>{string(price.bookmaker) || "NOT VERIFIED"}</b></span>
    </div>
    <small>Price captured: {localDate(price.captured_at)} · Historical snapshot, not actionable.</small>
    {string(decision.reason_display) && <p>{string(decision.reason_display)}</p>}
    {fixtureId !== null && Number.isInteger(fixtureId) && fixtureId > 0 &&
      <a href={"/app-v3-react/match/" + fixtureId}>View persisted matchup <span aria-hidden="true">↗</span></a>}
  </article>;
}

function Candidates({ data, section }: { data: Json; section: "picks" | "leans" }) {
  const group = object(data[section]);
  const items = list(group.rows);
  const expected = section === "picks" ? "BET" : "LEAN";
  const verified = items.filter(item => object(item.decision).classification === expected);
  return <>
    <div className="commercial-kpis">
      <div><small>{expected} in snapshot</small><strong>{count(group.total)}</strong></div>
      <div><small>Rows received</small><strong>{items.length}</strong></div>
      <div><small>Source model</small><strong className="commercial-kpi-small">{string(data.model_version) || "NOT VERIFIED"}</strong></div>
    </div>
    <p className="commercial-disclaimer">These are persisted classifications from the source contract, not current prices or betting instructions. No new picks are calculated here. A market requires fresh price, confirmed availability and server-side eligibility to become actionable.</p>
    {verified.length ? <div className="commercial-candidates">{verified.map((item, i) =>
      <CandidateRow key={i} item={item} expected={expected} />)}</div> :
      <div className="commercial-empty">
        <b>NO VERIFIED {expected} ROWS IN THE PERSISTED SNAPSHOT</b>
        <p>Zero eligible candidates is a valid outcome. Missing or stale odds are not replaced with example bets.</p>
      </div>}
    {verified.length !== items.length && <p className="commercial-caution">Some source rows lacked an explicit {expected} classification and were not displayed.</p>}
  </>;
}

function Performance({ data, historical }: { data: Json; historical: boolean }) {
  const track = object(data.bet_track_record);
  const lean = object(data.research_lean);
  const settlement = object(data.settlement);
  const clv = object(data.true_clv);
  const familyRows = list(track.families);
  const settled = numeric(track.settled);
  const min = numeric(track.directional_minimum);
  const sufficient = settled !== null && min !== null && settled >= min;
  return <>
    <div className="commercial-kpis">
      <div><small>Settled BET only</small><strong>{count(track.settled)}</strong></div>
      <div><small>BET wins / losses</small><strong>{count(track.win)} / {count(track.loss)}</strong></div>
      <div><small>BET pushes</small><strong>{count(track.push)}</strong></div>
      <div><small>BET ROI · units</small><strong>{settled && settled > 0 ? units(track.roi_units) : "NOT VERIFIED"}</strong></div>
    </div>
    <p className="commercial-disclaimer">
      {historical
        ? "Historical outcome summary based only on the canonical settled BET ledger. Individual fixture-level scores and graded bet rows are not exposed by this endpoint; no match history is reconstructed."
        : "Verified BET-only settlement summary. LEAN/OOS research is never counted as realized BET performance."}
    </p>
    <div className="commercial-maturity-notice">
      <b>{sufficient ? "MINIMUM SAMPLE REACHED · NOT AN APPROVAL" : "SMALL SAMPLE · DIRECTIONAL CLAIMS BLOCKED"}</b>
      <span>Directional minimum: {count(track.directional_minimum)} settled BET · Observed: {count(track.settled)}</span>
      <span>BET hit rate (excluding pushes): {settled && settled > 0 ? percentage(track.hit_rate_ex_push) : "NOT VERIFIED"}</span>
    </div>
    <div className="commercial-columns">
      <section className="commercial-section">
        <h3>Historical BET settlement by family</h3>
        {familyRows.length ? <div className="commercial-family-list">{familyRows.map((row, i) =>
          <div key={i}><strong>{string(row.market_family) || "NOT VERIFIED"}</strong><span>{count(row.settled)} settled</span><span>{count(row.win)} W / {count(row.loss)} L / {count(row.push)} push</span></div>)}</div> :
          <div className="commercial-empty"><b>NO FAMILY-LEVEL VERIFIED SETTLEMENT</b><p>The report has not supplied enough settlement history.</p></div>}
      </section>
      <section className="commercial-section">
        <h3>Research and source checks</h3>
        <div className="commercial-data-list">
          <div><span>LEAN settled (separate)</span><b>{count(lean.settled)}</b></div>
          <div><span>Settlement coverage</span><b>{percentage(settlement.coverage_rate)}</b></div>
          <div><span>Unsettled backlog</span><b>{count(settlement.backlog_rows)}</b></div>
          <div><span>Canonical True CLV sample</span><b>{count(clv.rows)}</b></div>
          <div><span>Source model</span><b>{string(object(data.source).bet_track_record_model_version) || "NOT VERIFIED"}</b></div>
        </div>
      </section>
    </div>
    {historical && <div className="commercial-empty">
      <b>FIXTURE-LEVEL RESULT HISTORY · NOT VERIFIED</b>
      <p>The current Performance API provides aggregated settlement only. A separate read-only, date-bounded historical results contract is required to show actual final scores, original picks, odds and outcomes per game.</p>
    </div>}
  </>;
}

export function CommercialPage({ section, onOpenAuth }: {
  section: CommercialSection;
  onOpenAuth: () => void;
}) {
  const [state, setState] = useState<State>({ status: "loading" });
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let current = true;
    // Role changes must not leave stale protected rows visible.
    setState({ status: "loading" });
    readOnlyResource(section).then(next => { if (current) setState(next); });
    return () => { current = false; };
  }, [section, reload]);
  return <section className="react-page commercial-page">
    <header className="slate-page-head">
      <div><span>SOCCER EDGE · VERIFIED COMMERCIAL EVIDENCE</span><h1>{TITLE[section]}</h1>
        <p>Sport first · Market second · No new provider requests</p></div>
      {state.status === "ready" && <button className="commercial-recheck" onClick={() => setReload(n => n + 1)}>Recheck persisted evidence</button>}
    </header>
    {state.status === "loading" ? <div className="commercial-empty" role="status">Checking access and persisted evidence…</div> :
      state.status === "locked" ? <div className="commercial-empty" role="status"><b>{state.detail}</b><p>Premium evidence is not displayed without authorization.</p><button className="commercial-recheck" onClick={onOpenAuth}>Sign in</button></div> :
      state.status === "unavailable" ? <div className="commercial-empty" role="alert"><b>{state.detail}</b><p>Source unavailable. No archived probabilities, odds or settlements have been invented.</p><button className="commercial-recheck" onClick={() => setReload(n => n + 1)}>Retry</button></div> :
      <>
        <div className="commercial-source-bar"><span>{sourceAge(section === "picks" || section === "leans" ? state.data.generated_at_utc : object(state.data.source).generated_at_utc)}</span>
          <DataMarker value={section === "picks" || section === "leans" ? state.data.generated_at_utc : object(state.data.source).generated_at_utc} /></div>
        {section === "picks" || section === "leans"
          ? <Candidates data={state.data} section={section}/>
          : <Performance data={state.data} historical={section === "results"}/>}
      </>}
  </section>;
}
