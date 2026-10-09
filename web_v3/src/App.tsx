import { useEffect, useState } from "react";
import { loadAuthState, signIn, signOut, signUp, type AuthState } from "./auth";
import { loadMatchCenter, loadSlate, type SlateItem } from "./live";
import type {
  MatchCenterViewModel,
  ScoreMatrix,
  SectionState,
  TeamScoringProfile,
} from "./model";

function TeamBadge({ team }: { team: MatchCenterViewModel["home"] }) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const showImage = !!team.logoUrl && failedUrl !== team.logoUrl;
  return (
    <div className={"team-badge " + team.side + (showImage ? " with-logo" : "")}>
      {showImage
        ? <img src={team.logoUrl!} alt={team.name + " crest"} onError={() => setFailedUrl(team.logoUrl ?? null)} />
        : <span>{team.shortName}</span>}
    </div>
  );
}

function MiniTeamLogo({ url, name }: { url: string | null; name: string }) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const showImage = !!url && failedUrl !== url;
  return showImage
    ? <img src={url!} alt={name + " crest"} onError={() => setFailedUrl(url)} />
    : <i aria-label={name + " crest unavailable"}>{name.slice(0,2).toUpperCase()}</i>;
}

function MissingPanel({ title, section }: { title: string; section: SectionState }) {
  return (
    <article className="panel missing-panel">
      <header><h3>{title}</h3><span>{section.state.split("_").join(" ")}</span></header>
      <div className="missing-state">
        <b>{section.state.split("_").join(" ")}</b>
        <p>{section.note || "No verified persisted value is available for this visual."}</p>
      </div>
    </article>
  );
}

function Hero({ match }: { match: MatchCenterViewModel }) {
  return (
    <section className="hero">
      <div className="hero-topline">
        <span>{match.league} · Today</span>
        <span>{match.kickoff}</span>
      </div>
      <div className="hero-main">
        <div className="team">
          <TeamBadge team={match.home} />
          <b>{match.home.name}</b><span>HOME</span>
        </div>
        <div className="vs">VS</div>
        <div className="team">
          <TeamBadge team={match.away} />
          <b>{match.away.name}</b><span>AWAY</span>
        </div>
        <aside className="quality">
          <div><span>Data Quality</span><b>{match.dataQuality}</b></div>
          <div><span>Lineup</span><b>{match.lineup}</b></div>
          <div><span>Market</span><b>{match.market}</b></div>
        </aside>
      </div>
    </section>
  );
}

function ProbabilityPanel({ match }: { match: MatchCenterViewModel }) {
  const p = match.probability;
  if (!p) return <MissingPanel title="Match Result Probability" section={match.sections.probability} />;
  const leader = Math.max(p.home, p.draw, p.away);
  const leaderLabel = leader === p.home ? "HOME" : leader === p.draw ? "DRAW" : "AWAY";
  const runner = [p.home, p.draw, p.away].sort((a, b) => b - a)[1] ?? 0;
  return (
    <article className="panel probability">
      <header><h3>Match Result Probability</h3><span>SPORT MODEL</span></header>
      <div className="prob-strip">
        <div className={"prob-cell " + (leaderLabel === "HOME" ? "home" : "")}><span>HOME</span><b>{p.home.toFixed(1)}%</b></div>
        <div className={"prob-cell " + (leaderLabel === "DRAW" ? "home" : "")}><span>DRAW</span><b>{p.draw.toFixed(1)}%</b></div>
        <div className={"prob-cell " + (leaderLabel === "AWAY" ? "home" : "")}><span>AWAY</span><b>{p.away.toFixed(1)}%</b></div>
      </div>
      <div className="panel-foot"><i /> {leaderLabel} LEADS MODEL <strong>{leader.toFixed(1)}% · +{(leader-runner).toFixed(1)} pp</strong></div>
    </article>
  );
}

function XgPanel({ match }: { match: MatchCenterViewModel }) {
  const xg = match.xg;
  if (!xg) return <MissingPanel title="Projected Goals (λ)" section={match.sections.xg} />;
  const total = xg.home !== null && xg.away !== null ? xg.home + xg.away : null;
  const delta = xg.home !== null && xg.away !== null ? xg.home - xg.away : null;
  return (
    <article className="panel xg-panel">
      <header><h3>Projected Goals (λ)</h3><span>SPORT MODEL</span></header>
      <div className="xg-values">
        <div><span>{match.home.name}</span><b>{xg.home === null ? "—" : xg.home.toFixed(2)}</b></div>
        <em>—</em>
        <div><span>{match.away.name}</span><b>{xg.away === null ? "—" : xg.away.toFixed(2)}</b></div>
      </div>
      <div className="xg-meta">
        <div><span>TOTAL λ</span><b>{total === null ? "—" : total.toFixed(2)}</b></div>
        <div><span>HOME DELTA</span><b>{delta === null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(2)}</b></div>
      </div>
      <div className="verified-copy">{match.modelProvenance?.goalRateSemantics === "POISSON_LAMBDA_NOT_XG" ? "Poisson scoring rates, not independently verified xG" : "Persisted model scoring inputs · source shown in Model"}</div>
    </article>
  );
}

function EdgePanel({ match }: { match: MatchCenterViewModel }) {
  const e = match.edge;
  if (!e) return <MissingPanel title="Edge Gap" section={match.sections.edge} />;
  const min = Math.floor(Math.min(e.market, e.model) / 10) * 10 - 10;
  const max = Math.ceil(Math.max(e.market, e.model) / 10) * 10 + 10;
  return (
    <article className="panel edge-panel">
      <header><h3>Edge Gap</h3><span>MARKET LAYER</span></header>
      <div className="edge-grid">
        <div className="edge-visual">
          <div className="edge-pill"><b>{e.gap >= 0 ? "+" : ""}{e.gap.toFixed(1)} pp</b><span>EDGE GAP</span></div>
          <div className="edge-labels"><span>MARKET {e.market.toFixed(1)}</span><span>{e.modelKind} {e.model.toFixed(1)}</span></div>
          <div className="edge-scale">
            <span className="dot market" />
            <span className="edge-line" />
            <span className="dot model" />
          </div>
          <div className="ticks"><span>{min}%</span><span>{Math.round((min+max)/2)}%</span><span>{max}%</span></div>
        </div>
        <aside className="edge-stats">
          <div><span>FAIR PRICE</span><b>{e.fairPrice === null ? "N/V" : e.fairPrice.toFixed(2)}</b><small>Market {e.marketPrice === null ? "NOT VERIFIED" : e.marketPrice.toFixed(2)}</small></div>
          <div><span>CONFIDENCE</span><b>{match.confidence === null ? "N/V" : match.confidence}</b><small>Availability confidence</small></div>
        </aside>
      </div>
    </article>
  );
}

function SportProfile({ match }: { match: MatchCenterViewModel }) {
  if (!match.sportProfile.length) return <MissingPanel title="Sport Profile" section={match.sections.sportProfile} />;
  return (
    <article className="panel sport-profile">
      <header><h3>Sport Profile</h3><span>FOOTBALL ONLY</span></header>
      <div className="profile-grid">
        {match.sportProfile.map((item) => (
          <div className="profile-row" key={item.label}>
            <span>{item.label}</span><b className={item.status.toLowerCase()}>{item.status}</b>
            <div><i style={{ width: item.score + "%" }} /></div>
          </div>
        ))}
      </div>
    </article>
  );
}

function Heatmap({ matrix, compact = false }: { matrix: ScoreMatrix; compact?: boolean }) {
  return (
    <div className={compact ? "heat compact" : "heat"}>
      <span />
      {matrix.labels.map((x) => <span className="axis" key={"x"+x}>{x}</span>)}
      {matrix.values.map((row, r) => [
        <span className="axis" key={"y"+r}>{matrix.labels[r]}</span>,
        ...row.map((v, c) => (
          <div
            key={r+"-"+c}
            className={
              v === null
                ? "heat-cell unavailable"
                : "heat-cell level-" + Math.min(5, Math.max(1, Math.ceil(v / 2))) +
                  (matrix.hot?.[0] === r && matrix.hot?.[1] === c ? " hot" : "")
            }
          >{v === null ? "·" : v}</div>
        ))
      ])}
    </div>
  );
}

function TeamHeatmap({ profile }: { profile: TeamScoringProfile }) {
  const matrix: ScoreMatrix = { labels: profile.labels, values: profile.values, hot: null };
  return (
    <div className={"team-heat " + profile.tone}>
      <div className="team-heat-head"><b>{profile.team}</b><span>GF × GA</span></div>
      <Heatmap matrix={matrix} compact />
      <small>X = GF · Y = GA</small>
    </div>
  );
}

function MatrixPanel({ match }: { match: MatchCenterViewModel }) {
  if (!match.scoreMatrix) return <MissingPanel title="Score Matrix (FT)" section={match.sections.scoreMatrix} />;
  const likely = match.scoreMatrix.mostLikely;
  return (
    <article className="panel matrix-panel">
      <header><h3>Score Matrix (FT)</h3><span>PERSISTED MODEL CELLS ONLY</span></header>
      <div className="matrix-layout">
        <div className="matrix-main">
          <Heatmap matrix={match.scoreMatrix} />
          <p>{likely ? <>Most likely persisted: <b>{likely.score} · {likely.probability.toFixed(1)}%</b></> : "No ranked scoreline persisted"}</p>
        </div>
        <div className="team-heats">
          {match.scoringProfiles.length
            ? match.scoringProfiles.map((p) => <TeamHeatmap key={p.team} profile={p} />)
            : <div className="inline-missing"><b>{match.sections.scoringProfiles.state.split("_").join(" ")}</b><span>{match.sections.scoringProfiles.note}</span></div>}
        </div>
      </div>
    </article>
  );
}

function GoalsPanel({ match }: { match: MatchCenterViewModel }) {
  const over = match.over25;
  if (!over) {
    const home = match.xg?.home, away = match.xg?.away;
    if (home !== null && home !== undefined && away !== null && away !== undefined &&
        home >= 0 && away >= 0 && match.modelProvenance?.goalRateSemantics === "POISSON_LAMBDA_NOT_XG") {
      const rate = home + away;
      const sportOver = 100 * (1 - Math.exp(-rate) * (1 + rate + rate * rate / 2));
      return <article className="panel goals-panel">
        <header><h3>Over / Under 2.5 Goals</h3><span>SPORT MODEL ONLY</span></header>
        <div className="sport-only-total"><small>Poisson · Sport projection Over 2.5</small><b>{sportOver.toFixed(1)}%</b></div>
        <p className="verified-copy">Derived from persisted λ {rate.toFixed(2)}. No verified market price, probability edge or EV.</p>
      </article>;
    }
    return <MissingPanel title="Over / Under 2.5 Goals" section={match.sections.over25} />;
  }
  return (
    <article className="panel goals-panel">
      <header><h3>Over / Under 2.5 Goals</h3><span>{over.modelKind} VS MARKET FAIR</span></header>
      <div className="ou-top">
        <div className="ou-copy"><span>{over.modelKind} OVER 2.5</span><b>{over.model.toFixed(1)}%</b><strong>{over.edge >= 0 ? "+" : ""}{over.edge.toFixed(1)} pp edge</strong><small>Market fair {over.market.toFixed(1)}%</small></div>
        <div className="ou-gauge">
          <span className="gauge-label market">MARKET {over.market.toFixed(1)}</span>
          <span className="gauge-label model">MODEL {over.model.toFixed(1)}</span>
          <div className="gauge-base"><i className="gauge-gap" /><i className="gauge-market" /><i className="gauge-model" /></div>
          <div className="gauge-ticks"><span>40%</span><span>50%</span><span>60%</span><span>70%</span><span>80%</span></div>
        </div>
      </div>
      <div className="verified-copy">Distribution is hidden until a verified persisted goal-distribution source is exposed.</div>
    </article>
  );
}


const MATCH_TABS = ["Overview","Goals","Corners","Cards","Players","Market","Model"] as const;
type MatchTab = typeof MATCH_TABS[number];

type SportEvidenceGroup = NonNullable<MatchCenterViewModel["evidenceSections"]>[number];
type SportEvidenceItem = SportEvidenceGroup["items"][number];

function sportValue(item: SportEvidenceItem | undefined): string {
  if (!item) return "—";
  if (typeof item.value === "boolean") return item.value ? "Yes" : "No";
  if (typeof item.value === "number") return Number.isInteger(item.value) ? String(item.value) : String(Number(item.value.toFixed(2)));
  return String(item.value);
}

function evidenceSourceName(raw: string | null): string {
  if (!raw) return "Source unverified";
  if (raw.startsWith("API_FOOTBALL")) return "API-Football";
  if (raw.startsWith("SOCCER_EDGE")) return "Soccer Edge model";
  return raw.replace(/_/g, " ").toLowerCase();
}

function EvidenceTechnicalDetails({ group }: { group: SportEvidenceGroup }) {
  return (
    <details className="evidence-technical">
      <summary><span>All {group.items.length} fields and sources</span><span aria-hidden="true">+</span></summary>
      <div className="evidence-technical-table">
        {group.items.map(item => (
          <div className="evidence-technical-row" key={item.key}>
            <div className="evidence-technical-value">
              <span>{item.label}</span><strong>{sportValue(item)}</strong>
            </div>
            <div className="evidence-technical-meta">
              {evidenceSourceName(item.source)}
              {item.sampleN !== null ? " · n=" + item.sampleN : ""}
              {item.capturedAt ? " · " + item.capturedAt.slice(0,16).replace("T"," ") + " UTC" : ""}
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}

function EvidenceForm({ form }: { form?: SportEvidenceItem }) {
  if (!form || typeof form.value !== "string") return <span className="evidence-form-none">Form not available</span>;
  const results = [...form.value.toUpperCase()].filter(letter => ["W","D","L"].includes(letter)).slice(-8);
  if (!results.length) return <span className="evidence-form-none">Form not available</span>;
  return <div className="evidence-form" aria-label={"Last eight form entries in provider order: " + results.join(" ")}>
    {results.map((result,i) => <span className={"evidence-form-chip " + result.toLowerCase()} key={i}>{result}</span>)}
  </div>;
}

function TeamEvidence({ group, match }: { group: SportEvidenceGroup; match: MatchCenterViewModel }) {
  const item = (side: "home" | "away", name: string) =>
    group.items.find(entry => entry.key === "team_performance." + side + "_" + name);
  const teamCard = (side: "home" | "away") => {
    const home = side === "home";
    const name = home ? match.home.name : match.away.name;
    const team = home ? match.home : match.away;
    const won = item(side, "wins_total");
    const drawn = item(side, "draws_total");
    const lost = item(side, "losses_total");
    const matches = won?.sampleN ?? drawn?.sampleN ?? lost?.sampleN ?? null;
    const w = typeof won?.value === "number" ? won.value : null;
    const d = typeof drawn?.value === "number" ? drawn.value : null;
    const l = typeof lost?.value === "number" ? lost.value : null;
    const total = w !== null && d !== null && l !== null ? w+d+l : 0;
    return <article className={"evidence-team-card " + side} key={side}>
      <div className="evidence-team-heading">
        {team.logoUrl ? <img src={team.logoUrl} alt="" /> : <span className="evidence-team-placeholder">{team.shortName}</span>}
        <div><small>{home ? "HOME TEAM" : "AWAY TEAM"}</small><h4>{name}</h4></div>
      </div>
      <div className="evidence-games"><strong>{matches ?? "—"}</strong><span>season matches</span></div>
      <div className="evidence-wdl">
        <div><strong>{sportValue(won)}</strong><small>WINS</small></div>
        <div><strong>{sportValue(drawn)}</strong><small>DRAWS</small></div>
        <div><strong>{sportValue(lost)}</strong><small>LOSSES</small></div>
      </div>
      {total > 0 && <div className="evidence-record-bar" aria-label={w + " wins, " + d + " draws, " + l + " losses"}>
        <span className="wins" style={{width:(100*(w??0)/total)+"%"}} />
        <span className="draws" style={{width:(100*(d??0)/total)+"%"}} />
        <span className="losses" style={{width:(100*(l??0)/total)+"%"}} />
      </div>}
      <div className="evidence-team-metrics">
        {[
          ["Goals for / game", "goals_for_avg"],
          ["Goals against / game", "goals_against_avg"],
          ["Clean sheets", "clean_sheets"],
          ["Failed to score", "failed_to_score"],
          [home ? "Home matches" : "Away matches", "played_split"],
        ].filter(([,metric])=>!!item(side,metric)).map(([label,metric])=>
          <div key={metric}><span>{label}</span><b>{sportValue(item(side,metric))}</b></div>
        )}
      </div>
      <div className="evidence-form-heading">FORM <small>Last 8 entries</small></div>
      <EvidenceForm form={item(side,"form")} />
    </article>;
  };
  const api = group.items.find(x=>x.source?.startsWith("API_FOOTBALL"));
  const model = group.items.some(x=>x.source?.startsWith("SOCCER_EDGE"));
  return <>
    <div className="evidence-team-comparison">{teamCard("home")}{teamCard("away")}</div>
    <div className="evidence-lineage">
      <span><i className="evidence-source-dot" /> {api ? "API-Football season statistics" : "Persisted team statistics"}</span>
      {api?.capturedAt && <span>Captured {api.capturedAt.slice(0,10)}</span>}
      {model && <span>Includes separate Soccer Edge model inputs</span>}
    </div>
    <EvidenceTechnicalDetails group={group} />
  </>;
}

function EvidenceGroupContent({ group, match }: { group: SportEvidenceGroup; match: MatchCenterViewModel }) {
  if (group.category === "TEAMS") return <TeamEvidence group={group} match={match} />;
  return <>
    <div className="evidence-smart-grid">
      {group.items.slice(0,6).map(item => (
        <div className="evidence-smart-metric" key={item.key}>
          <span>{item.label}</span>
          <strong>{sportValue(item).length > 32 ? sportValue(item).slice(-8) : sportValue(item)}</strong>
          <small>{evidenceSourceName(item.source)}{item.sampleN !== null ? " · n="+item.sampleN : ""}</small>
        </div>
      ))}
    </div>
    <EvidenceTechnicalDetails group={group}/>
  </>;
}

function EvidenceBoard({ match, category }: { match: MatchCenterViewModel; category?: string }) {
  const all = match.evidenceSections || [];
  const groups = category ? all.filter(group=>group.category===category) : all;
  if (!groups.length) return (
    <article className="evidence-board">
      <header className="evidence-board-heading"><div><span className="evidence-eyebrow">SPORT INTELLIGENCE</span><h3>{category || "Match evidence"}</h3></div><span className="evidence-count">NOT VERIFIED</span></header>
      <p className="evidence-absence">No persisted features for this section of this fixture. Missing data is not treated as zero.</p>
    </article>
  );
  const count = groups.reduce((sum,g)=>sum+g.items.length,0);
  return <section className="evidence-board">
    <header className="evidence-board-heading">
      <div><span className="evidence-eyebrow">SPORT INTELLIGENCE</span><h3>{category ? category[0]+category.slice(1).toLowerCase() : "Match evidence"}</h3><p>What the data actually supports</p></div>
      <span className="evidence-count">{count} fields saved</span>
    </header>
    {groups.map(group => <details className="evidence-group" key={group.category} open={category !== undefined || group.category === "TEAMS"}>
      <summary><span className="evidence-group-name">{group.category === "TEAMS" ? "Team comparison" : group.category[0]+group.category.slice(1).toLowerCase()}</span><small>{group.items.length} fields {group.dataTier ? "· "+group.dataTier.replace(/_/g," ").toLowerCase() : ""}</small></summary>
      <EvidenceGroupContent group={group} match={match}/>
    </details>)}
    <p className="evidence-footnote">Historical and persisted performance helps explain the sporting matchup; it does not verify lineups, injuries or an actionable betting price.</p>
  </section>;
}

function MarketBoard({ match }: { match: MatchCenterViewModel }) {
  const prices = match.marketRows || [];
  return <section className="evidence-board">
    <header><b>Market evidence</b><span>{match.finalResult ? "FINISHED / HISTORICAL" : match.edge ? "COMPARISON AVAILABLE" : "NOT ACTIONABLE / NOT VERIFIED"}</span></header>
    {match.finalResult ? <p>The event is final. Stored prices may be viewed for historical analysis only; no live market edge or actionable quote is inferred.</p> : match.edge ? <EdgePanel match={match}/> : <p>Market edge is not calculable without a fresh verified price, bookmaker/source, captured timestamp and a defensible fair-market probability. Missing prices are never displayed as zero.</p>}
    {prices.length ? <div className="evidence-items">{prices.map((row,i) =>
      <div className="evidence-item" key={row.name+"-"+row.selection+"-"+i}>
        <span>{row.family ? row.family+" · " : ""}{row.name}{row.line !== null ? " · "+row.line : ""}</span>
        <b>{row.selection}{row.price !== null ? " · "+row.price.toFixed(2) : " · Price NOT VERIFIED"}</b>
        <small>{row.bookmaker || row.source || "SOURCE NOT VERIFIED"}{row.capturedAt ? " · "+row.capturedAt : ""} · {row.fresh ? "FRESH FLAG" : "CURRENT PRICE NOT VERIFIED"}</small>
      </div>
    )}</div> : <p>No persisted market candidate rows are available for this fixture.</p>}
  </section>;
}

function ModelBoard({ match }: { match: MatchCenterViewModel }) {
  const provenance = match.modelProvenance;
  return <section className="evidence-board">
    <header><b>Model traceability</b><span>RAW SPORT BEFORE MARKET</span></header>
    <div className="evidence-items">
      {[
        ["Projection source", provenance?.source || "NOT VERIFIED"],
        ["Model version", provenance?.modelVersion || "NOT VERIFIED"],
        ["Captured at", provenance?.capturedAt || "NOT VERIFIED"],
        ["Goal-rate semantics", provenance?.goalRateSemantics || "NOT VERIFIED"],
        ["Data quality", match.dataQuality],
        ["Availability confidence", match.confidence === null ? "NOT VERIFIED" : String(match.confidence)+" / 100"],
      ].map(([label,value]) => <div className="evidence-item" key={label}><span>{label}</span><b>{value}</b></div>)}
    </div>
    <p>Model outputs are calculations based on persisted inputs; they are not direct measurements or guarantees. Missing model provenance stays NOT VERIFIED.</p>
  </section>;
}

function AuthModal({
  open,
  auth,
  onClose,
}: {
  open: boolean;
  auth: AuthState | null;
  onClose: () => void;
}) {
  const [email, setEmail] = useState(auth?.email || "");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const submitSignIn = async () => {
    try {
      setBusy(true);
      setStatus("Signing in…");
      await signIn(email.trim(), password);
      window.location.reload();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "SIGN_IN_FAILED");
      setBusy(false);
    }
  };

  const submitSignUp = async () => {
    try {
      setBusy(true);
      setStatus("Creating account…");
      const result = await signUp(email.trim(), password);
      setStatus(result.message);
      if (result.signedIn) window.location.reload();
      else setBusy(false);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "SIGN_UP_FAILED");
      setBusy(false);
    }
  };

  const submitSignOut = () => {
    signOut();
    window.location.reload();
  };

  return (
    <div className="auth-modal" role="dialog" aria-modal="true" aria-label="Soccer Edge account">
      <div className="auth-card">
        <div className="auth-head">
          <div><span>SOCCER EDGE</span><h2>{auth?.authenticated ? "Account" : "Sign in"}</h2></div>
          <button onClick={onClose} aria-label="Close">×</button>
        </div>

        {auth?.authenticated ? (
          <>
            <div className="account-summary">
              <span>PLAN</span><b>{auth.displayRole.toUpperCase()}</b>
              <small>{auth.email || "Authenticated account"}</small>
              <em>{auth.premiumUnlocked ? "Match Intelligence unlocked" : "Explorer access · premium evidence remains locked"}</em>
            </div>
            <div className="auth-actions">
              <button className="secondary" onClick={submitSignOut}>Sign out</button>
              <button className="primary" onClick={onClose}>Continue</button>
            </div>
          </>
        ) : (
          <>
            <p>Sign in with your Soccer Edge account to resolve your plan and unlock premium Match Intelligence when entitled.</p>
            <label>Email<input type="email" autoComplete="email" value={email} onChange={(e)=>setEmail(e.target.value)} placeholder="you@example.com" /></label>
            <label>Password<input type="password" autoComplete="current-password" value={password} onChange={(e)=>setPassword(e.target.value)} placeholder="••••••••" /></label>
            <div className="auth-actions">
              <button className="primary" disabled={busy || !email || !password} onClick={submitSignIn}>Sign in</button>
              <button className="secondary" disabled={busy || !email || !password} onClick={submitSignUp}>Create account</button>
            </div>
            <div className="auth-status">{status}</div>
          </>
        )}
      </div>
    </div>
  );
}

function coverageTone(item: SlateItem): string {
  if (item.sportEvidenceCount > 0 || item.analysisRows > 0) return "sport";
  if (item.marketEvidenceCount > 0) return "market";
  return "empty";
}

function SlatePage({
  mode,
  items,
}: {
  mode: "today" | "matches";
  items: SlateItem[];
}) {
  const sportCount = items.filter((x)=>x.sportEvidenceCount > 0 || x.analysisRows > 0).length;
  const marketOnly = items.filter((x)=>x.sportEvidenceCount === 0 && x.analysisRows === 0 && x.marketEvidenceCount > 0).length;
  const insufficient = items.length - sportCount - marketOnly;
  return (
    <section className="react-page">
      <div className="slate-page-head">
        <div>
          <span>SOCCER EDGE · LIVE SLATE</span>
          <h1>{mode === "today" ? "Today" : "Matches"}</h1>
          <p>{mode === "today" ? "Sport-first coverage before market evaluation." : "Every eligible fixture stays visible. Missing evidence stays NOT VERIFIED."}</p>
        </div>
        <b>{items.length} fixtures</b>
      </div>
      <div className="coverage-kpis">
        <div><span>SPORT DATA</span><b>{sportCount}</b></div>
        <div><span>MARKET ONLY</span><b>{marketOnly}</b></div>
        <div><span>INSUFFICIENT</span><b>{insufficient}</b></div>
      </div>
      <div className="slate-card">
        {items.length ? items.map((item)=>(
          <a className="react-slate-row" href={"/app-v3-react/match/"+item.fixtureId} key={item.fixtureId}>
            <div className="slate-time"><b>{item.kickoff}</b><span>{item.league}</span></div>
            <div className="slate-match">
              <div className="mini-team">
                <MiniTeamLogo url={item.homeLogoUrl} name={item.home} />
                <b>{item.home}</b>
              </div>
              <span className="mini-vs">vs</span>
              <div className="mini-team away">
                <MiniTeamLogo url={item.awayLogoUrl} name={item.away} />
                <b>{item.away}</b>
              </div>
            </div>
            <div className={"coverage-tag "+coverageTone(item)}>
              <b>{item.status}</b>
              <span>{item.sportEvidenceCount > 0 ? item.sportEvidenceCount+" sport evidence" : item.marketEvidenceCount > 0 ? item.marketEvidenceCount+" market snapshots" : "No deep analysis yet"}</span>
            </div>
          </a>
        )) : <div className="slate-empty">No eligible fixtures are available in the current persisted slate.</div>}
      </div>
    </section>
  );
}

function ProductNav({
  active,
  onNavigate,
  onOpenAuth,
}: {
  active: "today" | "matches" | "match";
  onNavigate: (view: "today" | "matches" | "match") => void;
  onOpenAuth: () => void;
}) {
  return (
    <>
      <nav className="mobile-product-nav">
        <button className={active==="today"?"active":""} onClick={()=>onNavigate("today")}><span>◉</span>Today</button>
        <button className={active==="matches"?"active":""} onClick={()=>onNavigate("matches")}><span>◎</span>Matches</button>
        <button className={active==="match"?"active":""} onClick={()=>onNavigate("match")}><span>▥</span>Match</button>
        <button onClick={onOpenAuth}><span>○</span>Account</button>
      </nav>
    </>
  );
}

function AppBody({
  match,
  auth,
  slate,
  view,
  onNavigate,
  onOpenAuth,
}: {
  match: MatchCenterViewModel;
  auth: AuthState | null;
  slate: SlateItem[];
  view: "today" | "matches" | "match";
  onNavigate: (view: "today" | "matches" | "match") => void;
  onOpenAuth: () => void;
}) {
  const modeLabel = match.sample ? "SAMPLE DESIGN MODE" : "LIVE CONTRACT";
  const classification = match.decision.classification || match.decision.displayBucket || "SPORT FIRST";
  const [activeTab,setActiveTab] = useState<MatchTab>("Overview");

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand"><span>SE</span><b>SOCCER<br/>EDGE</b></div>
        <nav>
          <button className={view==="today"?"active":""} onClick={()=>onNavigate("today")}>Today</button>
          <button className={view==="matches"?"active":""} onClick={()=>onNavigate("matches")}>Matches</button>
          <button className={view==="match"?"active":""} onClick={()=>onNavigate("match")}>Match Center</button>
          <span className="rail-label">NEXT</span>
          <button disabled>Edge Feed</button>
          <button disabled>Performance</button>
          <button disabled>My Edge</button>
          <button onClick={onOpenAuth}>Account</button>
        </nav>
      </aside>
      <main>
        <div className="topbar">
          <button className="back-btn" onClick={()=>onNavigate("matches")}>← {view==="match" ? "Back to matches" : "Soccer Edge"}</button>
          <div className="topbar-actions">
            <span className={match.sample ? "preview-mode" : "live"}>● {modeLabel}</span>
            <span className={"plan-chip " + (auth?.premiumUnlocked ? "pro" : "")}>{(auth?.displayRole || "Explorer").toUpperCase()}</span>
            <button className="account-btn" onClick={onOpenAuth}>{auth?.authenticated ? "Account" : "Sign in"}</button>
          </div>
        </div>

        {view === "today" ? <SlatePage mode="today" items={slate} /> :
         view === "matches" ? <SlatePage mode="matches" items={slate} /> :
         <>
          <Hero match={match} />
          <nav className="tabs" aria-label="Match evidence sections">{MATCH_TABS.map(x=><button type="button" className={activeTab===x?"active":""} aria-current={activeTab===x?"page":undefined} onClick={()=>setActiveTab(x)} key={x}>{x}</button>)}</nav>
          <section className="content">
            <div className="title-row"><div><h1>Match Center</h1><p>What do we know, what is missing, and what deserves attention?</p></div><span>{match.league} · {match.sample ? "sample snapshot" : "persisted snapshot"}</span></div>
            <div className="status-strip">
              <div><span>DATA QUALITY</span><b>{match.dataQuality}</b></div>
              <div><span>CONFIDENCE</span><b>{match.confidence === null ? "N/V" : match.confidence}</b></div>
              <div><span>XI</span><b>{match.lineup}</b></div>
              <div><span>MARKET</span><b>{match.market}</b></div>
            </div>
            {activeTab === "Overview" && <>
              <div className="deck top"><ProbabilityPanel match={match}/><XgPanel match={match}/>{match.edge && <EdgePanel match={match}/>}</div>
              {!match.edge && <div className="market-no-edge"><b>Market Edge · NOT VERIFIED</b><span>No fresh verified quote. Sporting projections remain visible without suggesting a wager.</span></div>}
              <div className="deck middle"><SportProfile match={match}/><MatrixPanel match={match}/><GoalsPanel match={match}/></div>
              <EvidenceBoard match={match}/>
            </>}
            {activeTab === "Goals" && <><div className="deck middle"><XgPanel match={match}/><MatrixPanel match={match}/><GoalsPanel match={match}/></div><EvidenceBoard match={match} category="GOALS"/></>}
            {activeTab === "Corners" && <EvidenceBoard match={match} category="CORNERS"/>}
            {activeTab === "Cards" && <EvidenceBoard match={match} category="CARDS"/>}
            {activeTab === "Players" && <EvidenceBoard match={match} category="PLAYERS"/>}
            {activeTab === "Market" && <MarketBoard match={match}/>}
            {activeTab === "Model" && <><ModelBoard match={match}/><EvidenceBoard match={match} category="CONTEXT"/><EvidenceBoard match={match} category="AVAILABILITY"/></>}
            <section className="primary-read">
              <span className="se">SE</span>
              <div><b>Primary read <em>{classification}</em></b><p>{match.decision.reason || "Sporting projection is built first. Market value is assessed only after the football case is established."}</p></div>
              <strong>{match.edge ? (match.edge.gap >= 0 ? "+" : "") + match.edge.gap.toFixed(1) + " pp" : "—"}</strong>
            </section>
            <footer>{match.disclosure}</footer>
          </section>
         </>
        }
      </main>
      <ProductNav active={view} onNavigate={onNavigate} onOpenAuth={onOpenAuth} />
    </div>
  );
}

export default function App() {
  const [match, setMatch] = useState<MatchCenterViewModel | null>(null);
  const [slate, setSlate] = useState<SlateItem[]>([]);
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [view, setView] = useState<"today" | "matches" | "match">(
    window.location.pathname.includes("/match/") || new URLSearchParams(window.location.search).has("fixture_id") ? "match" : "today"
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([loadMatchCenter(), loadSlate(), loadAuthState()])
      .then(([matchData, slateData, authData]) => {
        if (!active) return;
        setMatch(matchData);
        setSlate(slateData);
        setAuth(authData);
      })
      .catch((err: unknown) => {
        if (active) setError(err instanceof Error ? err.message : "MATCH_CENTER_UNAVAILABLE");
      });
    return () => { active = false; };
  }, []);

  const navigate = (next: "today" | "matches" | "match") => {
    setView(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (error) {
    return <div className="load-screen"><b>Soccer Edge unavailable</b><span>{error}</span><a href="/app-v3-react?sample=1">Open frozen sample design</a></div>;
  }
  if (!match) {
    return <div className="load-screen"><b>Loading verified Soccer Edge…</b><span>Sport first · market second</span></div>;
  }
  return (
    <>
      <AppBody match={match} auth={auth} slate={slate} view={view} onNavigate={navigate} onOpenAuth={()=>setAuthOpen(true)} />
      <AuthModal open={authOpen} auth={auth} onClose={()=>setAuthOpen(false)} />
    </>
  );
}
