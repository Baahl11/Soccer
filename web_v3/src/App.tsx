import { useEffect, useState } from "react";
import { loadMatchCenter } from "./live";
import type {
  MatchCenterViewModel,
  ScoreMatrix,
  SectionState,
  TeamScoringProfile,
} from "./model";

function TeamBadge({ team }: { team: MatchCenterViewModel["home"] }) {
  return (
    <div className={"team-badge " + team.side}>
      {team.logoUrl ? <img src={team.logoUrl} alt="" /> : <span>{team.shortName}</span>}
    </div>
  );
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
  if (!xg) return <MissingPanel title="Expected Goals (λ)" section={match.sections.xg} />;
  const total = xg.home !== null && xg.away !== null ? xg.home + xg.away : null;
  const delta = xg.home !== null && xg.away !== null ? xg.home - xg.away : null;
  return (
    <article className="panel xg-panel">
      <header><h3>Expected Goals (λ)</h3><span>SPORT</span></header>
      <div className="xg-values">
        <div><span>{match.home.name}</span><b>{xg.home === null ? "—" : xg.home.toFixed(2)}</b></div>
        <em>—</em>
        <div><span>{match.away.name}</span><b>{xg.away === null ? "—" : xg.away.toFixed(2)}</b></div>
      </div>
      <div className="xg-meta">
        <div><span>TOTAL XG</span><b>{total === null ? "—" : total.toFixed(2)}</b></div>
        <div><span>HOME DELTA</span><b>{delta === null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(2)}</b></div>
      </div>
      <div className="verified-copy">Persisted expected-goals inputs only</div>
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
      <header><h3>Score Matrix (FT)</h3><span>VERIFIED CELLS ONLY</span></header>
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
  if (!over) return <MissingPanel title="Over / Under 2.5 Goals" section={match.sections.over25} />;
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

function AppBody({ match }: { match: MatchCenterViewModel }) {
  const modeLabel = match.sample ? "SAMPLE DESIGN MODE" : "LIVE CONTRACT";
  const classification = match.decision.classification || match.decision.displayBucket || "SPORT FIRST";
  return (
    <div className="app-shell">
      <aside className="rail"><div className="brand"><span>SE</span><b>SOCCER<br/>EDGE</b></div><nav><a>Today</a><a>Edge Feed</a><a className="active">Matches</a><a>Markets</a><a>Performance</a><a>My Edge</a></nav></aside>
      <main>
        <div className="topbar"><span>← Back to matches</span><span className={match.sample ? "preview-mode" : "live"}>● {modeLabel}</span></div>
        <Hero match={match} />
        <nav className="tabs">{["Overview","Goals","Corners","Cards","Players","Market","Model"].map((x,i)=><button className={i===0?"active":""} key={x}>{x}</button>)}</nav>
        <section className="content">
          <div className="title-row"><div><h1>Match Center</h1><p>What do we know, what is missing, and what deserves attention?</p></div><span>{match.league} · {match.sample ? "sample snapshot" : "persisted snapshot"}</span></div>
          <div className="status-strip">
            <div><span>DATA QUALITY</span><b>{match.dataQuality}</b></div>
            <div><span>CONFIDENCE</span><b>{match.confidence === null ? "N/V" : match.confidence}</b></div>
            <div><span>XI</span><b>{match.lineup}</b></div>
            <div><span>MARKET</span><b>{match.market}</b></div>
          </div>
          <div className="deck top"><ProbabilityPanel match={match}/><XgPanel match={match}/><EdgePanel match={match}/></div>
          <div className="deck middle"><SportProfile match={match}/><MatrixPanel match={match}/><GoalsPanel match={match}/></div>
          <section className="primary-read">
            <span className="se">SE</span>
            <div><b>Primary read <em>{classification}</em></b><p>{match.decision.reason || "Sporting projection is built first. Market value is assessed only after the football case is established."}</p></div>
            <strong>{match.edge ? (match.edge.gap >= 0 ? "+" : "") + match.edge.gap.toFixed(1) + " pp" : "—"}</strong>
          </section>
          <footer>{match.disclosure}</footer>
        </section>
      </main>
    </div>
  );
}

export default function App() {
  const [match, setMatch] = useState<MatchCenterViewModel | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    loadMatchCenter()
      .then((data) => { if (active) setMatch(data); })
      .catch((err: unknown) => { if (active) setError(err instanceof Error ? err.message : "MATCH_CENTER_UNAVAILABLE"); });
    return () => { active = false; };
  }, []);

  if (error) {
    return <div className="load-screen"><b>Match Center unavailable</b><span>{error}</span><a href="/app-v3-react?sample=1">Open frozen sample design</a></div>;
  }
  if (!match) {
    return <div className="load-screen"><b>Loading verified Match Center…</b><span>Sport first · market second</span></div>;
  }
  return <AppBody match={match} />;
}
