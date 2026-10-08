import { sampleMatch as match } from "./sample";
import type { ScoreMatrix, TeamScoringProfile } from "./model";

function TeamBadge({ short, tone }: { short: string; tone: "home" | "away" }) {
  return <div className={"team-badge " + tone}>{short}</div>;
}

function Hero() {
  return (
    <section className="hero">
      <div className="hero-topline">
        <span>{match.league} · Today</span>
        <span>{match.kickoff}</span>
      </div>
      <div className="hero-main">
        <div className="team">
          <TeamBadge short={match.home.shortName} tone="home" />
          <b>{match.home.name}</b><span>HOME</span>
        </div>
        <div className="vs">VS</div>
        <div className="team">
          <TeamBadge short={match.away.shortName} tone="away" />
          <b>{match.away.name}</b><span>AWAY</span>
        </div>
        <aside className="quality">
          <div><span>Data Quality</span><b>A</b></div>
          <div><span>Lineup</span><b>Confirmed</b></div>
          <div><span>Market</span><b>Fresh</b></div>
        </aside>
      </div>
    </section>
  );
}

function ProbabilityPanel() {
  const p = match.probability;
  return (
    <article className="panel probability">
      <header><h3>Match Result Probability</h3><span>MODEL</span></header>
      <div className="prob-strip">
        <div className="prob-cell home"><span>HOME</span><b>{p.home}%</b></div>
        <div className="prob-cell"><span>DRAW</span><b>{p.draw}%</b></div>
        <div className="prob-cell"><span>AWAY</span><b>{p.away}%</b></div>
      </div>
      <div className="panel-foot"><i /> HOME LEADS MODEL <strong>{p.home}% · +46.8 pp vs draw</strong></div>
    </article>
  );
}

function XgPanel() {
  const total = (match.xg.home + match.xg.away).toFixed(2);
  const delta = (match.xg.home - match.xg.away).toFixed(2);
  return (
    <article className="panel xg-panel">
      <header><h3>Expected Goals (λ)</h3><span>SPORT</span></header>
      <div className="xg-values">
        <div><span>{match.home.name}</span><b>{match.xg.home.toFixed(2)}</b></div>
        <em>—</em>
        <div><span>{match.away.name}</span><b>{match.xg.away.toFixed(2)}</b></div>
      </div>
      <div className="xg-meta"><div><span>TOTAL XG</span><b>{total}</b></div><div><span>HOME DELTA</span><b>+{delta}</b></div></div>
      <svg viewBox="0 0 180 32" className="trend" aria-label="Illustrative attacking trend">
        <polyline points="4,25 31,21 58,22 85,15 112,17 140,10 176,7" />
      </svg>
    </article>
  );
}

function EdgePanel() {
  const e = match.edge;
  return (
    <article className="panel edge-panel">
      <header><h3>Edge Gap</h3><span>EXAMPLE MARKET</span></header>
      <div className="edge-grid">
        <div className="edge-visual">
          <div className="edge-pill"><b>+{e.gap} pp</b><span>EDGE GAP</span></div>
          <div className="edge-labels"><span>MARKET {e.market}</span><span>MODEL {e.model}</span></div>
          <div className="edge-scale">
            <span className="dot market" />
            <span className="edge-line" />
            <span className="dot model" />
          </div>
          <div className="ticks"><span>50%</span><span>60%</span><span>70%</span><span>80%</span></div>
        </div>
        <aside className="edge-stats">
          <div><span>FAIR PRICE</span><b>{e.fairPrice}</b><small>Model · Market {e.marketPrice}</small></div>
          <div><span>CONFIDENCE</span><b>{match.confidence}</b><small>High · signal confidence</small></div>
        </aside>
      </div>
    </article>
  );
}

function SportProfile() {
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
            className={"heat-cell level-" + Math.min(5, Math.max(1, Math.ceil(v / 2))) + (matrix.hot[0] === r && matrix.hot[1] === c ? " hot" : "")}
          >{v}</div>
        ))
      ])}
    </div>
  );
}

function TeamHeatmap({ profile }: { profile: TeamScoringProfile }) {
  const matrix: ScoreMatrix = { labels: profile.labels, values: profile.values, hot: [-1, -1] };
  return (
    <div className={"team-heat " + profile.tone}>
      <div className="team-heat-head"><b>{profile.team}</b><span>GF × GA</span></div>
      <Heatmap matrix={matrix} compact />
      <small>X = GF · Y = GA</small>
    </div>
  );
}

function MatrixPanel() {
  return (
    <article className="panel matrix-panel">
      <header><h3>Score Matrix (FT)</h3><span>FT + SCORING PROFILE</span></header>
      <div className="matrix-layout">
        <div className="matrix-main">
          <Heatmap matrix={match.scoreMatrix} />
          <p>Most likely: <b>2–1 · 11.4%</b></p>
        </div>
        <div className="team-heats">
          {match.scoringProfiles.map((p) => <TeamHeatmap key={p.team} profile={p} />)}
        </div>
      </div>
    </article>
  );
}

function GoalsPanel() {
  return (
    <article className="panel goals-panel">
      <header><h3>Over / Under 2.5 Goals</h3><span>MODEL VS MARKET</span></header>
      <div className="ou-top">
        <div className="ou-copy"><span>MODEL OVER 2.5</span><b>{match.over25.model}%</b><strong>+{match.over25.edge} pp edge</strong><small>Market fair {match.over25.market}%</small></div>
        <div className="ou-gauge">
          <span className="gauge-label market">MARKET {match.over25.market}</span>
          <span className="gauge-label model">MODEL {match.over25.model}</span>
          <div className="gauge-base"><i className="gauge-gap" /><i className="gauge-market" /><i className="gauge-model" /></div>
          <div className="gauge-ticks"><span>40%</span><span>50%</span><span>60%</span><span>70%</span><span>80%</span></div>
        </div>
      </div>
      <div className="dist">
        {[18,38,60,45,28,15].map((h,i) => <div key={i}><i className="home" style={{height:h+"px"}}/><i className="away" style={{height:[28,50,34,18,7,2][i]+"px"}}/><span>{i===5?"5+":i}</span></div>)}
      </div>
      <div className="legend"><span><i className="home"/>Arsenal</span><span><i className="away"/>Brighton</span></div>
    </article>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <aside className="rail"><div className="brand"><span>SE</span><b>SOCCER<br/>EDGE</b></div><nav><a>Today</a><a>Edge Feed</a><a className="active">Matches</a><a>Markets</a><a>Performance</a><a>My Edge</a></nav></aside>
      <main>
        <div className="topbar"><span>← Back to matches</span><span className="live">● LIVE · React migration preview</span></div>
        <Hero />
        <nav className="tabs">{["Overview","Goals","Corners","Cards","Players","Market","Model"].map((x,i)=><button className={i===0?"active":""} key={x}>{x}</button>)}</nav>
        <section className="content">
          <div className="title-row"><div><h1>Match Center</h1><p>Why does this matchup matter?</p></div><span>{match.league} · model snapshot</span></div>
          <div className="status-strip"><div><span>DATA QUALITY</span><b>A</b></div><div><span>CONFIDENCE</span><b>{match.confidence}</b></div><div><span>XI</span><b>CONFIRMED</b></div><div><span>MARKET</span><b>FRESH</b></div></div>
          <div className="deck top"><ProbabilityPanel/><XgPanel/><EdgePanel/></div>
          <div className="deck middle"><SportProfile/><MatrixPanel/><GoalsPanel/></div>
          <section className="primary-read"><span className="se">SE</span><div><b>Primary read <em>SPORT FIRST</em></b><p>Sporting projection is built first. Market value is assessed only after the football case is established.</p></div><strong>+11.7 pp</strong></section>
          <footer>DESIGN MIGRATION PREVIEW · SAMPLE DATA · NOT A LIVE PREDICTION</footer>
        </section>
      </main>
    </div>
  );
}
