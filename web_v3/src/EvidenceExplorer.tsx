import type { EvidenceItem, EvidenceSection } from "./model";

type InputProps = { group: EvidenceSection };
type Row = EvidenceItem;
const METRIC_NAMES: Record<string, string> = {
  wins_total: "Wins", draws_total: "Draws", losses_total: "Losses",
  goals_for_avg: "Goals scored / match", goals_against_avg: "Goals conceded / match",
  clean_sheets: "Clean sheets", failed_to_score: "Failed to score",
  played_split: "Matches at venue", recent_matches: "Recent matches",
  wins_split: "Wins at venue", draws_split: "Draws at venue", losses_split: "Losses at venue",
  goals_for_split: "Goals scored at venue", goals_against_split: "Goals conceded at venue",
};
const METRIC_ORDER = Object.keys(METRIC_NAMES);

function prettyText(str: string) {
  return str.toLowerCase().replace(/_/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());
}
function sourceLabel(key: string | null) {
  if (!key) return "Unverified source";
  if (key.startsWith("API_FOOTBALL")) return "API-Football";
  if (key.startsWith("SOCCER_EDGE")) return "Soccer Edge model";
  return prettyText(key);
}
function fmt(value: Row["value"]) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
  if (value === "RESEARCH_ONLY") return "Research only";
  if (value.length > 32) return value.slice(0,26) + "…";
  return value;
}
function meta(item: Row) {
  const sample = item.sampleN !== null ? " · sample " + item.sampleN : "";
  const date = item.capturedAt ? " · " + item.capturedAt.slice(0,10) : "";
  return sourceLabel(item.source) + sample + date;
}
function MatchForm({ value, long = false }: { value: string; long?: boolean }) {
  const codes = [...value.toUpperCase()].filter(c => c === "W" || c === "D" || c === "L");
  const entries = long ? codes : codes.slice(-8);
  return <div className="eex-form" aria-label={"Provider-order form sequence: " + entries.join(" ")}>
    {entries.map((v,i)=><span key={i} className={"eex-form-chip " + v.toLowerCase()}>{v}</span>)}
  </div>;
}
function SourceCard({ source, items }: { source: string | null; items: Row[] }) {
  const counts = Array.from(new Set(items.map(x=>x.sampleN).filter((x): x is number => x !== null)));
  const dates = Array.from(new Set(items.map(x=>x.capturedAt?.slice(0,10)).filter((x): x is string => !!x)));
  const versions = Array.from(new Set(items.map(x=>x.modelVersion).filter((x): x is string => !!x)));
  return <article className="eex-source">
    <div className="eex-source-head"><span className="eex-dot"/><strong>{sourceLabel(source)}</strong><span>{items.length} fields</span></div>
    <div className="eex-source-meta">
      {counts.length>0 && <span>Samples: {counts.join(", ")}</span>}
      {dates.length>0 && <span>Captured: {dates.join(", ")}</span>}
      {versions.length>0 && <span>{source?.startsWith("SOCCER_EDGE") ? "Model: " : "Processed by: "}{versions.join(", ")}</span>}
    </div>
  </article>;
}

function EvidenceBody({group}: InputProps) {
  const model = group.items.filter(item=>item.source?.startsWith("SOCCER_EDGE") || /goal_rate_blend/.test(item.key));
  const home = group.items.filter(item=>item.key.startsWith("team_performance.home_"));
  const away = group.items.filter(item=>item.key.startsWith("team_performance.away_"));
  const form = group.items.filter(item => /(?:^|_)form$/.test(item.key) && typeof item.value === "string");
  const season = [...home,...away].filter(item => !form.includes(item));
  const used = new Set([...model,...season,...form].map(item=>item.key));
  const other = group.items.filter(item=>!used.has(item.key));

  const homeFields = new Map(home.filter(item=>!form.includes(item) && !model.includes(item)).map(item=>[item.key.replace(/^team_performance\.home_/,""),item]));
  const awayFields = new Map(away.filter(item=>!form.includes(item) && !model.includes(item)).map(item=>[item.key.replace(/^team_performance\.away_/,""),item]));
  const venueMetricOrder = ["played_split","wins_split","draws_split","losses_split","goals_for_split","goals_against_split"];
  const venueKeys = venueMetricOrder.filter(key => homeFields.has(key) || awayFields.has(key));
  const allKeys = Array.from(new Set([...homeFields.keys(),...awayFields.keys()]))
    .filter(key => !venueMetricOrder.includes(key));
  allKeys.sort((a,b)=>{
    const i=METRIC_ORDER.indexOf(a),j=METRIC_ORDER.indexOf(b);
    return (i<0?100:i)-(j<0?100:j) || a.localeCompare(b);
  });
  const postKickoffRows = group.items.filter(item=>item.observationScope === "POSTGAME_OBSERVATION" || item.observationScope === "POST_KICKOFF_OBSERVATION");
  const sources = new Map<string, Row[]>();
  for (const item of group.items) {
    const key=item.source || "";
    sources.set(key,[...(sources.get(key) || []),item]);
  }

  return <div className="eex-content">
    {postKickoffRows.length>0 && <p className="eex-postgame-note">Some statistics were captured after kickoff. They describe the season at collection time, not the inputs available to the original pre-match prediction.</p>}
    {model.length>0 && <section className="eex-section">
      <div className="eex-section-title"><span className="eex-section-index">01</span><div><h4>Model inputs</h4><p>Projected scoring rates, not measured xG</p></div></div>
      <div className="eex-model-grid">
        {model.map(item=>{
          const k=item.key.toLowerCase();
          const label = k.includes("total_goal_rate") ? "Total projected (λ)" :
            k.includes("home_goal_rate") ? "Home (λ)" :
            k.includes("away_goal_rate") ? "Away (λ)" : item.label;
          return <div className="eex-model-tile" key={item.key} title={meta(item)}>
            <span>{label}</span><strong>{fmt(item.value)}</strong>
            <small>{sourceLabel(item.source)}{item.sampleN!==null?" · n="+item.sampleN:""}</small>
          </div>;
        })}
      </div>
    </section>}
    {venueKeys.length>0 && <section className="eex-section">
      <div className="eex-section-title"><span className="eex-section-index">02</span><div><h4>Home vs away performance</h4><p>Venue-specific records from stored statistics</p></div></div>
      <div className="eex-stats">
        <div className="eex-stats-head"><span>HOME</span><span>VENUE METRIC</span><span>AWAY</span></div>
        {venueKeys.map(k=><div className="eex-stat-line" key={k}>
          <strong className={homeFields.has(k)?"":"eex-null"} title={homeFields.has(k)?meta(homeFields.get(k)!):"Not verified"}>{homeFields.has(k)?fmt(homeFields.get(k)!.value):"—"}</strong>
          <span>{METRIC_NAMES[k] || prettyText(k)}</span>
          <strong className={awayFields.has(k)?"":"eex-null"} title={awayFields.has(k)?meta(awayFields.get(k)!):"Not verified"}>{awayFields.has(k)?fmt(awayFields.get(k)!.value):"—"}</strong>
        </div>)}
      </div>
      {(!homeFields.has("wins_split") || !awayFields.has("wins_split")) && <p className="eex-venue-caution">Some venue records are not persisted. Missing values are unverified; season totals are shown separately.</p>}
    </section>}
    {allKeys.length>0 && <section className="eex-section">
      <div className="eex-section-title"><span className="eex-section-index">03</span><div><h4>Full-season performance</h4><p>Totals and averages across all venues</p></div></div>
      <div className="eex-stats">
        <div className="eex-stats-head"><span>HOME</span><span>METRIC</span><span>AWAY</span></div>
        {allKeys.map(k=><div className="eex-stat-line" key={k}>
          <strong className={homeFields.has(k)?"":"eex-null"} title={homeFields.has(k)?meta(homeFields.get(k)!):"No disponible"}>{homeFields.has(k)?fmt(homeFields.get(k)!.value):"—"}</strong>
          <span>{METRIC_NAMES[k] || prettyText(k)}</span>
          <strong className={awayFields.has(k)?"":"eex-null"} title={awayFields.has(k)?meta(awayFields.get(k)!):"No disponible"}>{awayFields.has(k)?fmt(awayFields.get(k)!.value):"—"}</strong>
        </div>)}
      </div>
    </section>}
    {form.length>0 && <section className="eex-section">
      <div className="eex-section-title"><span className="eex-section-index">04</span><div><h4>Form history</h4><p>W win · D draw · L loss · sequence order supplied by provider</p></div></div>
      <div className="eex-form-grid">
        {form.map(item=>{
          const formValue=typeof item.value === "string" ? item.value : "";
          const total=[...formValue.toUpperCase()].filter(x=>["W","D","L"].includes(x)).length;
          const side=item.key.includes(".home_")?"HOME":item.key.includes(".away_")?"AWAY":"FORMA";
          return <div className="eex-form-card" key={item.key}>
            <div className="eex-form-title"><b>{side}</b><span>{Math.min(total,8)} displayed · {total} saved</span></div>
            <MatchForm value={formValue}/>
            {total>8 && <details className="eex-full-form"><summary>Show full history ({total} results)</summary><MatchForm long value={formValue}/></details>}
          </div>;
        })}
      </div>
    </section>}
    {other.length>0 && <section className="eex-section">
      <div className="eex-section-title"><span className="eex-section-index">05</span><div><h4>Other available data</h4><p>Persisted fields only</p></div></div>
      <div className="eex-extra-grid">{other.map(item=><div className="eex-extra" key={item.key} title={meta(item)}>
        <span>{item.label}</span>
        {typeof item.value === "string" && /[WDL]{12,}/.test(item.value)
          ? <MatchForm value={item.value}/>
          : <b>{fmt(item.value)}</b>}
      </div>)}</div>
    </section>}
    <section className="eex-section eex-provenance">
      <div className="eex-section-title"><span className="eex-section-index">i</span><div><h4>Data sources</h4><p>Source, sample and capture date for this section</p></div></div>
      <div className="eex-source-grid">{Array.from(sources.entries()).map(([source,items])=><SourceCard source={source||null} items={items} key={source}/>)}</div>
      <p className="eex-disclaimer">{group.items.length} original fields preserved. Historical data does not verify current lineups or prices.</p>
    </section>
  </div>;
}

export function EvidenceExplorer({group}: InputProps) {
  return <details className="eex-wrap">
    <summary><span className="eex-summary-icon" aria-hidden="true">▤</span><span className="eex-summary-copy"><strong>Explore evidence & sources</strong><small>Model inputs, performance and sources · {group.items.length} fields</small></span><span className="eex-chevron" aria-hidden="true">⌄</span></summary>
    <EvidenceBody group={group} />
  </details>;
}
