import type { EvidenceItem, MatchCenterViewModel } from "./model";

type MatchProps = { match: MatchCenterViewModel; category: "CORNERS" | "CARDS" | "STATS" };
const names: Record<string,string> = {
  corners:"Corner kicks", yellow_cards:"Yellow cards",red_cards:"Red cards",
  offsides:"Offsides",fouls:"Fouls committed",total_shots:"Total shots",
  shots_on_goal:"Shots on target",shots_off_goal:"Shots off target",
  blocked_shots:"Blocked shots",shots_inside_box:"Shots inside box",
  shots_outside_box:"Shots outside box",possession_pct:"Possession (%)",
  keeper_saves:"Goalkeeper saves",passes:"Total passes",
  passes_accurate:"Accurate passes",pass_accuracy_pct:"Pass accuracy (%)",
  formation:"Formation",
};
const playerLabels: Record<string,string> = {
  minutes:"Minutes",position:"Position",rating:"Rating",
  shots:"Shots",shots_on:"On target",goals:"Goals",assists:"Assists",
  yellow_cards:"Yellow",red_cards:"Red",key_passes:"Key passes",
  tackles:"Tackles",fouls:"Fouls",
};
function sourceContext(items: EvidenceItem[]) {
  const dates = [...new Set(items.map(x=>x.capturedAt).filter((x):x is string=>!!x))].sort();
  const after = items.some(x=>["POSTGAME_OBSERVATION","LIVE_OBSERVATION","POST_KICKOFF_OBSERVATION"].includes(x.observationScope || ""));
  return <div className="fixture-data-context">
    <span>API-Football · observed match data{after ? " · not pre-match evidence" : ""}</span>
    {dates.length > 0 && <span>Captured: {dates[dates.length-1].replace("T"," ").slice(0,16)} UTC</span>}
  </div>;
}
function unavailable(match: MatchCenterViewModel, title: string) {
  const pre = !match.finalResult && !["1H","HT","2H","ET","P","FT","AET","PEN"].includes(match.fixtureStatus || "");
  return <article className="fixture-stats-card">
    <header><h3>{title}</h3><span>NOT VERIFIED</span></header>
    <div className="fixture-coverage-message">
      <b>No provider observations saved</b>
      <p>{pre
        ? "Match events and player box scores cannot be known before kickoff. The section will populate when API-Football supplies data."
        : "API-Football has not supplied verified values for this fixture. Some competitions do not cover team statistics, player statistics or lineups."}
      </p>
      <small>Missing data ≠ zero. No fabricated estimates, market lines or player availability.</small>
    </div>
  </article>;
}
function renderNumber(value: EvidenceItem["value"]) {
  return typeof value === "boolean" ? (value ? "Yes":"No") : String(value);
}
export function FixtureMetricsPanel({match,category}: MatchProps) {
  const items = (match.evidenceSections||[]).filter(g=>g.category===category)
    .flatMap(g=>g.items).filter(row=>row.source?.startsWith("API_FOOTBALL_FIXTURE_"));
  const rows = new Map<string,{home?:EvidenceItem;away?:EvidenceItem}>();
  for (const item of items) {
    const m = item.key.match(/^[^.]+\.(home|away)_(.+)$/);
    if (!m) continue;
    const [,side,metric] = m;
    const line=rows.get(metric)||{};
    line[side as "home"|"away"]=item;
    rows.set(metric,line);
  }
  const title=category==="STATS"?"Match statistics":category==="CORNERS"?"Corners":"Cards";
  if (!rows.size) return unavailable(match,title);
  return <section className="fixture-stats-card">
    <header><h3>{title}</h3><span>PROVIDER OBSERVED · {rows.size} METRICS</span></header>
    <div className="fixture-stats-table">
      <div className="fixture-stats-table-head"><span>{match.home.name}</span><span>METRIC</span><span>{match.away.name}</span></div>
      {[...rows.entries()].map(([metric,line])=><div className="fixture-stats-line" key={metric}>
        <strong className={line.home?"":"not-reported"}>{line.home?renderNumber(line.home.value):"—"}</strong>
        <span>{names[metric] || metric.replace(/_/g," ")}</span>
        <strong className={line.away?"":"not-reported"}>{line.away?renderNumber(line.away.value):"—"}</strong>
      </div>)}
    </div>
    {sourceContext(items)}
    <p className="fixture-footnote">Only available provider observations are shown. Unreported values remain unknown, not zero. These figures are not pre-match projections.</p>
  </section>;
}

type PlayerRecord = {id:string;side:"home"|"away";fields:Record<string,EvidenceItem>};
export function FixturePlayersPanel({match}:{match:MatchCenterViewModel}) {
  const items = (match.evidenceSections||[]).filter(g=>g.category==="PLAYERS").flatMap(g=>g.items)
    .filter(x=>x.source?.startsWith("API_FOOTBALL_FIXTURE_"));
  const records=new Map<string,PlayerRecord>();
  for(const item of items){
    const m=item.key.match(/^players\.(home|away)_(\d+)_(.+)$/);
    if(!m)continue;
    const [,side,id,field]=m;const key=side+"_"+id;
    const rec=records.get(key)||{id,side:side as "home"|"away",fields:{}};
    rec.fields[field]=item;records.set(key,rec);
  }
  if(!records.size)return unavailable(match,"Players & lineups");
  const grouped={home:[] as PlayerRecord[],away:[] as PlayerRecord[]};
  for(const rec of records.values())grouped[rec.side].push(rec);
  for(const side of ["home","away"] as const)grouped[side].sort((a,b)=>{
    const starter=(v:PlayerRecord)=>(v.fields.starter?.value===true?0:1);
    return starter(a)-starter(b)||String(a.fields.name?.value||a.id).localeCompare(String(b.fields.name?.value||b.id));
  });
  return <section className="fixture-stats-card">
    <header><h3>Players & lineups</h3><span>PROVIDER OBSERVED · {records.size} PLAYERS</span></header>
    <div className="fixture-player-grid">
      {(["home","away"] as const).map(side=><div className="fixture-player-team" key={side}>
        <h4>{side==="home"?match.home.name:match.away.name}</h4>
        {grouped[side].length?grouped[side].map(player=><article className="fixture-player-row" key={player.id}>
          <div className="fixture-player-name">
            <b>{String(player.fields.name?.value||"Player #"+player.id)}</b>
            {player.fields.starter && <small>{player.fields.starter.value===true?"Starting XI":"Substitute"}</small>}
          </div>
          <div className="fixture-player-metrics">
            {Object.entries(playerLabels).filter(([key])=>player.fields[key]).map(([key,label])=>
              <span key={key}><small>{label}</small><b>{renderNumber(player.fields[key].value)}</b></span>)}
          </div>
        </article>):<p className="fixture-footnote">No player records supplied for this team</p>}
      </div>)}
    </div>
    {sourceContext(items)}
    <p className="fixture-footnote">Player box scores and reported lineups are historical/live observations only. No player status is inferred from absent provider rows.</p>
  </section>;
}
