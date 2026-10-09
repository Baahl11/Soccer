import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(resolve(here, "../src/teamGoalMatrices.ts"), "utf8");
const transpiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
}).outputText;
const exports = {};
runInNewContext(transpiled, { exports });
const { makeTeamGoalMatrices } = exports;

const snapshot = "2026-10-08T21:49:17Z";
const evidence = [
  {category:"TEAMS",items: [
    ...[
      ["home_goals_for_avg", 1.6, 31],
      ["home_goals_against_avg", 0.6, 31],
      ["away_goals_for_avg", 0.8, 29],
      ["away_goals_against_avg", 1.3, 29],
    ].map(([key,value,n])=>({
      key:"team_performance."+key, value,
      status:"PERSISTED",source:"API_FOOTBALL_TEAM_STATS",
      sampleN:n,capturedAt:snapshot,observationScope:"PREMATCH_OBSERVATION"
    }))
  ]}
];
// API-Football serializes goals.for.average.total as strings, not numbers.
evidence[0].items.forEach(row=>{
  if (row.key.endsWith("_goals_for_avg") || row.key.endsWith("_goals_against_avg"))
    row.value=String(row.value);
});
const profiles = makeTeamGoalMatrices("Boca Juniors Res.","Colón Res.",evidence);
assert.equal(profiles.length,2);
assert.equal(profiles[0].gfRate,1.6);
assert.equal(profiles[0].gaRate,0.6);
assert.equal(profiles[1].gfRate,0.8);
assert.equal(profiles[1].gaRate,1.3);
assert.deepEqual([...profiles[0].labels],["0","1","2","3","4+"]);
assert.equal(profiles[0].values[0][0],11.1);
assert.equal(profiles[1].values[0][0],12.2);
assert.notEqual(profiles[0].values[0][0],12.1,
  "Boca season matrix must not clone the 1.51/0.60 match score matrix");
for(const p of profiles){
  assert.equal(p.values.length,5);
  assert(p.values.every(row=>row.length===5 && row.every(x=>Number.isFinite(x)&&x>=0)));
  const sum=p.values.flat().reduce((total,cell)=>total+cell,0);
  assert(Math.abs(sum-100)<0.55, "Poisson joint matrix must retain the 4+ tails");
}
const withoutAway=evidence.map(group=>({...group,
  items:group.items.filter(item=>!item.key.startsWith("team_performance.away"))}));
assert.equal(makeTeamGoalMatrices("Boca","Colón",withoutAway).length,1,
  "Missing evidence must not be replaced with the opponent's rates");
const wrongSource=evidence.map(group=>({...group,
  items:group.items.map(item=>({...item,source:"UNKNOWN"}))}));
assert.equal(makeTeamGoalMatrices("Boca","Colón",wrongSource).length,0,
  "Unverified source cannot authorize a season matrix");
const wrongSnapshot=evidence.map(group=>({...group,
  items:group.items.map(item=>item.key==="team_performance.home_goals_against_avg"
    ? {...item,capturedAt:"2026-10-09T22:00:00Z"}:item)}));
assert.equal(makeTeamGoalMatrices("Boca","Colón",wrongSnapshot).length,1,
  "Do not combine a team's GF and GA from different snapshot times");
assert.equal(makeTeamGoalMatrices("Boca","Colón",[]).length,0);
console.log("Season GF x GA matrices: 8 regression checks passed");
