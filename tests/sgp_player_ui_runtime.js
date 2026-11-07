/* Offline Node verification of the actual JS shipped in rendered App V2 HTML.
 * No sportsbook requests, no browser navigation, no wagers.
 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const file = process.argv[2];
assert.ok(file, "expected path to rendered app JavaScript");
const app = fs.readFileSync(file, "utf8");
const a = app.indexOf("function sgpPlayerMarketName(");
const b = app.indexOf("async function loadResearchDesk()", a);
assert.ok(a >= 0 && b > a, "actual SGP player code is present");

const slip = { innerHTML: "" };
const sandbox = {
  num: v => {
    if (v === null || v === undefined || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  },
  esc: v => String(v ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;"
  })[c]),
  sgpSelectedPlayerIndexes: [],
  sgpPlayerObservations: [],
  $: id => id === "sgpBuilderSlip" ? slip : null,
};
vm.createContext(sandbox);
const oddsStart = app.indexOf("function americanOddsFromDecimal(");
const oddsEnd = app.indexOf("const pp=", oddsStart);
assert.ok(oddsStart >= 0 && oddsEnd > oddsStart, "actual price converter loaded");
vm.runInContext(app.slice(oddsStart,oddsEnd), sandbox);
vm.runInContext(app.slice(a,b), sandbox);
assert.equal(sandbox.americanOddsFromDecimal(2.10), "+110");
assert.equal(sandbox.americanOddsFromDecimal(2.00), "+100");
assert.equal(sandbox.americanOddsFromDecimal(1.91), "-110");
assert.equal(sandbox.americanOddsFromDecimal(1.50), "-200");
assert.equal(sandbox.americanOddsFromDecimal(1), "NOT VERIFIED");
assert.equal(sandbox.americanOdds(null,"DECIMAL"), "NOT VERIFIED");
assert.equal(sandbox.americanOdds(110,"AMERICAN"), "+110");
assert.equal(sandbox.americanOdds(-110,"AMERICAN"), "-110");
assert.equal(sandbox.americanOdds(2.10,"UNKNOWN"), "NOT VERIFIED");

// Test all other user-visible App V2 price surfaces against the actual
// rendered JS, not a second copy of the conversion formula.
sandbox.market = x => x.market || {};
sandbox.pct = v => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
const canonicalStart = app.indexOf("function priceText(");
const canonicalEnd = app.indexOf("function evidence(", canonicalStart);
assert.ok(canonicalStart >= 0 && canonicalEnd > canonicalStart);
vm.runInContext(app.slice(canonicalStart,canonicalEnd), sandbox);
assert.equal(sandbox.priceText({market:{price:{
  value:2.10,format:"DECIMAL",bookmaker:"Bet365"
}}}), "+110 · Bet365");
assert.equal(sandbox.priceText({market:{price:{
  value:1.91,format:"DECIMAL",bookmaker:"Bet365"
}}}), "-110 · Bet365");
assert.equal(sandbox.priceText({market:{price:{
  value:110,format:"AMERICAN",bookmaker:"Bet365"
}}}), "+110 · Bet365");
assert.equal(sandbox.priceText({market:{price:{
  value:2.10,format:"UNKNOWN",bookmaker:"Bet365"
}}}), "Cuota NO VERIFICADA");

const researchStart = app.indexOf("function researchPriceCard(");
const researchEnd = app.indexOf("let sgpPlayerObservations=", researchStart);
assert.ok(researchStart >= 0 && researchEnd > researchStart);
vm.runInContext(app.slice(researchStart,researchEnd), sandbox);
const comboStart = app.indexOf("function sgpComboCard(");
const catalogStart = app.indexOf("function sgpCatalogCard(", comboStart);
const catalogEnd = app.indexOf("function sgpPlayerMarketName(", catalogStart);
assert.ok(comboStart >= 0 && catalogStart > comboStart && catalogEnd > catalogStart);
vm.runInContext(app.slice(comboStart,catalogStart), sandbox);
vm.runInContext(app.slice(catalogStart,catalogEnd), sandbox);
const offer = {
  fixture_id:701231,home_team:"KVC Westerlo",away_team:"Charleroi",
  league:"Belgium",bookmaker:"Bet365",market_family:"Player Shots",
  provider_market:"Over/Under Goals",selection:"Observed",line:3,
  price_decimal:2.10,quote_provider_updated_at:"2026-11-07T05:28:17Z",
  quote_updated_at:"2026-11-07T05:28:17Z",
  breakeven_probability:0.47619,raw_sport_probability:null,
  market_shrunk_probability:null,hypothetical_ev_pct:null,
  raw_joint_probability:null,raw_hypothetical_ev_pct:null,
  source_quote_fresh:true,
};
assert.match(sandbox.researchPriceCard(offer), /\+110/);
assert.doesNotMatch(sandbox.researchPriceCard(offer), /@ 2\.10/);
assert.match(sandbox.sgpComboCard(offer), /Bet365 @ \+110/);
assert.match(sandbox.sgpCatalogCard({...offer,price_decimal:1.91}),
  /Cuota Bet365 observada: -110/);

const base = {
  player_label: "Min-Hyeok Yang",
  home_team: "KVC Westerlo",
  away_team: "Charleroi",
  observed_team_side: "HOME",
  fixture_id: 701231,
  provider_market: "Home Player Shots",
  selection: "Min-Hyeok Yang - 3",
  price_decimal: 2.1,
  source_quote_fresh: true,
  quote_updated_at: "2026-11-07T05:28:17+00:00",
};
assert.equal(sandbox.sgpPlayerMarketName(base), "Tiros del jugador");
assert.equal(sandbox.sgpPlayerQuantity(base), "3");
const card = sandbox.sgpPlayerCard(base,0);
assert.match(card, /Min-Hyeok Yang/);
assert.match(card, /KVC Westerlo vs Charleroi/);
assert.match(card, /<strong>3 tiros<\/strong>/);
assert.match(card, /<strong>\+110<\/strong>/);
assert.match(card, /Cuota americana individual/);
assert.match(card, /Equipo local/);
assert.match(card, /NO BET/);
assert.match(card, /condición exacta/);
assert.match(card, /titularidad y minutos/);
assert.match(card, /precio conjunto/);
assert.match(card, /CDMX/);
assert.doesNotMatch(card, /<strong>3 o más tiros/);
assert.doesNotMatch(card, /apuesta recomendada/i);

const sot = {...base, player_label:"Thomas Van Den Keybus",
  observed_team_side:"AWAY",
  provider_market:"Home Player Shots On Target Total",
  selection:"Thomas Van Den Keybus - 1",
  source_quote_fresh:false };
assert.equal(sandbox.sgpPlayerMarketName(sot), "Tiros a puerta");
assert.equal(sandbox.sgpPlayerShortSelection(sot), "1 tiro a puerta");
const sotCard = sandbox.sgpPlayerCard(sot,1);
assert.match(sotCard,/Precio anterior · reconfirmar/);
assert.match(sotCard,/Equipo visitante/);

const unknown = {...base, selection:"Jugador - Más de 2.5"};
assert.equal(sandbox.sgpPlayerQuantity(unknown), null);
assert.match(sandbox.sgpPlayerCard(unknown,2), /Ver selección original/);

const dangerous = {...base,player_label:'<img src=x onerror=alert(1)>',
  selection:'<script>alert("run")<\/script> - 3'};
const escaped = sandbox.sgpPlayerCard(dangerous,3);
assert.doesNotMatch(escaped,/<img src=x/);
assert.doesNotMatch(escaped,/<script>alert/);
assert.match(escaped,/&lt;img/);

sandbox.sgpPlayerObservations = [base, sot];
sandbox.sgpSelectedPlayerIndexes = [];
sandbox.renderSgpPlayerSlip();
assert.match(slip.innerHTML,/0\/3 jugadores/);
assert.match(slip.innerHTML,/No se suman ni multiplican cuotas/);
sandbox.sgpSelectedPlayerIndexes = [0,1];
sandbox.renderSgpPlayerSlip();
assert.match(slip.innerHTML,/2\/3 jugadores/);
assert.match(slip.innerHTML,/CUOTA COMBINADA: NOT VERIFIED/);
assert.match(slip.innerHTML,/La casa debe cotizar el boleto real/);
assert.match(slip.innerHTML,/Min-Hyeok Yang/);
assert.match(slip.innerHTML,/Thomas Van Den Keybus/);
assert.doesNotMatch(slip.innerHTML,/cuota combinada: 4\.41/i);

console.log("SGP player Spanish presentation + real rendered JS semantics PASS");
