import { chromium } from "playwright";
import assert from "node:assert/strict";

const base = "http://127.0.0.1:4173/";
const fixture = {
  fixture: {fixture_id: 1024, home_team: "Team One", away_team: "Team Two", league: "Research League", kickoff: "2026-10-01T20:00:00Z"},
  decision: {classification: "BET", reason_display: "Persisted historical selection."},
  market: {name:"Match Winner", selection:"Home", price:{value:2.05,bookmaker:"Verified Snapshot",captured_at:"2026-10-01T19:20:00Z"}},
};
const picks = {model_version:"SOCCER_V4_TEST_FIXTURE",generated_at_utc:"2026-10-01T18:00:00Z",
  picks:{total:2, rows:[fixture, {...fixture, decision:{classification:"WATCH"}}]}};
const leans = {model_version:"SOCCER_V4_TEST_FIXTURE",generated_at_utc:"2026-10-01T18:00:00Z",
  leans:{total:1, rows:[{...fixture,decision:{classification:"LEAN"}}]}};
const performance = {
  status:"VERIFIED_PERFORMANCE_SAMPLE_NOT_READY",
  performance_policy:{headline_scope:"CANONICAL_BET_SETTLEMENT_ONLY",leans_in_headline:false},
  bet_track_record:{settled:5,win:3,loss:2,push:0,ungraded:1,hit_rate_ex_push:0.6,
    roi_units:0.4,sample_status:"SAMPLE_TOO_SMALL",directional_minimum:20,
    families:[{market_family:"1X2",settled:5,win:3,loss:2,push:0}]},
  research_lean:{settled:43,win:38,loss:5},
  settlement:{coverage_rate:0.8,backlog_rows:1},
  true_clv:{rows:null},
  source:{generated_at_utc:"2026-10-01T18:00:00Z",bet_track_record_model_version:"CANONICAL_LEDGER_V4"},
};
const browser = await chromium.launch({headless:true});
async function visit(section, status=200, width=1440) {
  const context=await browser.newContext({viewport:{width,height:900}});
  const page=await context.newPage();
  await page.route("**/app/api/v2/today", route =>
    route.fulfill({status:200,contentType:"application/json",body:JSON.stringify({slate:{rows:[]}})}));
  await page.route("**/app/api/v2/account", route =>
    route.fulfill({status:401,contentType:"application/json",body:'{"error":"AUTH_REQUIRED"}'}));
  const endpoint = section==="picks"?"/picks":section==="leans"?"/leans":"/performance";
  await page.route("**/app/api/v2"+endpoint, route =>
    route.fulfill({status,contentType:"application/json",body:status===200?
      JSON.stringify(section==="picks"?picks:section==="leans"?leans:performance):
      JSON.stringify({error:status===403?"PRO_REQUIRED":"UNAVAILABLE"})}));
  await page.goto(base+"?view="+section,{waitUntil:"domcontentloaded"});
  return {page,context};
}
try {
  {
    const {page,context}=await visit("picks");
    await page.getByRole("heading",{name:"Persisted picks"}).waitFor();
    await page.locator(".commercial-candidate").first().waitFor();
    assert.equal(await page.locator(".commercial-candidate").count(),1,
      "Only explicit persisted BET may be displayed; WATCH cannot be promoted");
    assert.match(await page.locator(".commercial-page").innerText(),/ARCHIVED SNAPSHOT/);
    assert.match(await page.locator(".commercial-page").innerText(),/not actionable/i);
    assert.match(await page.locator(".commercial-page").innerText(),/Verified Snapshot/);
    await context.close();
  }
  {
    const {page,context}=await visit("leans");
    await page.locator(".commercial-candidate").first().waitFor();
    assert.equal(await page.locator(".commercial-candidate").count(),1);
    assert.match(await page.locator(".commercial-page").innerText(),/LEAN · STORED DECISION/);
    await context.close();
  }
  for (const section of ["performance","results"]) {
    const {page,context}=await visit(section);
    await page.locator(".commercial-kpis").waitFor();
    assert.match(await page.locator(".commercial-page").innerText(),/Settled BET only/);
    assert.match(await page.locator(".commercial-page").innerText(),/SMALL SAMPLE/);
    assert.match(await page.locator(".commercial-page").innerText(),/NOT VERIFIED/);
    assert(!(await page.locator(".commercial-page").innerText()).includes("38 W"),"LEAN wins must not inflate BET");
    if (section==="results")
      assert.match(await page.locator(".commercial-page").innerText(),/FIXTURE-LEVEL RESULT HISTORY · NOT VERIFIED/);
    await context.close();
  }
  for (const status of [401,403,503]) {
    const {page,context}=await visit("picks",status);
    await page.locator(".commercial-empty").first().waitFor();
    assert.equal(await page.locator(".commercial-candidate").count(),0);
    assert.match(await page.locator(".commercial-page").innerText(),
      status===401?/SIGN IN REQUIRED/:status===403?/EDGE PRO OR OWNER REQUIRED/:/SOURCE NOT VERIFIED/);
    await context.close();
  }
  {
    const {page,context}=await visit("results",200,390);
    await page.locator(".commercial-kpis").waitFor();
    const size=await page.evaluate(()=>({
      scroll:document.documentElement.scrollWidth,
      visible:document.documentElement.clientWidth,
      nav:document.querySelector(".mobile-product-nav")?.scrollWidth,
    }));
    assert(size.scroll<=size.visible+2,"mobile body must not overflow");
    assert(size.nav>size.visible-20,"mobile bottom nav scrolls instead of rendering microscopic links");
    await context.close();
  }
  console.log("OFFLINE_COMMERCIAL_E2E_SUCCESS: BET-only, LEAN isolation, small sample, stale, 401/403/503, mobile");
} finally {
  await browser.close();
}
