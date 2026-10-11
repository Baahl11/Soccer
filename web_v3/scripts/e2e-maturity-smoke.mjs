import { chromium } from "playwright";
import { readFileSync, mkdirSync } from "node:fs";
import assert from "node:assert/strict";

const payload = JSON.parse(readFileSync(new URL("./maturity-real-state-fixture.json", import.meta.url), "utf8"));
assert.equal(payload.market_rows.length, 21, "fixture from canonical research has 21 market rows");
const base = "http://127.0.0.1:4173/?sample=1";
const output = "/tmp/soccer-maturity-browser-qa";
mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true });

async function prepare(page, maturityStatus = 200) {
  await page.route("**/app/api/v2/today", route =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ slate: { rows: [] } }) }));
  await page.route("**/app/api/v2/account", route =>
    route.fulfill({ status: 401, contentType: "application/json", body: '{"error":"AUTH_REQUIRED"}' }));
  await page.route("**/app/api/v2/maturity", route => route.fulfill({
    status: maturityStatus,
    contentType: "application/json",
    body: maturityStatus === 200 ? JSON.stringify(payload) : JSON.stringify({ error: "NOT_AUTHORIZED" }),
  }));
  await page.goto(base, { waitUntil: "networkidle" });
}

try {
  for (const [label, viewport] of [["desktop", {width:1440,height:900}], ["mobile", {width:390,height:844}]]) {
    const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
    const page = await context.newPage();
    await prepare(page);
    const button = label === "desktop"
      ? page.getByRole("button", { name: "Market Maturity", exact: true })
      : page.locator(".mobile-product-nav button").filter({ hasText: "Maturity" });
    await button.click();
    await page.locator(".maturity-table tbody tr").first().waitFor({ timeout: 10000 });
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 21);
    assert.match(await page.locator(".maturity-caveat").innerText(), /RESEARCH.*BET/s);
    // A canonical zero is a recorded observation, whereas null is NOT VERIFIED.
    const countBy = predicate => payload.market_rows.filter(predicate).length;
    const canonicalCount = countBy(row => row.true_clv_rows !== null && row.true_clv_rows !== undefined);
    await page.locator("#maturity-filter").selectOption("canonical-clv");
    assert.equal(await page.locator(".maturity-table tbody tr").count(), canonicalCount);
    await page.locator("#maturity-filter").selectOption("missing-clv");
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 21 - canonicalCount);
    await page.locator("#maturity-filter").selectOption("missing-oos");
    assert.equal(await page.locator(".maturity-table tbody tr").count(),
      countBy(row => row.market_specific_evidence_verified !== true));
    await page.locator("#maturity-filter").selectOption("all");
    await page.locator("#maturity-search").fill("BTTS");
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 1);
    await page.locator("#maturity-search").fill("no-such-market");
    await page.getByText(/No markets match this filter/).waitFor({ timeout: 10000 });
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 0);
    await page.locator("#maturity-search").fill("");
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 21);
    const texts = await page.locator(".maturity-table").innerText();
    assert.match(texts, /canonical true clv/i);
    assert.match(texts, /NOT VERIFIED/);
    assert.match(texts, /BLOCKED/);
    const dimensions = await page.evaluate(() => ({
      body: document.documentElement.scrollWidth,
      viewport: document.documentElement.clientWidth,
      tableScroll: document.querySelector(".maturity-scroll")?.scrollWidth,
      tableClient: document.querySelector(".maturity-scroll")?.clientWidth,
    }));
    assert(dimensions.body <= dimensions.viewport + 2,
      `unwanted page overflow ${label}: ${JSON.stringify(dimensions)}`);
    if (label === "mobile") assert(dimensions.tableScroll > dimensions.tableClient,
      "wide market table must be scrollable independently on mobile");
    await page.screenshot({ path: `${output}/${label}.png`, fullPage: true });
    console.log(`${label} canonical-state browser QA passed: 21 markets, no body overflow`);
    await context.close();
  }
  for (const status of [401, 403, 503]) {
    const context = await browser.newContext({ viewport: {width: 1300,height:800} });
    const page = await context.newPage();
    await prepare(page, status);
    await page.getByRole("button", { name: "Market Maturity", exact: true }).click();
    const marker = status === 401 ? "SIGN IN REQUIRED" : status === 403 ? "EDGE PRO REQUIRED" : "MATURITY UNAVAILABLE";
    await page.getByText(new RegExp(marker)).waitFor({ timeout: 10000 });
    assert.equal(await page.locator(".maturity-table tbody tr").count(), 0,
      "locked/error content cannot show research rows");
    await context.close();
  }
  // A role transition must never leave previously authorized research visible.
  const context = await browser.newContext({ viewport: {width: 1300,height:800} });
  const page = await context.newPage();
  await prepare(page, 403);
  await page.getByRole("button", { name: "Market Maturity", exact: true }).click();
  await page.getByText(/EDGE PRO REQUIRED/).waitFor({ timeout: 10000 });
  await page.unroute("**/app/api/v2/maturity");
  await page.route("**/app/api/v2/maturity", route =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(payload) }));
  await page.getByRole("button", { name: "Recheck evidence" }).click();
  await page.locator(".maturity-table tbody tr").first().waitFor({ timeout: 10000 });
  assert.equal(await page.locator(".maturity-table tbody tr").count(), 21);
  await page.unroute("**/app/api/v2/maturity");
  await page.route("**/app/api/v2/maturity", route =>
    route.fulfill({ status: 403, contentType: "application/json", body: '{"error":"PREVIEW_REQUIRES_PRO"}' }));
  await page.getByRole("button", { name: "Recheck evidence" }).click();
  await page.getByText(/EDGE PRO REQUIRED/).waitFor({ timeout: 10000 });
  assert.equal(await page.locator(".maturity-table tbody tr").count(), 0,
    "revoked access must clear premium rows, never leak stale research");
  await context.close();
  console.log("React maturity browser QA completed: 21 canonical rows, filters, 401/403/503, role transition and safe recheck.");
} finally {
  await browser.close();
}
