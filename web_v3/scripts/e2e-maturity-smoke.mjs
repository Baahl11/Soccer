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
    const texts = await page.locator(".maturity-table").innerText();
    assert.match(texts, /Canonical True CLV/);
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
  console.log("React maturity browser QA completed: authorized fixture + 401/403/503 states.");
} finally {
  await browser.close();
}
