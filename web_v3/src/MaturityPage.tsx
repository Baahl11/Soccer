import { useEffect, useState } from "react";
import { hasStoredSession, refreshSession, storedAccessToken } from "./auth";

type Evidence = { current?: number | null; target?: number | null; unit?: string | null };
type Quality = { brier?: number | null; log_loss?: number | null; ece?: number | null; scope?: string | null };
type MarketRow = {
  key: string; label: string; parent_family: string | null; model_quality?: Quality;
  report_true_clv_rows?: number | null; source_model_version?: string | null; source_temporal_provenance?: string | null;
  model_evidence: Evidence | null;
  mapped_rows: number | null; priced_rows: number | null;
  true_clv_rows: number | null; true_clv_target: number | null;
  parent_research_stage: string | null; next_gate: string | null;
  source: string | null; report_status?: string | null; blockers?: string[];
  market_specific_evidence_verified?: boolean; production_promotion_allowed: boolean;
};
type MaturityPayload = {
  status?: string; families?: unknown[]; market_rows?: MarketRow[];
  comparable_true_clv_rows?: number | null; errors?: Record<string, string>;
  production_promotion_allowed?: boolean;
};

function verified(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === "" ? "NOT VERIFIED" : String(value);
}
function metric(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "NOT VERIFIED";
}
function gate(value: number | null | undefined, target: number | null | undefined): string {
  return verified(value) + (target == null ? "" : " / " + verified(target));
}

async function requestMaturity(token: string): Promise<Response> {
  return fetch("/app/api/v2/maturity", {
    headers: token ? { Authorization: "Bearer " + token } : {},
    cache: "no-store",
  });
}

export function MaturityPage() {
  const [payload, setPayload] = useState<MaturityPayload | null>(null);
  const [message, setMessage] = useState("Loading persisted market evidence…");
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        let token = storedAccessToken();
        let response = await requestMaturity(token);
        if (response.status === 401 && hasStoredSession()) {
          token = await refreshSession();
          response = await requestMaturity(token);
        }
        if (!response.ok) {
          if (active) setMessage(response.status === 403 ? "EDGE PRO REQUIRED — market research data is restricted." : response.status === 401 ? "SIGN IN REQUIRED — research access requires authentication." : "MATURITY UNAVAILABLE — could not verify the persisted report.");
          return;
        }
        const data: MaturityPayload = await response.json();
        if (active) {
          setPayload(data);
          setMessage("");
        }
      } catch {
        if (active) setMessage("MATURITY UNAVAILABLE — no research counters have been fabricated.");
      }
    })();
    return () => { active = false; };
  }, []);

  const rows = Array.isArray(payload?.market_rows) ? payload.market_rows : [];
  const errors = Object.keys(payload?.errors ?? {});
  return <section className="react-page">
    <div className="slate-page-head">
      <div><span>SOCCER EDGE · RESEARCH TRUTH</span><h1>Market Maturity</h1><p>Sport first. Market second. Reported OOS, observed price history and strict True CLV by market, with missing evidence flagged.</p></div>
      <b>{payload ? verified(payload.status) : "VERIFYING"}</b>
    </div>
    <div className="coverage-kpis">
      <div><span>MARKET INVENTORY</span><b>{payload ? rows.length : "—"}</b></div>
      <div><span>RESEARCH FAMILIES</span><b>{payload ? (payload.families?.length ?? "NOT VERIFIED") : "—"}</b></div>
      <div><span>STRICT TRUE CLV</span><b>{verified(payload?.comparable_true_clv_rows)}</b></div>
    </div>
    <div className="slate-card maturity-surface">
      <div className="maturity-caveat"><b>RESEARCH ≠ BET · PRODUCTION NOT AUTHORIZED</b><p>A gate met means eligibility for review, never automatic production promotion. Parent and submarket counts are not independent samples. Missing data and source freshness stay NOT VERIFIED without direct proof.</p></div>
      {message && <div className="slate-empty" role="status">{message}</div>}
      {!!errors.length && <div className="slate-empty">PARTIAL REPORTS — could not verify: {errors.join(", ")}</div>}
      {payload && rows.length === 0 && <div className="slate-empty">No verified market maturity reports available.</div>}
      {!!rows.length && <div className="maturity-scroll"><table className="maturity-table"><thead><tr><th>Market</th><th>Model sample</th><th>Model quality</th><th>Mapped</th><th>Priced</th><th>Canonical True CLV</th><th>Validation report CLV</th><th>Parent research stage</th><th>Market blockers</th><th>Source report</th><th>Production</th></tr></thead><tbody>
        {rows.map(row => <tr key={row.key}>
          <td><strong>{verified(row.label)}</strong><small>{verified(row.parent_family)}</small></td>
          <td>{gate(row.model_evidence?.current, row.model_evidence?.target)}<small>{verified(row.model_evidence?.unit)}</small><small>{row.market_specific_evidence_verified ? "MARKET OOS EVIDENCE REPORTED" : "INDEPENDENT OOS NOT VERIFIED"}</small></td>
          <td>Brier: {metric(row.model_quality?.brier)}<small>Log loss: {metric(row.model_quality?.log_loss)}</small><small>ECE: {metric(row.model_quality?.ece)}</small><small>{verified(row.model_quality?.scope)}</small></td>
          <td>{verified(row.mapped_rows)}</td><td>{verified(row.priced_rows)}</td>
          <td>{gate(row.true_clv_rows, row.true_clv_target)}</td><td>{verified(row.report_true_clv_rows)}<small>Research report; not a substitute for canonical CLV</small></td>
          <td>{verified(row.parent_research_stage)}</td><td>{verified(row.next_gate)}{(row.blockers?.length ?? 0) > 1 && <details className="maturity-blockers"><summary>{(row.blockers?.length ?? 0) - 1} more blockers</summary><small>{row.blockers?.slice(1).join(" · ")}</small></details>}</td><td><small>{verified(row.source)}</small><small>Report: {verified(row.report_status)}</small><small>Model: {verified(row.source_model_version)}</small><small>Timestamp: {verified(row.source_temporal_provenance)}</small></td><td>{row.production_promotion_allowed ? "OPERATOR VERIFICATION REQUIRED" : "BLOCKED"}</td>
        </tr>)}
      </tbody></table></div>}
    </div>
  </section>;
}
