import { useEffect, useState } from "react";
import { hasStoredSession, refreshSession, storedAccessToken } from "./auth";

type Evidence = { current?: number | null; target?: number | null; unit?: string | null };
type MarketRow = {
  key: string; label: string; parent_family: string | null;
  model_evidence: Evidence | null;
  mapped_rows: number | null; priced_rows: number | null;
  true_clv_rows: number | null; true_clv_target: number | null;
  parent_research_stage: string | null; next_gate: string | null;
  source: string | null; production_promotion_allowed: boolean;
};
type MaturityPayload = {
  status?: string; families?: unknown[]; market_rows?: MarketRow[];
  comparable_true_clv_rows?: number | null; errors?: Record<string, string>;
  production_promotion_allowed?: boolean;
};

function verified(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === "" ? "NOT VERIFIED" : String(value);
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
      <div><span>SOCCER EDGE · RESEARCH TRUTH</span><h1>Market Maturity</h1><p>Sport first. Market second. Independently verified OOS, priced history and strict True CLV by market.</p></div>
      <b>{payload ? verified(payload.status) : "VERIFYING"}</b>
    </div>
    <div className="coverage-kpis">
      <div><span>MARKET INVENTORY</span><b>{payload ? rows.length : "—"}</b></div>
      <div><span>RESEARCH FAMILIES</span><b>{payload ? (payload.families?.length ?? "NOT VERIFIED") : "—"}</b></div>
      <div><span>STRICT TRUE CLV</span><b>{verified(payload?.comparable_true_clv_rows)}</b></div>
    </div>
    <div className="slate-card maturity-surface">
      <div className="maturity-caveat"><b>RESEARCH ≠ BET</b><p>A gate met means eligibility for review, never automatic production promotion. Parent and submarket counts are not independent samples. Missing data stays NOT VERIFIED.</p></div>
      {message && <div className="slate-empty" role="status">{message}</div>}
      {!!errors.length && <div className="slate-empty">PARTIAL REPORTS — could not verify: {errors.join(", ")}</div>}
      {payload && rows.length === 0 && <div className="slate-empty">No verified market maturity reports available.</div>}
      {!!rows.length && <div className="maturity-scroll"><table className="maturity-table"><thead><tr><th>Market</th><th>OOS model</th><th>Mapped</th><th>Priced</th><th>True CLV</th><th>Research stage</th><th>Blocking gate</th><th>Source report</th></tr></thead><tbody>
        {rows.map(row => <tr key={row.key}>
          <td><strong>{verified(row.label)}</strong><small>{verified(row.parent_family)}</small></td>
          <td>{gate(row.model_evidence?.current, row.model_evidence?.target)}<small>{verified(row.model_evidence?.unit)}</small></td>
          <td>{verified(row.mapped_rows)}</td><td>{verified(row.priced_rows)}</td>
          <td>{gate(row.true_clv_rows, row.true_clv_target)}</td>
          <td>{verified(row.parent_research_stage)}</td><td>{verified(row.next_gate)}</td><td><small>{verified(row.source)}</small></td>
        </tr>)}
      </tbody></table></div>}
    </div>
  </section>;
}
