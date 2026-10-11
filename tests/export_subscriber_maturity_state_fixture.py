"""Export real canonical reports into a read-only browser QA fixture.

This is a test-only artifact. No model, database, or production API is modified.
"""
import json
from pathlib import Path

from mcp_gateway import subscriber_maturity_v232 as m

root = Path("state/soccer_edge_state/analysis")
clv = json.loads((root / "clv_v4_postgres_report.json").read_text(encoding="utf-8"))
reports = {
    family: json.loads((root / filename).read_text(encoding="utf-8"))
    for family, filename in m._REPORTS.items()
}
families = m._build_family_rows(clv, reports)
payload = {
    "status": "OK",
    "families": families,
    "market_rows": m._build_market_inventory(families, clv, reports),
    "comparable_true_clv_rows": clv.get("comparable_true_clv_rows"),
    "production_promotion_allowed": False,
    "errors": {},
}
path = Path("web_v3/scripts/maturity-real-state-fixture.json")
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Exported {len(payload['market_rows'])} canonical-source market rows for browser QA")
