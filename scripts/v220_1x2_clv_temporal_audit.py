from __future__ import annotations
import json, os
from datetime import datetime, timezone
import psycopg

DB=os.environ["DATABASE_URL"]
SQL=r"""
WITH signals AS (
  SELECT DISTINCT ON (
    (mm.row->>'fixture_id')::bigint,
    lower(trim(coalesce(mm.row->>'market',''))),
    lower(trim(coalesce(mm.row->>'selection','')))
  )
    (mm.row->>'fixture_id')::bigint fixture_id,
    mm.row->>'market' market,
    mm.row->>'selection' selection,
    p.generated_at_utc signal_at,
    f.kickoff
  FROM soccer_pipeline_runs p
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(coalesce(p.payload->'market_mismatch_rows','[]'::jsonb))='array'
         THEN coalesce(p.payload->'market_mismatch_rows','[]'::jsonb) ELSE '[]'::jsonb END
  ) mm(row)
  JOIN soccer_fixtures f ON f.fixture_id=(mm.row->>'fixture_id')::bigint
  WHERE upper(mm.row->>'market_family')='1X2'
    AND coalesce((mm.row->>'rankable')::boolean,false)=true
    AND p.generated_at_utc < f.kickoff
    AND nullif(mm.row->>'market','') IS NOT NULL
    AND nullif(mm.row->>'selection','') IS NOT NULL
  ORDER BY
    (mm.row->>'fixture_id')::bigint,
    lower(trim(coalesce(mm.row->>'market',''))),
    lower(trim(coalesce(mm.row->>'selection',''))),
    p.generated_at_utc ASC
),
classified AS (
 SELECT s.*,
   EXISTS (
     SELECT 1 FROM soccer_market_snapshots m
     WHERE m.fixture_id=s.fixture_id
       AND m.captured_at>s.signal_at AND m.captured_at<s.kickoff
       AND lower(trim(coalesce(m.market,'')))=lower(trim(coalesce(s.market,'')))
   ) later_snapshot,
   EXISTS (
     SELECT 1 FROM soccer_market_snapshots m
     WHERE m.fixture_id=s.fixture_id
       AND m.captured_at>s.signal_at AND m.captured_at<s.kickoff
       AND lower(trim(coalesce(m.market,'')))=lower(trim(coalesce(s.market,'')))
       AND m.provider_update IS NOT NULL AND m.provider_update>s.signal_at
   ) later_provider_update,
   (SELECT max(m.captured_at) FROM soccer_market_snapshots m
     WHERE m.fixture_id=s.fixture_id AND m.captured_at<s.kickoff
       AND lower(trim(coalesce(m.market,'')))=lower(trim(coalesce(s.market,'')))) latest_capture,
   (SELECT max(m.provider_update) FROM soccer_market_snapshots m
     WHERE m.fixture_id=s.fixture_id AND m.captured_at<s.kickoff
       AND lower(trim(coalesce(m.market,'')))=lower(trim(coalesce(s.market,'')))) latest_provider_update
 FROM signals s
)
SELECT fixture_id,market,selection,signal_at,kickoff,later_snapshot,later_provider_update,
       latest_capture,latest_provider_update,
       CASE WHEN later_provider_update THEN 'STRICT_LATER_PROVIDER_UPDATE'
            WHEN later_snapshot THEN 'LATER_SNAPSHOT_NO_LATER_PROVIDER_UPDATE'
            ELSE 'NO_LATER_SNAPSHOT' END classification
FROM classified
ORDER BY signal_at DESC;
"""
with psycopg.connect(DB) as conn:
    with conn.cursor() as cur:
        cur.execute(SQL)
        cols=[d.name for d in cur.description]
        rows=[dict(zip(cols,r)) for r in cur.fetchall()]
counts={}
fixtures={}
for r in rows:
    k=r["classification"]; counts[k]=counts.get(k,0)+1
    fixtures.setdefault(k,set()).add(int(r["fixture_id"]))
def iso(v): return v.isoformat() if hasattr(v,"isoformat") else v
examples={}
for k in counts:
    examples[k]=[{a:iso(v) for a,v in r.items()} for r in rows if r["classification"]==k][:10]
out={
 "schema_version":"1.0.0",
 "model_version":"SOCCER_1X2_CLV_TEMPORAL_AUDIT_V220_1.0.0",
 "generated_at_utc":datetime.now(timezone.utc).isoformat(),
 "scope":"1X2 rankable Phase16 signals; oldest exact fixture/market/selection anchor; pre-kickoff snapshots only",
 "signal_rows":len(rows),
 "classification_counts":dict(sorted(counts.items())),
 "classification_unique_fixture_counts":{k:len(v) for k,v in sorted(fixtures.items())},
 "examples":examples,
 "provider_requests_added":0,
 "strict_close_semantics_changed":False,
 "models_changed":False,
 "thresholds_changed":False,
 "gates_changed":False,
 "provider_budget_changed":False
}
print(json.dumps(out,indent=2,default=str))
