from mcp_gateway.settlement_merge_v4 import build_merge_report


def _row(fixture_id, generated_at, *, source, price=2.0):
    return {
        "fixture_id": fixture_id,
        "generated_at_local": generated_at,
        "classification": "BET",
        "market_family": "FT_TOTALS",
        "market": "Goals Over/Under",
        "selection": "Over 2.5",
        "line": 2.5,
        "decimal_price": price,
        "settlement_status": "WIN",
        "settled": True,
        "roi_units": price - 1.0,
        "source": source,
    }


def test_merge_retains_legacy_and_adds_postgres_rows():
    legacy = [_row(1, "2026-09-20T10:00:00+00:00", source="LEGACY")]
    postgres = [_row(2, "2026-09-24T10:00:00+00:00", source="POSTGRES")]
    report = build_merge_report(legacy, postgres)
    assert report["legacy_rows"] == 1
    assert report["postgres_rows"] == 1
    assert report["merged_rows"] == 2
    assert report["market_performance_summary"]["by_market_family"]["FT_TOTALS"]["settled"] == 2


def test_merge_dedupes_overlap_and_prefers_latest_postgres_row():
    legacy = [_row(1, "2026-09-20T10:00:00+00:00", source="LEGACY", price=1.9)]
    postgres = [_row(1, "2026-09-20T10:05:00+00:00", source="POSTGRES", price=2.1)]
    report = build_merge_report(legacy, postgres)
    assert report["merged_rows"] == 1
    assert report["deduped_overlap_rows"] == 1
    assert report["rows"][0]["source"] == "POSTGRES"
    assert report["rows"][0]["decimal_price"] == 2.1
