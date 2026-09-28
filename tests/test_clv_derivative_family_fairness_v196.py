from mcp_gateway import clv_postgres_v4 as clv


def _signal(fid, family, market, selection="Over", price=2.0):
    return {
        "fixture_id": fid,
        "generated_at": f"2026-09-28T10:{fid % 60:02d}:00+00:00",
        "market_candidate": {
            "market_family": family,
            "market": market,
            "selection": selection,
            "decimal_price": price,
        },
    }


def test_derivative_merge_round_robins_families_inside_global_cap():
    pipeline = [
        _signal(1, "BTTS", "Both Teams Score", "Yes"),
        _signal(2, "1X2", "Match Winner", "Home"),
    ]
    derivatives = [
        *[_signal(100 + i, "HOME_TT", "Total - Home") for i in range(30)],
        _signal(300, "1H", "Goals Over/Under - First Half"),
        _signal(301, "FT_CORNERS", "Corners Over Under"),
    ]

    merged = clv._merge_signals(pipeline, derivatives, [], max_rows=5)
    families = [clv._family(row["market_candidate"]) for row in merged]

    assert families[:2] == ["BTTS", "1X2"]
    assert "1H" in families
    assert "FT_CORNERS" in families
    assert "HOME_TT" in families
    assert len(merged) == 5


def test_derivative_merge_keeps_global_bound_and_dedupes():
    duplicate = _signal(400, "1H", "Goals Over/Under - First Half")
    derivatives = [duplicate, dict(duplicate), _signal(401, "FT_CORNERS", "Corners Over Under")]
    merged = clv._merge_signals([], derivatives, [], max_rows=2)
    assert len(merged) == 2
    assert {clv._family(row["market_candidate"]) for row in merged} == {"1H", "FT_CORNERS"}
