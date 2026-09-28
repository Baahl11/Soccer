from mcp_gateway import clv_postgres_v4 as clv


def _event():
    return {
        "fixture_id": 198001,
        "generated_at": "2026-09-28T10:00:00+00:00",
        "event_payload": {
            "two_h_goals_intelligence": {
                "observed_market_rows": [
                    {
                        "market": "Goals Over/Under - Second Half",
                        "selection": "OVER",
                        "line": 1.5,
                        "decimal_price": 1.95,
                    }
                ]
            }
        },
    }


def test_dedicated_two_h_intelligence_enters_derivative_clv_source():
    rows = clv._derivative_signals_from_events([_event()])
    assert len(rows) == 1
    row = rows[0]
    assert row["signal_source"] == "DERIVATIVE_INTELLIGENCE:two_h_goals_intelligence"
    assert row["market_candidate"]["market_family"] == "2H"
    assert clv._family(row["market_candidate"]) == "2H"


def test_dedicated_two_h_is_not_excluded_as_period_team_total():
    row = clv._derivative_signals_from_events([_event()])[0]
    assert clv._is_period_team_total_signal(row) is False


def test_team_total_origin_period_two_h_remains_excluded():
    row = {
        "signal_source": "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
        "market_candidate": {
            "market_family": "2H",
            "market": "Team Total Goals Over/Under - Second Half",
            "selection": "Over 0.5",
            "decimal_price": 1.8,
        },
    }
    assert clv._is_period_team_total_signal(row) is True


def test_fair_merge_preserves_dedicated_two_h_under_derivative_pressure():
    def signal(fid, family, source):
        return {
            "fixture_id": fid,
            "generated_at": f"2026-09-28T10:{fid % 60:02d}:00+00:00",
            "signal_source": source,
            "market_candidate": {
                "market_family": family,
                "market": "Goals Over/Under - Second Half" if family == "2H" else "Total - Home",
                "selection": "OVER",
                "line": 1.5,
                "decimal_price": 1.9,
            },
        }

    derivatives = [
        *[signal(100 + i, "HOME_TT", "DERIVATIVE_INTELLIGENCE:team_totals_intelligence") for i in range(30)],
        signal(900, "2H", "DERIVATIVE_INTELLIGENCE:two_h_goals_intelligence"),
    ]
    merged = clv._merge_signals([], derivatives, [], max_rows=2)
    assert {clv._family(row["market_candidate"]) for row in merged} == {"HOME_TT", "2H"}
