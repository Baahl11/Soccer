from mcp_gateway.oos_canonical_pair_inventory_v1 import card_counts, outcomes


def raw(metric, home, away):
    return [{"team": {"id": tid}, "statistics": [{"type": metric, "value": v}]}
            for tid, v in ((10, home), (20, away))]


def test_card_counts_requires_explicit_zero_and_both_teams():
    assert card_counts(raw("Yellow Cards", 0, 3), 10, 20, "Yellow Cards") == (0, 3)
    assert card_counts(raw("Yellow Cards", None, 3), 10, 20, "Yellow Cards") is None
    assert card_counts(raw("Yellow Cards", False, 3), 10, 20, "Yellow Cards") is None
    assert card_counts(raw("Yellow Cards", 0, 3), 10, 20, "Red Cards") is None
    assert card_counts(raw("Yellow Cards", 0, 3)[:1], 10, 20, "Yellow Cards") is None


def test_postgame_evidence_ignores_live_and_conflicts():
    fixture = dict(fixture_id=1, league_id=2, home_team_id=10, away_team_id=20,
                   kickoff="2026-11-17T00:00:00+00:00")
    def obs(value, status="FT"):
        return dict(event_type="FIXTURE_OBSERVATION_RESEARCH", stage="FIXTURE_OBSERVATION",
                    bet_eligible=False, fixture={**fixture, "status": status},
                    sporting=dict(collection_scope="POSTGAME_OBSERVATION",
                                  fixture_statistics=raw("Yellow Cards", value, 3)))
    assert outcomes([obs(0)], fixture) == {"yellow"}
    assert outcomes([obs(0, "1H")], fixture) == set()
    assert outcomes([obs(0), obs(1)], fixture) == set()
    assert outcomes([obs(0)] * 9, fixture) == set()


def test_fe11_corners_recover_absent_stats_but_not_conflicting_truth():
    fixture = dict(fixture_id=1, league_id=2, home_team_id=10, away_team_id=20,
                   kickoff="2026-11-17T00:00:00+00:00")
    assert outcomes([], fixture, [[2, 3]]) == {"corners"}
    assert outcomes([], fixture, [[2, 3], [3, 3]]) == set()
    obs = dict(event_type="FIXTURE_OBSERVATION_RESEARCH", stage="FIXTURE_OBSERVATION",
               bet_eligible=False, fixture={**fixture, "status": "FT"},
               sporting=dict(collection_scope="POSTGAME_OBSERVATION",
                             fixture_statistics=raw("Corner Kicks", 2, 3)))
    assert outcomes([obs], fixture, [[2, 3]]) == {"corners"}
    assert outcomes([obs], fixture, [[3, 3]]) == set()
