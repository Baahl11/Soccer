from mcp_gateway import analyze_corners_baseline as baseline_v
from mcp_gateway import analyze_team_corners as team_corners_v
from mcp_gateway import corners_oos_v4 as v


def _corner_eval(league_id, actual, base_lam, challenger_lam, prior_matchup_n=8):
    row = {
        "league_id": league_id,
        "prior_matchup_n": prior_matchup_n,
        "baseline_total_lambda": base_lam,
        "challenger_total_lambda": challenger_lam,
        "actual_total_corners": actual,
    }
    for line in baseline_v.REQUIRED_LINES:
        key = str(line).replace(".", "_")
        row[f"base_over_{key}"] = baseline_v.pois_over(base_lam, line)
        row[f"challenger_over_{key}"] = baseline_v.pois_over(challenger_lam, line)
    return row


def test_corners_baseline_materializes_review_sized_league_lift():
    rows = []
    for league_id in (100, 200):
        for _ in range(20):
            rows.append(_corner_eval(league_id, 12.0, 6.0, 12.0))

    report = baseline_v.formation_lift_by_league(rows)
    assert report["formation_adjusted_evaluations"] == 40
    assert report["review_eligible_leagues"] == ["100", "200"]
    assert report["stable_lift_leagues"] == ["100", "200"]
    assert report["negative_lift_leagues"] == []
    assert report["review_ready"] is True


def test_corners_baseline_does_not_call_thin_league_stable():
    rows = [_corner_eval(100, 10.0, 12.0, 10.0) for _ in range(19)]
    report = baseline_v.formation_lift_by_league(rows)
    assert report["review_eligible_leagues"] == []
    assert report["review_ready"] is False


def test_team_corners_materializes_league_venue_stability_without_inventing_performance_gate():
    rows=[]
    for fixture_id in range(1, 21):
        for line in team_corners_v.LINES:
            rows.append({
                "fixture_id":fixture_id,
                "league_id":39,
                "team_role":"HOME",
                "line":line,
                "p_over":0.55,
                "actual_over":1 if fixture_id % 2 else 0,
            })
    report=team_corners_v.league_venue_stability(rows)
    segment=report["segments"]["39|HOME"]
    assert segment["unique_fixtures"] == 20
    assert segment["review_eligible"] is True
    assert report["review_eligible_segments"] == ["39|HOME"]
    assert report["review_ready"] is False


def test_team_corners_league_venue_requires_two_review_sized_segments():
    rows=[]
    for role in ("HOME","AWAY"):
        for fixture_id in range(1, 21):
            for line in team_corners_v.LINES:
                rows.append({
                    "fixture_id":fixture_id,
                    "league_id":39,
                    "team_role":role,
                    "line":line,
                    "p_over":0.55,
                    "actual_over":1 if fixture_id % 2 else 0,
                })
    report=team_corners_v.league_venue_stability(rows)
    assert report["review_eligible_segments"] == ["39|AWAY","39|HOME"]
    assert report["review_ready"] is True


def test_v4_022_blocks_current_small_formation_sample_and_no_clv():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.612748,
                "lines": {
                    "8.5": {"brier": 0.253428, "log_loss": 0.701139},
                    "9.5": {"brier": 0.252232, "log_loss": 0.697662},
                    "10.5": {"brier": 0.221616, "log_loss": 0.63546},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.581188,
                "lines": {
                    "8.5": {"brier": 0.252486, "log_loss": 0.699201},
                    "9.5": {"brier": 0.248863, "log_loss": 0.690901},
                    "10.5": {"brier": 0.221125, "log_loss": 0.634773},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "promotion_gate": {"enabled": False},
        },
        [],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert report["ft_corners"]["all_required_lines_improve_brier_and_log_loss"] is True
    assert report["ft_corners"]["mae_improves"] is True
    assert "FORMATION_ADJUSTED_39_LT_100" in report["blockers"]
    assert "FT_CORNERS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert "TEAM_CORNERS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_022_true_clv_matcher_is_corners_only():
    rows = [
        {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "fixture_id": 3, "clv_probability_pp": 0.20},
    ]
    summary = v.summarize_true_clv(rows)
    ft = v.summarize_true_clv(rows, "FT_CORNERS")
    team = v.summarize_true_clv(rows, "TEAM_CORNERS")
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
    assert ft["rows"] == 1
    assert ft["unique_fixtures"] == 1
    assert ft["avg_probability_clv_pp"] == 0.02
    assert team["rows"] == 1
    assert team["unique_fixtures"] == 1
    assert team["avg_probability_clv_pp"] == -0.01


def test_v4_022_family_views_do_not_share_ft_and_team_clv():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.6,
                "lines": {
                    "8.5": {"brier": 0.25, "log_loss": 0.70},
                    "9.5": {"brier": 0.25, "log_loss": 0.70},
                    "10.5": {"brier": 0.22, "log_loss": 0.63},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.5,
                "lines": {
                    "8.5": {"brier": 0.24, "log_loss": 0.69},
                    "9.5": {"brier": 0.24, "log_loss": 0.69},
                    "10.5": {"brier": 0.21, "log_loss": 0.62},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "promotion_gate": {"enabled": False},
        },
        [
            *[
                {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
            *[
                {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(10)
            ],
        ],
    )
    assert report["family_views"]["FT_CORNERS"]["true_clv"]["rows"] == 60
    assert report["family_views"]["TEAM_CORNERS"]["true_clv"]["rows"] == 10
    assert "FT_CORNERS_TRUE_CLV_60_LT_50" not in report["family_views"]["FT_CORNERS"]["blockers"]
    assert "TEAM_CORNERS_TRUE_CLV_10_LT_50" in report["family_views"]["TEAM_CORNERS"]["blockers"]


def test_v4_022_replaces_historical_disabled_flags_with_explicit_evidence_blockers():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.6,
                "lines": {
                    "8.5": {"brier": 0.25, "log_loss": 0.70},
                    "9.5": {"brier": 0.25, "log_loss": 0.70},
                    "10.5": {"brier": 0.22, "log_loss": 0.63},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.5,
                "lines": {
                    "8.5": {"brier": 0.24, "log_loss": 0.69},
                    "9.5": {"brier": 0.24, "log_loss": 0.69},
                    "10.5": {"brier": 0.21, "log_loss": 0.62},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "by_league": {"46": {"n": 90}},
            "promotion_gate": {"enabled": False},
        },
        [
            *[
                {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
            *[
                {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
        ],
    )

    ft_blockers = report["family_views"]["FT_CORNERS"]["blockers"]
    team_blockers = report["family_views"]["TEAM_CORNERS"]["blockers"]
    assert "SOURCE_FT_CORNERS_PROMOTION_GATE_DISABLED" not in ft_blockers
    assert "SOURCE_TEAM_CORNERS_PROMOTION_GATE_DISABLED" not in team_blockers
    assert "FORMATION_ADJUSTED_39_LT_100" in ft_blockers
    assert "FT_CORNERS_LEAGUE_LIFT_NOT_MATERIALIZED" in ft_blockers
    assert "PARENT_FT_CORNERS_NOT_REVIEW_READY" in team_blockers
    assert "TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_MATERIALIZED" in team_blockers


def test_v4_022_uses_materialized_league_lift_without_removing_sample_gate():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.6,
                "lines": {
                    "8.5": {"brier": 0.25, "log_loss": 0.70},
                    "9.5": {"brier": 0.25, "log_loss": 0.70},
                    "10.5": {"brier": 0.22, "log_loss": 0.63},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.5,
                "lines": {
                    "8.5": {"brier": 0.24, "log_loss": 0.69},
                    "9.5": {"brier": 0.24, "log_loss": 0.69},
                    "10.5": {"brier": 0.21, "log_loss": 0.62},
                },
            },
            "formation_lift_by_league": {
                "review_ready": True,
                "review_eligible_leagues": ["100", "200"],
                "stable_lift_leagues": ["100", "200"],
                "negative_lift_leagues": [],
            },
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
        },
        [
            *[
                {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
            *[
                {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
        ],
    )
    ft_blockers = report["family_views"]["FT_CORNERS"]["blockers"]
    assert "FT_CORNERS_LEAGUE_LIFT_NOT_MATERIALIZED" not in ft_blockers
    assert "FT_CORNERS_LEAGUE_LIFT_NOT_STABLE" not in ft_blockers
    assert "FORMATION_ADJUSTED_39_LT_100" in ft_blockers


def test_v4_022_distinguishes_materialized_but_not_ready_team_corner_stability():
    report = v.build_report(
        {
            "walk_forward_evaluations":244,
            "formation_adjusted_evaluations":100,
            "baseline":{
                "mae_total_corners":2.6,
                "lines":{
                    "8.5":{"brier":0.25,"log_loss":0.70},
                    "9.5":{"brier":0.25,"log_loss":0.70},
                    "10.5":{"brier":0.22,"log_loss":0.63},
                },
            },
            "formation_challenger":{
                "mae_total_corners":2.5,
                "lines":{
                    "8.5":{"brier":0.24,"log_loss":0.69},
                    "9.5":{"brier":0.24,"log_loss":0.69},
                    "10.5":{"brier":0.21,"log_loss":0.62},
                },
            },
            "formation_lift_by_league":{
                "review_ready":True,
                "review_eligible_leagues":["100","200"],
                "stable_lift_leagues":["100","200"],
                "negative_lift_leagues":[],
            },
        },
        {
            "evaluated_fixtures":244,
            "evaluated_rows":1464,
            "by_role_line":{
                "HOME|3.5":{"n":244},
                "HOME|4.5":{"n":244},
                "HOME|5.5":{"n":244},
                "AWAY|3.5":{"n":244},
                "AWAY|4.5":{"n":244},
                "AWAY|5.5":{"n":244},
            },
            "by_league":{"39":{"n":100}},
            "league_venue_stability":{
                "review_ready":False,
                "review_eligible_segments":[],
                "segments":{},
            },
        },
        [
            *[{"market_family":"FT_CORNERS","market":"Corners Over/Under","fixture_id":i,"clv_probability_pp":0.1} for i in range(60)],
            *[{"market_family":"TEAM_CORNERS","market":"Home Team Corners","fixture_id":i,"clv_probability_pp":0.1} for i in range(60)],
        ],
    )
    blockers=report["family_views"]["TEAM_CORNERS"]["blockers"]
    assert "TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_MATERIALIZED" not in blockers
    assert "TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_READY" in blockers


