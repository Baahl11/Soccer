import unittest
from sports_edge.multi_market_sport_profile import (
    build_multimarket_sport_profile, describe_three_leg_historical_pattern,
)
from sports_edge.market_context import _fulltime_corner_pair, collect_historical_market_context

TARGET = "2026-11-07T20:00:00+00:00"

def past(fid, home, away, ft, ht, *, kickoff="2026-11-04T18:00:00+00:00", status="FT"):
    if ht is None:
        first = second = None
    else:
        first = {"home": ht[0], "away": ht[1]}
        second = {"home": ft[0]-ht[0], "away": ft[1]-ht[1]}
    return {
        "fixture_id":fid,"home_team_id":home,"away_team_id":away,
        "kickoff":kickoff,"status":status,
        "goals":{"home":ft[0],"away":ft[1]},
        "first_half_goals":first,"second_half_goals":second,
    }


def history():
    return [
        past(10,11,22,(3,1),(1,0)),  # real pattern input test only
        past(11,22,11,(1,0),(0,0)),  # team 11 lost by one => Asian +1 PUSH
        past(12,11,33,(2,1),(1,0)),
        past(13,11,22,(4,3),(2,1),status="AET"),
        past(14,11,22,(1,1),None),    # valid FT, no half-time
        past(15,11,22,(1,0),(2,0)), # impossible half > full => skip period
        past(16,11,22,(2,2),(1,1),kickoff="2026-11-09T18:00:00+00:00"),
        past(10,11,22,(3,1),(1,0)),  # duplicate
    ]


class MultiMarketSportFirstTests(unittest.TestCase):
    def profile(self):
        return build_multimarket_sport_profile(
            history(), team_id=11, target_kickoff=TARGET,
            corners_by_fixture={
                10:{"home":6,"away":3,"period":"FT_90"},
                11:{"home":5,"away":2,"period":"FT_90"},
                12:{"home":4,"away":4,"period":"FT_90"},
                14:{"home":3,"away":2,"period":"NOT_VERIFIED"}, # never use
                15:{"home":4,"away":3,"period":"NOT_VERIFIED"},
                16:{"home":25,"away":18,"period":"FT_90"},   # future => never use
            },
        )

    def test_periods_separated_and_90_min_only_no_leakage(self):
        p=self.profile()
        self.assertEqual(p["evidence_scope"],"PAST_VERIFIED_FT_ONLY")
        self.assertTrue(p["sports_first_no_market_inputs"])
        self.assertFalse(p["bet_eligible"])
        self.assertFalse(p["calibrated_prediction"])
        self.assertEqual(len(p["observations"]),5) # 10,11,12,14,15
        self.assertEqual(p["excluded"]["future_or_invalid"],1)
        self.assertEqual(p["excluded"]["not_ft"],1)
        self.assertEqual(p["excluded"]["duplicate"],1)
        h=p["splits"]["all"]["features"]
        self.assertEqual(h["first_half_goals_for"]["n"],3)
        self.assertEqual(h["second_half_goals_for"]["n"],3)
        self.assertEqual(h["second_half_goals_total"]["over_line_frequency_descriptive"],round(2/3,4))
        self.assertEqual(p["splits"]["all"]["period_goals_share_scored_second_half"],.6)
        self.assertIsNone(h["second_half_goals_total"]["model_probability"])
        self.assertFalse(h["second_half_goals_total"]["minimum_descriptive_sample_met"])
        self.assertEqual(h["total_corners_ft"]["n"],3)
        self.assertIsNone(next(x for x in p["observations"] if x["fixture_id"]==14)["total_corners_ft"])
        self.assertNotIn(13,{x["fixture_id"] for x in p["observations"]})
        self.assertNotIn(16,{x["fixture_id"] for x in p["observations"]})

    def test_away_and_home_splits_have_correct_team_perspective(self):
        p=self.profile()
        v=p["splits"]["away"]
        self.assertEqual(v["matches_with_valid_ft"],1)
        self.assertEqual(v["features"]["goals_for"]["mean"],0)
        self.assertEqual(v["features"]["goals_against"]["mean"],1)
        self.assertEqual(v["features"]["team_corners_ft"]["mean"],2)
        self.assertEqual(v["features"]["opponent_corners_ft"]["mean"],5)

    def test_joint_goals_corners_handicap_counts_use_same_complete_fixture(self):
        p=self.profile()
        asian=describe_three_leg_historical_pattern(
            p,goals_over=1.5,corners_over=6.5,
            team_handicap=1,handicap_settlement="ASIAN",
        )
        threeway=describe_three_leg_historical_pattern(
            p,goals_over=1.5,corners_over=6.5,
            team_handicap=1,handicap_settlement="THREE_WAY",
        )
        self.assertEqual(asian["complete_historical_rows"],3)
        self.assertEqual(asian["observed_joint_hits"],2)
        self.assertEqual(asian["observed_joint_misses"],1)
        self.assertEqual(asian["asian_handicap_leg_push_records"],1)
        self.assertEqual(asian["asian_handicap_push_records"],0)
        self.assertEqual(asian["resolved_without_push"],3)
        self.assertEqual(asian["observed_joint_hit_frequency"],round(2/3,4))
        self.assertEqual(threeway["observed_joint_hits"],2)
        self.assertEqual(threeway["observed_joint_misses"],1)
        self.assertEqual(threeway["observed_joint_hit_frequency"],round(2/3,4))
        self.assertEqual(threeway["asian_handicap_push_records"],0)
        for row in (asian,threeway):
            self.assertFalse(row["eligible_as_joint_model_probability"])
            self.assertFalse(row["bet_eligible"])
            self.assertIsNone(row["price_decimal"])
            self.assertIsNone(row["correlated_sgp_quote"])

    def test_unverified_handicap_rules_not_assumed(self):
        with self.assertRaises(ValueError):
            describe_three_leg_historical_pattern(self.profile(),goals_over=1.5,
                corners_over=6.5,handicap_settlement="UNKNOWN")
        with self.assertRaises(ValueError):
            describe_three_leg_historical_pattern(self.profile(),goals_over=1.5,
                corners_over=6.5,handicap_settlement="ASIAN",venue="invalid")
        with self.assertRaises(ValueError):
            describe_three_leg_historical_pattern(self.profile(),goals_over=2.0,
                corners_over=6.5,handicap_settlement="ASIAN")
        self.assertEqual(
            describe_three_leg_historical_pattern(
              self.profile(),goals_over=1.5,corners_over=6.5,
              handicap_settlement="ASIAN",venue="away",
            )["asian_handicap_leg_push_records"],1)


    def test_asian_push_with_other_legs_winning_is_unresolved_not_a_win(self):
        p=build_multimarket_sport_profile(
            [past(17,11,22,(1,2),(0,1))],
            team_id=11,target_kickoff=TARGET,
            corners_by_fixture={17:{"home":5,"away":4,"period":"FT_90"}},
        )
        asian=describe_three_leg_historical_pattern(
            p, goals_over=1.5, corners_over=6.5,
            team_handicap=1, handicap_settlement="ASIAN",
        )
        three=describe_three_leg_historical_pattern(
            p, goals_over=1.5, corners_over=6.5,
            team_handicap=1, handicap_settlement="THREE_WAY",
        )
        self.assertEqual(asian["asian_handicap_leg_push_records"],1)
        self.assertEqual(asian["asian_handicap_push_records"],1)
        self.assertEqual(asian["observed_joint_hits"],0)
        self.assertEqual(asian["observed_joint_misses"],0)
        self.assertIsNone(asian["observed_joint_hit_frequency"])
        self.assertEqual(three["observed_joint_misses"],1)
        self.assertEqual(three["asian_handicap_push_records"],0)
        self.assertFalse(asian["bet_eligible"])

    def test_missing_corners_are_not_zero_or_fake_joint_probabilities(self):
        p=build_multimarket_sport_profile(
            [past(1,11,22,(3,1),(1,0))],
            team_id=11,target_kickoff=TARGET,
            corners_by_fixture={1:{"home":5,"away":None,"period":"FT_90"}},
        )
        f=p["splits"]["all"]["features"]
        self.assertEqual(f["total_corners_ft"]["n"],0)
        self.assertIsNone(f["total_corners_ft"]["mean"])
        self.assertIsNone(f["total_corners_ft"]["over_line_frequency_descriptive"])
        j=describe_three_leg_historical_pattern(p,goals_over=1.5,corners_over=6.5,
                                                handicap_settlement="THREE_WAY")
        self.assertEqual(j["complete_historical_rows"],0)
        self.assertIsNone(j["observed_joint_hit_frequency"])

    def test_parse_two_side_corners_without_inventing_missing_opponent(self):
        raw={"response":[
          {"team":{"id":11},"statistics":[{"type":"Corner Kicks","value":6}]},
          {"team":{"id":22},"statistics":[{"type":"Corner Kicks","value":3}]},
        ]}
        self.assertEqual(_fulltime_corner_pair(raw,11,22)["period"],"FT_90")
        self.assertEqual(_fulltime_corner_pair(raw,11,22)["away"],3)
        self.assertIsNone(_fulltime_corner_pair(raw,11,33))
        raw["response"][1]["statistics"][0]["value"]={"first_half":1,"second_half":2}
        self.assertIsNone(_fulltime_corner_pair(raw,11,22))

    def test_existing_market_context_adds_profile_without_extra_api_calls(self):
        games=[past(10,11,22,(3,1),(1,0))]
        batch={"screened_fixtures":[{
            "fixture_id":300,"kickoff_utc":TARGET,"match":"A vs B",
            "home_team":{"team_id":11},"away_team":{"team_id":22},
            "multi_competition_history":{"home_fixtures":games,"away_fixtures":[]}
        }],"shortlists":{"merged_fixture_ids":[300]}}
        calls=[]
        def api(endpoint,params):
            calls.append((endpoint,params))
            if endpoint=="fixtures/statistics":
                return {"response":[
                  {"team":{"id":11},"statistics":[{"type":"Corner Kicks","value":6}]},
                  {"team":{"id":22},"statistics":[{"type":"Corner Kicks","value":3}]},
                ]}
            return {"response":[]}
        result=collect_historical_market_context(batch,api,max_fixtures=1)
        x=result["reports"][0]["home"]
        p=x["sport_first_multimarket_profile"]
        self.assertEqual(p["splits"]["all"]["features"]["second_half_goals_total"]["mean"],3)
        self.assertEqual(p["splits"]["all"]["features"]["total_corners_ft"]["mean"],9)
        self.assertEqual(x["example_goals_corners_handicap_pattern"]["ASIAN"]["observed_joint_hits"],1)
        self.assertFalse(result["bet_eligible"])
        self.assertEqual(result["historical_stats_requests"],1)
        self.assertEqual(sum(1 for path,_ in calls if path=="fixtures/statistics"),1)


if __name__=="__main__":
    unittest.main()
