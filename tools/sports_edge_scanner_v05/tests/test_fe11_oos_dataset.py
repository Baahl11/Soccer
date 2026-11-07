"""Offline FE11 Phase 2: time provenance, corner integrity and builder settlement."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from sports_edge.fe11_oos_dataset import (
    VERSION, CORNER_SOURCE, build_fe11_oos_dataset, chronological_holdout,
)

START = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)


def sample(n=28):
    fixtures, corners = [], {}
    for i in range(n):
        kickoff = START + timedelta(days=i)
        home, away = (11, 22) if i % 2 == 0 else (22, 11)
        ft = (2, 1)
        if i == 25:
            ft = (1, 0)  # Away +1 Asian leg PUSH; goal Over 1.5 loses.
        if i == 26:
            ft = (2, 1)  # Away +1 Asian leg PUSH; other legs both win.
        if i == 27:
            ft = (0, 2)  # Home +1 loses.
        fixtures.append({
            "fixture_id": 1000+i, "home_team_id": home, "away_team_id": away,
            "league_id": 39, "season": 2026, "status": "FT",
            "kickoff": kickoff.isoformat(),
            "source_available_at_utc": (kickoff+timedelta(hours=2)).isoformat(),
            "goals": {"home": ft[0], "away": ft[1]},
            "first_half_goals": {"home": 1 if ft[0] > 0 else 0, "away": 0},
            "second_half_goals": {"home": ft[0]-(1 if ft[0]>0 else 0),
                                  "away": ft[1]},
        })
        corners[1000+i] = {
            "home": 5, "away": 4, "period": "FT_90",
            "source": CORNER_SOURCE,
            "source_available_at_utc": (kickoff+timedelta(hours=3)).isoformat(),
        }
    return fixtures, corners


def dataset(fixtures, corners):
    return build_fe11_oos_dataset(
        fixtures, corners,
        research_cutoff_utc=(START+timedelta(days=29)).isoformat())


class Fe11OosDatasetTests(unittest.TestCase):
    def test_rolling_asof_dataset_is_research_only(self):
        games, cs = sample()
        d = dataset(games, cs)
        self.assertEqual(d["version"], VERSION)
        self.assertGreater(d["eligible_rows"], 0)
        self.assertFalse(d["bet_eligible"])
        self.assertFalse(d["market_inputs_used"])
        self.assertIsNone(d["calibrated_joint_probability"])
        self.assertEqual(d["rows"][0]["fixture_id"], 1008)
        first = d["rows"][0]
        self.assertEqual(first["features"]["home_prior_ft_all"], 8)
        self.assertEqual(first["features"]["away_prior_ft_all"], 8)
        self.assertEqual(first["features"]["home_prior_ft_venue"], 4)
        self.assertEqual(first["features"]["away_prior_ft_venue"], 4)
        self.assertEqual(first["features"]["home_prior_corners_venue"], 4)
        self.assertNotIn("odds", str(first["features"]).lower())

    def test_builder_push_only_when_other_legs_survive(self):
        games, cs = sample()
        rows = {r["fixture_id"]:r for r in dataset(games, cs)["rows"]}
        # 25 has FT 1-0, the away +1 Asian leg pushes, but goals Over 1.5 loses.
        self.assertEqual(rows[1025]["labels"]["away_asian_plus_one_leg"], "PUSH")
        self.assertEqual(rows[1025]["labels"]["away_asian_plus_one_joint"], "LOSS")
        # 26 has FT 2-1, total goals and corners hit, and Asian away +1 pushes.
        self.assertEqual(rows[1026]["labels"]["away_asian_plus_one_leg"], "PUSH")
        self.assertEqual(rows[1026]["labels"]["away_asian_plus_one_joint"],
                         "PUSH_REQUIRES_BOOKMAKER_RULE")
        self.assertEqual(rows[1026]["labels"]["away_three_way_plus_one_joint"], "LOSS")
        self.assertEqual(rows[1027]["labels"]["home_asian_plus_one_joint"], "LOSS")

    def test_late_backfill_never_enters_pregame_features(self):
        games, cs = sample()
        base = dataset(games, cs)
        target = next(r for r in base["rows"] if r["fixture_id"] == 1026)
        past = dict(games[4])
        past.update({
            "fixture_id": 9000, "goals": {"home": 20, "away": 0},
            "first_half_goals": {"home": 20, "away": 0},
            "second_half_goals": {"home": 0, "away": 0},
            "source_available_at_utc": (START+timedelta(days=28)).isoformat(),
        })
        games.append(past)
        cs[9000] = dict(cs[1004], source_available_at_utc=
                        (START+timedelta(days=28, hours=1)).isoformat())
        changed = dataset(games, cs)
        other = next(r for r in changed["rows"] if r["fixture_id"] == 1026)
        self.assertEqual(target["features"], other["features"])
        self.assertEqual(target["labels"], other["labels"])

    def test_missing_corner_or_invalid_time_is_not_implicitly_zero(self):
        games, cs = sample()
        cs[1026]["away"] = None
        d = dataset(games, cs)
        self.assertNotIn(1026, [r["fixture_id"] for r in d["rows"]])
        self.assertGreater(d["excluded"]["missing_or_unavailable_corners"], 0)
        cs[1026]["away"] = 4
        cs[1026]["period"] = "NOT_VERIFIED"
        self.assertNotIn(1026, [r["fixture_id"] for r in dataset(games, cs)["rows"]])

    def test_extra_time_future_leakage_duplicate_and_inconsistent_periods(self):
        games, cs = sample()
        games.append(dict(games[1]))  # repeated fixture
        aet = dict(games[2], fixture_id=9001, status="AET")
        future = dict(games[2], fixture_id=9002,
                      kickoff=(START+timedelta(days=32)).isoformat(),
                      source_available_at_utc=(START+timedelta(days=32,hours=3)).isoformat())
        contradictory = dict(games[2], fixture_id=9003,
                             first_half_goals={"home": 4, "away": 0})
        games.extend([aet, future, contradictory])
        d = dataset(games, cs)
        self.assertEqual(d["excluded"]["duplicate_fixture"], 1)
        self.assertGreaterEqual(d["excluded"]["invalid_or_unavailable_fixture"], 3)

    def test_market_input_contamination_rejected(self):
        games, cs = sample()
        games[3]["odds"] = {"home": 2.0}
        with self.assertRaisesRegex(ValueError, "MARKET_DATA_CONTAMINATES_SPORT_FIRST"):
            dataset(games, cs)

    def test_chronological_fold_requires_published_labels_and_embargo(self):
        games, cs = sample()
        d = dataset(games, cs)
        fold = chronological_holdout(
            d, cutoff_utc=(START+timedelta(days=24)).isoformat(),
            embargo_hours=72)
        self.assertFalse(fold["bet_eligible"])
        self.assertEqual(fold["calibration_status"], "NOT_TRAINED")
        self.assertTrue(fold["training_fixture_ids"])
        self.assertTrue(fold["holdout_fixture_ids"])
        self.assertFalse(set(fold["training_fixture_ids"]) & set(fold["holdout_fixture_ids"]))
        self.assertTrue(all(fid < 1021 for fid in fold["training_fixture_ids"]))
        self.assertTrue(all(fid >= 1024 for fid in fold["holdout_fixture_ids"]))


if __name__ == "__main__":
    unittest.main()
