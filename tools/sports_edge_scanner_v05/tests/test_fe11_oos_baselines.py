"""FE11 OOS research baselines: immutable chronological holdout, no market."""
import copy
from datetime import datetime, timedelta, timezone
import unittest

from sports_edge.fe11_oos_dataset import build_fe11_oos_dataset, CORNER_SOURCE
from sports_edge.fe11_oos_baselines import evaluate_fe11_frozen_baselines, VERSION

START = datetime(2025, 10, 1, 15, 0, tzinfo=timezone.utc)


def fixture_sample(n=175):
    fixtures, corners = [], {}
    for i in range(n):
        dt = START + timedelta(days=i)
        # Deliberately correlated synthetic fixture outcomes: these are TEST
        # fixtures, not actual teams, market quotes or model accuracy.
        hg, ag, hcorner, acorner = [
            (3, 1, 6, 5), (0, 0, 2, 1), (1, 2, 5, 3), (2, 0, 2, 2),
        ][i % 4]
        ht_h, ht_a = min(hg, 1), min(ag, 1)
        fixtures.append({
            "fixture_id": 30000+i, "home_team_id": 11, "away_team_id": 22,
            "league_id": 39, "season": 2025, "status": "FT",
            "kickoff": dt.isoformat(),
            "source_available_at_utc": (dt+timedelta(hours=2)).isoformat(),
            "goals": {"home": hg, "away": ag},
            "first_half_goals": {"home": ht_h, "away": ht_a},
            "second_half_goals": {"home": hg-ht_h, "away": ag-ht_a},
        })
        corners[30000+i] = {
            "period": "FT_90", "source": CORNER_SOURCE,
            "home": hcorner, "away": acorner,
            "source_available_at_utc": (dt+timedelta(hours=3)).isoformat(),
        }
    return fixtures, corners


def data(n=175):
    fixtures, corners = fixture_sample(n)
    return build_fe11_oos_dataset(
        fixtures, corners,
        research_cutoff_utc=(START+timedelta(days=n+1)).isoformat(),
    )


def study(d, **overrides):
    args = dict(cutoff_utc=(START+timedelta(days=130)).isoformat(),
                selection="home_three_way_plus_one_joint",
                min_training=100, min_holdout=30)
    args.update(overrides)
    return evaluate_fe11_frozen_baselines(d, **args)


class FrozenHoldoutBaselineTests(unittest.TestCase):
    def test_two_baselines_are_measured_on_same_frozen_holdout(self):
        result = study(data())
        self.assertEqual(result["version"], VERSION)
        self.assertEqual(result["status"], "BASELINES_EVALUATED_RESEARCH_ONLY")
        self.assertGreaterEqual(result["n_training"], 100)
        self.assertGreaterEqual(result["n_holdout"], 30)
        self.assertTrue(result["frozen_fit"])
        self.assertFalse(result["feature_adjusted"])
        self.assertFalse(result["bet_eligible"])
        self.assertFalse(result["production_promotion_allowed"])
        self.assertIsNone(result["calibrated_joint_probability"])
        self.assertIsNone(result["bookmaker_ticket_quote"])
        self.assertIsNone(result["estimated_ev"])
        self.assertNotEqual(result["naive_independent_estimate"],
                            result["same_fixture_joint_estimate"])
        self.assertTrue(0 <= result["holdout_observed_joint_win_rate"] <= 1)
        self.assertLess(result["metrics"]["same_fixture_joint"]["brier"], 1)
        self.assertLess(result["metrics"]["independent"]["log_loss"], 10)

    def test_embargo_and_published_labels_prevent_future_leakage(self):
        d = data()
        r = study(d)
        self.assertLess(max(r["train_fixture_ids"]), 30127)
        self.assertGreaterEqual(min(r["holdout_fixture_ids"]), 30130)
        self.assertFalse(set(r["train_fixture_ids"]) & set(r["holdout_fixture_ids"]))
        # Result known only AFTER fold cutoff cannot enter frozen training.
        changed = copy.deepcopy(d)
        t = next(x for x in changed["rows"] if x["fixture_id"] == 30028)
        t["target_source_available_at_utc"] = (
            START + timedelta(days=132)).isoformat()
        after = study(changed)
        self.assertNotIn(30028, after["train_fixture_ids"])
        self.assertEqual(r["n_training"]-1, after["n_training"])

    def test_baseline_is_never_promoted_when_sample_insufficient(self):
        r = study(data(), min_training=500)
        self.assertEqual(r["status"], "NOT_ENOUGH_OOS_DATA")
        self.assertIsNone(r["metrics"])
        self.assertIsNone(r["calibrated_joint_probability"])
        self.assertFalse(r["bet_eligible"])
        self.assertNotIn("same_fixture_joint_estimate", r)

    def test_selection_restricted_to_validated_threeway_no_asian_push(self):
        with self.assertRaisesRegex(ValueError, "FE11_OOS_CONFIGURATION_UNTRUSTED"):
            study(data(), selection="home_asian_plus_one_joint")

    def test_duplicate_fixture_and_modified_joint_label_are_rejected(self):
        d = data()
        d["rows"].append(copy.deepcopy(d["rows"][-1]))
        with self.assertRaisesRegex(ValueError, "FE11_DUPLICATE_FIXTURE"):
            study(d)
        d = data()
        d["rows"][-1]["labels"]["home_three_way_plus_one_joint"] = "PUSH"
        with self.assertRaisesRegex(ValueError, "FE11_LABELS_INCONSISTENT"):
            study(d)

    def test_market_feature_contamination_and_missing_source_rejected(self):
        d = data()
        d["rows"][0]["features"]["bookmaker_odds"] = 2.3
        with self.assertRaisesRegex(ValueError, "MARKET_OR_TARGET_LEAKAGE"):
            study(d)
        d = data()
        d["rows"][0]["target_corners_available_at_utc"] = None
        with self.assertRaisesRegex(ValueError, "FE11_ROW_IDENTITY_OR_PROVENANCE"):
            study(d)

    def test_cannot_read_snapshot_after_train_cutoff_or_missing_provenance(self):
        d = data()
        d["research_cutoff_utc"] = (START+timedelta(days=120)).isoformat()
        with self.assertRaisesRegex(ValueError, "FE11_CUTOFF_AFTER_RESEARCH_SNAPSHOT"):
            study(d)
        d = data()
        d["source_provenance_required"] = False
        with self.assertRaisesRegex(ValueError, "FE11_OOS_CONFIGURATION_UNTRUSTED"):
            study(d)

    def test_league_filter_fails_closed_when_no_such_league(self):
        r = study(data(), league_id=999)
        self.assertEqual(r["status"], "NOT_ENOUGH_OOS_DATA")
        self.assertEqual(r["n_training"], 0)
        self.assertEqual(r["n_holdout"], 0)


if __name__ == "__main__":
    unittest.main()
