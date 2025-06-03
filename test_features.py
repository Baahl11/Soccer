import unittest
from features import compute_total_goals, compute_corners, compute_cards, compute_btts

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.matches = [
            {"home_goals": 2, "away_goals": 1, "home_corners": 5, "away_corners": 3, "home_yellow_cards": 1, "away_yellow_cards": 2, "home_red_cards": 0, "away_red_cards": 1},
            {"home_goals": 1, "away_goals": 1, "home_corners": 4, "away_corners": 2, "home_yellow_cards": 0, "away_yellow_cards": 1, "home_red_cards": 0, "away_red_cards": 0},
            {"home_goals": 3, "away_goals": 2, "home_corners": 6, "away_corners": 4, "home_yellow_cards": 2, "away_yellow_cards": 0, "home_red_cards": 0, "away_red_cards": 0},
        ]

    def test_compute_total_goals(self):
        home_goals, away_goals = compute_total_goals(self.matches)
        self.assertEqual(home_goals, 6)
        self.assertEqual(away_goals, 4)

    def test_compute_corners(self):
        home_corners, away_corners = compute_corners(self.matches)
        self.assertEqual(home_corners, 15)
        self.assertEqual(away_corners, 9)

    def test_compute_cards(self):
        home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards = compute_cards(self.matches)
        self.assertEqual(home_yellow_cards, 3)
        self.assertEqual(away_yellow_cards, 3)
        self.assertEqual(home_red_cards, 0)
        self.assertEqual(away_red_cards, 1)

    def test_compute_btts(self):
        btts_count = compute_btts(self.matches)
        self.assertEqual(btts_count, 3)

if __name__ == '__main__':
    unittest.main()
