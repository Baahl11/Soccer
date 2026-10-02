from __future__ import annotations

import unittest

from mcp_gateway.player_props_shots_anchor_patch_v4 import _merge_oldest_shots, _shots_key


def _shot(
    *,
    fixture_id: int = 1001,
    player_id: int = 10,
    side: str = "OVER",
    line: float = 2.5,
    signal_at: str = "2026-10-02T12:00:00+00:00",
) -> dict:
    return {
        "fixture_id": fixture_id,
        "market_family": "SHOTS",
        "player_id": player_id,
        "side": side,
        "selection": side,
        "line": line,
        "signal_timestamp": signal_at,
    }


class PlayerPropsShotsAnchorPatchTests(unittest.TestCase):
    def test_oldest_exact_signal_replaces_later_recycled_tick(self) -> None:
        original = [
            _shot(signal_at="2026-10-02T12:20:00+00:00"),
            _shot(signal_at="2026-10-02T12:10:00+00:00"),
        ]
        recent = [_shot(signal_at="2026-10-02T11:50:00+00:00")]

        merged, diag = _merge_oldest_shots(original, recent)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["signal_timestamp"], "2026-10-02T11:50:00+00:00")
        self.assertEqual(diag["shots_duplicates_collapsed"], 1)
        self.assertEqual(diag["older_anchors_replaced"], 1)

    def test_same_line_different_players_remain_distinct(self) -> None:
        original = [_shot(player_id=10), _shot(player_id=11)]

        merged, diag = _merge_oldest_shots(original, [])

        self.assertEqual(len(merged), 2)
        self.assertEqual(diag["unique_shots_instruments"], 2)
        self.assertNotEqual(_shots_key(merged[0]), _shots_key(merged[1]))

    def test_same_player_different_line_or_side_remains_distinct(self) -> None:
        original = [
            _shot(line=1.5, side="OVER"),
            _shot(line=2.5, side="OVER"),
            _shot(line=2.5, side="UNDER"),
        ]

        merged, diag = _merge_oldest_shots(original, [])

        self.assertEqual(len(merged), 3)
        self.assertEqual(diag["unique_shots_instruments"], 3)

    def test_recent_only_instrument_does_not_expand_canonical_cohort(self) -> None:
        original = [_shot(player_id=10)]
        recent = [_shot(player_id=99, signal_at="2026-10-02T11:00:00+00:00")]

        merged, diag = _merge_oldest_shots(original, recent)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["player_id"], 10)
        self.assertEqual(diag["recent_matching_instruments"], 0)
        self.assertEqual(diag["older_anchors_replaced"], 0)

    def test_non_shots_rows_are_preserved(self) -> None:
        non_shot = {
            "fixture_id": 1001,
            "market_family": "SOT",
            "player_id": 10,
            "side": "OVER",
            "line": 0.5,
            "signal_timestamp": "2026-10-02T12:00:00+00:00",
        }
        original = [non_shot, _shot()]

        merged, _ = _merge_oldest_shots(original, [])

        self.assertIn(non_shot, merged)
        self.assertLessEqual(len(merged), len(original))


if __name__ == "__main__":
    unittest.main()
