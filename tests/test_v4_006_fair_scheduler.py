from mcp_gateway import fair_scheduler


def _item(category: str, index: int) -> dict:
    return {
        "fx": {"fixture_id": 1000 + index},
        "stage": "T-40",
        "tier": "A",
        "priority": (0, 1, 0, 0, 0, 0, 0, index, 0),
        "fairness_category": category,
        "fairness_state": {
            "deep_dive_count": 0 if category == "unseen" else 1,
            "last_deep_dive_at": f"2026-09-23T00:{index:02d}:00+00:00",
        },
    }


def test_v4_006_weighted_slots_exact_20():
    items = []
    items += [_item("actionable", i) for i in range(20)]
    items += [_item("unseen", 20 + i) for i in range(20)]
    items += [_item("exploratory", 40 + i) for i in range(20)]

    selected, deferred, metrics = fair_scheduler.fair_order(items, 20)

    counts = metrics["planned_slot_counts"]
    assert len(selected) == 20
    assert len(deferred) == 40
    assert counts["actionable"] == 12
    assert counts["unseen"] == 5
    assert counts["exploratory"] == 3


def test_v4_006_prefix_remains_mixed_under_early_budget_stop():
    items = []
    items += [_item("actionable", i) for i in range(20)]
    items += [_item("unseen", 20 + i) for i in range(20)]
    items += [_item("exploratory", 40 + i) for i in range(20)]

    selected, _, _ = fair_scheduler.fair_order(items, 20)
    prefix = selected[:10]
    categories = [item["fairness_category"] for item in prefix]

    assert "actionable" in categories
    assert "unseen" in categories
    assert "exploratory" in categories
    assert categories.count("actionable") >= 5


def test_v4_006_empty_category_reallocates_slots():
    items = [_item("unseen", i) for i in range(10)]
    selected, deferred, metrics = fair_scheduler.fair_order(items, 6)

    assert len(selected) == 6
    assert len(deferred) == 4
    assert metrics["planned_slot_counts"]["unseen"] == 6
    assert metrics["planned_slot_counts"]["actionable"] == 0
    assert metrics["planned_slot_counts"]["exploratory"] == 0


def test_v4_006_category_semantics():
    assert fair_scheduler.category(True, {"deep_dive_count": 4}) == "actionable"
    assert fair_scheduler.category(False, {"deep_dive_count": 0}) == "unseen"
    assert fair_scheduler.category(False, {"deep_dive_count": 2}) == "exploratory"
