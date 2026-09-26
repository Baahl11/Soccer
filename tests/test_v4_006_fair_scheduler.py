from mcp_gateway import fair_scheduler


def _item(category: str, index: int, league_id: int | None = None) -> dict:
    return {
        "fx": {
            "fixture_id": 1000 + index,
            "league_id": league_id if league_id is not None else 39,
        },
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



def test_v4_006_unseen_slots_round_robin_across_leagues():
    items = []
    items += [_item("unseen", i, league_id=39) for i in range(6)]
    items += [_item("unseen", 10 + i, league_id=140) for i in range(3)]
    items += [_item("unseen", 20 + i, league_id=78) for i in range(3)]

    selected, _, metrics = fair_scheduler.fair_order(items, 6)
    leagues = [item["fx"]["league_id"] for item in selected]

    assert set(leagues[:3]) == {39, 140, 78}
    assert metrics["eligible_unique_leagues"] == 3
    assert metrics["planned_unique_leagues"] == 3
    assert metrics["schema_version"] == "1.1.0"
    assert "LEAGUE_ROUND_ROBIN" in metrics["policy"]


def test_v4_006_actionable_priority_order_is_not_rewritten_by_league_round_robin():
    first = _item("actionable", 1, league_id=39)
    second = _item("actionable", 2, league_id=140)
    first["priority"] = (0, 0, 0, 0, 0, 0, 0, 1, 0)
    second["priority"] = (0, 1, 0, 0, 0, 0, 0, 2, 0)

    selected, _, _ = fair_scheduler.fair_order([second, first], 2)

    assert [item["fx"]["fixture_id"] for item in selected] == [
        first["fx"]["fixture_id"],
        second["fx"]["fixture_id"],
    ]


def test_v4_006_coverage_catchup_prioritizes_unseen_when_urgent_load_is_low():
    items = []
    # Three urgent shortlist fixtures should all remain ahead of non-urgent
    # repeats, while the large unseen backlog gets most of the early prefix.
    for i in range(3):
        item = _item("actionable", i, league_id=39 + i)
        item["stage"] = "T-20"
        items.append(item)
    for i in range(3, 43):
        item = _item("actionable", i, league_id=39)
        item["stage"] = "EARLY_RESEARCH"
        items.append(item)
    items += [_item("unseen", 100 + i, league_id=100 + (i % 25)) for i in range(120)]

    selected, _, metrics = fair_scheduler.fair_order(items, 48)
    prefix = selected[:9]
    categories = [item["fairness_category"] for item in prefix]
    urgent_ids = {
        1000 + i
        for i in range(3)
    }
    selected_actionable_ids = {
        item["fx"]["fixture_id"]
        for item in prefix
        if item["fairness_category"] == "actionable"
    }

    assert metrics["schema_version"] == "1.2.0"
    assert metrics["scheduling_mode"] == "COVERAGE_CATCHUP"
    assert metrics["effective_weights_pct"]["unseen"] == 60
    assert metrics["urgent_actionable_count"] == 3
    assert categories.count("unseen") >= 5
    assert urgent_ids.issubset(selected_actionable_ids)


def test_v4_006_coverage_catchup_disables_when_urgent_queue_is_large():
    items = []
    for i in range(20):
        item = _item("actionable", i, league_id=39)
        item["stage"] = "T-20"
        items.append(item)
    items += [_item("unseen", 100 + i, league_id=100 + (i % 25)) for i in range(120)]

    _, _, metrics = fair_scheduler.fair_order(items, 48)

    assert metrics["scheduling_mode"] == "NORMAL_WEIGHTED_FAIR"
    assert metrics["effective_weights_pct"] == fair_scheduler.CATEGORY_WEIGHTS
