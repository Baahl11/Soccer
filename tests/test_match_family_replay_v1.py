import pytest

from mcp_gateway.match_family_replay_v1 import summarize


@pytest.mark.parametrize("current,full,stored,truncated,state", [
    (1, 1, True, False, "ACEPTADO_EN_REPLAY_DE_FICHA"),
    (0, 1, True, False, "ACEPTADO_FUERA_DE_ULTIMOS_12"),
    (0, 0, True, False, "PERSISTIDO_PREPARTIDO_NO_ACEPTADO"),
    (0, 0, False, False, "SIN_CAMPO_PREPARTIDO_EN_REFRESH_EVENTS"),
    (0, 0, False, True, "ARCHIVO_TRUNCADO_NO_CONCLUYENTE"),
])
def test_replay_states(current, full, stored, truncated, state):
    archive = [{"payload": {"key": {"private": 0.5}}}] if stored else []
    result = summarize("key", [], archive, current, full, truncated)
    assert result["estado"] == state
    assert result["eventos_prepartido_con_campo"] == int(stored)
    assert "private" not in repr(result)


def test_discovery_preserves_first_page_on_second_page_error(monkeypatch):
    from contextlib import nullcontext
    from mcp_gateway import match_family_replay_v1 as mod

    class Connection:
        transaction = staticmethod(nullcontext)

    class Cursor:
        connection = Connection()

    class Timeout(Exception):
        sqlstate = "57014"

    def page(cur, now, anchor):
        start = 100 if anchor is None else anchor[1] - 1
        return [(x, 7, None) for x in range(start, start - mod.PAGE_SIZE, -1)]

    def fields(cur, rows):
        if rows[0][0] != 100:
            raise Timeout()
        return [(100, ["player_shots_intelligence"])]

    monkeypatch.setattr(mod, "candidate_page", page)
    monkeypatch.setattr(mod, "candidate_fields", fields)
    monkeypatch.setattr(mod, "audit_fixture", lambda *_: {
        "familias": {"SHOTS": {"estado": "ACEPTADO_FUERA_DE_ULTIMOS_12"}}})
    result = mod.discover(Cursor(), None)
    assert result["events_scanned"] == mod.PAGE_SIZE
    assert result["next_before_event_id"] == 77
    assert result["scan_failure"] == {"phase": "EVENT_PAYLOAD_KEYS", "reason": "QUERY_TIMEOUT"}
    assert result["families"]["SHOTS"]["positivo"]["fixture_id"] == 7

def test_historical_player_replay_is_separate_and_reads_past_twelve():
    from datetime import datetime, timezone
    from mcp_gateway import match_family_replay_v1 as mod

    events = [{"generated_at": "2026-11-14T01:00:00+00:00",
               "payload": {"corners_intelligence": {}}} for _ in range(12)]
    events.append({"generated_at": "2026-11-14T00:00:00+00:00",
                   "payload": {"player_sot_intelligence": {}}})
    fixture = {"kickoff": "2026-11-14T02:00:00+00:00"}
    now = datetime(2026, 11, 15, tzinfo=timezone.utc)

    def replay(rows, fx, *, now_utc):
        assert rows == [events[-1]] and fx is fixture
        assert now_utc.isoformat() == events[-1]["generated_at"]
        return {"rows": [{"family": "SOT"}]}

    assert mod.historical_player_families(events, fixture, now, replay=replay) == {"SOT"}

def test_player_diagnostics_separate_producer_and_validator_and_do_not_leak():
    from datetime import datetime, timezone
    from mcp_gateway import match_family_replay_v1 as mod

    stamp = "2026-11-14T01:00:00+00:00"
    events = [{"generated_at": stamp, "payload": {
        "player_shots_intelligence": {"status": "NOT_VERIFIED", "reason": "BOTH_STARTING_XI_NOT_CONFIRMED"},
        "player_sot_intelligence": {"status": "LIVE_RESEARCH_SOT_PROPS", "modeled_players": 2},
    }}]
    def replay(rows, fixture, *, now_utc):
        assert "player_shots_intelligence" not in rows[0]["payload"]
        assert now_utc.isoformat() == stamp
        return {"unverified_input_reasons": {"OPPONENT_CONTEXT_NOT_VERIFIED": 2, "private": 1}}
    result = mod.player_diagnostics(events, {}, datetime(2026, 11, 15, tzinfo=timezone.utc), replay)
    assert result["SHOTS"] == {"XI_NOT_CONFIRMED": 1}
    assert result["SOT"] == {"MODEL_OUTPUT_EVENTS": 1, "OPPONENT_CONTEXT_NOT_VERIFIED": 2,
                             "OTHER_VALIDATOR_REASON": 1}


@pytest.mark.parametrize("value, expected", [
    (None, "PLAYER_REGISTRY_TIMESTAMP_MISSING"),
    (" ", "PLAYER_REGISTRY_TIMESTAMP_MISSING"),
    ("bad", "PLAYER_REGISTRY_TIMESTAMP_INVALID"),
    ("2026-11-14T01:00:00", "PLAYER_REGISTRY_TIMESTAMP_INVALID"),
    ("2026-11-14T01:00:01Z", "PLAYER_REGISTRY_TIMESTAMP_AFTER_CAPTURE"),
    ("2026-11-13T19:00:00-06:00", None),
])
def test_registry_timestamp_rejection_breakdown(value, expected):
    from datetime import datetime
    from mcp_gateway.player_prop_sport_first_bridge_v1 import registry_time_blocker

    assert registry_time_blocker(value, datetime.fromisoformat("2026-11-14T01:00:00+00:00")) == expected

def test_fresh_capture_filter_excludes_old_and_boundary_events():
    from datetime import timedelta
    from mcp_gateway import match_family_replay_v1 as mod

    events = [{"generated_at": (mod.FRESH_CAPTURE_AFTER + timedelta(seconds=delta)).isoformat()}
              for delta in (-1, 0, 1)]
    assert mod.fresh_captures(events) == [events[-1]]


def test_panel_controls_isolate_failure_and_do_not_forward_payload():
    from contextlib import nullcontext
    from mcp_gateway import match_family_replay_v1 as mod

    class Connection:
        transaction = staticmethod(nullcontext)

    class Cursor:
        connection = Connection()

    def replay(cur, fixture, now):
        if fixture == 1647367:
            raise RuntimeError("private error")
        return {"eventos_en_ventana": 1, "archivo_truncado": False, "familias": {},
                "payload": {"secret": "private data"}}

    controls = mod.panel_controls(Cursor(), None, replay)
    assert controls["1647367"]["state"] == "DISCOVERY_ERROR"
    assert controls["1545887"]["state"] == "REPLAYED"
    assert "private" not in repr(controls)


def test_corners_evidence_is_pair_specific():
    from mcp_gateway.corners_evidence_inventory_v1 import bucket
    snap = {"teams": [{"team_id": 1, "formation": "4-4-2"},
                      {"team_id": 2, "formation": "4-2-3-1"}]}
    assert bucket(1, 2, [snap], True) == "paired"
    assert bucket(1, 2, [snap], False) == "formation_only"
    assert bucket(1, 3, [snap], True) == "official_ft_only"
    assert bucket(1, 2, [], False) == "neither"


def test_reuse_only_exact_regulation_provider_corners_and_reject_conflicts():
    from datetime import datetime
    from copy import deepcopy
    from mcp_gateway.corners_evidence_inventory_v1 import verified_observation_corners

    stamp = "2026-11-15T14:00:00+00:00"
    row = {"event_type": "FIXTURE_OBSERVATION_RESEARCH", "stage": "FIXTURE_OBSERVATION",
           "bet_eligible": False,
           "fixture": {"fixture_id": 5, "league_id": 6, "home_team_id": 1,
                       "away_team_id": 2, "status": "FT", "kickoff": stamp},
           "sporting": {"collection_scope": "POSTGAME_OBSERVATION", "fixture_statistics": [
               {"team": {"id": 2}, "statistics": [{"type": "Corner Kicks", "value": 0}]},
               {"team": {"id": 1}, "statistics": [{"type": "Corner Kicks", "value": 7}]},
           ]}}
    def lookup(*rows):
        return verified_observation_corners(rows, 5, datetime.fromisoformat(stamp), 6, 1, 2)
    assert lookup(row, row) == (7, 0)
    for field, bad in [("status", "AET"), ("fixture_id", 7), ("kickoff", "2026-11-15T15:00:00Z"),
                       ("away_team_id", 3)]:
        other = deepcopy(row)
        other["fixture"][field] = bad
        assert lookup(other) is None
    other = deepcopy(row)
    other["sporting"]["fixture_statistics"][0]["statistics"][0]["value"] = 2
    assert lookup(row, other) is None
    other["sporting"]["fixture_statistics"][0]["statistics"][0]["value"] = None
    assert lookup(other) is None
