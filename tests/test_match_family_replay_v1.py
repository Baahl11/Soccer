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
