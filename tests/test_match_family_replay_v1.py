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


def test_discovery_checks_each_family_and_reuses_fixture_audit(monkeypatch):
    from mcp_gateway import match_family_replay_v1 as mod

    calls = []
    monkeypatch.setattr(mod, "candidate_ids", lambda *_: {k: [7] for k in mod.FIELDS.values()})

    def audit(cur, fixture_id, now):
        calls.append(fixture_id)
        return {"familias": {"SHOTS": {"estado": "ACEPTADO_FUERA_DE_ULTIMOS_12"},
                             "SOT": {"estado": "PERSISTIDO_PREPARTIDO_NO_ACEPTADO"}}}

    monkeypatch.setattr(mod, "audit_fixture", audit)
    rows = mod.discover(None, None)["families"]
    assert calls == [7]
    assert rows["SHOTS"]["positivo"] == {"fixture_id": 7, "estado": "ACEPTADO_FUERA_DE_ULTIMOS_12"}
    assert rows["SOT"]["positivo"] is None
    assert rows["SOT"]["revisados"] == 1
    assert rows["CORNERS_FT"]["no_concluyentes"] == 1
