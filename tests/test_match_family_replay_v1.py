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
