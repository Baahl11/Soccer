"""Customer-readable FE11/SGP player market cards, research-only by contract.

Renders the actual HTML delivered to app-v2 and asserts that missing market
operators, player XI, ticket quotes and SGP joint prices remain unverified.
"""
from __future__ import annotations

from mcp_gateway import subscriber_frontend_v2


def test_player_section_explains_threshold_price_and_joint_market():
    html = subscriber_frontend_v2.render()
    assert "¿Qué significan estas tarjetas?" in html
    assert "Mercados de jugadores · Solo para investigar" in html
    assert "«3 tiros» es la referencia publicada" in html
    assert "La condición exacta" in html
    assert "cuota conjunta verificada" in html
    assert "EN OBSERVACIÓN · NO APOSTAR" in html
    assert "No son recomendaciones" in html


def test_player_card_does_not_mislabel_offer_as_predicted_probability():
    html = subscriber_frontend_v2.render()
    assert "function sgpPlayerMarketName(" in html
    assert "function sgpPlayerQuantity(" in html
    assert "function sgpPlayerShortSelection(" in html
    assert "const reference=quantity!==null?sgpPlayerShortSelection(x)" in html
    assert "Tiros a puerta" in html
    assert "Tiros del jugador" in html
    assert "Jugador goleador" in html
    assert "Cuota decimal individual" in html
    assert "Precio reciente · confirmar" in html
    assert "Precio anterior · reconfirmar" in html
    assert "titularidad y minutos" in html
    assert "reglas de liquidación" in html
    assert "Ver selección original del proveedor" in html
    assert "RECENTE · reconfirmar" not in html


def test_betslip_is_an_unpriced_draft_not_a_bookmaker_quote():
    html = subscriber_frontend_v2.render()
    assert 'id="sgpBuilderSlip"' in html
    assert 'role="status" aria-live="polite"' in html
    assert "Borrador de investigación" in html
    assert "CUOTA COMBINADA: NOT VERIFIED" in html
    assert "No se suman ni multiplican cuotas" in html
    assert "No se envían apuestas" in html
    assert "solo crea un borrador" in html
    assert "x.fixture_id!==row.fixture_id" in html
    assert "sgpSelectedPlayerIndexes.length>=3" in html


def test_responsive_accessible_safe_structural_markup():
    html = subscriber_frontend_v2.render()
    assert 'class="card-grid sgp-player-grid"' in html
    assert "sgp-leg-card.selected" in html
    assert "sgp-guide-steps" in html
    assert "@media(max-width:700px)" in html
    assert 'aria-label="Incluir ' in html
    assert "esc(x.player_label||'Jugador sin identificar')" in html
    assert "esc(x.selection||'Sin selección verificada')" in html
    assert "esc(sgpPlayerShortSelection(x))" in html
    assert "innerHTML=playerRows.length?" in html
    assert "No se inventarán cuotas" in html
