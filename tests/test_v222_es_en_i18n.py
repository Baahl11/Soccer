from __future__ import annotations

from mcp_gateway import subscriber_app_v4, subscriber_i18n_v4


def test_i18n_catalog_supports_en_and_es():
    catalog = subscriber_i18n_v4.public_catalog()
    assert catalog["supported_locales"] == ["en", "es"]
    assert catalog["translations"]["en"]["hero_title"] == "Signals without the noise."
    assert catalog["translations"]["es"]["hero_title"] == "Señales sin ruido."
    assert catalog["market_labels"]["es"]["BTTS"] == "Ambos anotan"
    assert catalog["status_labels"]["es"]["WAIT_PRICE"] == "Esperando precio"


def test_subscriber_app_injects_locale_switch_and_bilingual_copy(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    rendered = subscriber_app_v4._app_html()
    assert "SOCCER_SUBSCRIBER_I18N_V4" in rendered
    assert 'id="langEn"' in rendered
    assert 'id="langEs"' in rendered
    assert "Señales sin ruido." in rendered
    assert "Signals without the noise." in rendered
    assert "soccer_edge_locale" in rendered
    assert "navigator.language" in rendered


def test_i18n_adds_locale_aware_metadata_without_touching_product_payload(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    rendered = subscriber_app_v4._app_html()
    assert 'name="description"' in rendered
    assert 'property="og:title"' in rendered
    assert "Modelo vs Mercado" in rendered
    assert "Model vs Market" in rendered

    product = {
        "views": {
            "todays_slate": {"total": 1, "rows": [{"fixture_id": 1, "market_family": "FT_TOTALS", "status": "WAIT_PRICE"}]},
            "waiting_for_price": {"total": 1, "rows": []},
            "waiting_for_xi": {"total": 0, "rows": []},
            "control_tower": {},
        }
    }
    payload = subscriber_app_v4.build_subscriber_payload(product, subscriber_app_v4.anonymous_entitlement())
    row = payload["public"]["verified_slate"]["rows"][0]
    assert row["market_family"] == "FT_TOTALS"
    assert row["status"] == "WAIT_PRICE"


def test_i18n_layer_is_presentation_only():
    source = open(subscriber_i18n_v4.__file__, "r", encoding="utf-8").read().lower()
    assert "api-football" not in source
    assert "provider_requests" not in source
    assert "decision_weight" not in source
    assert "production_promotion_allowed" not in source
