from mcp_gateway import player_props_clv_postgres_v4 as clv


def _event_with_v200_sidecar():
    return {
        "fixture_id": 201,
        "generated_at": "2026-09-28T20:00:00+00:00",
        "stage": "T-20",
        "kickoff": "2026-09-28T20:30:00+00:00",
        "event_payload": {
            "stage": "T-20",
            "fixture": {
                "fixture_id": 201,
                "kickoff": "2026-09-28T20:30:00+00:00",
            },
            "market": {
                "observed_player_prop_subfamily_counts": {
                    "SHOTS": 2,
                    "PLAYER_CARDS": 1,
                },
                "kept_player_prop_subfamily_counts": {
                    "SHOTS": 1,
                    "PLAYER_CARDS": 1,
                },
                "research_cards_props_markets": [],
            },
        },
    }


def test_v201_capture_audit_distinguishes_observed_kept_dropped_and_not_offered():
    diagnostics = {}
    signals = clv.extract_shadow_signals([_event_with_v200_sidecar()], diagnostics)
    assert signals == []

    shots = diagnostics["families"]["SHOTS"]
    assert shots["sidecar_telemetry_event_rows"] == 1
    assert shots["sidecar_observed_event_rows"] == 1
    assert shots["sidecar_observed_market_rows"] == 2
    assert shots["sidecar_kept_event_rows"] == 1
    assert shots["sidecar_kept_market_rows"] == 1
    assert shots["sidecar_dropped_market_rows"] == 1
    assert shots["sidecar_capture_status"] == "OBSERVED_AND_PARTIALLY_KEPT"

    cards = diagnostics["families"]["PLAYER_CARDS"]
    assert cards["sidecar_observed_market_rows"] == 1
    assert cards["sidecar_kept_market_rows"] == 1
    assert cards["sidecar_dropped_market_rows"] == 0
    assert cards["sidecar_capture_status"] == "OBSERVED_AND_KEPT"

    sot = diagnostics["families"]["SOT"]
    assert sot["sidecar_telemetry_event_rows"] == 1
    assert sot["sidecar_observed_market_rows"] == 0
    assert sot["sidecar_kept_market_rows"] == 0
    assert sot["sidecar_capture_status"] == "NOT_OBSERVED_IN_PAID_ODDS"


def test_v201_capture_audit_keeps_model_namespace_and_research_only_scope():
    assert clv.MODEL_VERSION == "SOCCER_PLAYER_PROPS_TRUE_CLV_V4_1.3.0"
    assert callable(clv.build_from_postgres)
