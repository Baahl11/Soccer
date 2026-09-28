from pathlib import Path

path = Path("mcp_gateway/player_props_clv_postgres_v4.py")
text = path.read_text(encoding="utf-8")

old = '''                "signal_rows": 0,
                "probability_failure_reasons": {},
'''
new = '''                "signal_rows": 0,
                "sidecar_telemetry_event_rows": 0,
                "sidecar_observed_event_rows": 0,
                "sidecar_observed_market_rows": 0,
                "sidecar_kept_event_rows": 0,
                "sidecar_kept_market_rows": 0,
                "sidecar_dropped_market_rows": 0,
                "sidecar_capture_status": "NO_V200_SIDECAR_TELEMETRY",
                "probability_failure_reasons": {},
'''
assert old in text, "family diagnostics initializer missing"
text = text.replace(old, new, 1)

old = '''        diag["eligible_event_rows"] += 1
        for family, config in FAMILY_CONFIG.items():
'''
new = '''        diag["eligible_event_rows"] += 1
        event_market = event.get("market") if isinstance(event.get("market"), dict) else {}
        observed_sidecar_counts = event_market.get("observed_player_prop_subfamily_counts")
        kept_sidecar_counts = event_market.get("kept_player_prop_subfamily_counts")
        sidecar_telemetry_available = (
            isinstance(observed_sidecar_counts, dict)
            and isinstance(kept_sidecar_counts, dict)
        )
        for family, config in FAMILY_CONFIG.items():
'''
assert old in text, "eligible event loop marker missing"
text = text.replace(old, new, 1)

old = '''            })
            intel = event.get(config["intel_key"])
'''
new = '''            })
            if sidecar_telemetry_available:
                try:
                    observed_count = max(0, int(observed_sidecar_counts.get(family) or 0))
                except (TypeError, ValueError):
                    observed_count = 0
                try:
                    kept_count = max(0, int(kept_sidecar_counts.get(family) or 0))
                except (TypeError, ValueError):
                    kept_count = 0
                family_diag["sidecar_telemetry_event_rows"] += 1
                family_diag["sidecar_observed_market_rows"] += observed_count
                family_diag["sidecar_kept_market_rows"] += kept_count
                family_diag["sidecar_dropped_market_rows"] += max(0, observed_count - kept_count)
                if observed_count > 0:
                    family_diag["sidecar_observed_event_rows"] += 1
                if kept_count > 0:
                    family_diag["sidecar_kept_event_rows"] += 1

            intel = event.get(config["intel_key"])
'''
assert old in text, "post family initializer marker missing"
text = text.replace(old, new, 1)

old = '''                    })
    return signals


def _snapshot_values'''
new = '''                    })

    for family_diag in families_diag.values():
        telemetry_rows = int(family_diag.get("sidecar_telemetry_event_rows") or 0)
        observed_rows = int(family_diag.get("sidecar_observed_market_rows") or 0)
        kept_rows = int(family_diag.get("sidecar_kept_market_rows") or 0)
        dropped_rows = int(family_diag.get("sidecar_dropped_market_rows") or 0)
        if telemetry_rows <= 0:
            status = "NO_V200_SIDECAR_TELEMETRY"
        elif observed_rows <= 0:
            status = "NOT_OBSERVED_IN_PAID_ODDS"
        elif kept_rows <= 0:
            status = "OBSERVED_BUT_NOT_KEPT"
        elif dropped_rows > 0:
            status = "OBSERVED_AND_PARTIALLY_KEPT"
        else:
            status = "OBSERVED_AND_KEPT"
        family_diag["sidecar_capture_status"] = status
    return signals


def _snapshot_values'''
assert old in text, "extract signals return marker missing"
text = text.replace(old, new, 1)

old = '''            "STRICTLY_LATER PREKICKOFF PROVIDER UPDATE; SAME BOOK PREFERRED; "
            "ONE-WAY MARKETS TRACK PRICE CLV WITHOUT PRETENDING TO BE DEVIGGED; "
'''
new = '''            "STRICTLY_LATER PREKICKOFF PROVIDER UPDATE; SAME BOOK PREFERRED; "
            "V200 SIDECAR OBSERVED/KEPT/DROPPED CAPTURE TELEMETRY IS AUDIT-ONLY; "
            "ONE-WAY MARKETS TRACK PRICE CLV WITHOUT PRETENDING TO BE DEVIGGED; "
'''
assert old in text, "report policy marker missing"
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

Path("tests/test_player_props_capture_audit_v201.py").write_text('''from mcp_gateway import player_props_clv_postgres_v4 as clv


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
''', encoding="utf-8")
