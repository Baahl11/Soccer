import json
from typing import Any

from mcp_gateway import persistence as base
from mcp_gateway import feature_snapshot_v4


def persist_tick(tick: dict[str, Any]) -> bool:
    persisted = base.persist_tick(tick)
    if not persisted:
        return False

    with base._connect() as conn:
        with conn.cursor() as cur:
            for event in tick.get("events") or []:
                fx = event.get("fixture") or {}
                fixture_id = fx.get("fixture_id")
                raw = event.get("raw_projection")
                decision = event.get("market_decision") or {}
                best = decision.get("best_decision") or {}

                if fixture_id and event.get("event_type") == "SOCCER_REFRESH" and event.get("stage") != "POSTGAME":
                    snapshot = feature_snapshot_v4.build(tick, event)
                    validation_errors = feature_snapshot_v4.validate(snapshot)
                    if not validation_errors:
                        cur.execute(
                            """
                            INSERT INTO soccer_feature_snapshots (
                                fixture_id, captured_at, stage, schema_version,
                                model_version, data_tier, feature_count,
                                missing_feature_count, payload
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                            ON CONFLICT (fixture_id, captured_at, stage, schema_version) DO NOTHING
                            """,
                            (
                                fixture_id,
                                snapshot.get("captured_at"),
                                snapshot.get("stage"),
                                snapshot.get("schema_version"),
                                snapshot.get("model_version"),
                                snapshot.get("data_tier"),
                                snapshot.get("feature_count", 0),
                                snapshot.get("missing_feature_count", 0),
                                json.dumps(snapshot),
                            ),
                        )

                if not fixture_id or not isinstance(raw, dict):
                    continue
                cur.execute(
                    """
                    INSERT INTO soccer_model_runs (
                        fixture_id, model_version, run_type, run_timestamp,
                        raw_projection, shrunk_projection, model_prob,
                        market_fair_prob, prob_edge_pp, estimated_ev,
                        shrink_weight, availability_confidence,
                        classification, tier, payload
                    ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        fixture_id,
                        event.get("model_version") or tick.get("model_version") or "SOCCER EDGE ENGINE v1.0",
                        event.get("stage") or "UNKNOWN",
                        tick.get("generated_at_utc"),
                        json.dumps(raw),
                        json.dumps({
                            "p_shrunk": best.get("p_shrunk"),
                            "market": best.get("market"),
                            "selection": best.get("selection"),
                            "line": best.get("line"),
                            "price": best.get("decimal_price"),
                            "bookmaker": best.get("bookmaker"),
                        }),
                        best.get("p_shrunk"),
                        best.get("p_market_fair"),
                        best.get("prob_edge_pp"),
                        best.get("estimated_ev"),
                        best.get("shrink_weight"),
                        event.get("availability_confidence"),
                        event.get("classification"),
                        event.get("tier"),
                        # soccer_refresh_events already stores the full event payload.
                        # Keep model_runs focused on the model/projection snapshot to
                        # avoid serializing and storing the same large event twice.
                        json.dumps({
                            "fixture": fx,
                            "stage": event.get("stage"),
                            "raw_projection": raw,
                            "market_decision": decision,
                            "availability_confidence": event.get("availability_confidence"),
                            "classification": event.get("classification"),
                            "tier": event.get("tier"),
                        }),
                    ),
                )
    return True
