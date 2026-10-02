from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import persistence
from mcp_gateway import team_totals_close_provenance_v4 as provenance

MODEL_VERSION = "SOCCER_TEAM_TOTALS_CLOSE_PROVENANCE_HISTORY_V4_1.0.0"
QUERY_STRATEGY = "RECENT_FIXTURE_IDS_THEN_MINIMAL_DERIVATIVE_JSON_V1"
DEFAULT_LOOKBACK_HOURS = 12
MAX_RECENT_FIXTURES = 250
MAX_REFRESH_EVENTS = 600


def load_report(
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    max_recent_fixtures: int = MAX_RECENT_FIXTURES,
    max_refresh_events: int = MAX_REFRESH_EVENTS,
    max_samples: int = provenance.MAX_SAMPLES,
) -> dict[str, Any]:
    if not persistence.persistence_configured():
        report = provenance.build_historical_report(
            signal_rows=[], snapshot_rows=[], max_samples=max_samples
        )
        report.update(
            {
                "status": "NO_DATABASE",
                "source": "POSTGRES_RECENT_FIXTURE_SCOPED_TEAM_TOTALS_PROVENANCE",
                "query_strategy": QUERY_STRATEGY,
                "provider_requests_added": 0,
            }
        )
        return report

    persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, kickoff
                FROM soccer_fixtures
                WHERE kickoff > %s
                  AND kickoff <= %s
                ORDER BY kickoff DESC, fixture_id DESC
                LIMIT %s
                """,
                (cutoff, now, max(1, min(int(max_recent_fixtures), 500))),
            )
            recent_fixture_rows = cur.fetchall()

        recent_fixture_ids = [
            int(row[0]) for row in recent_fixture_rows if row and row[0] is not None
        ]
        refresh_rows: list[tuple[Any, ...]] = []
        if recent_fixture_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        e.fixture_id,
                        e.generated_at,
                        e.stage,
                        f.kickoff,
                        CASE
                            WHEN jsonb_typeof(
                                e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows'
                            ) = 'array'
                            THEN e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows'
                            ELSE '[]'::jsonb
                        END AS observed_exact_market_rows
                    FROM soccer_refresh_events e
                    JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                    WHERE e.fixture_id = ANY(%s)
                      AND e.generated_at >= %s
                      AND e.generated_at < f.kickoff
                      AND e.stage = ANY(%s)
                      AND e.payload -> 'team_totals_intelligence' IS NOT NULL
                    ORDER BY e.generated_at ASC, e.fixture_id ASC
                    LIMIT %s
                    """,
                    (
                        recent_fixture_ids,
                        cutoff,
                        list(provenance.TEAM_TOTALS_RESEARCH_STAGES),
                        max(1, min(int(max_refresh_events), 1000)),
                    ),
                )
                refresh_rows = cur.fetchall()

        oldest_exact: dict[tuple[Any, ...], dict[str, Any]] = {}
        for fixture_id, generated_at, stage, kickoff, observed_rows in refresh_rows:
            rows = observed_rows if isinstance(observed_rows, list) else []
            for candidate in rows:
                if not isinstance(candidate, dict):
                    continue
                market = candidate.get("market")
                selection = candidate.get("selection")
                line = provenance._signal_line(candidate)
                if not market or not selection or line is None:
                    continue
                key = (
                    int(fixture_id),
                    provenance._norm(market),
                    provenance._selection_norm(selection),
                    float(line),
                )
                oldest_exact.setdefault(
                    key,
                    {
                        "fixture_id": int(fixture_id),
                        "signal_generated_at": generated_at,
                        "stage": stage,
                        "kickoff": kickoff,
                        "market_candidate": {
                            **candidate,
                            "line": float(line),
                        },
                    },
                )

        signal_rows = list(oldest_exact.values())
        modeled_fixture_ids = sorted(
            {int(row["fixture_id"]) for row in signal_rows if row.get("fixture_id") is not None}
        )
        snapshot_rows: list[dict[str, Any]] = []
        if modeled_fixture_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        fixture_id,
                        captured_at,
                        stage,
                        bookmaker_id,
                        bookmaker,
                        market_id,
                        market,
                        values,
                        provider_update
                    FROM soccer_market_snapshots
                    WHERE fixture_id = ANY(%s)
                      AND captured_at >= %s
                      AND captured_at <= %s
                    ORDER BY fixture_id ASC, captured_at ASC
                    """,
                    (modeled_fixture_ids, cutoff, now),
                )
                columns = [desc.name for desc in cur.description]
                snapshot_rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    report = provenance.build_historical_report(
        signal_rows=signal_rows,
        snapshot_rows=snapshot_rows,
        now=now,
        max_samples=max_samples,
    )
    report.update(
        {
            "model_version": MODEL_VERSION,
            "source": "POSTGRES_RECENT_FIXTURE_SCOPED_TEAM_TOTALS_PROVENANCE",
            "query_strategy": QUERY_STRATEGY,
            "lookback_hours": max(1, int(lookback_hours)),
            "recent_fixture_count": len(recent_fixture_ids),
            "refresh_events_loaded": len(refresh_rows),
            "deduped_exact_signals": len(signal_rows),
            "audited_fixture_ids": modeled_fixture_ids,
            "snapshot_rows_loaded": len(snapshot_rows),
            "cap_independent": True,
            "global_signal_cap_dependency": False,
            "provider_requests_added": 0,
            "selection_logic_changed": False,
            "strict_close_semantics_changed": False,
            "historical_rows_mutated": False,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        }
    )
    return report
