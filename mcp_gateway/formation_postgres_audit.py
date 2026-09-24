from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any

from mcp_gateway import persistence as persistence_base


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _norm_formation(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "").replace("–", "-").replace("—", "-")
    nums = re.findall(r"\d+", text)
    return "-".join(nums) if len(nums) >= 3 else None


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _formations(
    payload: dict[str, Any],
    home_team_id: int | None,
    away_team_id: int | None,
) -> tuple[str | None, str | None, bool]:
    teams = payload.get("teams") or []
    by_id: dict[int, tuple[Any, str | None]] = {}
    any_unrecognized = False
    for row in teams:
        if not isinstance(row, dict) or row.get("team_id") is None:
            continue
        tid = int(row["team_id"])
        raw = row.get("formation")
        norm = _norm_formation(raw)
        if raw not in (None, "") and norm is None:
            any_unrecognized = True
        by_id[tid] = (raw, norm)
    home = by_id.get(int(home_team_id)) if home_team_id is not None else None
    away = by_id.get(int(away_team_id)) if away_team_id is not None else None
    return (
        home[1] if home else None,
        away[1] if away else None,
        any_unrecognized,
    )


def audit_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    clean: dict[int, dict[str, Any]] = {}
    for item in targets:
        if not isinstance(item, dict):
            continue
        try:
            fixture_id = int(item.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        kickoff = _parse_dt(item.get("kickoff_local") or item.get("kickoff"))
        if kickoff is None:
            continue
        clean[fixture_id] = {
            "fixture_id": fixture_id,
            "kickoff": kickoff,
            "kickoff_local": kickoff.isoformat(),
        }

    if not clean:
        return {
            "status": "NO_VALID_TARGETS",
            "target_count": 0,
            "recoverable_prekickoff_formations": 0,
            "reason_counts": {},
            "rows": [],
        }

    persistence_base.ensure_schema()
    fixture_ids = sorted(clean)
    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    f.fixture_id,
                    f.home_team_id,
                    f.away_team_id,
                    l.captured_at,
                    l.stage,
                    l.both_xi_confirmed,
                    l.payload
                FROM soccer_fixtures AS f
                LEFT JOIN soccer_lineup_snapshots AS l
                  ON l.fixture_id = f.fixture_id
                WHERE f.fixture_id = ANY(%s)
                ORDER BY f.fixture_id, l.captured_at
                """,
                (fixture_ids,),
            )
            db_rows = cur.fetchall()

    snapshots: dict[int, list[dict[str, Any]]] = {fid: [] for fid in fixture_ids}
    team_ids: dict[int, tuple[int | None, int | None]] = {}
    for fixture_id, home_id, away_id, captured_at, stage, both_confirmed, payload in db_rows:
        fid = int(fixture_id)
        team_ids[fid] = (
            int(home_id) if home_id is not None else None,
            int(away_id) if away_id is not None else None,
        )
        if captured_at is None:
            continue
        snapshots.setdefault(fid, []).append(
            {
                "captured_at": _parse_dt(captured_at),
                "stage": stage,
                "both_xi_confirmed": bool(both_confirmed),
                "payload": _payload(payload),
            }
        )

    reason_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for fid in fixture_ids:
        target = clean[fid]
        kickoff = target["kickoff"]
        home_id, away_id = team_ids.get(fid, (None, None))
        obs = snapshots.get(fid) or []

        pre = [x for x in obs if x.get("captured_at") is not None and x["captured_at"] <= kickoff]
        post = [x for x in obs if x.get("captured_at") is not None and x["captured_at"] > kickoff]

        valid_pre: list[dict[str, Any]] = []
        valid_post: list[dict[str, Any]] = []
        one_team_pre = False
        unrecognized_pre = False
        confirmed_without_both_pre = False
        unconfirmed_pre = False

        for snap in pre:
            hf, af, bad = _formations(snap["payload"], home_id, away_id)
            if hf and af and snap["both_xi_confirmed"]:
                valid_pre.append({**snap, "home_formation": hf, "away_formation": af})
            elif bool(hf) ^ bool(af):
                one_team_pre = True
            elif bad:
                unrecognized_pre = True
            elif snap["both_xi_confirmed"]:
                confirmed_without_both_pre = True
            else:
                unconfirmed_pre = True

        for snap in post:
            hf, af, _ = _formations(snap["payload"], home_id, away_id)
            if hf and af and snap["both_xi_confirmed"]:
                valid_post.append({**snap, "home_formation": hf, "away_formation": af})

        recoverable = False
        recovered: dict[str, Any] | None = None
        if valid_pre:
            chosen = sorted(valid_pre, key=lambda x: x["captured_at"])[-1]
            reason = "POSTGRES_RECOVERABLE_PREKICKOFF_FORMATION"
            recoverable = True
            recovered = {
                "captured_at": chosen["captured_at"].isoformat(),
                "home_formation": chosen["home_formation"],
                "away_formation": chosen["away_formation"],
            }
        elif one_team_pre:
            reason = "FORMATION_ONE_TEAM_ONLY"
        elif unrecognized_pre:
            reason = "FORMATION_MAPPING_UNRECOGNIZED"
        elif confirmed_without_both_pre:
            reason = "CONFIRMED_XI_WITHOUT_FORMATION_VALUE"
        elif unconfirmed_pre:
            reason = "LINEUP_NOT_CONFIRMED_AT_SNAPSHOT"
        elif valid_post:
            reason = "FORMATION_TIMESTAMP_POSTERIOR_TO_PREDICTION_POINT"
        elif obs:
            reason = "NO_PREKICKOFF_VERIFIABLE_FORMATION"
        else:
            reason = "LEAGUE_OR_FIXTURE_WITHOUT_LINEUP"

        reason_counts[reason] += 1
        rows.append(
            {
                "fixture_id": fid,
                "kickoff_local": target["kickoff_local"],
                "postgres_lineup_snapshots": len(obs),
                "postgres_prekickoff_snapshots": len(pre),
                "postgres_postkickoff_snapshots": len(post),
                "reason": reason,
                "recoverable_by_reconciliation": recoverable,
                "recovered_formation": recovered,
            }
        )

    return {
        "status": "POSTGRES_FORMATION_RECONCILIATION_AUDIT",
        "policy": (
            "READ_ONLY; USE ONLY SNAPSHOTS CAPTURED_AT_OR_BEFORE_FINAL_KICKOFF; "
            "POSTKICKOFF FORMATIONS ARE NEVER BACKFILLED INTO OOS"
        ),
        "target_count": len(fixture_ids),
        "recoverable_prekickoff_formations": sum(
            1 for row in rows if row["recoverable_by_reconciliation"]
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }
