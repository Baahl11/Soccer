"""Export only verified past-only Soccer FE11 profiles to the Render research bridge.

No provider calls, betting logic, betting prices, market odds or unstated data.
The original scanner writes its reports before the bookmaker market stage.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

VERSION = "SPORTS_EDGE_SCANNER_V15_HISTORICAL_CONTEXT"
PROFILE_VERSION = "SOCCER_SPORT_FIRST_MULTIMARKET_PROFILE_v1.0"


def _utc(value: Any) -> datetime | None:
    if not isinstance(value,str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z","+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def make_fe11_envelope(summary: dict[str, Any],
                       sport_screen: dict[str, Any],
                       historical_context: dict[str, Any],
                       *, max_reports: int = 3) -> dict[str, Any]:
    if not 0 <= max_reports <= 3:
        raise ValueError("MAX_REPORTS_EXCEEDED")
    generated = summary.get("generated_at_utc")
    if not isinstance(generated,str):
        raise ValueError("SOURCE_GENERATION_TIMESTAMP_MISSING")
    screen = {
        str(fx.get("fixture_id")): fx
        for fx in sport_screen.get("screened_fixtures",[])
        if isinstance(fx,dict) and fx.get("fixture_id") is not None
    }
    out = []
    skipped = []
    seen = set()
    for report in historical_context.get("reports",[]):
        if len(out) >= max_reports:
            break
        if not isinstance(report,dict):
            continue
        fixture_id = report.get("fixture_id")
        if type(fixture_id) is not int or fixture_id <= 0 or fixture_id in seen:
            continue
        seen.add(fixture_id)
        fx = screen.get(str(fixture_id))
        if not fx:
            skipped.append({"fixture_id":fixture_id,"reason":"FIXTURE_NOT_IN_SPORT_SCREEN"})
            continue
        kickoff=fx.get("kickoff_utc")
        ht=fx.get("home_team") or {}
        at=fx.get("away_team") or {}
        home,away=report.get("home") or {},report.get("away") or {}
        hp=home.get("sport_first_multimarket_profile")
        ap=away.get("sport_first_multimarket_profile")
        if (type(ht.get("team_id")) is not int or type(at.get("team_id")) is not int
            or not isinstance(kickoff,str)
            or not isinstance(hp,dict) or not isinstance(ap,dict)
            or hp.get("version") != PROFILE_VERSION
            or ap.get("version") != PROFILE_VERSION
            or hp.get("team_id") != ht["team_id"]
            or ap.get("team_id") != at["team_id"]
            or _utc(hp.get("target_kickoff")) is None
            or _utc(hp.get("target_kickoff")) != _utc(kickoff)
            or _utc(ap.get("target_kickoff")) != _utc(kickoff)
            or hp.get("bet_eligible") is not False
            or ap.get("bet_eligible") is not False
            or not isinstance(hp.get("observations"),list)
            or not isinstance(ap.get("observations"),list)
            or len(hp["observations"]) == 0
            or len(ap["observations"]) == 0):
            skipped.append({"fixture_id":fixture_id,"reason":"HISTORICAL_PROFILES_INCOMPLETE"})
            continue
        patterns=(home.get("example_goals_corners_handicap_pattern"),
                  away.get("example_goals_corners_handicap_pattern"))
        if any(not isinstance(x,dict)
               or set(x) != {"ASIAN","THREE_WAY"}
               for x in patterns):
            skipped.append({"fixture_id":fixture_id,"reason":"HANDICAP_SETTLEMENT_NOT_VERIFIED"})
            continue
        out.append({
            "fixture_id":fixture_id,
            "home_team_id":ht["team_id"],"away_team_id":at["team_id"],
            "kickoff":kickoff,
            "home_profile":hp,"away_profile":ap,
            "home_pattern":patterns[0],"away_pattern":patterns[1],
        })
    return {
        "source": VERSION,
        "generated_at_utc": generated,
        "reports": out,
        "validated_before_db": len(out),
        "not_verified": skipped,
        "bet_eligible": False,
        "calibrated_joint_probability": None,
        "bookmaker_ticket_quote": None,
    }


def export_from_daily_dir(output_dir: str | Path, date: str) -> dict[str, Any]:
    folder = Path(output_dir)
    def read(stem: str) -> dict[str, Any]:
        p = folder / f"{stem}_{date}.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    envelope = make_fe11_envelope(
        read("daily_summary"),read("sport_first"),read("historical_market_context")
    )
    (folder / f"fe11_multimarket_transport_{date}.json").write_text(
        json.dumps(envelope,ensure_ascii=False,separators=(",",":")),
        encoding="utf-8",
    )
    return {
        "date":date,
        "verified_reports_for_transport":len(envelope["reports"]),
        "discarded_unverified_reports":len(envelope["not_verified"]),
        "no_market_inputs":True,"bets":0,
    }
