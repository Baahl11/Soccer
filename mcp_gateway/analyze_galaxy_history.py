from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _final_score(result: dict[str, Any]) -> tuple[int | None, int | None]:
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    ft = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    home = goals.get("home", ft.get("home"))
    away = goals.get("away", ft.get("away"))
    try:
        home = int(home) if home is not None else None
        away = int(away) if away is not None else None
    except (TypeError, ValueError):
        return None, None
    return home, away


def _result_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    result = event.get("result")
    if isinstance(result, dict) and result:
        return result
    if fixture.get("status") in {"FT", "AET", "PEN"}:
        goals = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
        if goals.get("home") is not None and goals.get("away") is not None:
            return {"status": fixture.get("status"), "goals": goals, "score": fixture.get("score")}
    return None


def _leg_signature(leg: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(leg.get("fixture_id") or ""),
        _norm(leg.get("family")),
        _norm(leg.get("selection")),
        str(leg.get("line") if leg.get("line") is not None else ""),
    )


def _candidate_signature(row: dict[str, Any]) -> str:
    parts = sorted(_leg_signature(leg) for leg in (row.get("legs") or []) if isinstance(leg, dict))
    fixture_id = str(row.get("fixture_id") or "")
    payload = [str(row.get("type") or ""), fixture_id, parts]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _grade_leg(leg: dict[str, Any], result: dict[str, Any]) -> str:
    home, away = _final_score(result)
    if home is None or away is None:
        return "NO_FINAL"
    family = _norm(leg.get("family"))
    selection = _norm(leg.get("selection"))
    line = _num(leg.get("line"))
    total = home + away

    if family == "FT_GOALS":
        if line is None:
            return "UNGRADABLE"
        if math.isclose(total, line):
            return "PUSH"
        if selection == "OVER":
            return "WIN" if total > line else "LOSS"
        if selection == "UNDER":
            return "WIN" if total < line else "LOSS"
        return "UNGRADABLE"

    if family == "BTTS":
        yes = home > 0 and away > 0
        if selection == "YES":
            return "WIN" if yes else "LOSS"
        if selection == "NO":
            return "WIN" if not yes else "LOSS"
        return "UNGRADABLE"

    if family == "DOUBLE_CHANCE":
        if selection == "1X":
            return "WIN" if home >= away else "LOSS"
        if selection == "X2":
            return "WIN" if away >= home else "LOSS"
        if selection == "12":
            return "WIN" if home != away else "LOSS"
        return "UNGRADABLE"

    if family in {"1X2", "MATCH_WINNER"}:
        actual = "1" if home > away else "2" if away > home else "X"
        aliases = {"HOME": "1", "AWAY": "2", "DRAW": "X"}
        pick = aliases.get(selection, selection)
        return "WIN" if pick == actual else "LOSS"

    if family == "CORRECT_SCORE":
        target = selection.replace(" ", "")
        return "WIN" if target == f"{home}-{away}" else "LOSS"

    if family == "DNB":
        if home == away:
            return "PUSH"
        actual = "HOME" if home > away else "AWAY"
        return "WIN" if selection in {actual, "1" if actual == "HOME" else "2"} else "LOSS"

    return "UNGRADABLE"


def _grade_candidate(row: dict[str, Any], results: dict[int, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    graded = []
    for leg in row.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        fid = leg.get("fixture_id") or row.get("fixture_id")
        try:
            fid_i = int(fid)
        except (TypeError, ValueError):
            outcome = "NO_FINAL"
            score = None
        else:
            result = results.get(fid_i)
            outcome = _grade_leg(leg, result or {})
            home, away = _final_score(result or {})
            score = f"{home}-{away}" if home is not None and away is not None else None
        graded.append({**leg, "outcome": outcome, "final_score": score})

    outcomes = [x["outcome"] for x in graded]
    if not outcomes:
        return "UNGRADABLE", graded
    if "LOSS" in outcomes:
        return "LOSS", graded
    if "NO_FINAL" in outcomes:
        return "NO_FINAL", graded
    if "UNGRADABLE" in outcomes:
        return "UNGRADABLE", graded
    if "PUSH" in outcomes:
        return "PUSH", graded
    return "WIN", graded


def _summary(rows: list[dict[str, Any]], allow_reference_profit: bool) -> dict[str, Any]:
    counts = Counter(str(row.get("outcome") or "UNGRADABLE") for row in rows)
    decided = counts["WIN"] + counts["LOSS"]
    risked = 0.0
    profit = 0.0
    priced = 0
    if allow_reference_profit:
        for row in rows:
            outcome = row.get("outcome")
            price = _num(row.get("first_component_product_decimal_reference"))
            if outcome not in {"WIN", "LOSS", "PUSH"} or price is None or price <= 1:
                continue
            priced += 1
            risked += 1.0
            if outcome == "WIN":
                profit += price - 1.0
            elif outcome == "LOSS":
                profit -= 1.0
    return {
        "n": len(rows),
        "graded": counts["WIN"] + counts["LOSS"] + counts["PUSH"],
        "win": counts["WIN"],
        "loss": counts["LOSS"],
        "push": counts["PUSH"],
        "no_final": counts["NO_FINAL"],
        "ungradable": counts["UNGRADABLE"],
        "hit_rate_ex_push": round(counts["WIN"] / decided, 4) if decided else None,
        "reference_priced_graded": priced,
        "hypothetical_reference_profit_units": round(profit, 4) if allow_reference_profit else None,
        "hypothetical_reference_roi": round(profit / risked, 4) if risked else None,
        "reference_profit_warning": (
            "HYPOTHETICAL COMPONENT-PRODUCT REFERENCE ONLY; NOT EXECUTED PARLAY ODDS OR BANKROLL ROI"
            if allow_reference_profit else
            "NOT CALCULATED FOR SGP; SAME-GAME COMPONENT PRODUCTS ARE NOT EXECUTABLE COMBINED ODDS"
        ),
    }


def analyze(history_dir: str) -> dict[str, Any]:
    results: dict[int, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    ticks = 0
    bad_lines = 0

    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                ticks += 1
                ts = tick.get("generated_at_local") or tick.get("generated_at_utc")
                version = tick.get("version")
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    result = _result_from_event(event)
                    if fid is not None and result:
                        try:
                            results[int(fid)] = result
                        except (TypeError, ValueError):
                            pass

                builder = tick.get("galaxy_builder") if isinstance(tick.get("galaxy_builder"), dict) else {}
                for key in ("same_game_candidates", "multi_match_candidates"):
                    for raw in builder.get(key) or []:
                        if not isinstance(raw, dict):
                            continue
                        row = dict(raw)
                        signature = _candidate_signature(row)
                        ref = row.get("component_price_reference") if isinstance(row.get("component_price_reference"), dict) else {}
                        reference_decimal = (
                            _num(row.get("component_product_decimal_reference"))
                            or _num(ref.get("component_product_decimal_reference"))
                        )
                        rec = candidates.get(signature)
                        if rec is None:
                            rec = {
                                "signature": signature,
                                "type": row.get("type"),
                                "first_seen": ts,
                                "last_seen": ts,
                                "first_version": version,
                                "last_version": version,
                                "fixture_id": row.get("fixture_id"),
                                "match": row.get("match"),
                                "kickoff": row.get("kickoff"),
                                "legs": [dict(x) for x in (row.get("legs") or []) if isinstance(x, dict)],
                                "joint_probability": _num(row.get("joint_model_probability") or row.get("conservative_joint_probability")),
                                "fair_decimal": _num(row.get("fair_decimal")),
                                "minimum_decimal": _num(row.get("minimum_sgp_decimal_for_target_edge") or row.get("minimum_parlay_decimal_for_target_edge")),
                                "first_component_product_decimal_reference": reference_decimal,
                                "last_component_product_decimal_reference": reference_decimal,
                                "bookmakers_seen": [],
                                "block_reasons_seen": [],
                                "manual_price_check_eligible_seen": bool(row.get("manual_price_check_eligible")),
                            }
                            candidates[signature] = rec
                        else:
                            rec["last_seen"] = ts
                            rec["last_version"] = version
                            rec["last_component_product_decimal_reference"] = reference_decimal
                            rec["manual_price_check_eligible_seen"] = bool(
                                rec.get("manual_price_check_eligible_seen") or row.get("manual_price_check_eligible")
                            )
                        book = row.get("bookmaker") or ref.get("bookmaker")
                        if book and book not in rec["bookmakers_seen"]:
                            rec["bookmakers_seen"].append(book)
                        for reason in row.get("block_reasons") or []:
                            if reason not in rec["block_reasons_seen"]:
                                rec["block_reasons_seen"].append(reason)

    rows = []
    for rec in candidates.values():
        outcome, graded_legs = _grade_candidate(rec, results)
        rec["outcome"] = outcome
        rec["graded_legs"] = graded_legs
        rec["first_seen_date_local"] = str(rec.get("first_seen") or "")[:10] or None
        rows.append(rec)
    rows.sort(key=lambda row: (str(row.get("first_seen") or ""), str(row.get("signature") or "")))

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("type") or "UNKNOWN")].append(row)
        by_date[str(row.get("first_seen_date_local") or "UNKNOWN")].append(row)

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "soccer_edge_state/history/*.jsonl galaxy_builder snapshots + persisted final results",
        "ticks_parsed": ticks,
        "bad_json_lines": bad_lines,
        "candidate_dedupe": "LOGICAL LEGS/FIXTURES; SAME COMBINATION ACROSS TICKS COUNTS ONCE",
        "policy": (
            "GALAXY WATCH OUTCOMES ARE GRADED FOR LEARNING EVEN WHEN NOT EXECUTED. MULTI REFERENCE ROI IS "
            "HYPOTHETICAL COMPONENT-PRODUCT BOOKKEEPING ONLY. SGP COMPONENT PRODUCTS ARE NEVER USED AS ROI."
        ),
        "overall": _summary(rows, allow_reference_profit=False),
        "by_type": {
            key: _summary(group, allow_reference_profit=(key == "MULTI_MATCH_PARLAY"))
            for key, group in sorted(by_type.items())
        },
        "by_first_seen_date_local": {
            date: {
                "overall": _summary(group, allow_reference_profit=False),
                "by_type": {
                    key: _summary([row for row in group if str(row.get("type") or "UNKNOWN") == key], allow_reference_profit=(key == "MULTI_MATCH_PARLAY"))
                    for key in sorted({str(row.get("type") or "UNKNOWN") for row in group})
                },
            }
            for date, group in sorted(by_date.items())
        },
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", default="soccer_edge_state/history")
    parser.add_argument("--output", default="soccer_edge_state/analysis/galaxy_outcomes.json")
    args = parser.parse_args()
    report = analyze(args.history_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "output": args.output,
        "overall": report["overall"],
        "by_type": report["by_type"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
