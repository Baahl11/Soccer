from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo


CLASS_ORDER = {"BET": 0, "LEAN": 1, "WATCH": 2, "PASS": 3, "CLOSE": 4, "POSTGAME": 5}
VALID_CLASSES = set(CLASS_ORDER)
VALID_DATA_TIERS = {"A", "B", "C", "D"}
DISPLAY_TIMEZONE = ZoneInfo("America/Mexico_City")


def _text(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _kickoff(value: Any) -> str:
    if not value:
        return "—"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(DISPLAY_TIMEZONE)
        return dt.strftime("%H:%M")
    except ValueError:
        return text


def _match(row: dict[str, Any]) -> str:
    return f"{_text(row.get('home'))} vs {_text(row.get('away'))}"


def _sport_signal(row: dict[str, Any]) -> str:
    bits: list[str] = []
    for label, key in (("S", "side_score"), ("G", "goals_score"), ("2W", "two_way_score")):
        value = row.get(key)
        if value is not None:
            try:
                bits.append(f"{label} {float(value):.1f}")
            except (TypeError, ValueError):
                pass
    return " / ".join(bits) if bits else "—"


def _market(row: dict[str, Any]) -> str:
    parts = [
        _text(row.get("market"), ""),
        _text(row.get("selection"), ""),
        _text(row.get("line"), ""),
    ]
    text = " ".join(part for part in parts if part)
    return text or "—"


def _escape(value: Any) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")


def _blockers(row: dict[str, Any]) -> str:
    blockers = row.get("blockers")
    if not isinstance(blockers, list):
        return "—"
    cleaned = [_text(item, "") for item in blockers]
    return "; ".join(item for item in cleaned if item) or "—"


def _detail_table(rows: Iterable[dict[str, Any]]) -> str:
    header = (
        "| Hora CDMX | País / Liga | Partido | Tier | MODEL_SIGNAL | Scores | EXECUTION_STATUS | Blockers | Disp. | Mercado | Precio | Book | Tier bet | Stake | Razón |\n"
        "|---|---|---|---:|---|---|---|---|---:|---|---:|---|---|---:|---|"
    )
    body: list[str] = []
    for row in rows:
        league = f"{_text(row.get('country'))} / {_text(row.get('competition'))}"
        body.append(
            "| " + " | ".join(
                [
                    _escape(_kickoff(row.get("kickoff"))),
                    _escape(league),
                    _escape(_match(row)),
                    _escape(row.get("data_tier")),
                    _escape(row.get("model_signal")),
                    _escape(_sport_signal(row)),
                    _escape(row.get("execution_status")),
                    _escape(_blockers(row)),
                    _escape(_pct(row.get("availability_confidence"))),
                    _escape(_market(row)),
                    _escape(row.get("price")),
                    _escape(row.get("bookmaker")),
                    _escape(row.get("tier")),
                    _escape(row.get("stake_units")),
                    _escape(row.get("reason")),
                ]
            ) + " |"
        )
    return header + ("\n" + "\n".join(body) if body else "\n| — | — | Sin partidos | — | — | — | — | — | — | — | — | — | — | — | — |")


def _pass_summary(rows: Iterable[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (_text(row.get("country")), _text(row.get("competition")))
        tier = _text(row.get("data_tier"), "D")
        groups[key]["total"] += 1
        groups[key][tier] += 1

    header = (
        "| País / Liga | PASS | A | B | C | D |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    body: list[str] = []
    for (country, competition), counts in sorted(groups.items()):
        body.append(
            f"| {_escape(country)} / {_escape(competition)} | {counts['total']} | {counts['A']} | {counts['B']} | {counts['C']} | {counts['D']} |"
        )
    return header + ("\n" + "\n".join(body) if body else "\n| — | 0 | 0 | 0 | 0 | 0 |")


def render_match_tables(rows: list[dict[str, Any]]) -> str:
    """Render deterministic user-facing Markdown from persisted match_table_rows.

    Pure presentation helper: no network, no market lookup, no model mutation.
    BET/LEAN/WATCH receive full rows. PASS is summarized by competition.
    """
    valid = [row for row in rows if isinstance(row, dict)]
    valid.sort(
        key=lambda row: (
            CLASS_ORDER.get(str(row.get("classification") or "WATCH"), 9),
            _text(row.get("country")),
            _text(row.get("competition")),
            _text(row.get("kickoff")),
        )
    )

    sections: list[str] = []
    for classification, title in (
        ("BET", "## 🟢 BET"),
        ("LEAN", "## 🟡 LEAN"),
        ("WATCH", "## 🔵 WATCH / RECHECK"),
    ):
        subset = [row for row in valid if row.get("classification") == classification]
        sections.append(title + "\n\n" + _detail_table(subset))

    passes = [row for row in valid if row.get("classification") == "PASS"]
    sections.append("## ⚪ PASS — resumen por competición\n\n" + _pass_summary(passes))
    return "\n\n".join(sections)


def validate_presentation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the v2.8 presentation/coverage contract without mutating decisions."""
    errors: list[str] = []
    warnings: list[str] = []

    registry = payload.get("league_coverage_registry")
    if not isinstance(registry, dict):
        errors.append("league_coverage_registry missing or not an object")
        registry = {}
    else:
        if registry.get("provider_calls_per_league") != 0:
            errors.append("provider_calls_per_league must be 0")
        if registry.get("galaxy_activation_is_eligibility_gate") is not False:
            errors.append("Galaxy activation must not be an eligibility gate")
        count = registry.get("competition_count")
        if not isinstance(count, int) or count < 0:
            errors.append("competition_count must be a non-negative integer")
        competitions = registry.get("competitions")
        if not isinstance(competitions, list):
            errors.append("competitions must be a list")
            competitions = []
        attached = registry.get("competition_rows_attached", len(competitions))
        if attached != len(competitions):
            errors.append("competition_rows_attached does not match competitions length")
        if isinstance(count, int) and count < len(competitions):
            errors.append("competition_count cannot be smaller than attached rows")
        for idx, row in enumerate(competitions[:500]):
            if not isinstance(row, dict):
                errors.append(f"competition[{idx}] is not an object")
                continue
            if row.get("data_tier") not in VALID_DATA_TIERS:
                errors.append(f"competition[{idx}] has invalid data_tier")
            if row.get("galaxy_activation_required") is not False:
                errors.append(f"competition[{idx}] incorrectly requires Galaxy activation")

    rows = payload.get("match_table_rows")
    if not isinstance(rows, list):
        errors.append("match_table_rows missing or not a list")
        rows = []
    row_count = payload.get("match_table_row_count", len(rows))
    attached_rows = payload.get("match_table_rows_attached", len(rows))
    if attached_rows != len(rows):
        errors.append("match_table_rows_attached does not match row list length")
    if isinstance(row_count, int) and row_count < len(rows):
        errors.append("match_table_row_count cannot be smaller than attached rows")
    for idx, row in enumerate(rows[:500]):
        if not isinstance(row, dict):
            errors.append(f"match_table_rows[{idx}] is not an object")
            continue
        classification = str(row.get("classification") or "WATCH")
        if classification not in VALID_CLASSES:
            errors.append(f"match_table_rows[{idx}] has invalid classification")
        tier = row.get("data_tier")
        if tier is not None and tier not in VALID_DATA_TIERS:
            warnings.append(f"match_table_rows[{idx}] has unverified data_tier")

    contract = payload.get("presentation_contract")
    if not isinstance(contract, dict):
        errors.append("presentation_contract missing or not an object")
    elif contract.get("default_format") != "MARKDOWN_TABLES":
        errors.append("presentation_contract.default_format must be MARKDOWN_TABLES")

    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:20],
        "warnings": warnings[:20],
    }
