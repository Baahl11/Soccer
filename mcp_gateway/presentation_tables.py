from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable


CLASS_ORDER = {"BET": 0, "LEAN": 1, "WATCH": 2, "PASS": 3, "CLOSE": 4, "POSTGAME": 5}


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


def _detail_table(rows: Iterable[dict[str, Any]]) -> str:
    header = (
        "| Hora | País / Liga | Partido | Tier | Señal deportiva | Disp. | Mercado | Precio | Book | Tier bet | Stake | Razón |\n"
        "|---|---|---|---:|---|---:|---|---:|---|---|---:|---|"
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
                    _escape(_sport_signal(row)),
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
    return header + ("\n" + "\n".join(body) if body else "\n| — | — | Sin partidos | — | — | — | — | — | — | — | — | — |")


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
