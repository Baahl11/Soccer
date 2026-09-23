from __future__ import annotations

import html
import json
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PRODUCT_DASHBOARD_V4_1.0.0"

DISPLAY_VIEWS = (
    ("todays_slate", "Today's Slate"),
    ("strong_sport_signals", "Strong Sport Signals"),
    ("value_plays", "Value Plays"),
    ("waiting_for_price", "Waiting for Price"),
    ("waiting_for_xi", "Waiting for XI"),
    ("team_totals", "Team Totals"),
    ("first_half", "1H"),
    ("second_half", "2H"),
    ("corners", "Corners"),
    ("player_props", "Player Props"),
)


def _esc(value: Any) -> str:
    if value is None:
        return "N/V"
    return html.escape(str(value), quote=True)


def _row_html(row: dict[str, Any]) -> str:
    fixture = f"{_esc(row.get('home'))} vs {_esc(row.get('away'))}"
    market = _esc(row.get("market"))
    selection = _esc(row.get("selection"))
    line = _esc(row.get("line"))
    price = _esc(row.get("price"))
    signal = _esc(row.get("model_signal"))
    execution = _esc(row.get("execution_status"))
    league = _esc(row.get("league"))
    return (
        "<tr>"
        f"<td>{fixture}<div class='muted'>{league}</div></td>"
        f"<td>{market}<div class='muted'>{selection} {line}</div></td>"
        f"<td>{price}</td>"
        f"<td>{signal}</td>"
        f"<td>{execution}</td>"
        "</tr>"
    )


def _view_section(key: str, title: str, view: dict[str, Any]) -> str:
    rows = view.get("rows") if isinstance(view, dict) else []
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    total = int(view.get("total") or 0) if isinstance(view, dict) else 0
    if not rows:
        body = "<div class='empty'>No rows available in the latest persisted tick.</div>"
    else:
        body = (
            "<div class='table-wrap'><table>"
            "<thead><tr><th>Fixture</th><th>Market</th><th>Price</th><th>Signal</th><th>Execution</th></tr></thead>"
            "<tbody>" + "".join(_row_html(row) for row in rows) + "</tbody></table></div>"
        )
    return (
        f"<section id='{_esc(key)}'>"
        f"<div class='section-head'><h2>{_esc(title)}</h2><span>{total} total</span></div>"
        f"{body}</section>"
    )


def render_dashboard(product_payload: dict[str, Any]) -> str:
    views = product_payload.get("views") if isinstance(product_payload.get("views"), dict) else {}
    generated = _esc(product_payload.get("generated_at_utc"))
    pipeline_version = _esc(product_payload.get("pipeline_version"))
    status = _esc(product_payload.get("status"))

    sections = "".join(
        _view_section(key, title, views.get(key) if isinstance(views.get(key), dict) else {})
        for key, title in DISPLAY_VIEWS
    )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": product_payload.get("pipeline_version"),
        "generated_at_utc": product_payload.get("generated_at_utc"),
        "status": product_payload.get("status"),
        "production_promotion_allowed": False,
    }
    meta_json = html.escape(json.dumps(meta, ensure_ascii=False), quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soccer Edge v4 Dashboard</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#0b0d10; color:#eef2f6; }}
main {{ max-width:1480px; margin:0 auto; padding:28px; }}
header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-end; margin-bottom:24px; }}
h1 {{ margin:0; font-size:30px; letter-spacing:-.03em; }}
h2 {{ margin:0; font-size:18px; }}
.meta {{ color:#97a3b1; font-size:13px; text-align:right; }}
.badge {{ display:inline-block; border:1px solid #303844; border-radius:999px; padding:5px 9px; margin-top:6px; }}
nav {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 24px; }}
nav a {{ color:#cbd5df; text-decoration:none; border:1px solid #29313b; padding:7px 10px; border-radius:8px; }}
section {{ background:#11151a; border:1px solid #252c35; border-radius:14px; margin-bottom:18px; overflow:hidden; }}
.section-head {{ padding:15px 18px; display:flex; justify-content:space-between; border-bottom:1px solid #252c35; }}
.section-head span,.muted {{ color:#8c98a6; font-size:12px; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:760px; }}
th,td {{ text-align:left; padding:12px 18px; border-bottom:1px solid #1d232b; vertical-align:top; }}
th {{ color:#9eabb8; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
td {{ font-size:14px; }}
.empty {{ padding:20px 18px; color:#7f8a97; }}
footer {{ color:#7f8a97; font-size:12px; padding:10px 0 24px; }}
@media (max-width:700px) {{ main {{padding:16px}} header {{display:block}} .meta {{text-align:left;margin-top:10px}} }}
</style>
</head>
<body data-meta="{meta_json}">
<main>
<header>
<div><h1>Soccer Edge v4</h1><div class="badge">Research-first dashboard</div></div>
<div class="meta">Pipeline {pipeline_version}<br>Generated {generated}<br>{status}</div>
</header>
<nav>
{''.join(f'<a href="#{_esc(key)}">{_esc(title)}</a>' for key,title in DISPLAY_VIEWS)}
</nav>
{sections}
<footer>Read-only operational view. Research/validation status remains authoritative; this dashboard does not promote markets or alter betting logic.</footer>
</main>
</body>
</html>"""
