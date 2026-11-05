"""Static HTML dashboard for real sport-first source evidence; WATCH ONLY."""
from __future__ import annotations
from html import escape
from datetime import datetime, timezone
from pathlib import Path
import math
import json
from typing import Any


def _pct(value: Any) -> str:
    try:
        n = float(value)
        if math.isfinite(n) and 0 <= n <= 1:
            return f'{100*n:.1f}%'
    except (TypeError, ValueError):
        pass
    return 'NOT VERIFIED'


def _text(x: Any) -> str:
    return escape(str(x if x is not None else 'NOT VERIFIED'), quote=True)


def render_board(batch: dict[str, Any], summary: dict[str, Any]) -> str:
    """Rendering never creates betting recommendations or uses market prices."""
    matches = batch.get('screened_fixtures') or []
    if not isinstance(matches, list):
        raise ValueError('Invalid screened_fixtures')
    cards = []
    for row in matches:
        raw = row.get('raw_sport_projection') or {}
        status = raw.get('status') or 'NOT_CONNECTED'
        sample = (raw.get('sample') or {}).get('minimum_split_sample')
        sample_note = 'MUESTRA INSUFICIENTE' if not isinstance(sample, int) or sample < 5 else 'NO CALIBRADO'
        name = _text(row.get('match') or 'Partido no identificado')
        fixture_id = _text(row.get('fixture_id'))
        league = _text(row.get('league_id'))
        line = (
            f'<tr><td>Victoria local</td><td>{_pct(raw.get("raw_home_win_prob"))}</td></tr>'
            f'<tr><td>Empate</td><td>{_pct(raw.get("raw_draw_prob"))}</td></tr>'
            f'<tr><td>Victoria visitante</td><td>{_pct(raw.get("raw_away_win_prob"))}</td></tr>'
            f'<tr><td>Más de 2.5 goles</td><td>{_pct(raw.get("raw_over_2_5_prob"))}</td></tr>'
            f'<tr><td>Más de 3.5 goles</td><td>{_pct(raw.get("raw_over_3_5_prob"))}</td></tr>'
            f'<tr><td>Ambos marcan (Sí)</td><td>{_pct(raw.get("raw_btts_yes_prob"))}</td></tr>'
        )
        cards.append(f'''<details class="game"><summary><span><strong>{name}</strong><small>Fixture #{fixture_id} · Liga {league} · {sample_note}</small></span><b>WATCH</b></summary>
        <div class="detail"><p>Modelo: <code>{_text(raw.get('projection_model'))}</code> · estado <b>{_text(status)}</b> · muestra local/visitante <b>{_text(sample)}</b>.</p>
        <table><thead><tr><th>Mercado</th><th>Probabilidad RAW (NO calibrada)</th></tr></thead><tbody>{line}</tbody></table>
        <p class="note">No implica EV positivo. Alineaciones, disponibilidad, calibración OOS y precios de casa accionables: NOT VERIFIED. No BET.</p></div></details>''')
    day = _text(summary.get('match_date') or batch.get('match_date'))
    generated = _text(summary.get('generated_at_utc'))
    screened = int(summary.get('sports_screened') or 0)
    modeled = int(summary.get('raw_existing_engine_projections') or 0)
    low_sample = sum(1 for row in matches if (row.get('raw_sport_projection') or {}).get('sample_warning'))
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SPORTS EDGE · {day}</title>
<style>:root{{color-scheme:dark;font-family:system-ui,Arial,sans-serif}}*{{box-sizing:border-box}}body{{background:#0d1324;color:#eaf0fb;margin:0}}main{{margin:auto;max-width:1020px;padding:25px 18px}}h1{{margin:8px 0}}p,small{{color:#aebbd1}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.metric,.game{{border:1px solid #34415c;background:#17223b;border-radius:12px;padding:14px}}.metric strong{{font-size:1.75rem;display:block}}.warning{{border-left:4px solid #e2b06a;padding:13px;background:#292638;border-radius:8px;margin:20px 0}}.game{{margin:10px 0;padding:0}}summary{{display:flex;justify-content:space-between;gap:15px;align-items:center;padding:15px;cursor:pointer}}summary small{{display:block;margin-top:3px}}summary b{{color:#efcf95}}.detail{{padding:0 15px 15px}}table{{border-collapse:collapse;width:100%;max-width:650px}}th,td{{padding:10px;border-bottom:1px solid #3a455c;text-align:left}}th{{color:#b4c0d5}}.note{{font-size:.87rem}}input{{padding:10px;background:#233451;border:1px solid #4b6084;color:white;border-radius:8px;width:min(100%,400px)}}.hide{{display:none}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main>
<small>SPORTS EDGE ENGINE · MOTOR ORIGINAL CONECTADO · PAPER RESEARCH</small><h1>Jornada {day}</h1><p>Generado {generated}. Estadísticas deportivas antes de mercados · probabilidades RAW del módulo original. Sin operaciones de apuesta.</p>
<div class="metrics"><div class="metric"><strong>{screened}</strong><small>Examinados</small></div><div class="metric"><strong>{modeled}</strong><small>RAW calculados</small></div><div class="metric"><strong>{low_sample}</strong><small>Muestra local/visitante insuficiente</small></div><div class="metric"><strong>0</strong><small>BET aprobadas</small></div></div>
<div class="warning"><strong>ATENCIÓN:</strong> proyecciones Poisson limitadas y NO calibradas. <b>NO</b> extrapolar como recomendaciones. Para BET faltan ajuste histórico/OOS, XI y disponibilidad confirmados, y precios vigentes de la casa elegida.</div>
<input id="search" aria-label="Filtrar partido" placeholder="Buscar equipo o fixture..." oninput="document.querySelectorAll('.game').forEach(x=>x.classList.toggle('hide',!x.textContent.toLowerCase().includes(this.value.toLowerCase())))"><div>{''.join(cards)}</div>
<p class="note">Fuente: API-Football Pro; motor existente fijado a commit 108c193c50e59f12f19f8693fc05c5cc9c607db4, función build_raw_projection. Sin captura de Caliente/Playdoit confirmada.</p></main></body></html>'''


def write_board(batch: dict[str, Any], summary: dict[str, Any], filename: str | Path) -> None:
    path=Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(render_board(batch,summary),encoding='utf-8')
    tmp.replace(path)