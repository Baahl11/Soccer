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


def render_board(batch: dict[str, Any], summary: dict[str, Any], *,
                 market_registry: dict[str, Any] | None = None,
                 canonical: dict[str, Any] | None = None,
                 advanced_context: dict[str, Any] | None = None,
                 parlay_radar: dict[str, Any] | None = None,
                 decision_report: dict[str, Any] | None = None,
                 canonical_research: dict[str, Any] | None = None,
                 bookmaker_catalogue: dict[str, Any] | None = None,
                 market_matrix: dict[str, Any] | None = None) -> str:
    """Render verified decision output; never manufacture prices or probabilities."""
    matches = batch.get('screened_fixtures') or []
    if not isinstance(matches, list):
        raise ValueError('Invalid screened_fixtures')
    market_registry = market_registry or {}
    canonical = canonical or {}
    families = market_registry.get('families') or []
    deep = {r['fixture_id']:r for r in (advanced_context or {}).get('reports',[]) if isinstance(r,dict) and 'fixture_id' in r}
    family_rows = ''.join(
        '<tr><td><b>' + _text(f.get('label')) + '</b></td>'
        + '<td>' + _text(f.get('research_status')) + '</td>'
        + '<td>' + _text('; '.join((f.get('blockers') or [])[:2]) or 'Sin suficientes garantías para BET') + '</td></tr>'
        for f in families
    )
    history_ok = int(summary.get('history_multi_competition_verified') or 0)
    canonical_status = _text(summary.get('canonical_market_status') or 'NOT_VERIFIED')
    cards = []
    for row in matches:
        raw = row.get('raw_sport_projection') or {}
        status = raw.get('status') or 'NOT_CONNECTED'
        sample = (raw.get('sample') or {}).get('minimum_split_sample')
        sample_note = 'MUESTRA INSUFICIENTE' if not isinstance(sample, int) or sample < 5 else 'NO CALIBRADO'
        hist = row.get('multi_competition_history') or {}
        history_text = ('Partidos previos en TODAS las competiciones: '
                        + _text(hist.get('home_count', 0)) + ' local / '
                        + _text(hist.get('away_count', 0)) + ' visitante. Estado: '
                        + _text(hist.get('status') or 'NOT_VERIFIED'))
        formation_text = _text(hist.get('formation_matchup_status') or 'NOT_VERIFIED_NO_VERIFIED_XI')
        context = deep.get(row.get('fixture_id')) or {}
        advanced_rows = ''
        if context:
            for key, label in [('corners','Córners'),('fouls','Faltas'),('yellow_cards','Tarjetas amarillas'),('shots','Tiros totales'),('shots_on_target','Tiros a puerta')]:
                h=(context.get('home') or {}).get('observed_averages',{}).get(key,{})
                a=(context.get('away') or {}).get('observed_averages',{}).get(key,{})
                def field(y):
                    return 'N/V' if y.get('average') is None else f"{_text(y['average'])} (n={_text(y.get('sample_count'))})"
                advanced_rows += f'<tr><td>{label}</td><td>{field(h)}</td><td>{field(a)}</td></tr>'
        period_html = ''
        if context:
            for key, label in (('first_half_goals','Primera mitad: goles anotados'),
                               ('second_half_goals','Segunda mitad: goles anotados')):
                h=(context.get('home') or {}).get('period_descriptive',{}).get(key,{})
                a=(context.get('away') or {}).get('period_descriptive',{}).get(key,{})
                def shown(z):
                    return 'N/V' if z.get('scored_average') is None else f"{_text(z['scored_average'])} (n={_text(z.get('sample_count'))})"
                period_html += f'<tr><td>{label}</td><td>{shown(h)}</td><td>{shown(a)}</td></tr>'
        advanced_rows += period_html
        advanced_html = ('<h4>Historial de mercados — descriptivo, NO predicción</h4>'
                         '<table><thead><tr><th>Métrica</th><th>Local</th><th>Visitante</th></tr></thead>'
                         '<tbody>'+advanced_rows+'</tbody></table>'
                         '<p class="note">Formaciones previas verificadas: '
                         + _text((context.get('home') or {}).get('past_formation_samples',0))+' / '
                         + _text((context.get('away') or {}).get('past_formation_samples',0))
                         + '. Sin once ni formación actual confirmados: NO ajuste táctico cuantitativo.</p>') if advanced_rows else ''
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
        <div class="detail"><p>Modelo RAW: <code>{_text(raw.get('projection_model'))}</code> · estado <b>{_text(status)}</b> · muestra de esta liga <b>{_text(sample)}</b>.</p>
        <p class="note">{history_text}<br>Formaciones y estilo: {formation_text}. El estado del módulo es investigativo, no un ajuste automático de goles.</p>
        <table><thead><tr><th>Mercado</th><th>Probabilidad RAW (NO calibrada)</th></tr></thead><tbody>{line}</tbody></table>{advanced_html}
        <p class="note">No implica EV positivo. Alineaciones, disponibilidad, calibración OOS y precios de casa accionables: NOT VERIFIED. No BET.</p></div></details>''')
    market_matrix = market_matrix or {}
    matrix_market_rows = ''.join(
        '<tr><td>'+_text(q.get('event_name'))+'</td>'
        +'<td>'+_text(q.get('market'))+' '+_text(q.get('selection'))
        +' '+_text(q.get('point'))+'</td>'
        +'<td>'+_text(q.get('price_decimal'))+'</td>'
        +'<td>'+_pct(q.get('raw_sport_probability'))+'</td>'
        +'<td>'+_pct(q.get('breakeven_probability'))+'</td>'
        +'<td>'+_text(q.get('raw_minus_breakeven_pp_NOT_EV'))+' pp</td>'
        +'<td>'+_text(q.get('venue_sample_min'))+'</td>'
        +'<td>'+_text(q.get('raw_probability_source'))+'</td></tr>'
        for q in market_matrix.get('top_model_matched_markets',[])[:25]
    )
    matrix_pairs = ''.join(
        '<li><b>'+_text(p.get('theoretical_combined_decimal'))+'</b> · '
        +_text(p['legs'][0].get('event_name'))+' · '
        +_text(p['legs'][0].get('market'))+' '+_text(p['legs'][0].get('selection'))+' '
        +_text(p['legs'][0].get('point'))+' @ '+_text(p['legs'][0].get('price_decimal'))
        +' + '+_text(p['legs'][1].get('event_name'))+' · '
        +_text(p['legs'][1].get('market'))+' '+_text(p['legs'][1].get('selection'))+' '
        +_text(p['legs'][1].get('point'))+' @ '+_text(p['legs'][1].get('price_decimal'))
        +' · menor diferencia RAW por pierna '
        +_text(p.get('minimum_raw_margin_pp_NOT_EV'))+' pp'
        +' · muestra mínima '+_text(p.get('minimum_venue_sample'))
        +' · '+_text(p.get('research_sample_status'))
        +' · <b>WATCH · NO ES EV / NI RIESGO VALIDADO</b></li>'
        for p in market_matrix.get('sporting_screened_plus100_pairs',[])[:12]
    )
    matrix_html=('<h2>Bet365 · todos los mercados que el modelo deportivo puede comparar</h2>'
      +'<p>Partidos con cuota Bet365 descargada: '
      +_text(market_matrix.get('bookmaker_snapshots_fetched',0))
      +' · con cuotas recientes compatibles: '
      +_text(market_matrix.get('fresh_quoted_fixtures',0))
      +'. Cotizaciones Bet365 normalizadas: '
      +_text(market_matrix.get('total_normalized_quotes',0))
      +' · mercados respaldados por probabilidad RAW: '
      +_text(market_matrix.get('exact_or_poisson_derived_raw_quotes',0))
      +' · cotizaciones con referencia sin vigorish: '
      +_text(market_matrix.get('two_way_or_three_way_novig_quotes',0))
      +' · duplas +100 distintas con dos piernas RAW superiores a breakeven: '
      +_text(market_matrix.get('sporting_screened_plus100_pairs_found',0))
      +' · con al menos 5 partidos en cada muestra local/visitante: '
      +_text(market_matrix.get('sporting_screened_plus100_pairs_min_sample_5',0))
      +'. <b>Estas comparaciones no prueban ventaja ni seguridad.</b></p>'
      +'<h3>Mercados con proyección deportiva RAW</h3>'
      +('<div style="overflow-x:auto"><table class="market-table"><thead><tr>'
      +'<th>Partido</th><th>Mercado</th><th>Bet365</th><th>RAW</th>'
      +'<th>Breakeven</th><th>RAW–precio</th><th>Muestra</th><th>Origen</th>'
      +'</tr></thead><tbody>'+matrix_market_rows+'</tbody></table></div>'
      if matrix_market_rows else '<p>No hay cotizaciones Bet365 recientes con proyección exacta.</p>')
      +'<h3>Parlays SPORT-FIRST para investigar (2 piernas)</h3>'
      +('<ol>'+matrix_pairs+'</ol>' if matrix_pairs else
        '<p>Ninguna dupla supera simultáneamente el filtro RAW preliminar y la cuota objetivo.</p>')
      +'<p class="note">No se crea un modelo para mitades, córners, faltas ni tarjetas '
      +'a partir del Poisson de goles completos. Sin OOS, disponibilidad, dependencia '
      +'y ticket confirmado en México, todos siguen WATCH.</p>')
    canonical_research = canonical_research or {}
    bookmaker_catalogue = bookmaker_catalogue or {}
    canonical_rows_html = ''.join(
        '<tr><td>'+_text(x.get('match'))+'</td>'
        +'<td>'+_text(x.get('market'))+' '+_text(x.get('selection'))
        +' '+_text(x.get('line'))+'</td>'
        +'<td>'+_pct(x.get('research_probability_for_selection'))+'</td>'
        +'<td>'+_text(x.get('indicative_price'))+' ('+_text(x.get('reference_bookmaker'))+')</td>'
        +'<td>'+_text(x.get('research_gap_pp_NOT_EV'))+' pp</td>'
        +'<td>'+_text(', '.join(x.get('blockers') or []))+'</td></tr>'
        for x in canonical_research.get('rows',[])[:30]
    )
    source_html = ('<h2>Motor original + precios de referencia: evidencia por mercado</h2>'
       +'<p><b>Fuente:</b> '+_text(canonical_research.get('status','NOT_VERIFIED'))
       +' · Mercados emparejados: '+_text(canonical_research.get('total_matching_rows',0))
       +' · Calibraciones experimentales por selección: '
       +_text(canonical_research.get('selection_calibration_research_rows',0))
       +' · Precios de referencia vigentes según su publicación: '
       +_text(canonical_research.get('reference_price_rows',0))
       +' · Directorio de casas: '+_text(bookmaker_catalogue.get('status','NOT_VERIFIED'))
       +' · Ofertas verificadas Caliente/Playdoit: '
       +_text(bookmaker_catalogue.get('actionable_offers',0))+'</p>'
       +'<p class="note">La calibración experimental NO es producción. El porcentaje '
       +'de OVER 2.5 no equivale al de UNDER 2.5; solo se utiliza su complemento '
       +'para el mismo umbral binario. La brecha de investigación NO es EV '
       +'ni probabilidad conjunta de una combinada. Una casa en API-Football '
       +'NO significa que el precio pueda aceptarse en ella.</p>'
       +('<div style="overflow-x:auto"><table class="market-table">'
       +'<thead><tr><th>Encuentro</th><th>Mercado</th><th>P investigación</th>'
       +'<th>Precio de referencia</th><th>Brecha (NO EV)</th><th>Bloqueos</th>'
       +'</tr></thead><tbody>'+canonical_rows_html+'</tbody></table></div>'
       if canonical_rows_html else '<p>No hay mercados comparables del estado canónico actual.</p>'))
    decision_report = decision_report or {}
    diag = decision_report.get('canonical_research') or {}
    diag_blockers = diag.get('blocker_counts') or {}
    main_blockers_html = ''.join(
        '<li>'+_text(key)+': '+_text(n)+' registros</li>'
        for key,n in sorted(diag_blockers.items(),key=lambda x:-x[1])[:8]
    ) or '<li>Sin registros canónicos para esta selección.</li>'
    singles_html = ''.join(
        '<li><b>'+_text(x.get('status'))+'</b> · '+_text(x.get('event_name'))
        +' · '+_text(x.get('market'))+' '+_text(x.get('selection'))
        +' · cuota '+_text(x.get('price_decimal'))
        +' · EV conservador '+_text(x.get('conservative_ev'))+'</li>'
        for x in decision_report.get('singles',[]) if x.get('status') in ('BET','LEAN')
    ) or '<li>No se ha validado una selección BET ni LEAN.</li>'
    tickets_html = ''.join(
        '<li><b>'+_text(x.get('status'))+'</b> · '+_text(x.get('bookmaker'))
        +' · ticket '+_text(x.get('ticket_reference_id'))
        +' · cuota '+_text(x.get('ticket_price_decimal'))
        +' · EV conservador '+_text(x.get('conservative_joint_ev'))+'</li>'
        for x in decision_report.get('parlays',[]) if x.get('status') == 'BET'
    ) or '<li>Sin combinadas certificadas +100.</li>'
    decision_html = ('<h2>Motor final: BET / LEAN / WATCH / PASS</h2>'
      +'<p>Estado de evidencia: <b>'+_text(decision_report.get('feed_status','NOT_VERIFIED'))
      +'</b>. BET individuales '+_text(decision_report.get('actionable_bets',0))
      +'; BET combinadas '+_text(decision_report.get('actionable_parlays',0))
      +'. Estos conteos los calcula el evaluador y no están fijados a cero.</p>'
      +'<h4>Bloqueos del motor canónico</h4><ul>'+main_blockers_html+'</ul>'
      +'<h4>Mercados aprobados o LEAN</h4><ul>'+singles_html+'</ul>'
      +'<h4>Tickets combinados aprobados</h4><ul>'+tickets_html+'</ul>'
      +'<p class="note">Las pruebas sintéticas de BET sirven exclusivamente para validar '
      +'el código. No representan partidos, cuotas ni apuestas reales.</p>')
    parlay_radar = parlay_radar or {}
    radar_pairs = parlay_radar.get('pairs') or []
    radar_cards = []
    for pair in radar_pairs[:15]:
        legs=pair.get('legs') or []
        if len(legs)!=2:continue
        def label(leg):
            point=(' '+_text(leg.get('point'))) if leg.get('point') else ''
            return ('<b>'+_text(leg.get('event_name') or 'Fixture '+str(leg.get('event_id')))
                    +'</b> · '+_text(leg.get('market'))+' · '+_text(leg.get('selection'))+point
                    +' · '+_text(leg.get('price_american')))
        def evidence(leg):
            raw = leg.get('raw_sport_probability')
            pbe = leg.get('individual_price_breakeven')
            delta = leg.get('raw_minus_individual_breakeven_pp')
            diff = 'N/V' if delta is None else f'{float(delta):+.2f} pp'
            warning = (' <b>RAW INFERIOR A BREAKEVEN</b>' if delta is not None and delta <= 0 else '')
            return ('<small>RAW no calibrada: '+_pct(raw)
                    +' · breakeven de esta pierna: '+_pct(pbe)
                    +' · diferencia RAW (NO EV): '+_text(diff)
                    +' · muestra local/visitante: '+_text(leg.get('raw_sample_venue_split_minimum'))
                    +' · validación: '+_text(leg.get('market_validation_status'))
                    +warning+'</small>')
        status_msg = ('<b>ALERTA: al menos una pierna RAW queda bajo su breakeven</b>'
                      if pair.get('joint_raw_sanity_status') == 'AT_LEAST_ONE_LEG_RAW_BELOW_ITS_BREAK_EVEN'
                      else '<b>Sin probabilidad conjunta calibrada</b>')
        radar_cards.append('<li>'+label(legs[0])+' <b>+</b> '+label(legs[1])
                           +'<br>Casa de referencia: '+_text(pair.get('bookmaker'))
                           +' · cuota matemática '+_text(pair.get('theoretical_combined_decimal'))
                           +' ('+_text(pair.get('theoretical_combined_american'))+')'
                           +' · breakeven '+_text(round(100*float(pair.get('joint_breakeven') or 0),1))+'%'
                           +' · <b>WATCH, EV NO VERIFICADO</b>'
                           +'<br>'+evidence(legs[0])+'<br>'+evidence(legs[1])
                           +'<br>'+status_msg
                           +'</li>')
    radar_html=('<h2>Radar parlay cercano a +100 — WATCH</h2>'
                '<p>Primero investigamos los partidos; luego exploramos posibles combinaciones de precios '
                'en dos encuentros distintos y la misma casa de referencia. Esto no mide riesgo real '
                'ni ofrece probabilidades conjuntas calibradas.</p>'
                + ('<ol>'+''.join(radar_cards)+'</ol>' if radar_cards else
                   '<p>No se encontraron combinaciones con precios de origen verificados y suficientemente recientes '
                   'en los mercados consultados. Sin inventar oportunidades.</p>')
                + '<p class="note">API-Football no verifica que el parlay sea ofrecido por Caliente/Playdoit. '
                  'Independencia, disponibilidad, calibración OOS, precio ofrecido y EV: NOT VERIFIED. La comparación RAW no mide riesgo. '
                  'Nunca BET por el mero hecho de sumar +100.</p>')

    day = _text(summary.get('match_date') or batch.get('match_date'))
    generated = _text(summary.get('generated_at_utc'))
    screened = int(summary.get('sports_screened') or 0)
    modeled = int(summary.get('raw_existing_engine_projections') or 0)
    pregame_text = ('Comprobaciones T-90/T-40/T-20: '
                    + _text(summary.get('pregame_refresh_completed', 0))
                    + ' / ' + _text(summary.get('pregame_candidates_in_window', 0))
                    + ' candidatos en ventana. XI oficial verificado: NO. '
                    + 'Reporte pregame_availability_'+_text(summary.get('match_date'))+'.json.')
    low_sample = sum(1 for row in matches if (row.get('raw_sport_projection') or {}).get('sample_warning'))
    market_section = ('<h2>Mercados del motor completo — evidencia, NO recomendaciones</h2>'
                     '<p>Los reportes de validación pertenecen al motor existente; pueden ser históricos. '
                     'No equivalen a pronósticos de cada partido ni autorizan apuestas.</p>'
                     '<table class="market-table"><thead><tr><th>Familia</th><th>Estado de investigación</th><th>Bloqueos principales</th></tr></thead>'
                     '<tbody>'+family_rows+'</tbody></table>') if family_rows else '<p>No se cargaron reportes de madurez de mercados.</p>'
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SPORTS EDGE · {day}</title>
<style>:root{{color-scheme:dark;font-family:system-ui,Arial,sans-serif}}*{{box-sizing:border-box}}body{{background:#0d1324;color:#eaf0fb;margin:0}}main{{margin:auto;max-width:1020px;padding:25px 18px}}h1{{margin:8px 0}}p,small{{color:#aebbd1}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.metric,.game{{border:1px solid #34415c;background:#17223b;border-radius:12px;padding:14px}}.metric strong{{font-size:1.75rem;display:block}}.warning{{border-left:4px solid #e2b06a;padding:13px;background:#292638;border-radius:8px;margin:20px 0}}.game{{margin:10px 0;padding:0}}summary{{display:flex;justify-content:space-between;gap:15px;align-items:center;padding:15px;cursor:pointer}}summary small{{display:block;margin-top:3px}}summary b{{color:#efcf95}}.detail{{padding:0 15px 15px}}table{{border-collapse:collapse;width:100%;max-width:650px}}th,td{{padding:10px;border-bottom:1px solid #3a455c;text-align:left}}th{{color:#b4c0d5}}.note{{font-size:.87rem}}.market-table{{max-width:none;font-size:.88rem}}.market-table td{{vertical-align:top}}input{{padding:10px;background:#233451;border:1px solid #4b6084;color:white;border-radius:8px;width:min(100%,400px)}}.hide{{display:none}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main>
<small>SPORTS EDGE ENGINE · MOTOR ORIGINAL CONECTADO · PAPER RESEARCH</small><h1>Jornada {day}</h1><p>Generado {generated}. Estadísticas deportivas antes de mercados · probabilidades RAW del módulo original. Sin operaciones de apuesta.</p>
<div class="metrics"><div class="metric"><strong>{screened}</strong><small>Examinados</small></div><div class="metric"><strong>{modeled}</strong><small>RAW calculados</small></div><div class="metric"><strong>{low_sample}</strong><small>Muestra local/visitante insuficiente</small></div><div class="metric"><strong>{_text(decision_report.get('actionable_bets',0))}</strong><small>BET aprobadas</small></div></div>
<p>Historial multicompetición verificado para <b>{history_ok}/{screened}</b> partidos. Snapshot del motor canónico: <b>{canonical_status}</b>. Nunca usar datos de otro día como señal actual.</p><p class="note">{pregame_text}</p>
<div class="warning"><strong>ATENCIÓN:</strong> proyecciones Poisson limitadas y NO calibradas. <b>NO</b> extrapolar como recomendaciones. Para BET faltan ajuste histórico/OOS, XI y disponibilidad confirmados, y precios vigentes de la casa elegida.</div>
{source_html}
{decision_html}
{matrix_html}
{radar_html}
{market_section}
<h2>Partidos — perfiles y modelos RAW</h2>
<input id="search" aria-label="Filtrar partido" placeholder="Buscar equipo o fixture..." oninput="document.querySelectorAll('.game').forEach(x=>x.classList.toggle('hide',!x.textContent.toLowerCase().includes(this.value.toLowerCase())))"><div>{''.join(cards)}</div>
<p class="note">Fuente: API-Football Pro; motor existente fijado a commit 108c193c50e59f12f19f8693fc05c5cc9c607db4, función build_raw_projection. Sin captura de Caliente/Playdoit confirmada.</p></main></body></html>'''


def write_board(batch: dict[str, Any], summary: dict[str, Any], filename: str | Path, *,
                market_registry: dict[str, Any] | None = None,
                canonical: dict[str, Any] | None = None,
                advanced_context: dict[str, Any] | None = None,
                parlay_radar: dict[str, Any] | None = None,
                decision_report: dict[str, Any] | None = None,
                canonical_research: dict[str, Any] | None = None,
                bookmaker_catalogue: dict[str, Any] | None = None,
                market_matrix: dict[str, Any] | None = None) -> None:
    path=Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(render_board(batch,summary,market_registry=market_registry,canonical=canonical,advanced_context=advanced_context,parlay_radar=parlay_radar,decision_report=decision_report,canonical_research=canonical_research,bookmaker_catalogue=bookmaker_catalogue,market_matrix=market_matrix),encoding='utf-8')
    tmp.replace(path)