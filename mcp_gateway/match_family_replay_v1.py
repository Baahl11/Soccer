"""Bounded, private replay of persisted research outputs; no provider calls."""
from datetime import datetime, timedelta, timezone

FIXTURES = (1647362, 1640518, 1647364)
WINDOW = 12
ARCHIVE_LIMIT = 200
FIELDS = {
    "GOLES_EQUIPO": "team_totals_intelligence",
    "CORNERS_FT": "corners_intelligence",
    "AMARILLAS_FT": "cards_intelligence_live",
    "AMARILLAS_EQUIPO": "team_cards_intelligence",
    "ROJA_SI_NO": "red_cards_intelligence",
    "SHOTS": "player_shots_intelligence",
    "SOT": "player_sot_intelligence",
}


def summarize(field, window, archive, current, full, truncated):
    def count(events):
        return sum(event["payload"].get(field) is not None for event in events)

    if current:
        state = "ACEPTADO_EN_REPLAY_DE_FICHA"
    elif full:
        state = "ACEPTADO_FUERA_DE_ULTIMOS_12"
    elif truncated:
        state = "ARCHIVO_TRUNCADO_NO_CONCLUYENTE"
    elif count(archive):
        state = "PERSISTIDO_PREPARTIDO_NO_ACEPTADO"
    else:
        state = "SIN_CAMPO_PREPARTIDO_EN_REFRESH_EVENTS"
    return {"estado": state, "eventos_con_campo_ventana": count(window),
            "eventos_prepartido_con_campo": count(archive),
            "unidades_aceptadas_ventana": current, "unidades_aceptadas_archivo": full}


def accepted(events, fixture, now):
    from mcp_gateway import (cards_intelligence_live as cards, corners_intelligence as corners,
                             player_prop_sport_first_bridge_v1 as players, red_cards_intelligence as reds,
                             team_totals_intelligence as goals)

    grid = goals.saved_match_projection(events, fixture)
    props = players.saved_match_profiles(events, fixture, now_utc=now)
    return {
        "GOLES_EQUIPO": len(grid.get("rows", [])) if grid["status"] == "PERSISTED_RESEARCH_PROJECTION" else 0,
        "CORNERS_FT": int(corners.saved_match_rates(events, fixture)["status"] == "PERSISTED_RESEARCH_RATES"),
        "AMARILLAS_FT": int(cards.saved_match_rates(events, fixture)["status"] == "PERSISTED_RESEARCH_RATES"),
        "AMARILLAS_EQUIPO": int(cards.saved_match_rates(events, fixture, by_team=True)["status"] == "PERSISTED_RESEARCH_RATES"),
        "ROJA_SI_NO": int(reds.saved_match_probability(events, fixture)["status"] == "PERSISTED_RESEARCH_PROJECTION"),
        **{f: sum(row.get("family") == f for row in props.get("rows", [])) for f in ("SHOTS", "SOT")},
    }


def read_events(cur, fixture_id, kickoff, now, *, archive):
    query = ("SELECT stage, event_type, generated_at, payload FROM soccer_refresh_events "
             "WHERE fixture_id = %s")
    params = (fixture_id,)
    if archive:
        query += " AND generated_at < %s AND generated_at <= %s AND payload ?| %s::text[]"
        params += (kickoff, now, list(FIELDS.values()))
    cur.execute(query + " ORDER BY generated_at DESC, event_id DESC LIMIT %s",
                (*params, ARCHIVE_LIMIT + 1 if archive else WINDOW))
    return [{"stage": stage, "event_type": event, "generated_at": stamp.isoformat(),
             "payload": payload if isinstance(payload, dict) else {}}
            for stage, event, stamp, payload in cur.fetchall()]


def audit_fixture(cur, fixture_id, now):
    cur.execute("""SELECT fixture_id, league_id, season, home_team_id, away_team_id,
                          kickoff, home_team, away_team
                   FROM soccer_fixtures WHERE fixture_id = %s""", (fixture_id,))
    row = cur.fetchone()
    if (not row or any(type(x) is not int or x <= 0 for x in row[:5])
            or row[3] == row[4] or not row[5] or row[5].utcoffset() is None):
        return {"fixture_id": fixture_id, "estado": "IDENTIDAD_CANONICA_NO_VERIFICADA"}
    keys = ("fixture_id", "league_id", "season", "home_team_id", "away_team_id",
            "kickoff", "home_team", "away_team")
    fixture = dict(zip(keys, row))
    fixture["kickoff"] = row[5].isoformat()
    window = read_events(cur, fixture_id, row[5], now, archive=False)
    archive = read_events(cur, fixture_id, row[5], now, archive=True)
    truncated = len(archive) > ARCHIVE_LIMIT
    archive = archive[:ARCHIVE_LIMIT]
    current, full = accepted(window, fixture, now), accepted(archive, fixture, now)
    return {"fixture_id": fixture_id, "eventos_en_ventana": len(window),
            "archivo_truncado": truncated,
            "familias": {f: summarize(key, window, archive, current[f], full[f], truncated)
                         for f, key in FIELDS.items()}}


LOOKBACK_DAYS = 30
CANDIDATE_LIMIT = 8


def candidate_ids(cur, now):
    """Recent matching fixture IDs, independently capped per family."""
    cur.execute("""
        WITH ranked AS (
            SELECT k.field, e.fixture_id, max(e.generated_at) AS latest
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f USING (fixture_id)
            CROSS JOIN unnest(%s::text[]) AS k(field)
            WHERE e.generated_at >= %s AND e.generated_at <= %s
              AND e.generated_at < f.kickoff
              AND jsonb_typeof(e.payload -> k.field) = 'object'
            GROUP BY k.field, e.fixture_id
        ), numbered AS (
            SELECT *, row_number() OVER (
                PARTITION BY field ORDER BY latest DESC, fixture_id
            ) AS rn
            FROM ranked
        )
        SELECT field, fixture_id FROM numbered WHERE rn <= %s
        ORDER BY field, rn
        """, (list(FIELDS.values()), now - timedelta(days=LOOKBACK_DAYS), now, CANDIDATE_LIMIT + 1))
    out = {key: [] for key in FIELDS.values()}
    for field, fixture_id in cur.fetchall():
        out[field].append(fixture_id)
    return out


def discover(cur, now):
    by_field = candidate_ids(cur, now)
    cache = {}
    output = {}
    for family, field in FIELDS.items():
        ids = by_field[field]
        checked = 0
        incomplete = 0
        positive = None
        for fixture_id in ids[:CANDIDATE_LIMIT]:
            if fixture_id not in cache:
                cache[fixture_id] = audit_fixture(cur, fixture_id, now)
            row = cache[fixture_id]
            checked += 1
            state = row.get("familias", {}).get(family, {}).get("estado")
            if state in {"ACEPTADO_EN_REPLAY_DE_FICHA", "ACEPTADO_FUERA_DE_ULTIMOS_12"}:
                positive = {"fixture_id": fixture_id, "estado": state}
                break
            incomplete += int(state in {None, "ARCHIVO_TRUNCADO_NO_CONCLUYENTE"})
        output[family] = {"candidatos_en_corte": min(len(ids), CANDIDATE_LIMIT),
                          "hay_mas_candidatos": len(ids) > CANDIDATE_LIMIT,
                          "revisados": checked, "no_concluyentes": incomplete, "positivo": positive}
    return {"lookback_days": LOOKBACK_DAYS, "candidate_limit": CANDIDATE_LIMIT, "families": output}


def run_audit():
    from mcp_gateway import persistence

    now = datetime.now(timezone.utc)
    with persistence._connect() as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '2500ms'")
            cur.execute("SET LOCAL lock_timeout = '750ms'")
            results = [audit_fixture(cur, fixture_id, now) for fixture_id in FIXTURES]
            discovery = discover(cur, now)
    return {"audit_version": 1, "audit_at_utc": now.isoformat(), "window": WINDOW,
            "scope": "POSTGRES_VALIDATOR_REPLAY_NOT_OWNER_HTTP_OR_BROWSER", "fixtures": results,
            "discovery": discovery}
