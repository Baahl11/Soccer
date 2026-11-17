"""Bounded, private replay of persisted research outputs; no provider calls."""
from collections import Counter
from datetime import datetime, timedelta, timezone

FRESH_CAPTURE_AFTER = datetime.fromisoformat("2026-11-16T19:52:31.554902+00:00")

FIXTURES = (1647362, 1640518, 1647364)
PANEL_CONTROLS = (1647367, 1545887)
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


def historical_player_families(events, fixture, now, replay=None):
    """Validate archived inputs at capture, never imply present OWNER visibility."""
    if replay is None:
        from mcp_gateway.player_prop_sport_first_bridge_v1 import saved_match_profiles as replay

    found = set()
    for event in events:
        payload = event["payload"]
        relevant = {f for f in ("SHOTS", "SOT") if f not in found and isinstance(payload.get(FIELDS[f]), dict)}
        if not relevant:
            continue
        stamp = datetime.fromisoformat(event["generated_at"])
        if stamp > now:
            continue
        result = replay([event], fixture, now_utc=stamp)
        found.update(row.get("family") for row in result.get("rows", []) if row.get("family") in relevant)
        if len(found) == 2:
            break
    return found


PLAYER_DIAGNOSTIC_KEYS = ['XI_NOT_CONFIRMED','OTHER_PRODUCER_STATE','MODEL_OUTPUT_EVENTS','NO_MODELED_PLAYERS','NO_MATCHED_PREGAME_PLAYER_SNAPSHOT','ACCEPTED_EVENTS','OTHER_VALIDATOR_REASON','SNAPSHOT_NOT_OBJECT','NOT_VERIFIED_ASOF_FIXTURE','MODEL_FIXTURE_MISMATCH','MODEL_PLAYER_LIST_MISSING','MODEL_PLAYER_ID_XI_NOT_VERIFIED','PLAYER_REGISTRY_ASOF_NOT_VERIFIED','PLAYER_REGISTRY_TIMESTAMP_MISSING','PLAYER_REGISTRY_TIMESTAMP_INVALID','PLAYER_REGISTRY_TIMESTAMP_AFTER_CAPTURE','HISTORICAL_PLAYER_SAMPLE_INSUFFICIENT','HISTORICAL_PLAYER_MINUTES_INSUFFICIENT','RAW_SHOTS_RATE_NOT_VERIFIED','EXPECTED_MINUTES_NOT_VERIFIED','OPPONENT_CONTEXT_NOT_VERIFIED','MODEL_LINE_DISTRIBUTION_NOT_VERIFIED']


def player_diagnostics(events, fixture, now, replay=None):
    """Count saved event states and family-specific first validator blockers."""
    if replay is None:
        from mcp_gateway.player_prop_sport_first_bridge_v1 import saved_match_profiles
        from functools import partial
        replay = partial(saved_match_profiles, audit_registry_time=True)

    counters = {family: Counter() for family in ("SHOTS", "SOT")}
    for event in events:
        stamp = datetime.fromisoformat(event["generated_at"])
        if stamp > now:
            continue
        for family, counts in counters.items():
            key = FIELDS[family]
            intel = event["payload"].get(key)
            if not isinstance(intel, dict):
                continue
            if intel.get("reason") == "BOTH_STARTING_XI_NOT_CONFIRMED":
                counts["XI_NOT_CONFIRMED"] += 1
                continue
            if intel.get("status") != f"LIVE_RESEARCH_{family}_PROPS":
                counts["OTHER_PRODUCER_STATE"] += 1
                continue
            counts["MODEL_OUTPUT_EVENTS"] += 1
            if intel.get("modeled_players") == 0:
                counts["NO_MODELED_PLAYERS"] += 1
            other = FIELDS["SOT" if family == "SHOTS" else "SHOTS"]
            payload = {k: v for k, v in event["payload"].items() if k != other}
            result = replay([{**event, "payload": payload}], fixture, now_utc=stamp)
            if result.get("rows"):
                counts["ACCEPTED_EVENTS"] += 1
            if result.get("reason") == "NO_MATCHED_PREGAME_PLAYER_SNAPSHOT":
                counts["NO_MATCHED_PREGAME_PLAYER_SNAPSHOT"] += 1
            for reason, count in result.get("unverified_input_reasons", {}).items():
                counts[reason if reason in PLAYER_DIAGNOSTIC_KEYS else "OTHER_VALIDATOR_REASON"] += count
    return counters


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


def fresh_captures(events):
    return [event for event in events
            if datetime.fromisoformat(event["generated_at"]) > FRESH_CAPTURE_AFTER]


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
    fresh_archive = fresh_captures(archive)
    historical = historical_player_families(fresh_archive, fixture, now)
    return {"fixture_id": fixture_id, "eventos_en_ventana": len(window),
            "historical_player_families": sorted(historical),
            "player_diagnostics": player_diagnostics(fresh_archive, fixture, now),
            "archivo_truncado": truncated,
            "familias": {f: summarize(key, window, archive, current[f], full[f], truncated)
                         for f, key in FIELDS.items()}}


LOOKBACK_DAYS = 30
CANDIDATE_LIMIT = 8
PAGE_SIZE = 24
EVENT_SCAN_LIMIT = 96


def failure_reason(exc):
    return {"57014": "QUERY_TIMEOUT", "55P03": "LOCK_TIMEOUT"}.get(
        getattr(exc, "sqlstate", None), "DISCOVERY_ERROR")


def candidate_page(cur, now, cursor):
    params = [max(now - timedelta(days=LOOKBACK_DAYS), FRESH_CAPTURE_AFTER), now]
    clause = ""
    if cursor is not None:
        clause = " AND (generated_at, event_id) < (%s, %s)"
        params.extend(cursor)
    cur.execute("SELECT event_id, fixture_id, generated_at FROM soccer_refresh_events "
                "WHERE generated_at > %s AND generated_at <= %s" + clause
                + " ORDER BY generated_at DESC, event_id DESC LIMIT %s", (*params, PAGE_SIZE))
    return cur.fetchall()


def candidate_fields(cur, page):
    if not page:
        return []
    cur.execute("""
        SELECT e.event_id, ARRAY(
            SELECT field FROM unnest(%s::text[]) AS keys(field)
            WHERE jsonb_typeof(e.payload -> field) = 'object'
        )
        FROM unnest(%s::bigint[]) AS ids(id)
        CROSS JOIN LATERAL (
            SELECT payload, fixture_id, generated_at, event_id
            FROM soccer_refresh_events WHERE event_id = ids.id LIMIT 1
        ) e
        JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
        WHERE e.generated_at < f.kickoff
        """, (list(FIELDS.values()), [row[0] for row in page]))
    return cur.fetchall()


def discover(cur, now, cursor=None):
    by_field = {key: [] for key in FIELDS.values()}
    scanned = 0
    complete = False
    failure = None
    for _ in range(EVENT_SCAN_LIMIT // PAGE_SIZE):
        phase = "EVENT_METADATA"
        try:
            with cur.connection.transaction():
                page = candidate_page(cur, now, cursor)
                phase = "EVENT_PAYLOAD_KEYS"
                by_event = dict(candidate_fields(cur, page))
        except Exception as exc:
            failure = {"phase": phase, "reason": failure_reason(exc)}
            break
        for event_id, fixture_id, _ in page:
            for field in by_event.get(event_id, ()):
                if fixture_id not in by_field[field]:
                    by_field[field].append(fixture_id)
        scanned += len(page)
        if page:
            cursor = (page[-1][2], page[-1][0])
        if len(page) < PAGE_SIZE:
            complete = True
            break
    errors = {}
    cache = {}
    output = {}
    for family, field in FIELDS.items():
        ids = by_field[field]
        checked = 0
        incomplete = 0
        positive = None
        historical_positive = None
        diagnostics = Counter()
        for fixture_id in ids[:CANDIDATE_LIMIT]:
            if len(errors) >= 2 and fixture_id not in cache and fixture_id not in errors:
                break
            if fixture_id not in cache and fixture_id not in errors:
                try:
                    with cur.connection.transaction():
                        cache[fixture_id] = audit_fixture(cur, fixture_id, now)
                except Exception as exc:
                    errors[fixture_id] = failure_reason(exc)
            row = cache.get(fixture_id, {})
            checked += 1
            diagnostics.update(row.get("player_diagnostics", {}).get(family, {}))
            if historical_positive is None and family in row.get("historical_player_families", []):
                historical_positive = fixture_id
            state = row.get("familias", {}).get(family, {}).get("estado")
            if state in {"ACEPTADO_EN_REPLAY_DE_FICHA", "ACEPTADO_FUERA_DE_ULTIMOS_12"}:
                positive = {"fixture_id": fixture_id, "estado": state}
                break
            incomplete += int(state in {None, "ARCHIVO_TRUNCADO_NO_CONCLUYENTE"})
        output[family] = {"candidatos_en_corte": min(len(ids), CANDIDATE_LIMIT),
                          "hay_mas_candidatos": len(ids) > CANDIDATE_LIMIT,
                          "revisados": checked, "no_concluyentes": incomplete, "positivo": positive,
                          "historical_positive_fixture_id": historical_positive,
                          "diagnostics": dict(diagnostics)}
    return {"lookback_days": LOOKBACK_DAYS, "candidate_limit": CANDIDATE_LIMIT,
            "event_scan_limit": EVENT_SCAN_LIMIT, "families": output,
            "events_scanned": scanned, "scan_complete": complete,
            "next_before_event_id": cursor[1] if cursor and not complete else None,
            "scan_failure": failure, "fixture_failures": [
                {"fixture_id": fixture_id, "reason": reason} for fixture_id, reason in errors.items()]}


def panel_controls(cur, now, replay=None):
    replay = replay or audit_fixture
    output = {}
    for fixture_id in PANEL_CONTROLS:
        try:
            with cur.connection.transaction():
                result = replay(cur, fixture_id, now)
            output[str(fixture_id)] = {
                "state": result.get("estado", "REPLAYED"),
                "window_events": result.get("eventos_en_ventana"),
                "archive_truncated": result.get("archivo_truncado"),
                "families": result.get("familias", {}),
            }
        except Exception as exc:
            output[str(fixture_id)] = {"state": failure_reason(exc), "window_events": None,
                                       "archive_truncated": None, "families": {}}
    return output


class DiscoveryFailure(RuntimeError):
    pass


def run_audit(before_id=None):
    from mcp_gateway import persistence

    now = datetime.now(timezone.utc)
    with persistence._connect() as conn:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '2500ms'")
            cur.execute("SET LOCAL lock_timeout = '750ms'")
            try:
                cursor = None
                if before_id is not None:
                    cur.execute("SELECT generated_at, event_id FROM soccer_refresh_events WHERE event_id = %s", (before_id,))
                    cursor = cur.fetchone()
                    if cursor is None:
                        raise ValueError("unknown cursor")
                discovery = discover(cur, now, cursor)
                controls = panel_controls(cur, now)
            except Exception as exc:
                raise DiscoveryFailure(failure_reason(exc)) from exc
            from mcp_gateway.corners_evidence_inventory_v1 import inventory
            try:
                corners_evidence = inventory(cur, now)
                corners_evidence_error = None
            except Exception as exc:
                corners_evidence, corners_evidence_error = [], failure_reason(exc)
            from mcp_gateway.oos_canonical_pair_inventory_v1 import inventory as pair_inventory
            try:
                with cur.connection.transaction():
                    pairs = pair_inventory(cur, now)
                pair_error = None
            except Exception as exc:
                pairs, pair_error = None, failure_reason(exc)
    return {"audit_version": 10, "canonical_pair_coverage": pairs, "canonical_pair_error": pair_error, "panel_controls": controls, "fresh_capture_after_utc": FRESH_CAPTURE_AFTER.isoformat(),
            "audit_at_utc": now.isoformat(), "window": WINDOW,
            "scope": "POSTGRES_VALIDATOR_REPLAY_NOT_OWNER_HTTP_OR_BROWSER", "discovery": discovery,
            "corners_evidence": corners_evidence, "corners_evidence_error": corners_evidence_error}
