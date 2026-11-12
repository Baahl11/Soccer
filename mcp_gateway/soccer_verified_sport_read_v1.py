"""Evidence-first descriptive soccer analysis from verified existing sport sources.

Owner/private research only. Source redisplay/OOS is unverified; this product
reading is NEVER an executable recommendation, 1X2 probability, xG or a
post-event reconstruction of a pregame model. It uses only verified counts
already returned by an authorized league/season statistics lookup.
"""
from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

VERSION = "SOCCER_VERIFIED_SPORT_READ_v1.0.0"
MIN_VENUE_SAMPLE_FOR_PROJECTION = 5
MAX_FACTS = 12


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nonnegative(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 1000 else None


def _season_team(value: Any) -> dict[str, Any] | None:
    """Never create a fixture-level fact from malformed/mismatched samples."""
    if not isinstance(value, dict):
        return None
    played = _nonnegative(value.get("played"))
    wins = _nonnegative(value.get("wins"))
    draws = _nonnegative(value.get("draws"))
    losses = _nonnegative(value.get("losses"))
    gf = _nonnegative(value.get("goals_for"))
    ga = _nonnegative(value.get("goals_against"))
    venue = _nonnegative(value.get("played_venue"))
    if (played is None or not played or wins is None or draws is None
            or losses is None or gf is None or ga is None or venue is None
            or wins + draws + losses != played or venue > played):
        return None
    team_id = value.get("team_id")
    if type(team_id) is not int or team_id <= 0:
        return None
    return {
        "team_id": team_id,
        "team_name": str(value.get("team_name") or "")[:90],
        "competition": str(value.get("competition") or "")[:120],
        "season": value.get("season") if type(value.get("season")) is int else None,
        "venue": value.get("venue") if value.get("venue") in ("home", "away") else None,
        "played": played,
        "played_venue": venue,
        "wins": wins, "draws": draws, "losses": losses,
        "goals_for": gf, "goals_against": ga,
        "goals_for_venue": _nonnegative(value.get("goals_for_venue")),
        "goals_against_venue": _nonnegative(value.get("goals_against_venue")),
    }


def read_owner_context(packet: Any, fixture: Any) -> dict[str, Any]:
    """Compile descriptive, user-readable facts without requesting any data.

    Packet must come from the existing owner/admin-only API-Football retrieval,
    which validated team IDs and league/season. Do not use this for FREE/PRO.
    """
    blank = {
        "status": "NO_VERIFIED_SEASON_CONTEXT",
        "evidence_status": "NOT_VERIFIED",
        "headline": "No hay estadísticas suficientes para una lectura documentada.",
        "facts": [],
        "warnings": ["Estadísticas de temporada no verificadas para ambos equipos."],
        "teams": [],
        "minimum_venue_sample": None,
        "sufficient_for_probability": False,
        "model_probabilities": None,
        "source": "NOT_VERIFIED",
        "observed_at_utc": None,
        "source_redisplay_rights_verified": False,
        "commercial_display_allowed": False,
        "owner_admin_only": True,
        "bet_eligible": False,
        "market_queried": False,
        "provenance": "SOURCE_READ_ONLY_NOT_ASOF_PREGAME_MODEL",
    }
    source = _dict(packet)
    fx = _dict(fixture)
    if (source.get("status") != "SPORTING_SEASON_CONTEXT_ONLY"
            or source.get("source") not in ("API_FOOTBALL_TEAM_SEASON_STATISTICS",
                                            "API_FOOTBALL_TEAM_SEASON_SPLITS")
            or source.get("owner_admin_only") is not True
            or source.get("commercial_display_allowed") is not False):
        return blank
    rows = [r for r in (_season_team(v) for v in source.get("teams") or [])
            if r is not None]
    home_id = fx.get("home_team_id")
    away_id = fx.get("away_team_id")
    home = next((r for r in rows if r["team_id"] == home_id and r["venue"] == "home"), None)
    away = next((r for r in rows if r["team_id"] == away_id and r["venue"] == "away"), None)
    valid = [x for x in (home, away) if x is not None]
    if not valid:
        return blank
    facts: list[str] = []
    for row, fallback in ((home, "Local"), (away, "Visitante")):
        if row is None:
            continue
        name = row["team_name"] or fallback
        facts.append(
            f"{name}: {row['wins']} victorias, {row['draws']} empates y "
            f"{row['losses']} derrotas en {row['played']} partidos "
            f"de {row['competition']} {row['season']}; "
            f"{row['goals_for']} goles a favor y {row['goals_against']} en contra."
        )
        if row["played_venue"] > 0:
            location = "local" if row["venue"] == "home" else "visitante"
            gf = row["goals_for_venue"]
            ga = row["goals_against_venue"]
            if gf is not None and ga is not None:
                facts.append(
                    f"Como {location}: {row['played_venue']} partidos, "
                    f"{gf} goles marcados y {ga} recibidos. "
                    "Son resultados observados, no xG."
                )
    both = bool(home and away)
    comparable = (both and home["competition"] == away["competition"]
                  and home["season"] == away["season"])
    minimum = min((r["played_venue"] for r in valid), default=0)
    small = minimum < MIN_VENUE_SAMPLE_FOR_PROJECTION
    warnings = []
    if small:
        warnings.append(
            f"Muestra local/visitante mínima de {minimum} partido(s); "
            f"se requieren al menos {MIN_VENUE_SAMPLE_FOR_PROJECTION} por equipo "
            "para intentar un baseline limitado."
        )
    if not both:
        warnings.append("Faltan estadísticas verificadas de uno de los equipos.")
    if both and not comparable:
        warnings.append(
            "Equipos con competiciones o temporadas distintas: "
            "las cifras no son directamente comparables."
        )
    warnings.extend([
        "Calidad de rivales, xG, lesiones y alineaciones: NOT VERIFIED.",
        "Estas cifras son contexto histórico observado, NO pronóstico 1X2 "
        "ni recomendación de valor.",
    ])
    return {
        **blank,
        "status": "VERIFIED_SEASON_CONTEXT" if both else "PARTIAL_SEASON_CONTEXT",
        "evidence_status": ("LOW_SAMPLE_EVIDENCE_ONLY" if small else
                            "OBSERVED_SEASON_EVIDENCE_ONLY"),
        "headline": ("Lectura histórica verificada · muestra pequeña"
                     if small else "Lectura histórica verificada · sin modelo 1X2"),
        "facts": facts[:MAX_FACTS],
        "warnings": warnings,
        "teams": valid,
        "minimum_venue_sample": minimum,
        "sufficient_for_probability": False,  # Requires independently calibrated model.
        "source": source["source"],
        "observed_at_utc": source.get("fetched_at_utc"),
        "same_competition_and_season": comparable,
        "season_context_verified": both,
    }


def _source_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def _as_p0_team(
    stats: Any, *, team_id: int, team_name: str,
    venue: str, league_name: str, season: int,
) -> dict[str, Any] | None:
    """Only verified compact API-Football season fields, never raw projections."""
    d = _dict(stats)
    f = _dict(d.get("fixtures"))
    g = _dict(d.get("goals"))
    wins = _dict(f.get("wins"))
    draws = _dict(f.get("draws"))
    loses = _dict(f.get("loses"))
    played = _dict(f.get("played"))
    goals_for = _dict(_dict(g.get("for")).get("total"))
    goals_against = _dict(_dict(g.get("against")).get("total"))
    return _season_team({
        "team_id": team_id, "team_name": team_name, "venue": venue,
        "competition": league_name, "season": season,
        "played": played.get("total"), "played_venue": played.get(venue),
        "wins": wins.get("total"), "draws": draws.get("total"),
        "losses": loses.get("total"), "goals_for": goals_for.get("total"),
        "goals_against": goals_against.get("total"),
        "goals_for_venue": goals_for.get(venue),
        "goals_against_venue": goals_against.get(venue),
    })


def source_context_from_persisted_p0(
    events: Any, fixture: Any,
) -> dict[str, Any]:
    """Reuse immutable as-of sport evidence from PostgreSQL, never market data.

    The raw projection and quote fields are deliberately discarded. The
    source must match the registered fixture identities and pre-kickoff time.
    Output is owner/admin private only, not a commercial right to redistribute.
    """
    fx = _dict(fixture)
    fid, league, season = (fx.get("fixture_id"),
                           fx.get("league_id"), fx.get("season"))
    home, away = fx.get("home_team_id"), fx.get("away_team_id")
    kickoff = _source_datetime(fx.get("kickoff"))
    empty = {"status": "NOT_VERIFIED", "teams": [],
             "source_redisplay_rights_verified": False,
             "commercial_display_allowed": False,
             "owner_admin_only": True, "provider_requests_added": 0}
    if (not all(type(x) is int and x > 0 for x in (fid,league,season,home,away))
            or kickoff is None or not isinstance(events, list)):
        return empty
    for stored in events[:12]:
        e = _dict(stored)
        row = _dict(e.get("payload"))
        event_fixture = _dict(row.get("fixture"))
        if (row.get("event_type") != "SOCCER_REFRESH"
                or row.get("stage") != "SPORT_PREVIEW"
                or row.get("source") != "API_FOOTBALL_TEAM_SEASON_SPLITS"
                or row.get("quote_requested") is not False
                or row.get("bet_eligible") is not False
                or any(event_fixture.get(k) != fx.get(k)
                       for k in ("fixture_id", "home_team_id",
                                 "away_team_id", "league_id", "season"))):
            continue
        observed = _source_datetime(row.get("source_observed_at_utc"))
        if observed is None or observed >= kickoff:
            continue
        sporting = _dict(row.get("sporting"))
        h = _as_p0_team(sporting.get("home_stats"),
                         team_id=home, team_name=str(fx.get("home_team") or ""),
                         venue="home", league_name=str(fx.get("league") or ""),
                         season=season)
        a = _as_p0_team(sporting.get("away_stats"),
                         team_id=away, team_name=str(fx.get("away_team") or ""),
                         venue="away", league_name=str(fx.get("league") or ""),
                         season=season)
        teams = [x for x in (h, a) if x is not None]
        if teams:
            return {
                "status": "SPORTING_SEASON_CONTEXT_ONLY",
                "source": "API_FOOTBALL_TEAM_SEASON_SPLITS",
                "teams": teams,
                "owner_admin_only": True,
                "commercial_display_allowed": False,
                "source_redisplay_rights_verified": False,
                "fetched_at_utc": observed.isoformat(),
                "provider_requests_added": 0,
                "projection": None,
                "bet_eligible": False,
                "market_queried": False,
                "from_persisted_pregame": True,
            }
    return empty
