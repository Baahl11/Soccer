"""Past-only, all-competition team form inputs for existing Soccer raw model.

No bookmaker price, predicted future result or head-to-head shortcut enters this layer.
No history-based signal is a calibrated betting probability by itself.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable
import os

from .providers import ProviderFailure, now_iso


def _dt(value: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def normalize_finished_history(payload: dict[str, Any], *, team_id: int,
                               target_fixture: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """Fail closed on non-past, non-final, identity-mismatched match data."""
    if not isinstance(payload, dict) or not isinstance(payload.get('response'), list):
        raise ProviderFailure('HISTORY_PROVIDER_RESPONSE_UNVERIFIED')
    kickoff = _dt(target_fixture.get('kickoff'))
    if kickoff is None or not isinstance(team_id, int) or team_id <= 0:
        raise ProviderFailure('HISTORY_TARGET_IDENTITY_UNVERIFIED')
    found, ids = [], set()
    for item in payload['response']:
        if not isinstance(item, dict):
            continue
        f, teams = item.get('fixture') or {}, item.get('teams') or {}
        home, away = teams.get('home') or {}, teams.get('away') or {}
        goals = item.get('goals') or {}
        fid = f.get('id')
        start = _dt(f.get('date'))
        # Only completed games STRICTLY before target, no match outcomes from the future.
        if (not isinstance(fid, int) or fid in ids or fid == target_fixture.get('fixture_id')
                or start is None or start >= kickoff or (f.get('status') or {}).get('short') != 'FT'
                or team_id not in (home.get('id'), away.get('id'))):
            continue
        gh, ga = goals.get('home'), goals.get('away')
        if not all(isinstance(v, int) and 0 <= v <= 30 for v in (gh, ga)):
            continue
        ids.add(fid)
        found.append({
            'fixture_id': fid, 'kickoff': start.isoformat(), 'status': 'FT',
            'home_team_id': home.get('id'), 'away_team_id': away.get('id'),
            'league_id': (item.get('league') or {}).get('id'),
            'league_name': (item.get('league') or {}).get('name'),
            'goals': {'home': gh, 'away': ga},
        })
    found.sort(key=lambda row: row['kickoff'], reverse=True)
    return found[:limit]


def get_team_history(team_id: int, target_fixture: dict[str, Any],
                     api_get: Callable[[str, dict[str, Any]], dict[str, Any]],
                     *, limit: int = 12, ttl_minutes: int = 90) -> list[dict[str, Any]]:
    """Query past results across leagues, not just this competition or direct H2H."""
    if not 3 <= limit <= 20 or not 1 <= ttl_minutes <= 1440:
        raise ValueError('Invalid history limit or TTL')
    cache = Path(os.getenv('API_FOOTBALL_CACHE_DIR', 'data/api-football-cache'))
    key = cache / f'recent_all_competitions_{team_id}_{limit}.json'
    reply = None
    if key.exists():
        try:
            x = json.loads(key.read_text(encoding='utf-8'))
            checked = _dt(x['cached_at_utc'])
            age = datetime.now(timezone.utc) - checked if checked else timedelta(days=100)
            if timedelta(0) <= age <= timedelta(minutes=ttl_minutes):
                reply = x['provider_payload']
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if reply is None:
        reply = api_get('fixtures', {'team': team_id, 'last': limit, 'status': 'FT'})
        if not isinstance(reply, dict) or not isinstance(reply.get('response'), list):
            raise ProviderFailure('HISTORY_PROVIDER_RESPONSE_UNVERIFIED')
        cache.mkdir(parents=True, exist_ok=True)
        tmp = key.with_suffix('.tmp')
        tmp.write_text(json.dumps({'cached_at_utc': now_iso(), 'provider_payload': reply}, ensure_ascii=False), encoding='utf-8')
        tmp.replace(key)
    return normalize_finished_history(reply, team_id=team_id, target_fixture=target_fixture, limit=limit)