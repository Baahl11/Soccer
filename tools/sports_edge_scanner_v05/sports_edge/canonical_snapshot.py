"""Read-only observation of the existing deployed Soccer pipeline market views.

Must NOT be used to build an independent raw sport projection or grant BET.
In particular a persisted October pipeline row is not a November fixture forecast.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Any, Callable

CANONICAL_URL = 'https://soccer-edge-api.onrender.com/product/views?limit=500'
VIEW_NAMES = ('todays_slate', 'strong_sport_signals', 'value_plays', 'waiting_for_price',
              'waiting_for_xi', 'team_totals', 'first_half', 'second_half', 'corners',
              'player_props', 'model_health', 'data_health', 'control_tower')
SAFE_FIELDS = ('fixture_id','home','away','market','market_family','selection', 'line',
               'execution_status','model_signal','model_signal_score','stage','status',
               'raw_sport_probability','raw_projection_model_version','p_shrunk',
               'phase16_calibration_status','blockers','bookmaker','source_updated_at',
               'price','reason', 'kickoff')


def _dt(value: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def fetch_canonical_market_views(fetch_fn: Callable[[], dict[str, Any]] | None = None,
                                 *, now: datetime | None = None, max_age_hours: float = 6) -> dict[str, Any]:
    """Fetch public read-only persisted output; do not trigger /internal/tick."""
    snapshot = {'source': CANONICAL_URL, 'no_provider_requests_added_by_endpoint': True,
                'status': 'NOT_VERIFIED', 'is_fresh': False, 'bet_eligible': False,
                'matching_fixture_rows': [], 'views': {}, 'error': None}
    try:
        if fetch_fn is None:
            req = Request(CANONICAL_URL, headers={'Accept':'application/json', 'User-Agent':'SportsEdgeResearch/0.6'})
            with urlopen(req, timeout=20) as resp:
                if int(resp.status) != 200:
                    raise ValueError('CANONICAL_HTTP_NON_200')
                body = resp.read(4_000_001)
                if len(body) > 4_000_000:
                    raise ValueError('CANONICAL_RESPONSE_TOO_LARGE')
                payload = json.loads(body)
        else:
            payload = fetch_fn()
        if not isinstance(payload, dict) or not isinstance(payload.get('views'), dict):
            raise ValueError('CANONICAL_SCHEMA_MISMATCH')
        if payload.get('production_promotion_allowed') is True:
            raise ValueError('UNEXPECTED_CANONICAL_PRODUCTION_PROMOTION')
        captured = _dt(payload.get('generated_at_utc'))
        ref = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (ref-captured).total_seconds()/3600 if captured else None
        if age is None or age < 0 or age > max_age_hours:
            snapshot['status'] = 'STALE_OR_UNDATED_CANONICAL_PIPELINE'
        else:
            snapshot['status'] = 'FRESH_RESEARCH_ONLY'
            snapshot['is_fresh'] = True
        snapshot['generated_at_utc'] = payload.get('generated_at_utc')
        snapshot['age_hours'] = round(age, 3) if age is not None else None
        snapshot['schema_version'] = payload.get('schema_version')
        snapshot['model_version'] = payload.get('model_version')
        for key in VIEW_NAMES:
            v = payload['views'].get(key)
            if not isinstance(v, dict):
                continue
            rows = v.get('rows')
            parsed_rows = []
            for row in rows if isinstance(rows,list) else []:
                if not isinstance(row, dict):
                    continue
                parsed_rows.append({f:row[f] for f in SAFE_FIELDS if f in row})
            snapshot['views'][key] = {'total': v.get('total'), 'rows': parsed_rows[:500]}
        return snapshot
    except (ValueError, OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        snapshot['error'] = type(exc).__name__
        return snapshot


def match_canonical_fixtures(snapshot: dict[str, Any], fixture_ids: set[int]) -> dict[str, Any]:
    """Only exact fixture IDs from a *fresh* persisted pipeline can be shown as current research."""
    if not snapshot.get('is_fresh'):
        return {**snapshot, 'matching_fixture_rows': [], 'status': snapshot.get('status','NOT_VERIFIED')}
    matches = []
    for view, node in snapshot.get('views', {}).items():
        for row in node.get('rows', []):
            try:
                fid = int(row['fixture_id'])
            except (KeyError, ValueError, TypeError):
                continue
            if fid in fixture_ids:
                matches.append({'view':view, **row, 'classification':'WATCH',
                                'bet_eligible':False, 'source':'CANONICAL_READ_ONLY'})
    return {**snapshot, 'matching_fixture_rows':matches}