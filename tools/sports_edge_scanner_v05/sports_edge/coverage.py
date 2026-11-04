"""Coverage and access auditing, including renewed API-Football Pro access.

Key rules: never infer that a response truncated at 100 is a full slate;
current fixtures/odds availability is distinct from league-season stats access;
zero lineup/injury records do not prove confirmed starters or healthy squads.
No external requests are made by this module.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def audit_fixture_response(raw: dict[str, Any], *, source_kind: str = 'connected') -> dict[str, Any]:
    """Accept either connector payload (fixtures) or direct /fixtures (response).

    `eligible_count` is only a count of *observed* not-started games.
    If the source is incomplete, all league-level coverage assertions are UNKNOWN.
    """
    if not isinstance(raw, dict):
        raise ValueError('Fixture payload must be a JSON object')
    if source_kind == 'connected':
        fixtures = raw.get('fixtures')
        reported = _as_int(raw.get('total_results'), -1)
        already_truncated = raw.get('truncated') is True
    elif source_kind == 'direct':
        fixtures = raw.get('response')
        reported = _as_int(raw.get('results'), -1)
        # /fixtures?date normally returns the complete array; do not assume that.
        pages = _as_int((raw.get('paging') or {}).get('total'), 1)
        already_truncated = pages > 1
    else:
        raise ValueError('source_kind must be connected or direct')
    if not isinstance(fixtures, list) or reported < 0:
        raise ValueError('Missing fixtures/results; cannot audit completeness')
    received = len(fixtures)
    complete = not already_truncated and received == reported
    statuses: dict[str, int] = {}
    observed_leagues: dict[str, int] = {}
    for item in fixtures:
        if source_kind == 'connected':
            status = str(item.get('status') or 'UNKNOWN')
            lid = item.get('league_id')
            league_name = item.get('league', 'UNKNOWN')
        else:
            status = str((item.get('fixture') or {}).get('status', {}).get('short') or 'UNKNOWN')
            l = item.get('league') or {}
            lid, league_name = l.get('id'), l.get('name', 'UNKNOWN')
        statuses[status] = statuses.get(status, 0) + 1
        league_key = f'{lid}:{league_name}'
        observed_leagues[league_key] = observed_leagues.get(league_key, 0) + 1
    result = {
        'source_kind': source_kind,
        'match_date': raw.get('match_date') if source_kind == 'connected' else None,
        'timezone': raw.get('timezone') if source_kind == 'connected' else None,
        'retrieved_at_utc': raw.get('retrieved_at_utc'),
        'total_reported': reported,
        'received': received,
        'missing_minimum': max(0, reported - received),
        'complete_slate_verified': complete,
        'coverage_status': 'COMPLETE' if complete else 'INCOMPLETE',
        'count_observed_not_started': statuses.get('NS', 0),
        'observed_status_counts': statuses,
        'observed_league_counts': observed_leagues,
        'league_coverage_verified': complete,
        'recommended_action': ('SPORTING_SCREEN_ALLOWED_AFTER_LEAGUE_ELIGIBILITY_FILTER' if complete
            else 'DO_NOT_CLAIM_FULL_SLATE; USE_DIRECT_AUTHORIZED_FIXTURES_DATE_WITH_EXISTING_KEY_OR_DOCUMENT_PARTIAL_COVERAGE'),
    }
    return result


def _error_note(error: Any) -> str:
    if error is None:
        return ''
    if isinstance(error, dict):
        return '; '.join(f'{k}: {v}' for k, v in error.items())
    return str(error)


def audit_capabilities(probes: dict[str, Any]) -> dict[str, Any]:
    """Read-only interpretation of observed probe replies.

    Probe entries may contain a normal connector JSON payload or an error object
    {error:{plan:'Free plans do not have access ...'}}; never guess status.
    This is *not* a subscription management API.
    """
    if not isinstance(probes, dict):
        raise ValueError('Probe input must be a JSON object')
    expected = ['fixture', 'odds', 'lineups', 'injuries', 'match_stats', 'league_coverage', 'team_stats']
    out: dict[str, Any] = {}
    remaining = []
    for name in expected:
        item = probes.get(name)
        if item is None:
            out[name] = {'state':'NOT_TESTED', 'interpretation':'NOT_VERIFIED'}
            continue
        if not isinstance(item, dict):
            out[name] = {'state':'INVALID_PROBE', 'interpretation':'NOT_VERIFIED'}
            continue
        error = item.get('error') or item.get('errors')
        err_note = _error_note(error)
        if error:
            out[name] = {'state':'BLOCKED_BY_PLAN' if 'plan' in err_note.lower() or 'free plans' in err_note.lower()
                         else 'PROVIDER_ERROR', 'interpretation':'NOT_VERIFIED', 'detail':err_note}
        else:
            rows = item.get('response', [])
            if not isinstance(rows, (list,dict)):
                out[name] = {'state':'INVALID_PROBE', 'interpretation':'NOT_VERIFIED'}
            elif rows:
                # /teams/statistics returns an object, not an array.
                count = len(rows) if isinstance(rows,list) else 1
                out[name] = {'state':'RETURNED_DATA', 'records':count,'interpretation':'PROVIDER_DATA_PRESENT_NOT_SPORTING_EDGE'}
            elif name == 'lineups':
                out[name] = {'state':'EMPTY', 'records':0,'interpretation':'LINEUP_NOT_VERIFIED'}
            elif name == 'injuries':
                out[name] = {'state':'EMPTY', 'records':0,'interpretation':'NO_PROVIDER_RECORDS_NOT_HEALTH_VERIFICATION'}
            else:
                out[name] = {'state':'EMPTY', 'records':0,'interpretation':'NOT_VERIFIED'}
        quota = item.get('quota') or {}
        if isinstance(quota, dict) and quota.get('daily_remaining') is not None:
            remaining.append(_as_int(quota.get('daily_remaining'), -1))
    return {
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_source':'SAVED_CONNECTOR_PROBES_NO_ADDITIONAL_REQUESTS',
        'endpoints':out,
        'last_observed_daily_remaining':min((x for x in remaining if x >= 0), default=None),
        'note':'Observed API quota is shared across callers; the scanner local counter is not the authoritative remaining quota.',
        'current_season_team_strength_ready':out.get('team_stats',{}).get('state') == 'RETURNED_DATA',
        'bet_eligible_from_access_audit_alone':False,
    }


def audit_connector_counts(raw: dict[str, Any]) -> dict[str, Any]:
    """Audit the connector's reported counters without inventing missing rows.

    Use when only a summary of the live response has been persisted, not the
    individual fixture objects. Such an observation cannot assert league-level
    completeness even when the reported counts match.
    """
    total = _as_int(raw.get('total_results'), -1)
    received = _as_int(raw.get('returned_results'), -1)
    if total < 0 or received < 0 or received > total:
        raise ValueError('Invalid connector counters')
    truncated = bool(raw.get('truncated')) or received < total
    return {
        'source_kind':'connector_counts_only',
        'match_date':raw.get('match_date'),
        'timezone':raw.get('timezone'),
        'retrieved_at_utc':raw.get('retrieved_at_utc'),
        'total_reported':total,
        'received':received,
        'missing_minimum':max(0,total-received),
        'complete_slate_verified':False,
        'coverage_status':'INCOMPLETE' if truncated else 'UNVERIFIED_NO_ROWS',
        'count_observed_not_started':None,
        'league_coverage_verified':False,
        'recommended_action':'USE_AUTHORIZED_DIRECT_FIXTURES_DATE_OR_RETAIN_INCOMPLETE_STATUS; NO_GUESSED_MISSING_FIXTURES',
    }