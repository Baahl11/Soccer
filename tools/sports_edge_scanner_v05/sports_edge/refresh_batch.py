"""Bounded T-90/T-40/T-20 refresh of sport-first Soccer shortlists.

Only checks pregame availability through the original dedicated refresh tool.
Never converts a provider XI into independently verified official availability.
Never places bets. Excludes fixtures without verified complete-slate provenance.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Callable
from pathlib import Path

from .core import parse_iso


def refresh_due_shortlist(slate: dict[str, Any], batch: dict[str, Any],
                          directory: str | Path, *, now: datetime | None = None,
                          max_fixtures: int = 5,
                          refresh_fn: Callable[..., dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """Refresh only near-kickoff sport-screened candidates (0..5 by default).

    Fails closed without making network calls if slate or screening provenance
    is absent or incomplete. All provider requests occur in pregame_refresh.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Time must be timezone-aware')
    if not 0 <= max_fixtures <= 5:
        raise ValueError('At most five pregame refreshes per run')
    result: dict[str, Any] = {
        'status': 'RESEARCH_ONLY', 'generated_at_utc': now.astimezone(timezone.utc).isoformat(),
        'eligible_candidates_in_window': 0, 'attempted': 0, 'completed': 0,
        'official_lineup_verified': 0, 'reports': [], 'errors': [],
        'actionable_bets': 0, 'wagers_placed': 0,
    }
    if (not batch.get('source_slate_complete_verified') or
        not batch.get('all_fixtures_attempted') or
        not slate.get('eligible_slate_verified') or
        not batch.get('complete_sport_screen')):
        return {**result, 'status': 'BLOCKED_INCOMPLETE_SPORT_FIRST_SCREEN'}
    if refresh_fn is None:
        from .refresh import pregame_refresh
        refresh_fn = pregame_refresh
    approved_ids = {int(x) for x in batch.get('shortlists', {}).get('merged_fixture_ids', [])}
    screened = {int(x['fixture_id']): x for x in batch.get('screened_fixtures', [])
                if isinstance(x, dict) and x.get('fixture_id') is not None}
    candidates = []
    for fx in slate.get('eligible_fixtures', []):
        fid = int(fx.get('fixture_id') or -1)
        row = screened.get(fid)
        if not row or fid not in approved_ids or row.get('min_sample_met') is not True:
            continue
        if str(fx.get('status') or '') != 'NS':
            continue
        try:
            remaining = (parse_iso(str(fx['kickoff'])) - now.astimezone(timezone.utc)).total_seconds() / 60
        except (TypeError, ValueError, KeyError):
            continue
        if 10 <= remaining <= 95:
            candidates.append((remaining, fid))
    candidates.sort(key=lambda x:(x[0],x[1]))
    result['eligible_candidates_in_window'] = len(candidates)
    for _, fixture_id in candidates[:max_fixtures]:
        result['attempted'] += 1
        try:
            r = refresh_fn(batch, fixture_id, directory, now=now, reference_odds=False)
            if int(r.get('fixture_id',-1)) != fixture_id:
                raise ValueError('Refresh returned wrong fixture ID')
            result['reports'].append({
                'fixture_id': fixture_id,
                'refresh_stage': r.get('refresh_stage'),
                'minutes_until_kickoff': r.get('minutes_until_kickoff'),
                'provider_lineups_complete': (r.get('provider_lineups') or {}).get('provider_lineups_complete'),
                'official_lineup_verified': False,
                'classification': 'WATCH',
                'reason': r.get('reason'),
                'report_file': r.get('report_file'),
            })
            result['completed'] += 1
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            result['errors'].append({'fixture_id':fixture_id,'reason':str(exc)[:200]})
    if result['errors']:
        result['status'] = 'PARTIAL_REFRESH_NOT_VERIFIED'
    return result