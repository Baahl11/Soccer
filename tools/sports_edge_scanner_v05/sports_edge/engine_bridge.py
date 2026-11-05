"""Audited read-only adapter to the existing SOCCER EDGE ENGINE raw sport model.

Does not build a model from prices, use bookmaker odds, or promote BET.
The model source must be a separately checked-out, pinned existing engine commit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
from typing import Any

# Sole allowed existing engine source. No fallback fabricated probabilities.
ENGINE_MODULE = 'mcp_gateway/soccer_model.py'
PROJECTION_FIELDS = (
    'status', 'model_version', 'projection_model', 'reason', 'missing',
    'advanced_metrics', 'model_limitations', 'sample',
    'raw_home_goal_rate', 'raw_away_goal_rate', 'raw_total_goals',
    'raw_home_win_prob', 'raw_draw_prob', 'raw_away_win_prob',
    'raw_btts_yes_prob', 'raw_over_1_5_prob', 'raw_over_2_5_prob',
    'raw_over_3_5_prob', 'scoring_path', 'screen_scores', 'top_scorelines',
)


def _refuse(reason: str) -> dict[str, Any]:
    return {
        'status': 'NOT_CONNECTED' if reason == 'ENGINE_SOURCE_NOT_CONFIGURED' else 'NOT_MODELED',
        'reason': reason,
        'model_version': None,
        'calibrated': False,
        'bet_eligible': False,
        'market_odds_used': False,
        'wagers_placed': 0,
    }


def _validated_stats(payload: dict[str, Any], team_id: int, league_id: int, season: int) -> bool:
    if not isinstance(payload, dict):
        return False
    r = payload.get('response')
    if not isinstance(r, dict) or not r:
        return False
    t, league = r.get('team') or {}, r.get('league') or {}
    return (t.get('id') == team_id and league.get('id') == league_id and league.get('season') == season)


def from_existing_engine(fixture: dict[str, Any], home_payload: dict[str, Any],
                         away_payload: dict[str, Any], *, engine_root: str | None = None) -> dict[str, Any]:
    """Evaluate exact existing soccer_model.py on independent API-Football stats.

    This raw GOAL_RATE_POISSON_LIMITED output is not OOS calibrated or betting
    eligible. The model remains maintained in soccer-edge-mcp-v1, not forked here.
    """
    root = engine_root if engine_root is not None else os.environ.get('SOCCER_ENGINE_SOURCE_DIR', '')
    if not root:
        return _refuse('ENGINE_SOURCE_NOT_CONFIGURED')
    model_path = Path(root) / ENGINE_MODULE
    if not model_path.is_file():
        return _refuse('EXISTING_ENGINE_SOURCE_NOT_FOUND')
    if fixture.get('status') != 'NS':
        return _refuse('FIXTURE_NOT_PREMATCH')
    try:
        home_id, away_id = int(fixture['home_team_id']), int(fixture['away_team_id'])
        league_id, season = int(fixture['league_id']), int(fixture['season'])
        fixture_id = int(fixture['fixture_id'])
    except (KeyError, TypeError, ValueError):
        return _refuse('MISSING_FIXTURE_IDENTITY')
    if not (_validated_stats(home_payload, home_id, league_id, season)
            and _validated_stats(away_payload, away_id, league_id, season)):
        return _refuse('TEAM_STATS_IDENTITY_MISMATCH_OR_UNVERIFIED')
    spec = importlib.util.spec_from_file_location('_sports_edge_existing_soccer_model', model_path)
    if spec is None or spec.loader is None:
        return _refuse('EXISTING_ENGINE_MODULE_UNLOADABLE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, 'build_raw_projection', None)
    if not callable(fn):
        return _refuse('EXISTING_ENGINE_RAW_PROJECTION_NOT_FOUND')
    # No odds or market-price inputs enter the existing raw sport model.
    output = fn(dict(fixture), {
        'home_stats': home_payload['response'],
        'away_stats': away_payload['response'],
        'home_recent': [],
        'away_recent': [],
    }, None)
    if not isinstance(output, dict):
        return _refuse('EXISTING_ENGINE_INVALID_MODEL_OUTPUT')
    projection = {key: output[key] for key in PROJECTION_FIELDS if key in output}
    projection.update({
        'fixture_id': fixture_id,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'engine_git_sha': os.environ.get('SOCCER_ENGINE_COMMIT_SHA') or 'NOT_VERIFIED',
        'model_source': f'existing_soccer_edge_engine:{ENGINE_MODULE}',
        'source': 'API-Football season team stats (confirmed team/league/season)',
        'home_recent': 'NOT_VERIFIED',
        'away_recent': 'NOT_VERIFIED',
        'availability': 'NOT_VERIFIED',
        'calibrated': False,
        'bet_eligible': False,
        'market_odds_used': False,
        'wagers_placed': 0,
        'classification': 'WATCH',
    })
    if projection.get('status') == 'MODELED_LIMITED':
        sample = projection.get('sample') or {}
        if (sample.get('minimum_split_sample') or 0) < 5:
            projection['sample_warning'] = 'LOW_LEAGUE_SPLIT_SAMPLE_UNDER_5'
    return projection