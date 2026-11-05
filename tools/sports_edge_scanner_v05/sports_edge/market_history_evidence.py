"""Past-only event-rate evidence alongside Bet365 exact goals-market legs.

Historical frequency is descriptive and NEVER a calibrated prediction, EV,
price replacement, injury inference, opponent-strength adjustment or BET gate.
"""
from __future__ import annotations
from datetime import datetime,timezone
from math import isfinite
from typing import Any


def _dt(value):
    try:
        dt=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError,TypeError,OverflowError):
        return None


def _past_event_observation(row:dict[str,Any], *,event_kickoff:str, team_id:int,
                            family:str,selection:str,point:str)->bool|None:
    if family!='FULL_TIME_GOALS' or selection.casefold() not in ('over','under'):
        return None
    try:
        threshold=float(point)
        if not isfinite(threshold) or abs(threshold%1-.5)>1e-9 or not 0.5<=threshold<=6.5:
            return None
    except (ValueError,TypeError):return None
    past=_dt(row.get('kickoff'));future=_dt(event_kickoff)
    if not past or not future or past>=future or row.get('status')!='FT':return None
    if team_id not in (row.get('home_team_id'),row.get('away_team_id')):return None
    goals=row.get('goals') or {}
    hg=goals.get('home');ag=goals.get('away')
    if not isinstance(hg,int) or isinstance(hg,bool) or not isinstance(ag,int) or isinstance(ag,bool):return None
    if not 0<=hg<=30 or not 0<=ag<=30:return None
    total=hg+ag
    return total>threshold if selection.casefold()=='over' else total<threshold


def past_goals_market_evidence(leg:dict[str,Any], fixture:dict[str,Any])->dict[str,Any]:
    """Report both teams' prior full-match totals separately, without fabricating future match odds."""
    base={'status':'NOT_VERIFIED','source':'PAST_ALL_COMPETITIONS_FINISHED_MATCHES',
          'market':'FULL_TIME_GOALS_DESCRIPTIVE_ONLY','historical_probability_calibrated':False,
          'opponent_quality_adjusted':False,'formation_adjusted':False,
          'teams':[],'sample_minimum':0,'both_team_directional_support':None,
          'reason':'NO_VERIFIED_MATCH_IDENTITIES_OR_MARKET'}
    if leg.get('family')!='FULL_TIME_GOALS':
        base['reason']='MARKET_FAMILY_REQUIRES_DIFFERENT_HISTORICAL_EVIDENCE';return base
    history=fixture.get('multi_competition_history') or {}
    if history.get('status')!='VERIFIED_PAST_MATCHES':
        base['reason']='PAST_MULTICOMPETITION_FIXTURES_NOT_VERIFIED';return base
    raw=fixture.get('raw_sport_projection') or {}
    if raw.get('status')!='MODELED_LIMITED':
        base['reason']='MATCH_GOALS_SPORT_PROJECTION_NOT_VERIFIED';return base
    target=fixture.get('kickoff') or leg.get('starts_at_utc')
    if not _dt(target):
        base['reason']='TARGET_KICKOFF_UNVERIFIED';return base
    # A historical fixture contains IDs; only those matching the team's
    # independently verified schedule count towards its observed rate.
    for side in ('home','away'):
        roster=history.get(side+'_fixtures') or []
        expected=fixture.get(side+'_team_id')
        if not isinstance(expected,int) or not expected>0:
            base['reason']='TARGET_TEAM_ID_UNVERIFIED';return base
        vals=[];seen=set()
        for row in roster:
            if not isinstance(row,dict):continue
            fid=row.get('fixture_id')
            if not isinstance(fid,int) or fid in seen:continue
            success=_past_event_observation(row,event_kickoff=target,team_id=expected,
                         family=leg['family'],selection=str(leg.get('selection','')),
                         point=str(leg.get('point','')))
            if success is None:continue
            seen.add(fid);vals.append(success)
        n=len(vals);wins=sum(vals)
        base['teams'].append({'team_side':side,'team_id':expected,'historical_games':n,
                              'matching_totals':wins,'observed_rate_NOT_PREDICTION':round(wins/n,4) if n else None})
    base['sample_minimum']=min(row['historical_games'] for row in base['teams'])
    if not base['sample_minimum']:
        base['reason']='NO_MATCHING_FINISHED_PAST_GAMES';return base
    base['status']='DESCRIPTIVE_HISTORY_AVAILABLE'
    base['reason']='HISTORICAL_MATCH_TOTALS_ARE_NOT_OPPONENT_ADJUSTED_PROBABILITIES'
    base['both_team_directional_support']=(
        all(row['historical_games']>=6 and row['observed_rate_NOT_PREDICTION']>=.5
            for row in base['teams']))
    return base