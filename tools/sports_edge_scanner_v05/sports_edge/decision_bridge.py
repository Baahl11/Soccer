"""Connect canonical Soccer research, audited decision feed and daily reporting.

This bridge never converts API-Football reference odds or an unpromoted model
into a playable price/edge. A trusted upstream adapter supplies both a
production-approved sporting model and a sportsbook offer, independently.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

from .decision_gate import adjudicate_offers, inspect_canonical_research
from .core import parse_iso

INPUT_SCHEMA = 'SOCCER_TRUSTED_DECISION_FEED_v1.0'


def _safe_payload(path: str | Path, now: datetime, signing_key: str | None=None) -> dict[str, Any]:
    filename=Path(path)
    if not filename.is_file() or filename.stat().st_size > 2_000_000:
        raise ValueError('TRUSTED_INPUT_FILE_UNAVAILABLE_OR_TOO_LARGE')
    data=json.loads(filename.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or data.get('schema_version') != INPUT_SCHEMA:
        raise ValueError('TRUSTED_INPUT_SCHEMA_INVALID')
    if data.get('trusted_producer') != 'SOCCER_EDGE_CANONICAL_SIGNED_EXPORT':
        raise ValueError('UNTRUSTED_MODEL_PRODUCER')
    key = signing_key or os.getenv('SOCCER_DECISION_FEED_HMAC_KEY')
    if not key or len(key) < 32:
        raise ValueError('SIGNED_DECISION_FEED_KEY_NOT_CONFIGURED')
    claimed = data.get('signature_hmac_sha256')
    unsigned = {k: v for k, v in data.items() if k != 'signature_hmac_sha256'}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    expected = hmac.new(key.encode('utf-8'), canonical, hashlib.sha256).hexdigest()
    if not isinstance(claimed,str) or not hmac.compare_digest(expected,claimed):
        raise ValueError('SIGNED_DECISION_FEED_AUTHENTICATION_FAILED')
    updated=parse_iso(str(data.get('generated_at_utc')))
    age=(now-updated).total_seconds()
    if not -60 <= age <= 6*3600:
        raise ValueError('TRUSTED_DECISION_FEED_STALE')
    for name in ('offers','models','parlay_offers','joint_evidence'):
        if not isinstance(data.get(name),list):
            raise ValueError('TRUSTED_INPUT_LIST_REQUIRED_'+name.upper())
    return data


def adjudicate_daily(canonical: dict[str, Any], *,
                     slate_complete: bool, now: datetime | None=None,
                     trusted_file: str | Path | None=None,
                     signing_key: str | None=None) -> dict[str, Any]:
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    diagnostic=inspect_canonical_research(canonical)
    source_path=trusted_file or os.getenv('SOCCER_TRUSTED_DECISION_INPUT')
    if not source_path:
        report=adjudicate_offers([],[],now=now,slate_complete=slate_complete)
        report['feed_status']='TRUSTED_MODEL_AND_SPORTSBOOK_OFFERS_NOT_CONNECTED'
        report['feed_error']=None
    else:
        try:
            payload=_safe_payload(source_path,now,signing_key=signing_key)
            report=adjudicate_offers(payload['offers'],payload['models'],
                                      parlay_offers=payload['parlay_offers'],
                                      joint_reports=payload['joint_evidence'],
                                      now=now,slate_complete=slate_complete)
            report['feed_status']='TRUSTED_PRODUCER_ASSERTION_SUPPLIED'
            report['feed_error']=None
        except (ValueError,TypeError,OSError,json.JSONDecodeError) as exc:
            report=adjudicate_offers([],[],now=now,slate_complete=slate_complete)
            report['feed_status']='TRUSTED_DECISION_FEED_INVALID'
            report['feed_error']=str(exc)[:140]
    report['canonical_research']=diagnostic
    report['blocker_counts']=dict(Counter(
        b for x in report['singles']+report['parlays'] for b in x.get('blockers',[])))
    report['no_implicit_upstream_promotion']=True
    report['wagers_placed']=0
    return report