"""SOCCER commercial release and privacy/notifications gates, deny-by-default.

Not a compliance opinion, license approval, or Stripe transaction. Named
evidence records and independent reviewers are required outside the code.
All flags default false; a single billing env toggle never activates sales.
"""
from __future__ import annotations
import os
from typing import Any

VERSION = "SOCCER_COMMERCIAL_LAUNCH_GATES_v1.0"
REQUIRED = {
    "SOCCER_MODEL_FAMILY_OOS_VERIFIED":"Independent FT/OOS calibration per offered family",
    "SOCCER_EXECUTABLE_QUOTES_VERIFIED":"Same-instrument directly executable bookmaker snapshots",
    "SOCCER_MARKET_SETTLEMENT_CLV_VERIFIED":"Settlements and true direct closing-line evidence",
    "SOCCER_DATA_REDISPLAY_LICENSES_APPROVED":"Licensing, team marks, vendor and league rights approved",
    "SOCCER_PRIVACY_RLS_AUDITED":"Auth, isolation, RLS and account/data-deletion audited",
    "SOCCER_PAYMENT_SANDBOX_APPROVED":"Stripe sandbox, idempotent webhooks, refunds and portal QA",
    "SOCCER_JURISDICTION_LEGAL_REVIEW_APPROVED":"Territory-specific betting-content and consumer-law review",
    "SOCCER_CONSENTED_ALERTS_QA_APPROVED":"Opt-in alerts, deduplication, cancellation, and unsubscribe QA",
    "SOCCER_PUBLIC_PRODUCT_APPROVED":"Explicit final pricing, disclosures, responsible gaming and signoff",
}


def enabled(value: Any) -> bool:
    return str(value or "").strip().lower() in {"yes", "1", "true", "on"}


def audit_launch_gates(env: dict[str,str] | None = None) -> dict[str, Any]:
    e=dict(os.environ if env is None else env)
    requested=enabled(e.get("SOCCER_PUBLIC_BILLING_ENABLED"))
    gates={name:{
        "met":enabled(e.get(name)),
        "requirement":description,
    } for name,description in REQUIRED.items()}
    missing=[name for name,x in gates.items() if not x["met"]]
    # An environment flag is only an operational assertion; no check here can
    # confer real licensed rights or verify historical external data.
    return {
        "version":VERSION,
        "status":"READY_FOR_FINAL_MANUAL_REVIEW" if not missing else "COMMERCIAL_LAUNCH_BLOCKED",
        "billing_requested":requested,
        "all_internal_attestations_present":not missing,
        "billing_enabled":requested and not missing,
        "missing_gates":missing,
        "gates":gates,
        "legal_certification_from_code":False,
        "operator_evidence_review_required":True,
        "no_automatic_market_family_bet_promotion":True,
    }


def allow_research_alert(
    *,
    user_opted_in: bool,
    family_maturity: str,
    identity_verified: bool,
    cached_snapshot_fresh: bool,
    notification_deduped: bool,
    display_license_approved: bool,
) -> dict[str,Any]:
    reasons=[]
    if user_opted_in is not True:reasons.append("USER_OPT_IN_NOT_VERIFIED")
    if family_maturity not in {"SHADOW","RESEARCH"}:reasons.append("FAMILY_MATURITY_NOT_SUPPORTED")
    if identity_verified is not True:reasons.append("MATCH_IDENTITY_NOT_VERIFIED")
    if cached_snapshot_fresh is not True:reasons.append("SNAPSHOT_STALE_OR_MISSING")
    if notification_deduped is not True:reasons.append("ALERT_IDEMPOTENCY_NOT_VERIFIED")
    if display_license_approved is not True:reasons.append("SOURCE_DISPLAY_RIGHTS_NOT_VERIFIED")
    return {
        "status":"RESEARCH_NOTICE_ALLOWED" if not reasons else "ALERT_BLOCKED",
        "reasons":reasons,
        "bet_signal_allowed":False,
        "price_claim_allowed":False,
        "notification_sent":False,
    }
