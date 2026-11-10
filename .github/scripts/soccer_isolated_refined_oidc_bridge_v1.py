"""Send prospective refined SPORT ONLY research to the private OIDC append-only store.

Runs only on GitHub Actions main, after the canonical v5 sport worker has
finished and passed quota checks. No odds, CLV, bets or provider raw input
cross the research intake boundary. Offline self-tests make no requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

MAX_BODY_BYTES = 250_000
MAX_ROWS_PER_POST = 12
MAX_SPORT_ROWS = 20
ENDPOINT = "https://soccer-edge-api.onrender.com/internal/isolated-sport-refined-pg-v1/import"
AUDIENCE = "soccer-edge-render"
ALLOWED_SOURCE_FIELDS = (
    "schema_version", "status", "fixture_id", "league_id",
    "home_team_id", "away_team_id", "kickoff_utc",
    "sport_analysis_started_at_utc", "sport_analysis_finished_at_utc",
    "source_runtime_commit", "model_version", "projection_model",
    "sporting_inputs_sha256", "source_stage", "availability",
    "raw_sport_projection", "full_deep_dive_verified",
    "availability_verified_for_bet", "market_used_to_generate_raw_probability",
    "market_price_verified", "canonical_clv_eligible", "bet_eligible",
    "production_promotion_allowed", "retroactive_forecast_created",
)
PROOF_FIELDS = (
    "runtime_sha", "actual_provider_calls", "quota_headers_verified",
    "last_daily_remaining", "transport_budget_denied",
    "markets_queried", "price_resolver_invocations", "clv_rows_added",
    "db_writes", "bet_eligible", "production_promotion_allowed",
    "full_deep_dives_verified", "provider_calls_by_endpoint",
)


def _digest(source):
    return hashlib.sha256(json.dumps(
        source, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def prepare(report: dict) -> list[bytes]:
    if not isinstance(report, dict) or report.get("status") != "SPORT_ONLY_RESEARCH_CANARY":
        raise ValueError("CANONICAL_SPORT_ONLY_REPORT_NOT_VERIFIED")
    for key, expected in (
        ("markets_queried", 0), ("price_resolver_invocations", 0),
        ("clv_rows_added", 0), ("db_writes", 0),
        ("transport_budget_denied", 0), ("bet_eligible", False),
        ("production_promotion_allowed", False),
        ("full_deep_dives_verified", 0),
    ):
        if report.get(key) != expected:
            raise ValueError("CANONICAL_SPORT_ONLY_GATES_FAILED")
    calls = report.get("actual_provider_calls")
    count = report.get("quota_headers_verified")
    if (type(calls) is not int or not 1 <= calls <= 40 or type(count) is not int
            or count < calls or type(report.get("last_daily_remaining")) is not int
            or report["last_daily_remaining"] < 1000):
        raise ValueError("OFFICIAL_PROVIDER_QUOTA_PROOF_NOT_VERIFIED")
    raw = report.get("raw_research_snapshots")
    if not isinstance(raw, list) or len(raw) > MAX_SPORT_ROWS:
        raise ValueError("UNBOUNDED_OR_MISSING_RESEARCH_ROWS")
    if len(raw) != report.get("real_frozen_raw_count"):
        raise ValueError("MODEL_COUNT_DIFFERENCE")
    proof = {k: report.get(k) for k in PROOF_FIELDS}
    batches = []
    sanitized = []
    for row in raw:
        if not isinstance(row, dict) or "sporting_inputs" not in row:
            raise ValueError("SOURCE_FEATURES_MISSING_FOR_HASH_VERIFICATION")
        if _digest(row["sporting_inputs"]) != row.get("sporting_inputs_sha256"):
            raise ValueError("MODEL_SOURCE_FEATURE_FINGERPRINT_CONFLICT")
        if row.get("source_runtime_commit") != report.get("runtime_sha"):
            raise ValueError("MODEL_RUNTIME_SHA_NOT_PINNED")
        slim = {k: row.get(k) for k in ALLOWED_SOURCE_FIELDS}
        if slim.get("source_stage") is None or slim.get("raw_sport_projection") is None:
            raise ValueError("MODEL_NOT_FROZEN")
        sanitized.append(slim)
    for i in range(0, len(sanitized), MAX_ROWS_PER_POST):
        request = {
            "transport_proof": proof,
            "snapshots": sanitized[i:i+MAX_ROWS_PER_POST],
        }
        body = json.dumps(request, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("REFINED_RESEARCH_BATCH_TOO_LARGE")
        batches.append(body)
    return batches


def _oidc_token():
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("OIDC_WORKFLOW_MUST_RUN_ON_MAIN")
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not bearer:
        raise RuntimeError("GITHUB_WORKFLOW_OIDC_IDENTITY_UNAVAILABLE")
    joiner = "&" if "?" in url else "?"
    req = urllib.request.Request(
        url+joiner+urllib.parse.urlencode({"audience": AUDIENCE}),
        headers={"Authorization": "bearer "+bearer},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        value = json.load(response).get("value")
    if not isinstance(value, str) or not value:
        raise RuntimeError("OIDC_SIGNED_IDENTITY_NOT_AVAILABLE")
    return value


def _post(body: bytes, token: str) -> dict:
    req = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={
            "Authorization": "Bearer "+token,
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            result = json.loads(response.read(12000))
    except urllib.error.HTTPError as exc:
        # Never expose the signed token, provider credentials, or raw source.
        raise RuntimeError("REFINED_RESEARCH_STORE_HTTP_"+str(exc.code)) from None
    return result


def run(source: Path) -> dict:
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("REFINED_SPORT_IMPORT_MAIN_ONLY")
    if os.environ.get("DATABASE_URL"):
        raise RuntimeError("WORKER_MUST_NOT_HAVE_DATABASE_URL")
    if not source.is_file() or source.stat().st_size > 2_000_000:
        raise RuntimeError("SPORT_ONLY_SOURCE_UNAVAILABLE_OR_TOO_LARGE")
    report = json.loads(source.read_text())
    batches = prepare(report)
    receipt = {
        "status": "REFINED_RESEARCH_LIVE_PRIVATE_IMPORT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prospective_source_rows": len(report["raw_research_snapshots"]),
        "model_stages": report.get("model_stages"),
        "provider_calls_original_run": report["actual_provider_calls"],
        "new_research_rows": 0, "identical_existing_rows": 0,
        "posts_sent": 0, "no_new_rows": not bool(batches),
        "canonical_clv_rows_added": 0, "canonical_deep_dives_added": 0,
        "market_prices_added": 0, "bet_eligible": False,
        "production_promotion_allowed": False,
        "upstream_provider_inputs_transmitted": False,
        "private_storage": "RESEARCH_ONLY_AUTHENTICATED_OIDC",
    }
    if not batches:
        return receipt
    token = _oidc_token()
    for body in batches:
        response = _post(body, token)
        expected = len(json.loads(body)["snapshots"])
        if (response.get("status") != "AUTHENTIC_REFINED_SPORT_RESEARCH_IMPORTED"
                or response.get("snapshots_validated") != expected
                or response.get("new_research_rows", 0) + response.get("identical_existing_rows", 0) != expected
                or response.get("canonical_clv_rows_added") != 0
                or response.get("canonical_deep_dives_added") != 0
                or response.get("market_prices_added") != 0
                or response.get("provider_requests_added") != 0
                or response.get("production_promotion_allowed") is not False
                or response.get("bet_eligible") is not False):
            raise RuntimeError("REFINED_RESEARCH_STORE_ACK_NOT_VERIFIED")
        receipt["new_research_rows"] += response["new_research_rows"]
        receipt["identical_existing_rows"] += response["identical_existing_rows"]
        receipt["posts_sent"] += 1
    return receipt


def self_test():
    source = {"away_stats": {"goals": 1.0}, "home_recent": []}
    digest = _digest(source)
    now = datetime(2026, 11, 10, 19, 0, tzinfo=timezone.utc)
    one = {
        "schema_version": "1.0.0",
        "status": "CANONICAL_V5_SPORT_ONLY_RESEARCH_LIMITED",
        "fixture_id": 11, "league_id": 12,
        "home_team_id": 13, "away_team_id": 14,
        "kickoff_utc": "2026-11-10T20:00:00+00:00",
        "sport_analysis_started_at_utc": "2026-11-10T18:59:00+00:00",
        "sport_analysis_finished_at_utc": now.isoformat(),
        "source_runtime_commit": "a"*40,
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "projection_model": "GOAL_RATE_POISSON_LIMITED_v0.1",
        "source_stage": "T-60",
        "availability": {"availability_confidence": 0.5},
        "raw_sport_projection": {"status": "MODELED_LIMITED",
                                  "projection_model": "GOAL_RATE_POISSON_LIMITED_v0.1"},
        "sporting_inputs": source, "sporting_inputs_sha256": digest,
        "full_deep_dive_verified": False,
        "availability_verified_for_bet": False,
        "market_used_to_generate_raw_probability": False,
        "market_price_verified": False, "canonical_clv_eligible": False,
        "bet_eligible": False, "production_promotion_allowed": False,
        "retroactive_forecast_created": False,
    }
    report = {
        "status": "SPORT_ONLY_RESEARCH_CANARY", "runtime_sha": "a"*40,
        "markets_queried": 0, "price_resolver_invocations": 0,
        "clv_rows_added": 0, "db_writes": 0, "transport_budget_denied": 0,
        "actual_provider_calls": 22, "quota_headers_verified": 22,
        "last_daily_remaining": 7180, "provider_calls_by_endpoint": {"/fixtures": 22},
        "bet_eligible": False, "production_promotion_allowed": False,
        "full_deep_dives_verified": 0, "real_frozen_raw_count": 1,
        "raw_research_snapshots": [one],
    }
    batches = prepare(report)
    assert len(batches) == 1
    payload = json.loads(batches[0])
    assert payload["snapshots"][0]["fixture_id"] == 11
    assert "sporting_inputs" not in payload["snapshots"][0]
    assert payload["snapshots"][0]["sporting_inputs_sha256"] == digest
    assert payload["transport_proof"]["markets_queried"] == 0
    try:
        prepare({**report, "raw_research_snapshots":
                 [{**one, "sporting_inputs_sha256": "0"*64}]})
        raise AssertionError("HASH_CORRUPTION_WAS_NOT_BLOCKED")
    except ValueError as exc:
        assert "FINGERPRINT" in str(exc)
    assert prepare({**report, "real_frozen_raw_count": 0,
                    "raw_research_snapshots": []}) == []
    print("SPORT_ONLY_OIDC_PRIVATE_IMPORT_OFFLINE_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source is None or args.receipt is None:
        parser.error("--source and --receipt required")
    result = run(args.source)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    # Public CI never prints fixture IDs, team statistics, forecasts, or tokens.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
