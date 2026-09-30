from __future__ import annotations

import html
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PUBLIC_PERFORMANCE_V4_1.0.0"
CACHE_TTL_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 3.0

_REPORTS = {
    "market_performance": "market_performance_summary.json",
    "settlement_coverage": "settlement_coverage_report.json",
    "true_clv": "true_clv_summary.json",
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT = 0.0


def _state_config() -> tuple[str, str] | None:
    repo = os.getenv("STATE_REPO", "").strip()
    branch = os.getenv("STATE_BRANCH", "").strip()
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/"):]
    if not repo or not branch:
        return None
    return repo, branch


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_reports(repo: str, branch: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reports: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    base = f"https://raw.githubusercontent.com/{repo}/{branch}/soccer_edge_state/analysis"
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for key, filename in _REPORTS.items():
            try:
                response = client.get(f"{base}/{filename}", headers={"Accept": "application/json"})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    reports[key] = payload
                else:
                    errors[key] = "unexpected_json_shape"
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return reports, errors


def _scope_from_classification(perf: dict[str, Any], classification: str) -> dict[str, Any]:
    nested = _dict(perf.get("by_market_family_and_classification"))
    wins = losses = pushes = settled = ungraded = 0
    roi_units = 0.0
    families: list[dict[str, Any]] = []
    for family, by_class in nested.items():
        node = _dict(_dict(by_class).get(classification))
        if not node:
            continue
        family_settled = _int(node.get("settled")) or 0
        family_wins = _int(node.get("win")) or 0
        family_losses = _int(node.get("loss")) or 0
        family_pushes = _int(node.get("push")) or 0
        family_ungraded = _int(node.get("ungraded")) or 0
        family_roi = _float(node.get("roi_units"))
        wins += family_wins
        losses += family_losses
        pushes += family_pushes
        settled += family_settled
        ungraded += family_ungraded
        if family_roi is not None:
            roi_units += family_roi
        families.append({
            "market_family": str(family),
            "settled": family_settled,
            "win": family_wins,
            "loss": family_losses,
            "push": family_pushes,
            "ungraded": family_ungraded,
            "hit_rate_ex_push": _float(node.get("hit_rate_ex_push")),
            "roi_units": family_roi,
            "status": str(node.get("status") or "RESEARCH_ONLY_SAMPLE_TOO_SMALL"),
        })
    families.sort(key=lambda row: (-int(row.get("settled") or 0), str(row.get("market_family"))))
    return {
        "classification": classification,
        "settled": settled,
        "win": wins,
        "loss": losses,
        "push": pushes,
        "ungraded": ungraded,
        "roi_units": round(roi_units, 4),
        "hit_rate_ex_push": (wins / (wins + losses)) if (wins + losses) else None,
        "families": families,
    }


def _build(reports: dict[str, dict[str, Any]], errors: dict[str, str]) -> dict[str, Any]:
    perf = reports.get("market_performance", {})
    coverage = reports.get("settlement_coverage", {})
    clv = reports.get("true_clv", {})
    bet = _scope_from_classification(perf, "BET")
    lean = _scope_from_classification(perf, "LEAN")
    directional_min = _int(_dict(perf.get("minimum_sample_policy")).get("directional_read")) or 20
    bet_status = "VERIFIED_SAMPLE" if int(bet.get("settled") or 0) >= directional_min else "SAMPLE_TOO_SMALL"
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "ACTIVE" if perf else "NO_VERIFIED_SETTLEMENT_SAMPLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bet_only": {**bet, "sample_status": bet_status, "directional_minimum": directional_min},
        "research_lean": lean,
        "settlement": {
            "ledger_rows": _int(coverage.get("settlement_rows")),
            "source_actionable_rows": _int(coverage.get("source_actionable_rows")),
            "coverage_rate": _float(coverage.get("settlement_coverage_rate")),
            "backlog_rows": _int(coverage.get("backlog_rows")),
        },
        "true_clv": {
            "status": clv.get("status"),
            "rows": _int(clv.get("true_clv_rows")),
            "avg_probability_pp": _float(clv.get("avg_true_clv_probability_pp")),
            "positive": _int(clv.get("positive_clv")),
            "negative": _int(clv.get("negative_clv")),
            "flat": _int(clv.get("flat_clv")),
            "same_book_rows": _int(clv.get("same_book_true_clv_rows")),
            "directional_minimum": _int(_dict(clv.get("minimum_sample_policy")).get("directional_read")),
        },
        "sources": dict(_REPORTS),
        "errors": errors,
        "provider_requests_added": 0,
        "historical_reconstruction_performed": False,
        "settlement_backfill_performed": False,
        "production_promotion_allowed": False,
        "decision_weight": 0.0,
    }


def load_snapshot(*, force: bool = False) -> dict[str, Any]:
    global _CACHE, _CACHE_AT
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE is not None and now - _CACHE_AT < CACHE_TTL_SECONDS:
            return dict(_CACHE)
        config = _state_config()
        if config is None:
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION,
                "status": "NO_VERIFIED_SETTLEMENT_SAMPLE",
                "reason": "STATE_REPO_OR_STATE_BRANCH_NOT_CONFIGURED",
                "provider_requests_added": 0,
                "historical_reconstruction_performed": False,
                "settlement_backfill_performed": False,
                "production_promotion_allowed": False,
                "decision_weight": 0.0,
            }
        else:
            reports, errors = _fetch_reports(*config)
            snapshot = _build(reports, errors)
        _CACHE = dict(snapshot)
        _CACHE_AT = now
        return dict(snapshot)


def _esc(value: Any) -> str:
    return "N/V" if value is None else html.escape(str(value), quote=True)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%" if value is not None else "N/V"
    except (TypeError, ValueError):
        return "N/V"


def _pp(value: Any) -> str:
    try:
        return f"{float(value):+.2f} pp" if value is not None else "N/V"
    except (TypeError, ValueError):
        return "N/V"


def render_fragment(snapshot: dict[str, Any] | None = None) -> str:
    node = snapshot if isinstance(snapshot, dict) else load_snapshot()
    bet = _dict(node.get("bet_only"))
    settlement = _dict(node.get("settlement"))
    clv = _dict(node.get("true_clv"))
    sample_status = bet.get("sample_status") or "NO_VERIFIED_SETTLEMENT_SAMPLE"
    settled = _int(bet.get("settled"))
    wins = _int(bet.get("win"))
    losses = _int(bet.get("loss"))
    pushes = _int(bet.get("push"))
    roi = _float(bet.get("roi_units"))
    record = "N/V" if settled is None else f"{wins or 0}-{losses or 0}-{pushes or 0}"
    roi_text = "N/V" if roi is None else f"{roi:+.2f}u"
    coverage = _pct(settlement.get("coverage_rate"))
    clv_rows = _int(clv.get("rows"))
    families = bet.get("families") if isinstance(bet.get("families"), list) else []
    family_html = "".join(
        f"<tr><td>{_esc(row.get('market_family'))}</td><td>{_esc(row.get('settled'))}</td><td>{_esc(row.get('win'))}-{_esc(row.get('loss'))}</td><td>{_esc('N/V' if row.get('roi_units') is None else f\"{float(row.get('roi_units')):+.2f}u\")}</td></tr>"
        for row in families if isinstance(row, dict)
    ) or "<tr><td colspan='4'>No verified BET-family settlement sample.</td></tr>"
    return f"""
<style>
.v216{{margin:14px 0;border:1px solid #1b3448;background:linear-gradient(155deg,#091620,#080f16);border-radius:14px;overflow:hidden}}
.v216-head{{padding:16px 18px;display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #172a3b}}
.v216-head p{{margin:4px 0 0;color:#7890a5;font-size:11px;line-height:1.5}}
.v216-badge{{height:max-content;padding:5px 8px;border:1px solid #5a431f;background:#362914;color:#f6ca73;border-radius:999px;font-size:9px;font-weight:900}}
.v216-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;padding:13px}}
.v216-metric{{background:#08141e;border:1px solid #153047;border-radius:10px;padding:12px}}
.v216-metric span{{display:block;color:#607a90;font-size:9px;text-transform:uppercase}}
.v216-metric strong{{display:block;font-size:20px;margin-top:5px}}
.v216-table{{padding:0 13px 13px;overflow:auto}}.v216 table{{width:100%;min-width:560px;border-collapse:collapse}}.v216 th,.v216 td{{padding:9px 10px;border-bottom:1px solid #132634;font-size:10px}}.v216 th{{color:#607a90;text-transform:uppercase}}.v216-note{{padding:0 13px 13px;color:#60788c;font-size:10px;line-height:1.5}}
@media(max-width:900px){{.v216-grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:560px){{.v216-grid{{grid-template-columns:repeat(2,1fr)}}.v216-head{{display:block}}.v216-badge{{display:inline-block;margin-top:9px}}}}
</style>
<section class="v216" id="verified-performance">
<div class="v216-head"><div><div class="eyebrow">VERIFIED PERFORMANCE · V216</div><h2>BET-only Track Record</h2><p>Canonical settlement evidence only. LEAN rows are excluded from the headline record. No historical reconstruction or settlement backfill is performed by this view.</p></div><span class="v216-badge">{_esc(sample_status)}</span></div>
<div class="v216-grid">
<div class="v216-metric"><span>Graded BETs</span><strong class="mono">{_esc(settled)}</strong></div>
<div class="v216-metric"><span>W-L-P</span><strong class="mono">{_esc(record)}</strong></div>
<div class="v216-metric"><span>Hit rate</span><strong class="mono">{_pct(bet.get('hit_rate_ex_push'))}</strong></div>
<div class="v216-metric"><span>ROI units</span><strong class="mono">{_esc(roi_text)}</strong></div>
<div class="v216-metric"><span>Settlement coverage</span><strong class="mono">{_esc(coverage)}</strong></div>
<div class="v216-metric"><span>True CLV</span><strong class="mono">{_esc(clv_rows)}</strong></div>
</div>
<div class="v216-table"><table><thead><tr><th>BET family</th><th>Settled</th><th>W-L</th><th>ROI units</th></tr></thead><tbody>{family_html}</tbody></table></div>
<div class="v216-note">True-CLV average: {_pp(clv.get('avg_probability_pp'))} across {_esc(clv_rows)} rows · same-book {_esc(clv.get('same_book_rows'))}. Settlement ledger coverage: {_esc(settlement.get('ledger_rows'))}/{_esc(settlement.get('source_actionable_rows'))}. Minimum directional sample for performance cells: {_esc(bet.get('directional_minimum'))}; current BET sample remains research-only until that minimum is reached.</div>
</section>
"""
