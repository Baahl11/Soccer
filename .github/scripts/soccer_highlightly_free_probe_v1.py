#!/usr/bin/env python3
"""One-shot, secrets-safe Highlightly free-tier research. Max SIX GETs; no writes.

Purpose: prove whether GitHub Actions secret exists and is accepted, inspect
real match list + one historical finished fixture's statistics and lineups.
Never print secret, response body, PII, raw provider data, or predicted scores.
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from unicodedata import normalize

BASE = "https://soccer.highlightly.net"
DATES = ("2026-11-09", "2026-11-10", "2026-11-11")
MAX_REQUESTS = 6
MIN_QUOTA_RESERVE = 70
calls = 0

# Verified identifiers from the connected canonical API-Football (UTC kickoffs).
REFERENCES = (
    (1570011, "Las Palmas", "Eldense"),
    (1576529, "Torreense", "União de Leiria"),
    (1608737, "Pineto", "Vis Pesaro"),
    (1636031, "Chapulineros", "Cordobés"),
    (1600289, "Nova Cidade", "Carapebus"),
    (1638304, "Häcken W", "Chelsea W"),
    (1638332, "Real Madrid W", "Paris FC W"),
    (1602548, "Metalul Buzău", "CSA Steaua Bucureşti"),
    (1619186, "Hull City U21", "Fleetwood Town U21"),
    (1581906, "Chernihiv", "Inhulets"),
    (1648706, "India", "Maldives"),
    (1559458, "Germany U17", "Azerbaijan U17"),
)


def clean(s):
    if not isinstance(s, str): return ""
    return re.sub(r"[^a-z0-9]", "", normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def request(path, key, params=None):
    global calls
    assert path.startswith("/") and calls < MAX_REQUESTS
    calls += 1
    q = urllib.parse.urlencode(params or {})
    url = BASE + path + ("?" + q if q else "")
    req = urllib.request.Request(url, headers={
        "x-rapidapi-key": key, "Accept": "application/json",
        "User-Agent": "SoccerEdgeResearch/1.0",
    }, method="GET")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=14) as resp:
            body = resp.read(2_000_001)
            headers = resp.headers
            status = resp.status
            data = json.loads(body) if len(body) <= 2_000_000 else None
            size_limited = len(body) > 2_000_000
    except urllib.error.HTTPError as e:
        headers, status, data, size_limited = e.headers, e.code, None, False
    except Exception as e:
        return {"http": None, "network_error_type": type(e).__name__,
                "ms": round((time.monotonic()-start)*1000), "data": None}
    def quota_header(name):
        value = headers.get(name)
        return int(value) if value and value.isdigit() else None
    return {"http": status, "ms": round((time.monotonic()-start)*1000),
            "remaining": quota_header("x-ratelimit-requests-remaining"),
            "limit": quota_header("x-ratelimit-requests-limit"),
            "body_too_large": size_limited, "data": data}


def arr(v):
    return v if isinstance(v, list) else []


def team_name(v):
    return v.get("name") if isinstance(v, dict) else None


def family_labels(v):
    labels = set()
    groups = {"xG": ("expected goals", "xg"), "CORNERS": ("corner",),
              "SHOTS": ("shot",), "CARDS": ("yellow card", "red card", "cards"),
              "POSSESSION": ("possession",), "PASSES": ("pass",),
              "SAVES": ("save",), "FOULS": ("foul",)}
    queue = [v]
    scanned = 0
    while queue and scanned < 2000:
        node = queue.pop()
        scanned += 1
        if isinstance(node, dict):
            label = str(node.get("displayName") or node.get("name") or "").lower()
            val = node.get("value")
            if val is not None:
                for k, terms in groups.items():
                    if any(t in label for t in terms):
                        labels.add(k)
            for child in node.values():
                if isinstance(child, (list, dict)):
                    queue.append(child)
        elif isinstance(node, list):
            queue.extend(node[:35])
    return sorted(labels)


def main():
    key = os.getenv("SOCCER_HIGHLIGHTLY_API_KEY", "").strip()
    report = {"schema": "SOCCER_HIGHLIGHTLY_FREE_SECRET_PROBE_v1",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "secret_present": bool(key), "source": "soccer.highlightly.net",
              "requested_dates": list(DATES),
              "scope": "FREE_TIER_SAMPLE_NOT_TOTAL_SLATE; POSTMATCH_STATISTICS_NOT_PREGAME",
              "requests_cap": MAX_REQUESTS, "quota_reserve_stop_at": MIN_QUOTA_RESERVE,
              "canonical_db_writes": 0, "odds_calls": 0,
              "public_billing_enabled": False, "bet_eligible": False}
    if not key:
        report.update(status="GITHUB_SECRET_NOT_VISIBLE_TO_WORKFLOW", requests_executed=0)
        print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
        return 0
    pages, candidates, stopped = [], [], False
    for date in DATES:
        if calls >= MAX_REQUESTS or stopped: break
        r = request("/matches", key, {"date": date, "limit": 100, "offset": 0})
        obj = r.get("data") if isinstance(r.get("data"), dict) else {}
        matches = arr(obj.get("data"))
        pagination = obj.get("pagination") if isinstance(obj.get("pagination"), dict) else {}
        plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
        matches_found = []
        for m in matches:
            if not isinstance(m, dict): continue
            h, a = clean(team_name(m.get("homeTeam"))), clean(team_name(m.get("awayTeam")))
            for fid, rh, ra in REFERENCES:
                if h == clean(rh) and a == clean(ra):
                    matches_found.append(fid)
            mid = m.get("id")
            if isinstance(mid, int) and mid > 0:
                league = m.get("league") if isinstance(m.get("league"), dict) else {}
                l_name = str(league.get("name") or "")
                league_priority = int(any(t in l_name.lower() for t in ("liga mx", "segunda", "serie", "champions", "premier", "bundesliga")))
                candidates.append((league_priority, date == "2026-11-09", mid, l_name))
        pages.append({"date": date, "http": r.get("http"),
                      "rows_returned": len(matches),
                      "total_count_declared": pagination.get("totalCount"),
                      "plan_tier": plan.get("tier"),
                      "plan_restrictions_present": "hidden" in str(plan.get("message") or "").lower(),
                      "strict_name_only_reference_ids": sorted(set(matches_found)),
                      "quota_limit": r.get("limit"), "quota_remaining": r.get("remaining"),
                      "ms": r.get("ms"), "body_too_large": r.get("body_too_large", False)})
        if r.get("http") != 200:
            report["status"] = "AUTH_OR_PROVIDER_ERROR" if r.get("http") in (401,403) else "PROVIDER_RESPONSE_NOT_OK"
            stopped = True
        remaining = r.get("remaining")
        if remaining is not None and remaining <= MIN_QUOTA_RESERVE:
            stopped = True
            report["stop_reason"] = "QUOTA_RESERVE_REACHED"
        if stopped: break

    report["fixture_pages"] = pages
    report["strict_reference_fixture_matches"] = sorted({fid for p in pages for fid in p["strict_name_only_reference_ids"]})
    # This is an availability probe only. Post-match data never enter RAW pregame.
    if candidates and not stopped and calls + 3 <= MAX_REQUESTS:
        _, _, match_id, league_name = sorted(candidates, reverse=True)[0]
        report["detail_sample"] = {"match_id": match_id, "league": league_name}
        for path, kind in (
            (f"/matches/{match_id}", "match_detail"),
            (f"/statistics/{match_id}", "postmatch_statistics"),
            (f"/lineups/{match_id}", "lineups"),
        ):
            r = request(path, key)
            obj = r.get("data")
            report["detail_sample"][kind] = {
                "http": r.get("http"), "ms": r.get("ms"),
                "fields_observed_non_null": family_labels(obj),
                "array_count": len(obj) if isinstance(obj, list) else None,
                "quota_remaining": r.get("remaining"),
            }
            rem = r.get("remaining")
            if rem is not None and rem <= MIN_QUOTA_RESERVE: break

    report["requests_executed"] = calls
    report.setdefault("status", "SECRET_PRESENT_AND_API_HTTP_200" if any(p["http"] == 200 for p in pages)
                      else "SECRET_PRESENT_BUT_API_NOT_VERIFIED")
    report["secret_value_disclosed"] = False
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    assert calls <= MAX_REQUESTS and report["canonical_db_writes"] == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
