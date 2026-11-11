#!/usr/bin/env python3
"""Bounded REAL public API probe. No private keys, no canonical writes, no odds.

Research-only. API-Football references were observed on 2026-11-11 UTC using
Soccer Edge API get_today_fixtures. A deliberately nonrepresentative 12-fixture
sample spanning 2026-11-09..11 is NOT a supplier performance estimate.

Requests: 6 OpenFoot official public-demo; 4 TheSportsDB official shared-free;
1 OpenLigaDB public Bundesliga sanity. Hard total <= 11.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

SCHEMA = "SOCCER_PUBLIC_FREE_PROVIDER_PROBE_v1.0"
SAMPLE = (
    (1570011, "2026-11-09T19:30:00Z", "Las Palmas", "Eldense", "Spain Segunda"),
    (1576529, "2026-11-09T20:15:00Z", "Torreense", "União de Leiria", "Portugal Segunda"),
    (1608737, "2026-11-09T19:30:00Z", "Pineto", "Vis Pesaro", "Italy Serie C"),
    (1636031, "2026-11-10T02:05:00Z", "Chapulineros", "Cordobés", "Mexico Primera Premier"),
    (1600289, "2026-11-09T17:00:00Z", "Nova Cidade", "Carapebus", "Brazil Carioca 2"),
    (1638304, "2026-11-10T17:45:00Z", "Häcken W", "Chelsea W", "UEFA Champions League Women"),
    (1638332, "2026-11-10T17:45:00Z", "Real Madrid W", "Paris FC W", "UEFA Champions League Women"),
    (1602548, "2026-11-10T17:30:00Z", "Metalul Buzău", "CSA Steaua Bucureşti", "Romania Liga II"),
    (1619186, "2026-11-10T13:00:00Z", "Hull City U21", "Fleetwood Town U21", "England Development League"),
    (1581906, "2026-11-11T09:30:00Z", "Chernihiv", "Inhulets", "Ukraine Persha Liga"),
    (1648706, "2026-11-11T09:30:00Z", "India", "Maldives", "SAFF Championship"),
    (1559458, "2026-11-11T10:00:00Z", "Germany U17", "Azerbaijan U17", "UEFA U17 qualifying"),
)
DAYS = ("2026-11-09", "2026-11-10", "2026-11-11")
ALLOWED_HOSTS = {"openfootapi.com", "www.thesportsdb.com", "api.openligadb.de"}
HEADERS = {"Accept": "application/json", "User-Agent": "SoccerEdgeResearch/1.0 public-provider-coverage-check"}
MAX_CALLS = 11


def dt(x):
    if not isinstance(x, str) or not x:
        return None
    try:
        d = datetime.fromisoformat(x.strip().replace("Z", "+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def norm(value):
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", value)


def fetch(host, path, params=None, bearer=None):
    """One safe bounded request; never print raw body or public-demo credential."""
    global calls
    assert host in ALLOWED_HOSTS and path.startswith("/") and calls < MAX_CALLS
    calls += 1
    query = urllib.parse.urlencode(params or {})
    url = "https://" + host + path + ("?" + query if query else "")
    headers = dict(HEADERS)
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    start = time.monotonic()
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.status
            # Guard response volume; no raw provider JSON is persisted/reprinted.
            body = response.read(1_500_001)
            if len(body) > 1_500_000:
                return {"status": "RESPONSE_TOO_LARGE", "http": status, "data": None, "ms": round((time.monotonic()-start)*1000)}
            parsed = json.loads(body)
            return {"status": "HTTP_OK" if status == 200 else "HTTP_ERROR",
                    "http": status, "data": parsed, "ms": round((time.monotonic()-start)*1000)}
    except urllib.error.HTTPError as e:
        return {"status": "HTTP_ERROR", "http": int(e.code), "data": None,
                "ms": round((time.monotonic()-start)*1000)}
    except Exception as e:
        return {"status": "NETWORK_ERROR", "http": None,
                "error_type": type(e).__name__, "data": None,
                "ms": round((time.monotonic()-start)*1000)}


def take_list(payload, key):
    if not isinstance(payload, dict):
        return []
    node = payload.get(key)
    if isinstance(node, list):
        return node
    if isinstance(node, dict):
        for name in ("items", "events", "matches"):
            if isinstance(node.get(name), list):
                return node[name]
    return []


def names_and_time(event, source):
    if source == "OPENFOOT":
        h, a = event.get("homeTeam"), event.get("awayTeam")
        home = h.get("name") if isinstance(h, dict) else h
        away = a.get("name") if isinstance(a, dict) else a
        when = dt(event.get("kickoffAt"))
        return home, away, when, str(event.get("id") or "")
    home, away = event.get("strHomeTeam"), event.get("strAwayTeam")
    when = dt(event.get("strTimestamp") or
              ((event.get("dateEvent") or "") + "T" + (event.get("strTime") or "00:00:00") + "Z"))
    return home, away, when, str(event.get("idEvent") or "")


def compare(rows, source):
    verified = set()
    date_only = set()
    inspected = 0
    for event in rows:
        if not isinstance(event, dict):
            continue
        inspected += 1
        home, away, observed, match_id = names_and_time(event, source)
        if not norm(home) or not norm(away):
            continue
        for fid, start, ref_home, ref_away, league in SAMPLE:
            # Strict identity, no fuzzy/substring/league shortcuts. False negatives possible.
            if norm(ref_home) != norm(home) or norm(ref_away) != norm(away):
                continue
            target = dt(start)
            if observed is None:
                date_only.add(fid)
            elif abs((observed - target).total_seconds()) <= 900:
                verified.add(fid)
            elif source == "SPORTSDB" and abs((observed - target).total_seconds()) <= 86400:
                # Date-only or wrong timezone data are *not* verified matches.
                date_only.add(fid)
    return {"rows_inspected": inspected, "strict_name_and_kickoff_15min_match_ids": sorted(verified),
            "name_only_or_date_ambiguous_ids": sorted(date_only - verified)}


calls = 0


def main():
    report = {
        "schema_version": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_reference_source": "API-FOOTBALL v3; Soccer_Edge_API get_today_fixtures",
        "fixture_source_retrieved_at_utc": "2026-11-11T04:41:40Z",
        "fixture_sample_size": len(SAMPLE),
        "sampling_status": "NONREPRESENTATIVE_CONVENIENCE_SAMPLE; NOT_ALL_316; NOT_COST_COVERAGE_ESTIMATE",
        "methods": "STRICT_NORMALIZED_TEAM_NAMES_AND_KICKOFF_WITHIN_15MIN; NO_FUZZY_MATCH",
        "run_is_prospective_sport_prediction": False,
        "source_rights_verified_for_commercial_redistribution": False,
        "providers": {},
        "http_request_cap": MAX_CALLS,
        "canonical_db_writes": 0,
        "market_calls": 0,
        "bet_eligible": False,
    }

    # Official public demonstration key is intentionally published at openfootapi.com/docs.
    openfoot = []
    o_requests = []
    for day in DAYS:
        r = fetch("openfootapi.com", "/v1/matches", {"date": day},
                  bearer="of_demo_openfootapi_docs")
        rows = take_list(r.get("data"), "data") if r["http"] == 200 else []
        openfoot.extend(rows)
        o_requests.append({"day": day, "http": r["http"], "status": r["status"],
                           "rows": len(rows), "ms": r["ms"]})
        time.sleep(0.3)
    oc = compare(openfoot, "OPENFOOT")
    comp = fetch("openfootapi.com", "/v1/competitions", bearer="of_demo_openfootapi_docs")
    comps = take_list(comp.get("data"), "data") if comp["http"] == 200 else []
    sanity = fetch("openfootapi.com", "/v1/matches", {"date": "2026-08-28"},
                   bearer="of_demo_openfootapi_docs")
    sanity_rows = take_list(sanity.get("data"), "data") if sanity["http"] == 200 else []
    xg = fetch("openfootapi.com", "/v1/matches/match_olg_83156/xg",
               bearer="of_demo_openfootapi_docs")
    report["providers"]["OPENFOOT_DEMO"] = {
        "access": "OFFICIAL_PUBLIC_STARTER_SCOPE_ONLY",
        "per_day": o_requests, **oc,
        "competitions_http": comp["http"], "competitions_array_size": len(comps),
        "sanity_docs_2026_08_28_http": sanity["http"],
        "sanity_docs_2026_08_28_rows": len(sanity_rows),
        "xg_restricted_endpoint_http": xg["http"],
        "xg_verified_in_sample": False,
        "scope_note": "Demo does not grant Developer xG/shotmaps; free-tier failure is not paid-provider coverage failure.",
    }

    sportsdb = []
    s_requests = []
    for day in DAYS:
        r = fetch("www.thesportsdb.com", "/api/v1/json/123/eventsday.php",
                  {"d": day, "s": "Soccer"})
        rows = take_list(r.get("data"), "events") if r["http"] == 200 else []
        sportsdb.extend(rows)
        s_requests.append({"day": day, "http": r["http"], "status": r["status"],
                           "rows": len(rows), "ms": r["ms"]})
        time.sleep(0.3)
    sc = compare(sportsdb, "SPORTSDB")
    ss = fetch("www.thesportsdb.com", "/api/v1/json/123/searchteams.php", {"t": "Arsenal"})
    ss_rows = take_list(ss.get("data"), "teams") if ss["http"] == 200 else []
    report["providers"]["THESPORTSDB_SHARED_FREE"] = {
        "access": "OFFICIAL_SHARED_PUBLIC_KEY_123_V1_CENSORED",
        "per_day": s_requests, **sc,
        "sanity_arsenal_http": ss["http"], "sanity_team_count": len(ss_rows),
        "xg_verified_in_sample": False,
        "scope_note": "Free endpoint may cap results to ONE; impossible to infer complete slate coverage from it.",
    }

    ol = fetch("api.openligadb.de", "/getmatchdata/bl1/2026/10")
    ol_rows = ol["data"] if ol["http"] == 200 and isinstance(ol.get("data"), list) else []
    report["providers"]["OPENLIGADB_PUBLIC"] = {
        "access": "NO_KEY_ODBL_GERMAN_BUNDESLIGA_MATCHDAY",
        "matchday_http": ol["http"], "rows": len(ol_rows), "ms": ol["ms"],
        "not_comparable_with_sample": True,
        "xg_verified_in_sample": False,
        "scope_note": "German league scores/fixtures control only; no claims about xG or broad international coverage.",
    }

    report["providers"]["HIGHLIGHTLY"] = {
        "access": "PRIVATE_KEY_REQUIRED",
        "live_sample_count": None,
        "scope_note": "No API key supplied; published league count is NOT a tested coverage count.",
    }
    report["providers"]["FOOTYSTATS"] = {
        "access": "CUSTOM_KEY_REQUIRED_FOR_GENERAL_FIXTURE_BENCHMARK",
        "live_sample_count": None,
    }
    report["providers"]["SPORTMONKS"] = {
        "access": "PRIVATE_KEY_REQUIRED",
        "live_sample_count": None,
    }

    report["http_requests_executed"] = calls
    assert calls == MAX_CALLS and report["market_calls"] == 0
    assert all(len(SAMPLE) >= len(v.get("strict_name_and_kickoff_15min_match_ids", []))
               for v in report["providers"].values())
    # Output only aggregate, no provider raw responses, no private credentials.
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
