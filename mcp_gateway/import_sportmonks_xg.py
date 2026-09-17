from __future__ import annotations

import argparse
import glob
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

SPORTMONKS_BASE = "https://api.sportmonks.com/v3/football"
XG_TYPE_ID = 5304


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _api_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).date().isoformat()
        return dt.date().isoformat()
    except ValueError:
        return text[:10] if len(text) >= 10 else None


def _load_aliases(path: str | None) -> dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    source = payload.get("sportmonks_to_api") if isinstance(payload, dict) and isinstance(payload.get("sportmonks_to_api"), dict) else payload
    if not isinstance(source, dict):
        return {}
    return {_norm(k): _norm(v) for k, v in source.items() if _norm(k) and _norm(v)}


def _history_index(history_dir: str) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    seen: set[int] = set()
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    if fid is None:
                        continue
                    try:
                        fid_int = int(fid)
                    except (TypeError, ValueError):
                        continue
                    if fid_int in seen:
                        continue
                    home = _norm(fixture.get("home_team"))
                    away = _norm(fixture.get("away_team"))
                    date = _api_date(fixture.get("kickoff"))
                    if not home or not away or not date:
                        continue
                    row = {
                        "api_fixture_id": fid_int,
                        "kickoff": fixture.get("kickoff"),
                        "home_team_id": fixture.get("home_team_id"),
                        "home_team": fixture.get("home_team"),
                        "away_team_id": fixture.get("away_team_id"),
                        "away_team": fixture.get("away_team"),
                    }
                    index.setdefault((date, home, away), []).append(row)
                    seen.add(fid_int)
    return index


def _fixture_names(item: dict[str, Any]) -> tuple[str | None, str | None]:
    participants = item.get("participants") if isinstance(item.get("participants"), list) else []
    home = away = None
    for p in participants:
        if not isinstance(p, dict):
            continue
        meta = p.get("meta") if isinstance(p.get("meta"), dict) else {}
        location = str(meta.get("location") or p.get("location") or "").lower()
        if location == "home":
            home = p.get("name")
        elif location == "away":
            away = p.get("name")
    if home and away:
        return str(home), str(away)
    name = str(item.get("name") or "")
    if " vs " in name:
        left, right = name.split(" vs ", 1)
        return left.strip() or None, right.strip() or None
    return None, None


def _xg(item: dict[str, Any]) -> tuple[float | None, float | None]:
    rows = item.get("xgfixture")
    if not isinstance(rows, list):
        rows = item.get("expected") if isinstance(item.get("expected"), list) else []
    home = away = None
    for row in rows:
        if not isinstance(row, dict) or int(row.get("type_id") or 0) != XG_TYPE_ID:
            continue
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        try:
            value = float(data.get("value"))
        except (TypeError, ValueError):
            continue
        location = str(row.get("location") or "").lower()
        if location == "home":
            home = value
        elif location == "away":
            away = value
    return home, away


def _get_page(token: str, start: str, end: str, page: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "api_token": token,
        "include": "xGFixture;participants",
        "per_page": 50,
        "page": page,
        "order": "asc",
    })
    url = f"{SPORTMONKS_BASE}/fixtures/between/{start}/{end}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "soccer-edge-engine/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Sportmonks request failed on page {page}: {type(exc).__name__}") from exc
    return payload if isinstance(payload, dict) else {}


def _existing(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not os.path.exists(path):
        return rows
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("api_fixture_id") is not None:
                    try:
                        rows[int(row["api_fixture_id"])] = row
                    except (TypeError, ValueError):
                        pass
    except OSError:
        pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Sportmonks xG type 5304 and map it conservatively to persisted API-Football fixtures.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--aliases", default="soccer_edge_state/external/sportmonks_team_aliases.json")
    ap.add_argument("--output", default="soccer_edge_state/external/xg_fixture_observations.jsonl")
    args = ap.parse_args()

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("SPORTMONKS_API_TOKEN is required; no external xG request was made.")

    aliases = _load_aliases(args.aliases)
    history = _history_index(args.history_dir)
    observations = _existing(args.output)
    fetched = with_xg = mapped = ambiguous = 0
    unmapped: list[dict[str, Any]] = []

    page = 1
    while True:
        payload = _get_page(token, args.start, args.end, page)
        items = payload.get("data") if isinstance(payload.get("data"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            fetched += 1
            home_xg, away_xg = _xg(item)
            if home_xg is None or away_xg is None:
                continue
            with_xg += 1
            source_home, source_away = _fixture_names(item)
            date = str(item.get("starting_at") or "")[:10]
            h = aliases.get(_norm(source_home), _norm(source_home))
            a = aliases.get(_norm(source_away), _norm(source_away))
            candidates = history.get((date, h, a), [])
            if len(candidates) != 1:
                if len(candidates) > 1:
                    ambiguous += 1
                elif len(unmapped) < 100:
                    unmapped.append({
                        "source_fixture_id": item.get("id"),
                        "starting_at": item.get("starting_at"),
                        "home": source_home,
                        "away": source_away,
                    })
                continue
            target = candidates[0]
            if target.get("home_team_id") is None or target.get("away_team_id") is None:
                continue
            fid = int(target["api_fixture_id"])
            observations[fid] = {
                **target,
                "source": "Sportmonks",
                "source_fixture_id": item.get("id"),
                "metric": "EXPECTED_GOALS",
                "metric_type_id": XG_TYPE_ID,
                "home_xg": round(home_xg, 6),
                "away_xg": round(away_xg, 6),
                "source_starting_at": item.get("starting_at"),
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            mapped += 1

        pagination = payload.get("pagination") if isinstance(payload.get("pagination"), dict) else {}
        if not pagination.get("has_more"):
            break
        page += 1
        if page > 200:
            raise RuntimeError("Sportmonks pagination safety limit reached")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for fid in sorted(observations):
            fh.write(json.dumps(observations[fid], ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "status": "SPORTMONKS_XG_IMPORT_COMPLETE",
        "start": args.start,
        "end": args.end,
        "fixtures_fetched": fetched,
        "fixtures_with_xg": with_xg,
        "mapped_this_run": mapped,
        "total_observations": len(observations),
        "ambiguous": ambiguous,
        "unmapped_sample": unmapped,
        "mapping_policy": "EXACT_NORMALIZED_DATE_HOME_AWAY_ONLY_PLUS_EXPLICIT_ALIAS_FILE; NO FUZZY_AUTO_MATCH",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
