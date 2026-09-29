from pathlib import Path

path = Path('mcp_gateway/player_props_clv_postgres_v4.py')
text = path.read_text(encoding='utf-8')
start = text.index('def _load_snapshots(')
end = text.index('\ndef build_from_postgres(', start)
old = text[start:end]
assert 'LEFT JOIN LATERAL' in old

new_block = r'''def _load_snapshots(conn, fixture_ids: list[int], *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidate_snapshots AS MATERIALIZED (
                SELECT
                    m.snapshot_id,
                    m.fixture_id,
                    m.captured_at,
                    m.stage,
                    m.bookmaker_id,
                    m.bookmaker,
                    m.market_id,
                    m.market,
                    m.provider_update,
                    f.kickoff
                FROM soccer_market_snapshots m
                JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                WHERE m.fixture_id = ANY(%s)
                  AND m.captured_at >= %s
                  AND m.captured_at < f.kickoff
                  AND (
                    LOWER(COALESCE(m.market,'')) LIKE '%%player%%'
                    OR LOWER(COALESCE(m.market,'')) LIKE '%%scorer%%'
                    OR LOWER(COALESCE(m.market,'')) LIKE '%%goalkeeper save%%'
                    OR LOWER(COALESCE(m.market,'')) LIKE '%%keeper save%%'
                  )
                ORDER BY m.fixture_id, m.captured_at
                LIMIT %s
            )
            SELECT
                c.fixture_id,
                c.captured_at,
                c.stage,
                c.bookmaker_id,
                c.bookmaker,
                c.market_id,
                c.market,
                m.values,
                c.provider_update,
                c.kickoff
            FROM candidate_snapshots c
            JOIN soccer_market_snapshots m ON m.snapshot_id = c.snapshot_id
            """,
            (fixture_ids, cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_confirmed_lineups(conn, fixture_ids: list[int], *, lookback_days: int) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                l.fixture_id,
                l.captured_at,
                jsonb_build_object(
                    'both_xi_confirmed', TRUE,
                    'teams', COALESCE((
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'team_id', team->'team_id',
                                'team', team->'team',
                                'starters', COALESCE((
                                    SELECT jsonb_agg(
                                        jsonb_build_object(
                                            'id', player->'id',
                                            'name', player->'name',
                                            'pos', player->'pos'
                                        )
                                    )
                                    FROM jsonb_array_elements(
                                        COALESCE(team->'starters', '[]'::jsonb)
                                    ) AS player
                                ), '[]'::jsonb)
                            )
                        )
                        FROM jsonb_array_elements(
                            COALESCE(l.payload->'teams', '[]'::jsonb)
                        ) AS team
                    ), '[]'::jsonb)
                ) AS payload
            FROM soccer_lineup_snapshots l
            WHERE l.fixture_id = ANY(%s)
              AND l.captured_at >= %s
              AND l.both_xi_confirmed IS TRUE
            """,
            (fixture_ids, cutoff),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _attach_confirmed_lineups(
    snapshots: list[dict[str, Any]],
    lineups: list[dict[str, Any]],
) -> None:
    lineups_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    snapshots_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in lineups:
        try:
            fixture_id = int(row.get('fixture_id'))
        except (TypeError, ValueError):
            continue
        if _dt(row.get('captured_at')) is not None:
            lineups_by_fixture[fixture_id].append(row)
    for row in snapshots:
        try:
            fixture_id = int(row.get('fixture_id'))
        except (TypeError, ValueError):
            continue
        row['confirmed_lineup_payload'] = None
        if _dt(row.get('captured_at')) is not None:
            snapshots_by_fixture[fixture_id].append(row)

    for fixture_id, fixture_snapshots in snapshots_by_fixture.items():
        fixture_lineups = lineups_by_fixture.get(fixture_id, [])
        fixture_lineups.sort(key=lambda row: _dt(row.get('captured_at')) or datetime.min.replace(tzinfo=timezone.utc))
        fixture_snapshots.sort(key=lambda row: _dt(row.get('captured_at')) or datetime.min.replace(tzinfo=timezone.utc))
        lineup_index = 0
        latest_payload = None
        for snapshot in fixture_snapshots:
            snapshot_at = _dt(snapshot.get('captured_at'))
            if snapshot_at is None:
                continue
            while lineup_index < len(fixture_lineups):
                lineup_at = _dt(fixture_lineups[lineup_index].get('captured_at'))
                if lineup_at is None or lineup_at > snapshot_at:
                    break
                latest_payload = fixture_lineups[lineup_index].get('payload')
                lineup_index += 1
            snapshot['confirmed_lineup_payload'] = latest_payload


'''
text = text[:start] + new_block + text[end:]
needle = '''        snapshots = _load_snapshots(\n            conn,\n            fixture_ids,\n            lookback_days=lookback_days,\n            max_rows=max_rows,\n        )\n'''
replacement = needle + '''        confirmed_lineups = _load_confirmed_lineups(\n            conn,\n            fixture_ids,\n            lookback_days=lookback_days,\n        )\n        _attach_confirmed_lineups(snapshots, confirmed_lineups)\n        del confirmed_lineups\n        gc.collect()\n'''
assert needle in text
text = text.replace(needle, replacement, 1)
assert 'LEFT JOIN LATERAL' not in text[start:text.index('\ndef build_from_postgres(', start)]
assert '_load_confirmed_lineups' in text
assert '_attach_confirmed_lineups(snapshots, confirmed_lineups)' in text
path.write_text(text, encoding='utf-8')
print('V2014_PATCH_APPLIED')
