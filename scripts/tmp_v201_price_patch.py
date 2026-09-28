from pathlib import Path

path = Path("mcp_gateway/price_resolver_v4.py")
text = path.read_text(encoding="utf-8")

marker = "\n\ndef _load_player_props_clv_maturation_backlog(\n"
assert marker in text, "player props backlog marker missing"
helper = '''

def _player_prop_maturation_family_accounting(
    signals: list[dict[str, Any]],
    matured_families: set[str],
) -> tuple[set[str], set[str]]:
    """Return evaluated Player Props families and those still missing a later quote."""
    supported = {
        "SHOTS",
        "SOT",
        "GOALSCORER_ANYTIME",
        "ASSISTS",
        "PLAYER_CARDS",
        "GK_SAVES",
    }
    evaluated = {
        str(signal.get("market_family") or "").upper()
        for signal in signals
        if str(signal.get("market_family") or "").upper() in supported
    }
    matured = {
        str(family).upper()
        for family in matured_families
        if str(family).upper() in evaluated
    }
    return evaluated, evaluated - matured


def _player_prop_missing_maturation_signals(
    signals: list[dict[str, Any]],
    later_markets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Keep only families that still lack a strict later provider quote."""
    matured_families = _player_prop_markets_with_later_provider_quote(
        later_markets,
        signals,
    )
    missing = [
        signal
        for signal in signals
        if str(signal.get("market_family") or "").upper() not in matured_families
    ]
    return missing, matured_families
'''
text = text.replace(marker, helper + marker, 1)

start = text.index("    # Suppress fixtures that already have a later provider-updated Player Props\n")
end = text.index("\n    events: list[dict[str, Any]] = []", start)
new_block = '''    # Suppress only families that already have a strictly later provider quote.
    # Fixture-level suppression is unsafe: one family can update while another
    # remains stale or missing.
    candidates: list[dict[str, Any]] = []
    candidate_family_counts: dict[str, int] = defaultdict(int)
    already_matured_family_counts: dict[str, int] = defaultdict(int)
    if grouped:
        with persistence._connect() as conn:
            with conn.cursor() as cur:
                for fixture_id, record in grouped.items():
                    signal_times = [
                        row.get("signal_generated_at")
                        for row in record["signals"]
                        if row.get("signal_generated_at")
                    ]
                    if not signal_times:
                        continue
                    earliest_signal = min(signal_times)
                    cur.execute(
                        """
                        SELECT m.market, m.provider_update
                        FROM soccer_market_snapshots m
                        JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                        WHERE m.fixture_id = %s
                          AND m.captured_at > %s
                          AND m.captured_at < f.kickoff
                          AND m.provider_update IS NOT NULL
                          AND m.provider_update > %s
                          AND (
                                LOWER(COALESCE(m.market,'')) LIKE '%%player shot%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%scorer%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%assist%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%goalkeeper save%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%keeper save%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%player card%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%player book%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%to be booked%%'
                             OR LOWER(COALESCE(m.market,'')) LIKE '%%to be carded%%'
                          )
                        ORDER BY m.captured_at DESC
                        """,
                        (fixture_id, earliest_signal, earliest_signal),
                    )
                    later_markets = [
                        {"market": raw[0], "provider_update": raw[1]}
                        for raw in cur.fetchall()
                        if raw and raw[0]
                    ]
                    missing_signals, matured_families = _player_prop_missing_maturation_signals(
                        list(record["signals"]),
                        later_markets,
                    )
                    for family in matured_families:
                        already_matured_family_counts[family] += 1
                    if not missing_signals:
                        continue
                    candidate = dict(record)
                    candidate["signals"] = missing_signals
                    candidates.append(candidate)
                    for signal in missing_signals:
                        family = str(signal.get("market_family") or "").upper()
                        if family:
                            candidate_family_counts[family] += 1
'''
text = text[:start] + new_block + text[end:]

old = '''        "candidate_family_counts": dict(sorted(family_counts.items())),
        "source": "POSTGRES_PLAYER_PROPS_CLV_MATURATION_BACKLOG",
'''
new = '''        "candidate_family_counts": dict(sorted(candidate_family_counts.items())),
        "signal_family_counts": dict(sorted(family_counts.items())),
        "already_matured_family_counts": dict(sorted(already_matured_family_counts.items())),
        "family_aware_suppression": True,
        "source": "POSTGRES_PLAYER_PROPS_CLV_MATURATION_BACKLOG_V2",
'''
assert old in text, "backlog return block missing"
text = text.replace(old, new, 1)

old = '''    player_props_maturation_family_refresh_counts: dict[str, int] = defaultdict(int)
    player_props_maturation_unchanged_provider_updates = 0
'''
new = '''    player_props_maturation_family_refresh_counts: dict[str, int] = defaultdict(int)
    player_props_maturation_family_evaluation_counts: dict[str, int] = defaultdict(int)
    player_props_maturation_not_matured_family_counts: dict[str, int] = defaultdict(int)
    player_props_maturation_unchanged_provider_updates = 0
'''
assert old in text, "metric init missing"
text = text.replace(old, new, 1)

old = '''                matured_families = _player_prop_markets_with_later_provider_quote(
                    markets,
                    signals,
                )
                if not matured_families:
                    player_props_maturation_unchanged_provider_updates += 1
                    continue
'''
new = '''                matured_families = _player_prop_markets_with_later_provider_quote(
                    markets,
                    signals,
                )
                evaluated_families, not_matured_families = _player_prop_maturation_family_accounting(
                    signals,
                    matured_families,
                )
                for family in evaluated_families:
                    player_props_maturation_family_evaluation_counts[family] += 1
                for family in not_matured_families:
                    player_props_maturation_not_matured_family_counts[family] += 1
                if not matured_families:
                    player_props_maturation_unchanged_provider_updates += 1
                    continue
'''
assert old in text, "matured block missing"
text = text.replace(old, new, 1)

old = '''        "player_props_clv_maturation_family_refresh_counts": dict(
            sorted(player_props_maturation_family_refresh_counts.items())
        ),
        "player_props_clv_maturation_unchanged_provider_updates": player_props_maturation_unchanged_provider_updates,
'''
new = '''        "player_props_clv_maturation_family_refresh_counts": dict(
            sorted(player_props_maturation_family_refresh_counts.items())
        ),
        "player_props_clv_maturation_family_evaluation_counts": dict(
            sorted(player_props_maturation_family_evaluation_counts.items())
        ),
        "player_props_clv_maturation_not_matured_family_counts": dict(
            sorted(player_props_maturation_not_matured_family_counts.items())
        ),
        "player_props_clv_maturation_signal_family_counts": dict(
            player_props_maturation.get("signal_family_counts") or {}
        ),
        "player_props_clv_maturation_already_matured_family_counts": dict(
            player_props_maturation.get("already_matured_family_counts") or {}
        ),
        "player_props_clv_maturation_family_aware_suppression": (
            player_props_maturation.get("family_aware_suppression") is True
        ),
        "player_props_clv_maturation_unchanged_provider_updates": player_props_maturation_unchanged_provider_updates,
'''
assert old in text, "payload metrics missing"
text = text.replace(old, new, 1)

old = '"player_props_clv_maturation_policy": "PRIMARY_TARGETS_FIRST;PRIMARY_CLV_SECOND;THEN_EXISTING_PLAYER_PROP_SIGNAL_LATER_REAL_QUOTE_MAX4;CACHE_REPLAY_NOT_CLOSE;THEN_TEAM_TOTALS;RESEARCH_ONLY",'
new = '"player_props_clv_maturation_policy": "PRIMARY_TARGETS_FIRST;PRIMARY_CLV_SECOND;THEN_EXISTING_PLAYER_PROP_SIGNAL_LATER_REAL_QUOTE_MAX4;FAMILY_AWARE_BACKLOG_SUPPRESSION;CACHE_REPLAY_NOT_CLOSE;THEN_TEAM_TOTALS;RESEARCH_ONLY",'
assert old in text, "policy missing"
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

# Append focused resolver tests.
test_path = Path("tests/test_price_resolver_v4.py")
test_text = test_path.read_text(encoding="utf-8")
assert "test_v201_player_props_partial_family_maturation_keeps_missing_family" not in test_text
addition = '''


def test_v201_player_props_partial_family_maturation_keeps_missing_family():
    signals = [
        {"market_family": "SHOTS", "signal_generated_at": "2026-09-28T20:00:00+00:00"},
        {"market_family": "GOALSCORER_ANYTIME", "signal_generated_at": "2026-09-28T20:00:00+00:00"},
    ]
    later_markets = [{
        "market": "Anytime Goal Scorer",
        "provider_update": "2026-09-28T20:10:00+00:00",
    }]
    missing, matured = v._player_prop_missing_maturation_signals(signals, later_markets)
    assert matured == {"GOALSCORER_ANYTIME"}
    assert [row["market_family"] for row in missing] == ["SHOTS"]


def test_v201_player_props_maturation_family_accounting_tracks_partial_close():
    signals = [
        {"market_family": "SHOTS"},
        {"market_family": "GK_SAVES"},
        {"market_family": "OTHER"},
    ]
    evaluated, missing = v._player_prop_maturation_family_accounting(signals, {"GK_SAVES"})
    assert evaluated == {"SHOTS", "GK_SAVES"}
    assert missing == {"SHOTS"}


def test_v201_player_props_backlog_uses_family_aware_suppression():
    import inspect
    source = inspect.getsource(v._load_player_props_clv_maturation_backlog)
    assert "_player_prop_missing_maturation_signals" in source
    assert "SELECT m.market, m.provider_update" in source
    assert 'candidate["signals"] = missing_signals' in source
    assert '"family_aware_suppression": True' in source
    assert "POSTGRES_PLAYER_PROPS_CLV_MATURATION_BACKLOG_V2" in source
'''
test_path.write_text(test_text.rstrip() + addition + "\n", encoding="utf-8")
