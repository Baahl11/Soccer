from pathlib import Path

SOURCE = Path("mcp_gateway/price_resolver_v4.py")
TESTS = Path("tests/test_price_resolver_v4.py")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    fn_anchor = text.index("def _load_team_totals_maturation_backlog(")

    start_marker = (
        "                  AND NOT EXISTS (\n"
        "                      SELECT 1\n"
        "                      FROM soccer_market_snapshots m\n"
    )
    start = text.index(start_marker, fn_anchor)
    end_marker = "                  )\n                ORDER BY f.kickoff ASC"
    end = text.index(end_marker, start) + len("                  )\n")

    replacement = '''                  AND NOT EXISTS (
                      SELECT 1
                      FROM soccer_market_snapshots m
                      JOIN soccer_refresh_events se
                        ON se.fixture_id = f.fixture_id
                       AND se.generated_at = ms.signal_generated_at
                      CROSS JOIN LATERAL jsonb_array_elements(
                          CASE
                              WHEN jsonb_typeof(
                                  COALESCE(
                                      se.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                                      '[]'::jsonb
                                  )
                              ) = 'array'
                              THEN COALESCE(
                                  se.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                                  '[]'::jsonb
                              )
                              ELSE '[]'::jsonb
                          END
                      ) AS sig(row)
                      CROSS JOIN LATERAL jsonb_array_elements(
                          CASE
                              WHEN jsonb_typeof(COALESCE(m.values, '[]'::jsonb)) = 'array'
                              THEN COALESCE(m.values, '[]'::jsonb)
                              ELSE '[]'::jsonb
                          END
                      ) AS q(value)
                      WHERE m.fixture_id = f.fixture_id
                        AND m.captured_at > ms.signal_generated_at
                        AND m.provider_update IS NOT NULL
                        AND m.provider_update > ms.signal_generated_at
                        AND m.captured_at < f.kickoff
                        AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(sig.row ->> 'market', '')))
                        AND (
                              CASE
                                  WHEN LOWER(TRIM(COALESCE(q.value ->> 'selection', ''))) LIKE 'over%%' THEN 'over'
                                  WHEN LOWER(TRIM(COALESCE(q.value ->> 'selection', ''))) LIKE 'under%%' THEN 'under'
                                  ELSE LOWER(TRIM(COALESCE(q.value ->> 'selection', '')))
                              END
                            ) = (
                              CASE
                                  WHEN LOWER(TRIM(COALESCE(sig.row ->> 'selection', ''))) LIKE 'over%%' THEN 'over'
                                  WHEN LOWER(TRIM(COALESCE(sig.row ->> 'selection', ''))) LIKE 'under%%' THEN 'under'
                                  ELSE LOWER(TRIM(COALESCE(sig.row ->> 'selection', '')))
                              END
                            )
                        AND COALESCE(q.value ->> 'line', '') ~ '^[0-9]+([.][0-9]+)?$'
                        AND COALESCE(sig.row ->> 'line', '') ~ '^[0-9]+([.][0-9]+)?$'
                        AND ABS(
                              (q.value ->> 'line')::NUMERIC
                              - (sig.row ->> 'line')::NUMERIC
                            ) < 0.000001
                  )
'''
    text = text[:start] + replacement + text[end:]
    SOURCE.write_text(text, encoding="utf-8")

    tests = TESTS.read_text(encoding="utf-8")
    marker = "def test_team_totals_maturation_backlog_requires_exact_comparable_quote():"
    if marker not in tests:
        tests += '''\n\ndef test_team_totals_maturation_backlog_requires_exact_comparable_quote():\n    import inspect\n\n    source = inspect.getsource(v._load_team_totals_maturation_backlog)\n    assert "JOIN soccer_refresh_events se" in source\n    assert "se.generated_at = ms.signal_generated_at" in source\n    assert "sig.row ->> 'market'" in source\n    assert "sig.row ->> 'selection'" in source\n    assert "q.value ->> 'selection'" in source\n    assert "sig.row ->> 'line'" in source\n    assert "q.value ->> 'line'" in source\n    assert "m.provider_update > ms.signal_generated_at" in source\n    assert "m.captured_at > ms.signal_generated_at" in source\n'''
        TESTS.write_text(tests, encoding="utf-8")


if __name__ == "__main__":
    main()
