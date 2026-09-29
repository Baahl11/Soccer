from pathlib import Path

path = Path('.github/workflows/soccer-edge-scheduler.yml')
text = path.read_text(encoding='utf-8')
needle = 'price_resolution_checkpoint:(.price_resolution_checkpoint // {}),'
assert text.count(needle) >= 2, text.count(needle)
text = text.replace(
    needle,
    needle + 'player_props_clv_maturation_funnel:(.player_props_clv_maturation_funnel // {}),',
)
validation = "and (.price_resolution_v4 | type == \"object\")' soccer_edge_state/latest.json >/dev/null"
assert validation in text
text = text.replace(
    validation,
    "and (.price_resolution_v4 | type == \"object\") and (.player_props_clv_maturation_funnel | type == \"object\")' soccer_edge_state/latest.json >/dev/null",
    1,
)
assert text.count('player_props_clv_maturation_funnel:(.player_props_clv_maturation_funnel // {})') >= 2
assert '(.player_props_clv_maturation_funnel | type == "object")' in text
path.write_text(text, encoding='utf-8')
print('V20112_SCHEDULER_STATE_PATCH_APPLIED')
