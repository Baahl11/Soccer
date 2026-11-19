from mcp_gateway.full_slate_research_v1 import goal_model, index_slate, quote_inventory

def test_pilot_and_explicit_missing():
    h={"played":{"home":7},"gf":{"home":6},"ga":{"home":7}}
    a={"played":{"away":7},"gf":{"away":10},"ga":{"away":9}}
    m=goal_model(h,a)
    assert m["probabilities"]["1X2"] == {"home":.322749,"draw":.283808,"away":.393444}
    assert m["probabilities"]["Goles FT"]["2.5"] == .400169
    assert "Córners" not in m["probabilities"]
    assert goal_model({},a) is None

def test_slate_not_capped_or_filtered_by_league():
    rows=[{"fixture_id":i,"league_id":i,"status":"PST" if i==1 else "NS"} for i in range(1,113)]
    assert len(index_slate(rows, lambda x:x)) == 112

def test_market_reference_and_empty():
    assert not any(quote_inventory([]).values())
    x=quote_inventory([{"update":"t","bookmakers":[{"name":"book","bets":[{"name":"Goals Over/Under","values":[{"value":"Over 2.5","odd":"2.00"}]}]}]}])
    assert x["Goles FT"][0]["source_updated_at"]=="t"
    assert not x["Tarjetas"]

def test_postgame_grading_preserves_as_of_and_requires_both_teams():
    from mcp_gateway.full_slate_research_v1 import evaluate
    h={"played":{"home":7},"gf":{"home":6},"ga":{"home":7}}
    a={"played":{"away":7},"gf":{"away":10},"ga":{"away":9}}
    model=goal_model(h,a)
    card={"fixture":{"timestamp":1795111200,"home_team_id":1,"away_team_id":2},
          "model_captured_at":"2026-11-19T17:49:00+00:00","model":model}
    final={"home_team_id":1,"away_team_id":2,"home":1,"away":1,"observed_at":"2026-11-19T20:00:00+00:00"}
    grade=evaluate(card,final)
    assert grade["actual_1x2"]=="draw" and grade["actual_btts"] and not grade["actual_over_2_5"]
    assert evaluate(card,dict(final,away_team_id=3)) is None
    assert evaluate(dict(card,model_captured_at="2026-11-19T18:00:00+00:00"),final) is None
    assert card["model"] == model
