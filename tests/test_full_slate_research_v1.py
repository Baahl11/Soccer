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
