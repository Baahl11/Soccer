from datetime import datetime, timezone
from mcp_gateway import soccer_sport_queue_v1 as queue
from mcp_gateway import subscriber_match_notes_v1 as notes

class Cursor:
    def __init__(self):
        self.ops=[]
    def execute(self, query, params=None):
        self.ops.append((query, params))


def test_queue_writes_all_eligible_registry_fixtures_without_network():
    cur=Cursor()
    queue.sync_pending(cur)
    sql, params=cur.ops[0]
    assert "FROM soccer_fixtures" in sql
    assert "kickoff > CURRENT_TIMESTAMP" in sql
    assert "ON CONFLICT (fixture_id)" in sql
    assert params==(queue.MAX_REGISTRY,)


def test_queue_never_counts_a_market_event_as_sport_screen():
    cur=Cursor()
    queue.record_event(cur, {"event_type":"SOCCER_REFRESH","stage":"T-40",
                             "fixture":{"fixture_id":17,"kickoff":"2026-11-12T17:00:00Z"}},at="2026-11-12T13:00:00Z")
    assert not cur.ops
    queue.record_event(cur, {"event_type":"SOCCER_REFRESH","stage":"SPORT_PREVIEW",
                             "raw_sport_show_as_forecast":True,"raw_sport_screen_status":"MODELED_LIMITED",
                             "model_version":"P0-v1",
                             "fixture":{"fixture_id":17,"kickoff":"2026-11-12T17:00:00Z"}},at="2026-11-12T13:00:00Z")
    assert len(cur.ops)==1
    assert cur.ops[0][1][0]==17
    assert cur.ops[0][1][2]=="MODELED_LIMITED"
    assert "sport_status='MODELED_LIMITED'" in cur.ops[0][0]


def test_queue_coverage_counts_unknown_instead_of_hiding_unmodeled_fixtures():
    fixtures=[{"fixture":{"fixture_id":7}},{"fixture":{"fixture_id":8}},{"fixture":{"fixture_id":9}}]
    summary=queue.coverage_summary({"7":{"status":"MODELED_LIMITED"},
                                    "8":{"status":"DATA_INSUFFICIENT"}},fixtures)
    assert summary["total_eligible_fixtures"]==3
    assert summary["by_sport_queue_status"]["MODELED_LIMITED"]==1
    assert summary["by_sport_queue_status"]["DATA_INSUFFICIENT"]==1
    assert summary["by_sport_queue_status"]["NOT_VERIFIED"]==1
    assert summary["all_fixtures_accounted_for"] is True


def test_note_sanitization_and_no_model_input():
    assert notes.validate_note({"fixture_id":991,"note_text":"  revisar portero  "})==(991,"revisar portero")
    for invalid in ({"fixture_id":991,"note_text":""},
                    {"fixture_id":991,"note_text":"x"*2001},
                    {"fixture_id":991,"note_text":None},
                    {"fixture_id":"991","note_text":"not verified"}):
        try:
            notes.validate_note(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid note was accepted")


class Response:
    def __init__(self,status,rows):
        self.status_code=status
        self.rows=rows
    def json(self):
        return self.rows


class FakeHTTP:
    def __init__(self):
        self.calls=[]
    def get(self, url, **kwargs):
        self.calls.append(("GET",url,kwargs))
        return Response(200,[{"fixture_id":991,"note_text":"revisar portero"}])
    def post(self,url,**kwargs):
        self.calls.append(("POST",url,kwargs))
        return Response(201,[{"fixture_id":991,"note_text":"revisar portero"}])
    def delete(self,url,**kwargs):
        self.calls.append(("DELETE",url,kwargs))
        return Response(204,[])


def test_notes_enforce_user_scope_and_never_enter_model(monkeypatch):
    monkeypatch.setattr(notes.supabase_auth_v4,"auth_config",
                        lambda:{"configured":True,"project_url":"https://example.supabase.co",
                                "publishable_key":"test-public-key"})
    http=FakeHTTP()
    uid="11111111-1111-4111-8111-111111111111"
    for method,body,fixture in [
        ("GET",None,991),
        ("POST",{"fixture_id":991,"note_text":"revisar portero"},None),
        ("DELETE",None,991),
    ]:
        result=notes.operate(method,"test-user-jwt",uid,fixture_id=fixture,body=body,client=http)
        assert result["ok"] is True
        assert result["model_inputs_changed"] is False
        assert result["provider_requests_added"]==0
    assert all(req[2]["params"].get("user_id")==("eq."+uid)
               for req in http.calls if req[0] in ("GET","DELETE"))
    assert http.calls[1][2]["json"]["user_id"]==uid
    assert "api-test-secret" not in str(http.calls)


def test_authenticated_free_notes_are_separate_from_pro_saved_recommendations(monkeypatch):
    import asyncio
    import json
    from starlette.requests import Request
    from mcp_gateway import subscriber_contract_v2 as customer
    from mcp_gateway import subscription_entitlements_v4 as subscriptions

    async def free_entitlement(request):
        return ({
            "authenticated": True,
            "effective_plan": "FREE",
            "user": {"id": "11111111-1111-4111-8111-111111111111"},
        }, None)

    def operate(method, token, user_id, *, fixture_id=None, body=None):
        assert token == "test-access-token"
        assert user_id == "11111111-1111-4111-8111-111111111111"
        assert method == "GET"
        assert fixture_id == 991
        return {"ok": True, "status": "NOTES_READY",
                "rows": [{"fixture_id": 991, "note_text": "posible cambio tactico"}]}

    monkeypatch.setattr(customer, "_resolve_entitlement", free_entitlement)
    monkeypatch.setattr(customer, "_my_edge_allowed", lambda *_: False)
    monkeypatch.setattr(customer.supabase_auth_v4, "bearer_token", lambda h: "test-access-token")
    monkeypatch.setattr(customer.subscriber_match_notes_v1, "operate", operate)
    scope = {
        "type": "http", "method": "GET", "path": "/app/api/v2/my-edge",
        "scheme": "https", "server": ("localhost", 443),
        "headers": [(b"authorization", b"Bearer test-access-token")],
        "query_string": b"resource=notes&fixture_id=991",
    }
    response = asyncio.run(customer.my_edge(Request(scope)))
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["rows"][0]["note_text"] == "posible cambio tactico"
    assert payload["canonical_bet_logic_changed"] is False

    # No note resource: the original paid saved-item entitlement remains intact.
    scope = {**scope, "query_string": b""}
    denied = asyncio.run(customer.my_edge(Request(scope)))
    assert denied.status_code == 403
    assert json.loads(denied.body)["error"] == "PRO_REQUIRED"


def test_league_coverage_retains_unknown_and_does_not_claim_model():
    fixtures = [
        {"fixture":{"fixture_id":71,"league_id":1,"league":"A"}},
        {"fixture":{"fixture_id":72,"league_id":1,"league":"A"}},
        {"fixture":{"fixture_id":73,"league_id":2,"league":"B"}},
    ]
    summary = queue.coverage_summary({
        "71":{"status":"MODELED_LIMITED"},
        "72":{"status":"DATA_INSUFFICIENT"},
    },fixtures)
    assert summary["all_fixtures_accounted_for"] is True
    assert summary["by_league_total"] == 2
    assert sum(row["eligible"] for row in summary["by_league"]) == 3
    a, b = summary["by_league"]
    assert a["league"] == "A"
    assert a["by_status"]["MODELED_LIMITED"] == 1
    assert a["by_status"]["DATA_INSUFFICIENT"] == 1
    assert b["by_status"]["NOT_VERIFIED"] == 1
    assert summary["no_market_inputs"] is True
