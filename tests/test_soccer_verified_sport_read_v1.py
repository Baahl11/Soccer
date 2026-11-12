from mcp_gateway import soccer_verified_sport_read_v1 as sport

FIXTURE = {"fixture_id": 1545869, "home_team_id": 1508,
           "away_team_id": 1499, "league_id": 36, "season": 2027}


def team(team_id, name, venue, wins, losses, goals_for, goals_against,
         played=2, played_venue=1, competition="Africa Cup of Nations - Qualification",
         season=2027):
    return {
        "team_id": team_id, "team_name": name, "venue": venue,
        "competition": competition, "season": season,
        "played": played, "played_venue": played_venue,
        "wins": wins, "draws": played-wins-losses, "losses": losses,
        "goals_for": goals_for, "goals_against": goals_against,
        "goals_for_venue": 2 if venue == "home" else 0,
        "goals_against_venue": 0 if venue == "home" else 1,
    }


def packet(*rows):
    return {"status": "SPORTING_SEASON_CONTEXT_ONLY",
            "source": "API_FOOTBALL_TEAM_SEASON_STATISTICS",
            "owner_admin_only": True,
            "commercial_display_allowed": False,
            "source_redisplay_rights_verified": False,
            "fetched_at_utc": "2026-11-12T15:01:00+00:00",
            "teams": list(rows)}


def test_congo_sierra_leone_two_games_are_real_evidence_not_empty_screen():
    home = team(1508, "Congo DR", "home", wins=2, losses=0, goals_for=4, goals_against=1)
    away = team(1499, "Sierra Leone", "away", wins=0, losses=2, goals_for=2, goals_against=4)
    read = sport.read_owner_context(packet(home, away), FIXTURE)
    assert read["status"] == "VERIFIED_SEASON_CONTEXT"
    assert read["evidence_status"] == "LOW_SAMPLE_EVIDENCE_ONLY"
    assert read["minimum_venue_sample"] == 1
    assert read["same_competition_and_season"] is True
    assert "2 victorias" in read["facts"][0]
    assert "4 goles a favor y 1 en contra" in read["facts"][0]
    assert "0 victorias" in read["facts"][2]
    assert any("muestra" in msg.lower() for msg in read["warnings"])
    assert read["sufficient_for_probability"] is False
    assert read["model_probabilities"] is None
    assert read["market_queried"] is False
    assert read["bet_eligible"] is False
    assert read["commercial_display_allowed"] is False
    assert read["source_redisplay_rights_verified"] is False


def test_cross_competition_and_one_team_only_are_not_comparable():
    home = team(1508, "Congo DR", "home", 3, 2, 8, 5, played=6, played_venue=3)
    away = team(1499, "Sierra Leone", "away", 1, 4, 5, 9, played=6,
                played_venue=3, competition="Another competition")
    # W+D+L must equal played: this sample has 6-3-2=1 draw.
    r = sport.read_owner_context(packet(home, away), FIXTURE)
    assert r["status"] == "VERIFIED_SEASON_CONTEXT"
    assert r["same_competition_and_season"] is False
    assert any("no son directamente comparables" in m for m in r["warnings"])
    only = sport.read_owner_context(packet(home), FIXTURE)
    assert only["status"] == "PARTIAL_SEASON_CONTEXT"
    assert only["model_probabilities"] is None


def test_mismatched_identity_or_missing_owner_permission_is_not_analysis():
    a = team(9876, "Not Congo", "home", 2, 0, 4, 1)
    b = team(1499, "Sierra Leone", "away", 0, 2, 2, 4)
    r = sport.read_owner_context(packet(a, b), FIXTURE)
    assert r["status"] == "PARTIAL_SEASON_CONTEXT"
    assert len(r["teams"]) == 1
    assert not r["season_context_verified"]
    r = sport.read_owner_context(
        {**packet(a,b), "owner_admin_only": False}, FIXTURE
    )
    assert r["status"] == "NO_VERIFIED_SEASON_CONTEXT"
    assert r["facts"] == []


def test_invalid_wins_losses_and_missing_goals_do_not_become_fake_facts():
    home = team(1508, "Congo DR", "home", 2, 0, 4, 1)
    home["wins"] = 3
    r = sport.read_owner_context(packet(home), FIXTURE)
    assert r["status"] == "NO_VERIFIED_SEASON_CONTEXT"
    home["wins"] = 2
    home["goals_for"] = None
    assert sport.read_owner_context(packet(home), FIXTURE)["facts"] == []


def test_frontend_prioritizes_verified_evidence_over_pending_banners():
    from mcp_gateway import subscriber_frontend_v2 as frontend
    html = frontend.render()
    assert "function verifiedSportAssessment(packet)" in html
    assert "const ownerSportRead=verifiedSportAssessment(d)" in html
    assert "sportN>0||hasOwnerSportRead" in html
    assert "ownerSportRead+ownerSeasonResearch" in html
    assert "muestra pequeña" in html.lower() or "Lectura deportiva histórica" in html
    assert "Fuente:" in html



def test_valid_persisted_pregame_p0_season_context_is_reused_without_provider_calls():
    import copy
    home=team(1508,"Congo DR","home",2,0,4,1)
    away=team(1499,"Sierra Leone","away",0,2,2,4)
    def compact(r):
        venue=r["venue"]
        return {
            "fixtures": {
                "played":{"total":r["played"],venue:r["played_venue"]},
                "wins":{"total":r["wins"]},
                "draws":{"total":r["draws"]},
                "loses":{"total":r["losses"]},
            },
            "goals":{
                "for":{"total":{"total":r["goals_for"],venue:r["goals_for_venue"]}},
                "against":{"total":{"total":r["goals_against"],venue:r["goals_against_venue"]}},
            },
        }
    fx={**FIXTURE,"kickoff":"2026-11-12T16:00:00+00:00",
        "home_team":"Congo DR","away_team":"Sierra Leone",
        "league":"Africa Cup of Nations - Qualification"}
    row={
        "event_type":"SOCCER_REFRESH","stage":"SPORT_PREVIEW",
        "source":"API_FOOTBALL_TEAM_SEASON_SPLITS",
        "fixture": copy.deepcopy(fx),
        "source_observed_at_utc":"2026-11-12T14:00:00+00:00",
        "quote_requested":False,"bet_eligible":False,
        "raw_projection":{"raw_home_win_prob":0.999}, # never shown
        "sporting":{"home_stats":compact(home),"away_stats":compact(away)},
    }
    context=sport.source_context_from_persisted_p0([{"payload":row}],fx)
    assert context["from_persisted_pregame"] is True
    assert context["provider_requests_added"] == 0
    assert context["projection"] is None
    read=sport.read_owner_context(context,fx)
    assert read["status"]=="VERIFIED_SEASON_CONTEXT"
    assert read["model_probabilities"] is None
    assert read["facts"]
    # False fixture identity or captured after kickoff cannot show history.
    wrong=copy.deepcopy(row)
    wrong["fixture"]["away_team_id"]=9876
    assert sport.source_context_from_persisted_p0([{"payload":wrong}],fx)["teams"] == []
    late=copy.deepcopy(row)
    late["source_observed_at_utc"]="2026-11-12T16:05:00+00:00"
    assert sport.source_context_from_persisted_p0([{"payload":late}],fx)["teams"] == []
    price=copy.deepcopy(row)
    price["quote_requested"]=True
    assert sport.source_context_from_persisted_p0([{"payload":price}],fx)["teams"] == []
