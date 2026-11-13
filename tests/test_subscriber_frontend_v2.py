from __future__ import annotations

from mcp_gateway import subscriber_frontend_v2


def test_fe2_shell_contains_no_design_time_mock_data():
    html = subscriber_frontend_v2.render()

    for forbidden in (
        "Arsenal vs Brighton",
        "Man City vs Nottm Forest",
        "Mockup Preview",
        "const markets=[['1X2'",
        "const rows=[['Arsenal vs Brighton'",
    ):
        assert forbidden not in html


def test_fe2_primary_information_architecture_is_customer_first():
    html = subscriber_frontend_v2.render()

    for label in (
        "Today",
        "Picks",
        "Leans",
        "Matches",
        "Performance",
        "My Edge",
        "Account",
    ):
        assert label in html

    assert "Control Tower" not in html
    assert "Research Lab" not in html


def test_fe2_visually_preserves_sport_first_market_second_separation():
    html = subscriber_frontend_v2.render()

    assert "RAW SPORT" in html
    assert "MARKET SHRUNK" in html
    assert "MARKET FAIR" in html
    assert "PROB EDGE" in html
    assert "Sport first. Market second." in html


def test_fe2_consumes_only_v2_customer_contract_for_decision_data():
    html = subscriber_frontend_v2.render()

    assert '"/app/api/v2"' in html
    assert "api('/today')" in html
    assert "api('/performance')" in html
    assert "api('/match/'" in html
    assert "/app-preview/data" not in html
    assert "/app/data" not in html


def test_fe2_billing_surface_keeps_price_ids_out_and_launch_default_off():
    html = subscriber_frontend_v2.render()
    contract = subscriber_frontend_v2.contract()

    assert "price_1ULUp1BE8LWeAYkQBNijFI03" not in html
    assert "price_1ULUp7BE8LWeAYkQbanCnbG7" not in html
    assert contract["billing_mutations_enabled"] is False
    assert contract["public_billing_launch_default"] is False
    assert contract["browser_sends_stripe_price_id"] is False

def test_fe2_has_one_customer_bootstrap_controller():
    html = subscriber_frontend_v2.render()

    assert html.count('id="soccer-edge-v2-app"') == 1
    assert html.count('id="boot"') == 1
    assert "MutationObserver" not in html
    assert "setTimeout(" not in html


def test_fe2_contract_preserves_engine_firewall():
    contract = subscriber_frontend_v2.contract()

    assert contract["mock_data"] is False
    assert contract["single_bootstrap_controller"] is True
    assert contract["operator_surfaces_in_primary_navigation"] is False
    assert contract["billing_mutations_enabled"] is False
    assert contract["frontend_creates_bet_or_lean"] is False
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False

def test_fe2_full_slate_exposes_coverage_and_insufficient_data_copy():
    html = subscriber_frontend_v2.render()

    assert "Every eligible fixture stays visible, even when analysis is incomplete" in html
    assert "Estado de análisis: NOT VERIFIED" in html
    assert "Sport data: " in html
    assert 'data-intel="' in html

def test_fe2_every_slate_fixture_is_openable_for_data_review():
    html = subscriber_frontend_v2.render()

    assert "Open data →" in html
    assert "Sport-first read" in html
    assert "Goal-rate baseline" in html
    assert "Datos deportivos que sí tenemos" in html
    assert 'class="slate-row" href="' in html
    assert "matchPath(f.fixture_id||'')" in html
    assert '[data-intel="1"]' not in html

def test_fe2_match_navigation_uses_dedicated_url_and_direct_boot_route():
    html = subscriber_frontend_v2.render()

    assert "'/app/match/'" in html
    assert "routeMatchId()" in html
    assert "history.pushState" in html
    assert "← Matches" in html
    assert "matchesBrowsePanel" in html

def test_fe2_identity_enrichment_chunks_large_slates_and_shows_relational_evidence():
    html = subscriber_frontend_v2.render()

    assert "for(let i=0;i<ids.length;i+=80)" in html
    assert "Latest market activity" in html
    assert "Market snapshots" in html
    assert "Model runs" in html
    assert "Evidence quality" in html


def test_fe2_match_detail_is_summary_first_and_mobile_safe():
    html = subscriber_frontend_v2.render()

    assert "intel-metrics" in html
    assert "intel-lens" in html
    assert "intel-tabs" in html
    assert "intel-accordion" in html
    assert "overflow-x:auto" in html
    assert "Raw current-snapshot market table" in html
    assert "Input coverage" in html
    assert "Datos actualizados" in html
    assert "Match result probability" in html
    assert "Expected goals" in html


def test_fe2_match_intelligence_is_sport_first_not_market_first():
    html = subscriber_frontend_v2.render()

    assert "Sport-first read" in html
    assert "Goal-rate baseline" in html
    assert "Team form & scoring" in html
    assert "Home form" in html
    assert "Market layer" in html
    assert "context only · not an edge by itself" in html
    assert "Market data exists, but verified sporting evidence is not sufficient" in html


def test_fe2_visible_sport_input_count_matches_rendered_inputs():
    html = subscriber_frontend_v2.render()

    assert "CUSTOMER_SPORT_FEATURES" in html
    assert "Verified sporting inputs" in html
    assert "core sport inputs" in html
    assert "core sport inputs" in html
    assert "No customer-facing sporting inputs are verified" in html


def test_fe2_kickoff_display_is_pinned_to_mexico_central_time():
    html = subscriber_frontend_v2.render()

    assert "APP_TIMEZONE='America/Mexico_City'" in html
    assert "timeZone:APP_TIMEZONE" in html


def test_fe2_match_center_visual_components_follow_blueprint_phase_1_and_2():
    html = subscriber_frontend_v2.render()

    for component in (
        "visual-grid",
        "coverage-gauge",
        "rate-bars",
        "form-dots",
        "prob-tiles",
        "prob-stack",
        "takeaway-list",
        "sport-stat-grid",
    ):
        assert component in html

    assert "Market rows do not increase it." in html
    assert "values are not labeled xG unless a verified xG source exists." in html
    assert "No synthetic probabilities." not in html


def test_fe2_match_center_rejects_zero_sample_team_stats_as_verified():
    html = subscriber_frontend_v2.render()

    assert "source==='API_FOOTBALL_TEAM_STATS'" in html
    assert "n!==null&&n<=0" in html
    assert "Market rows do not increase it." in html


def test_fe2_match_center_mobile_uses_premium_dense_composition():
    html = subscriber_frontend_v2.render()

    assert "match-brand-row" in html
    assert "mini-visual-grid" in html
    assert "form-spark" in html
    assert "goal-card" in html
    assert "body:has(#matches.active .match-detail) .topbar" in html


def test_fe2_sport_tab_matches_premium_visual_hierarchy():
    html = subscriber_frontend_v2.render()

    assert "Match result probability" in html
    assert "Expected goals" in html
    assert "Score matrix" in html
    assert "Goal distribution" in html
    assert "Team sport profile" in html
    assert "from score matrix" in html
    assert "goal-dist-bars" in html


def test_fe10_research_desk_owner_preview_preserves_canonical_picks():
    html = subscriber_frontend_v2.render()
    assert 'id="researchStatus"' in html
    assert 'id="researchPlus100"' in html
    assert 'id="researchRegular"' in html
    assert "api('/research-price-desk')" in html
    assert "EDGE -110 o mejor" in html
    assert "loadResearchDesk()" in html
    assert "WATCH · NO BET" in html
    assert "notification" not in html.lower() or "notificaciones push habilitadas" in html.lower()


def test_fe10_sgp_radar_has_bookmaker_composites_and_unpriced_multi_player_slip():
    html = subscriber_frontend_v2.render()
    assert 'id="sgpQuotedComposites"' in html
    assert 'id="sgpPlayerLegs"' in html
    assert 'id="sgpBuilderSlip"' in html
    assert "function sgpComboCard(" in html
    assert "function sgpPlayerCard(" in html
    assert "function wireSgpPlayerSelection(" in html
    assert "same game" in html.lower() or "Same Game" in html
    assert "CUOTA COMBINADA: NOT VERIFIED" in html
    assert "2.1" in html
    assert "BET" in html


def test_fe10_sgp_competing_outcomes_are_only_in_collapsed_market_catalog():
    html = subscriber_frontend_v2.render()
    assert '<details id="sgpMarketCatalogDetails"' in html
    assert '<summary>Ver catálogo de cuotas combinadas (NO son picks)</summary>' in html
    assert 'id="sgpMarketCatalog"' in html
    assert 'id="sgpQuotedComposites"' in html
    assert "const catalog=Array.isArray(data.sgp_market_catalog)" in html
    assert "function sgpCatalogCard(" in html
    assert "CATÁLOGO DE MERCADO, NO PICK" in html
    assert "Máximo UNA señal por encuentro" in html
    assert "NO BET" in html


def test_fe11_sport_first_period_corners_handicap_tab_never_creates_bet():
    html = subscriber_frontend_v2.render()
    assert "['periods','1H / 2H / Córners']" in html
    assert "data-detail-section=\"periods\"" in html
    assert "function fe11MultimarketSection(" in html
    assert "function fe11TeamProfile(" in html
    assert "function fe11PatternCard(" in html
    assert "d?.multi_market_research||null" in html
    assert "NO BET" in html
    assert "MUESTRA BAJA" in html
    assert "probabilidad conjunta futura" in html
    assert "NOT VERIFIED: no hay perfil histórico" in html
    assert "bookmaker" in html.lower()


def test_edge_sport_radar_is_visible_with_real_evidence_reasons_and_safe_fallback():
    html = subscriber_frontend_v2.render()
    assert 'id="edgeSportRadar"' in html
    assert 'id="edgeRadarStats"' in html
    assert 'id="edgeRadarStatus"' in html
    assert "Edge Radar · Cobertura deportiva real" in html
    assert "PERSISTED_EVIDENCE_AUDIT_READY" in html
    assert "renderEdgeSportRadar()" in html
    assert "renderWatchSlate(){renderEdgeSportRadar()" in html
    assert "r?.screening||{}" in html
    assert "e.missing_evidence_codes" in html
    assert "e.reason||s.reason_display" in html
    assert "Cobertura del registro NO VERIFICADA" in html
    assert "sin consultas nuevas a la API" in html
    assert "INVESTIGACIÓN · NO BET" in html
    assert "data-intel=" in html
    assert "api('/today')" in html
    assert "renderEdgeSportRadar" in html
    assert "renderEdgeSportRadar(){" in html
    assert "No synthetic probabilities" not in html


def test_edge_radar_highlightly_owner_only_metadata_never_becomes_bet():
    html=subscriber_frontend_v2.render()
    assert "OWNER_RESEARCH_COVERAGE_AVAILABLE" in html
    assert "supplementary_sport_research" in html
    assert "Highlightly BASIC" in html
    assert "(ACCESS?.owner||ACCESS?.admin)" in html
    assert "sin evidencia apta para modelar" in html
    assert "sin licencia de publicación" in html or "licencia pendiente" in html
    assert "NO BET" in html


def test_match_center_sporting_forecast_is_first_and_diagnostics_are_collapsed():
    html = subscriber_frontend_v2.render()
    overview = html.split('const overview=', 1)[1].split('const marketsSection=', 1)[0]

    assert "Partido identificado · análisis pendiente" in overview
    assert 'class="match-reading-banner"' in overview
    assert "Victoria · Empate · Derrota" in overview
    assert "Datos deportivos que sí tenemos" in overview
    assert overview.index('class="match-reading-banner"') < overview.index("Input coverage")
    assert overview.index("Victoria · Empate · Derrota") < overview.index("Contexto de goles registrado")
    assert "<details><summary>Forma reciente y datos disponibles" in overview
    assert "no mide confianza del modelo ni porcentaje de partidos analizados" in html
    assert "['overview','Resumen']" in html


def test_match_center_missing_numbers_do_not_become_zero_or_fake_xg():
    html = subscriber_frontend_v2.render()

    assert "if(v===null||v===undefined||v===''||typeof v==='boolean')return null" in html
    assert "featureVerified(features,'team_performance.home_goal_rate_blend')" in html
    assert "featureVerified(features,'team_performance.away_goal_rate_blend')" in html
    assert "POISSON_LAMBDA_NOT_XG" in html
    assert "Expected goals (xG) NOT VERIFIED sin fuente independiente" in html
    assert "No hay tasas de gol verificadas." in html


def test_match_center_readability_at_desktop_and_mobile():
    html = subscriber_frontend_v2.render()

    assert "#matchDetail .intel-tab{font-size:12px" in html
    assert "#matchDetail .takeaway small{font-size:12px" in html
    assert "#matchDetail .prob-tile b{font-size:27px" in html
    assert "@media(max-width:800px){#matchDetail .overview-primary-grid{grid-template-columns:1fr}}" in html
    assert "#matchDetail .mini-visual-grid,#matchDetail .sport-two-up{grid-template-columns:1fr}" in html


def test_customer_home_displays_real_sport_screening_not_an_empty_pick_lab():
    html=subscriber_frontend_v2.render()
    assert "let focusFilter='all';" in html
    assert "const screen=DATA?.sport_screening||{};" in html
    assert "SPORT_FIRST_SCREEN_RECORDED" in html
    assert "proyecciones deportivas limitadas" in html
    assert "No hay pronósticos previos al inicio" not in html
    assert "Las proyecciones de investigación no autorizadas para publicación" in html
    assert "focus-empty-compact" in html
    assert "canReviewPrivateResearch=ACCESS?.owner===true||ACCESS?.admin===true" in html


def test_universal_match_remains_available_without_pro_and_pass_cannot_hide_sport():
    html=subscriber_frontend_v2.render()
    assert "function showPublicMatch(fid)" in html
    assert "if(!premium()){showPublicMatch(fid);return}" in html
    assert "row?.sport_analysis" in html
    assert "row?.opportunity" in html
    assert "MERCADO_NO_EVALUADO" in html
    assert "SPORT_SCREEN_PASS" in html
    assert "Un PASS no oculta el análisis" in html
    assert "El análisis deportivo se evalúa con independencia del mercado." in html
    assert "proyección numérica" not in html.lower() or "licencia" in html.lower()


def test_analyst_notebook_uses_authenticated_api_and_does_not_become_model_input():
    html=subscriber_frontend_v2.render()
    assert 'id="analystNoteText"' in html
    assert "maxlength=\"2000\"" in html
    assert "async function loadAnalystNote(fid)" in html
    assert "item_type:'NOTE'" in html
    assert "/my-edge?resource=notes&fixture_id=" in html
    assert "No modifica el modelo" in html
    assert "loadAnalystNote(f.fixture_id)" in html


def test_today_shows_per_fixture_sport_queue_counts_without_fabricating_forecasts():
    html = subscriber_frontend_v2.render()
    assert "Estado deportivo acumulado de la jornada" in html
    assert "queueOverview.all_fixtures_accounted_for===true" in html
    assert "queueCounts.MODELED_LIMITED" in html
    assert "queueCounts.DATA_INSUFFICIENT" in html
    assert "queueCounts.QUEUED" in html
    assert "queueCounts.RETRY_REQUIRED" in html
    assert "queueCounts.NOT_VERIFIED" in html
    assert "NO predicciones publicables ni recomendaciones" in html
    assert "Estado de cobertura no verificado" in html


def test_live_cancelled_fixture_detail_is_a_real_readable_explanation_not_blank_odds():
    html = subscriber_frontend_v2.render()
    assert "d?.live_fixture_status?.source==='API_FOOTBALL_FIXTURE_STATUS_BY_ID'" in html
    assert "const blockedEvent=" in html
    assert "Este partido está cancelado" in html
    assert "Estado de oportunidad: evento no vigente" in html
    assert "No corresponde emitir una proyección vigente" in html
    assert "Estado de oportunidad: evento no vigente" in html
    assert "void loadAnalystNote(f.fixture_id)" in html
    assert "f={...known,...rawF}" in html


def test_match_mobile_must_fit_viewport_and_empty_probability_panels_are_not_repeated():
    html = subscriber_frontend_v2.render()
    assert "#matchDetail .intel-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}" in html
    assert "#matchDetail .universal-status-grid{grid-template-columns:minmax(0,1fr)}" in html
    assert "#matchDetail .intel-tabs{overflow-x:auto" in html
    assert "hasSportProb?'<div class=\"visual-card primary-forecast\"" in html
    assert "sportN>0?'<div class=\"visual-card primary-takeaways\"" in html
    assert "No hay estadísticas verificadas en el expediente" in html


def test_owner_private_verified_team_stats_visible_without_xg_or_public_promotion():
    html=subscriber_frontend_v2.render()
    assert "function internalSeasonFacts(packet)" in html
    assert "packet?.private_team_context" in html
    assert "INVESTIGACIÓN PRIVADA" in html
    assert "Datos históricos verificados" in html
    assert "NO xG ni un modelo 1X2" in html
    assert "No tienen validación OOS ni permiso de redistribución comercial." in html
    assert "ownerSeasonResearch+pastFacts" in html
    assert "ownerSeasonResearch+" in html
    assert ".season-facts-grid{grid-template-columns:minmax(0,1fr)}" in html


def test_today_has_plain_language_expandable_league_coverage():
    html = subscriber_frontend_v2.render()
    assert "Cobertura deportiva por competición" in html
    assert "const leagueRows=Array.isArray(queueOverview.by_league)" in html
    assert "focus-league-detail" in html
    assert "sin datos suficientes" in html
    assert "pendientes/no verificados" in html
    assert "no son pronósticos publicables" in html


def test_owner_match_uses_only_exact_asof_private_frozen_model_no_public_promotion():
    html=subscriber_frontend_v2.render()
    assert "const ownerP0=(ACCESS?.owner===true||ACCESS?.admin===true)?" in html
    assert "[matchFrozen,...p0Entries()].find(x=>x&&String(x.fixture_id)===String(f.fixture_id)" in html
    assert "const matchFrozen=d?.private_frozen_sport_projection" in html
    assert "x.market_queried===false&&x.bet_eligible===false" in html
    assert "x.commercial_display_allowed===false" in html
    assert "Math.abs(Date.parse(x.kickoff_utc)-Date.parse(f.kickoff))<60*1000" in html
    assert "Math.abs(ownerRaw.raw_home_win_prob+ownerRaw.raw_draw_prob+" in html
    assert "Modelo deportivo preliminar guardado" in html
    assert "ownerResearchCard+ownerRecent+ownerSportRead+ownerSeasonResearch" in html
    assert "No validado fuera de muestra" in html
    assert "Precio de mercado: NO CONSULTADO" in html


def test_owner_existing_sport_evidence_joins_by_fixture_without_new_collection():
    html = subscriber_frontend_v2.render()
    assert "function ownerPersistedSportEvidence(packet,fixture)" in html
    assert "ownerPersistedSportEvidence(d,f)" in html
    assert "packet?.private_pregame_evidence_sections" in html
    assert "Observaciones prepartido guardadas" in html
    assert "FE11 se muestra aparte" in html
    assert "Solo OWNER" in html
    assert "No son predicciones RAW, OOS ni recomendaciones. NO BET." in html
    assert "Sin observaciones prepartido adicionales verificadas en PostgreSQL" in html

    function = html.split(
        "function ownerPersistedSportEvidence(packet,fixture)", 1
    )[1].split("function internalSeasonFacts(packet)", 1)[0]
    # Permissions, source, timestamp and sample are checked before a row can
    # be shown. The code only formats stored evidence from the existing match.
    assert "ACCESS?.owner!==true&&ACCESS?.admin!==true" in function
    assert "item?.status==='PERSISTED'" in function
    assert "!!item?.source" in function
    assert "captured<kickoff&&captured<=now" in function
    assert "Number.isInteger(sample)&&sample>0" in function
    assert "new Set(['TEAMS','GOALS','STATS','CORNERS','CARDS'" in function
    assert "fetch(" not in function
    assert "api(" not in function
    assert "bet_eligible=true" not in function


def test_owner_matchup_dossier_executes_with_raw_fe11_and_independent_blockers():
    import subprocess
    from shutil import which
    import pytest

    if not which("node"):
        pytest.skip("Node is required to execute the matchup renderer")
    html = subscriber_frontend_v2.render()
    marker = "  function ownerMatchupResearch("
    source = marker + html.split(marker, 1)[1].split("  function renderMatch(", 1)[0]
    assert "['matchup-research','Expediente deportivo']" in html
    assert "(privateMatchup?matchupResearch:multiMarket)" in html
    script = r"""
const assert = require('node:assert/strict');
let ACCESS = {owner:true};
const pct = p => (100*p).toFixed(1)+'%';
const intelKV = rows => JSON.stringify(rows);
const esc = x => String(x);
const fe11MultimarketSection = x => x.history_fresh ? 'FE11_DETAIL' : 'FE11_MISSING';
""" + source + r"""
const fixture = {fixture_id:42,kickoff:'2026-11-14T12:00:00Z'};
Date.now = () => Date.parse('2026-11-14T11:00:00Z');
const raw = {raw_home_win_prob:.5,raw_draw_prob:.3,raw_away_win_prob:.2,
             raw_over_2_5_prob:.42,raw_btts_yes_prob:0};
const fe11 = {fixture_id:42,kickoff:fixture.kickoff,source_generated_at_utc:'2026-11-14T10:00:00Z',
  status:'HISTORICAL_DESCRIPTIVE_ONLY',history_fresh:true,
  observations_not_forecasts:true,bet_eligible:false,
  home:{splits:{home:{features:{total_corners_ft:{n:8}}}}}};
const teamTotals={status:'PERSISTED_RESEARCH_PROJECTION',fixture_id:42,
  kickoff_utc:fixture.kickoff,captured_at_utc:'2026-11-14T10:00:00Z',
  source:'POSTGRES_REFRESH_EVENTS_TEAM_TOTALS',bet_eligible:false,oos_certified_by_this_view:false,
  rows:[{team_role:'HOME',line:1.5,p_over:.4,p_under:.6}]};
const corners={status:'PERSISTED_RESEARCH_RATES',fixture_id:42,
  kickoff_utc:fixture.kickoff,captured_at_utc:'2026-11-14T10:00:00Z',
  source:'POSTGRES_REFRESH_EVENTS_FT_CORNERS',bet_eligible:false,oos_certified_by_this_view:false,
  home_lambda:5.5,away_lambda:4,total_lambda:9.5,sample:{prior_home_home_matches:0}};
const cards={status:'PERSISTED_RESEARCH_RATES',fixture_id:42,
  kickoff_utc:fixture.kickoff,captured_at_utc:'2026-11-14T10:00:00Z',
  source:'POSTGRES_REFRESH_EVENTS_FT_YELLOW_CARDS',bet_eligible:false,oos_certified_by_this_view:false,
  total_yellow_lambda:4.2,sample:{referee_prior_n:0}};
const shot={fixture_id:42,kickoff_utc:fixture.kickoff,pregame_snapshot_at_utc:'2026-11-14T10:00:00Z',
  classification:'RESEARCH_ONLY',bet_eligible:false,oos_validation_complete:false,family:'SOT',
  player:'Jugador de prueba',lines:[{line:1.5,p_over_raw:.25,p_under_raw:.75}]};
const props={status:'VERIFIED_SPORTING_INPUTS_RESEARCH_ONLY',fixture_id:42,
  source:'POSTGRES_REFRESH_EVENTS_PLAYER_SHOTS_SOT',bet_eligible:false,rows:[shot]};
const red={status:'PERSISTED_RESEARCH_PROJECTION',fixture_id:42,kickoff_utc:fixture.kickoff,
  captured_at_utc:'2026-11-14T10:00:00Z',source:'POSTGRES_REFRESH_EVENTS_ANY_RED_CARD',
  bet_eligible:false,oos_certified_by_this_view:false,p_any_red:.14,p_no_red:.86};
const render = (research,projection=raw,blocked=false,totals=null,corner=null,yellow=null,playerProps=null,redCard=null,teamYellow=null) =>
  ownerMatchupResearch({multi_market_research:research,
    private_team_total_projection:totals,
    private_ft_corners_rates:corner,
    private_ft_yellow_card_rate:yellow,
    private_pregame_shot_props:playerProps,
    private_red_card_projection:redCard,
    private_team_yellow_card_rates:teamYellow,
    private_frozen_sport_projection:{raw_probabilities:raw,
      raw_goal_rates:{raw_home_goal_rate:1.52,raw_away_goal_rate:0.81}},
    private_pregame_evidence_sections:[{category:'CARDS',items:[{}]}]},fixture,projection,'SPORT_EVIDENCE',blocked);
const good=render(fe11);
const teamYellow={status:'PERSISTED_RESEARCH_RATES',fixture_id:42,kickoff_utc:fixture.kickoff,
  captured_at_utc:'2026-11-14T10:00:00Z',source:'POSTGRES_REFRESH_EVENTS_TEAM_YELLOW_CARDS',
  bet_eligible:false,oos_certified_by_this_view:false,home_yellow_lambda:1.8,away_yellow_lambda:2.3};
const teamCard=(x)=>render(fe11,raw,false,null,null,null,null,null,x);
assert.match(teamCard(teamYellow),/Amarillas por equipo · tasas guardadas/);
assert.match(teamCard(teamYellow),/1.80/);
assert.match(teamCard(teamYellow),/2.30/);
assert.doesNotMatch(teamCard({...teamYellow,fixture_id:43}),/Amarillas por equipo · tasas guardadas/);
assert.doesNotMatch(teamCard({...teamYellow,captured_at_utc:fixture.kickoff}),/Amarillas por equipo · tasas guardadas/);
assert.match(render(fe11,raw,false,null,null,null,null,red),/Al menos una roja · proyección guardada/);
assert.doesNotMatch(render(fe11,raw,false,null,null,null,null,{...red,fixture_id:43}),/Al menos una roja · proyección guardada/);
assert.doesNotMatch(render(fe11,raw,false,null,null,null,null,{...red,p_no_red:.8}),/Al menos una roja · proyección guardada/);
assert.match(render(fe11,raw,false,null,null,null,props),/Más: 25.0% · Menos: 75.0%/);
assert.doesNotMatch(render(fe11,raw,false,null,null,null,{...props,rows:[{...shot,fixture_id:43}]}),/Más: 25.0%/);
assert.doesNotMatch(render(fe11,raw,false,null,null,null,{...props,rows:[{...shot,pregame_snapshot_at_utc:fixture.kickoff}]}),/Más: 25.0%/);
assert.match(render(fe11,raw,false,null,null,cards),/Amarillas FT · tasa del modelo guardado/);
assert.doesNotMatch(render(fe11,raw,false,null,null,{...cards,fixture_id:43}),/Amarillas FT · tasa del modelo guardado/);
assert.doesNotMatch(render(fe11,raw,false,null,null,{...cards,captured_at_utc:fixture.kickoff}),/Amarillas FT · tasa del modelo guardado/);
const both=render(fe11,raw,false,null,corners);
assert.match(both,/Córners FT · tasas del modelo guardado/);
assert.match(both,/5.50 · n localía 0/);
assert.match(both,/FE11_DETAIL/);
assert.doesNotMatch(render(fe11,raw,false,null,{...corners,fixture_id:43}),/tasas del modelo guardado/);
assert.doesNotMatch(render(fe11,raw,false,null,{...corners,captured_at_utc:fixture.kickoff}),/tasas del modelo guardado/);
assert.match(render(fe11,raw,false,teamTotals),/Más: 40.0% · Menos: 60.0%/);
assert.doesNotMatch(render(fe11,raw,false,{...teamTotals,fixture_id:43}),/Más: 40.0%/);
assert.doesNotMatch(render(fe11,raw,false,{...teamTotals,captured_at_utc:fixture.kickoff}),/Más: 40.0%/);
assert.match(good,/50.0% \/ 30.0% \/ 20.0%/);
assert.match(good,/RAW preliminar: 42.0%/);
assert.match(good,/RAW preliminar: 0.0%/);
assert.match(good,/n local 8 · n visita N\/D/);
assert.match(good,/FE11_DETAIL/);
assert.match(good,/SPORT_EVIDENCE/);
assert.match(good,/local 1.52 · visita 0.81 · sin proyección verificada por línea/);
assert.doesNotMatch(render(fe11, {...raw}),/Tasas RAW guardadas/);
assert.match(good,/1 campos prepartido guardados/);
assert.match(good,/Props .*Sin proyección específica verificada/);
assert.match(good,/Sin muestra FE11 por localía/);
for (const wrong of [
    {...fe11,fixture_id:99},{...fe11,kickoff:'2026-11-14T12:30:00Z'},
    {...fe11,history_fresh:false},{...fe11,source_generated_at_utc:fixture.kickoff},
    {...fe11,source_generated_at_utc:'2026-11-14T11:30:00Z'}
]) {
    assert.match(render(wrong),/FE11_MISSING/);
    assert.doesNotMatch(render(wrong),/n local 8/);
}
assert.doesNotMatch(render(fe11,null),/RAW preliminar:/);
assert.equal(render(fe11,raw,true),'');
ACCESS = {owner:false,admin:false};
assert.equal(render(fe11),'');
ACCESS = {admin:true};
assert.match(render(fe11),/FE11_DETAIL/);
"""
    result = subprocess.run(["node", "-"], input=script, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
