CREATE TABLE IF NOT EXISTS soccer_pipeline_runs (
    run_id BIGSERIAL PRIMARY KEY,
    generated_at_utc TIMESTAMPTZ NOT NULL,
    generated_at_local TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL,
    fixture_scan_count INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0,
    actionable_refresh_count INTEGER NOT NULL DEFAULT 0,
    quota_remaining INTEGER,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS soccer_fixtures (
    fixture_id BIGINT PRIMARY KEY,
    league_id BIGINT,
    league TEXT,
    country TEXT,
    season INTEGER,
    round TEXT,
    kickoff TIMESTAMPTZ,
    status TEXT,
    status_long TEXT,
    home_team_id BIGINT,
    home_team TEXT,
    away_team_id BIGINT,
    away_team TEXT,
    venue TEXT,
    city TEXT,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS soccer_refresh_events (
    event_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    classification TEXT,
    availability_confidence NUMERIC(4,3),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    data_tier TEXT,
    generated_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (fixture_id, stage, generated_at)
);

CREATE TABLE IF NOT EXISTS soccer_lineup_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    stage TEXT,
    lineup_state TEXT,
    both_xi_confirmed BOOLEAN,
    both_goalkeepers_confirmed BOOLEAN,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS soccer_availability_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    stage TEXT,
    availability_confidence NUMERIC(4,3),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS soccer_market_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    stage TEXT,
    bookmaker_id BIGINT,
    bookmaker TEXT,
    market_id BIGINT,
    market TEXT,
    values JSONB NOT NULL,
    provider_update TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS soccer_primary_clv_signal_anchors (
    anchor_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    market_family TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    line_text TEXT,
    price_text TEXT,
    signal_generated_at TIMESTAMPTZ NOT NULL,
    candidate_source TEXT NOT NULL,
    source_priority INTEGER NOT NULL,
    source_row JSONB NOT NULL,
    materialized_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_soccer_primary_clv_signal_anchor_identity
ON soccer_primary_clv_signal_anchors (
    fixture_id,
    market_family,
    (LOWER(TRIM(market))),
    (LOWER(TRIM(selection))),
    (COALESCE(line_text, '')),
    signal_generated_at,
    candidate_source
);

CREATE INDEX IF NOT EXISTS idx_soccer_primary_clv_signal_anchor_live
ON soccer_primary_clv_signal_anchors (
    signal_generated_at DESC,
    fixture_id,
    market_family
);

CREATE TABLE IF NOT EXISTS soccer_feature_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    stage TEXT,
    schema_version TEXT NOT NULL,
    model_version TEXT,
    data_tier TEXT,
    feature_count INTEGER NOT NULL DEFAULT 0,
    missing_feature_count INTEGER NOT NULL DEFAULT 0,
    payload JSONB NOT NULL,
    UNIQUE (fixture_id, captured_at, stage, schema_version)
);

CREATE TABLE IF NOT EXISTS soccer_model_runs (
    model_run_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    model_version TEXT NOT NULL,
    run_type TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    raw_projection JSONB,
    shrunk_projection JSONB,
    model_prob NUMERIC(7,6),
    market_fair_prob NUMERIC(7,6),
    prob_edge_pp NUMERIC(8,4),
    estimated_ev NUMERIC(10,6),
    shrink_weight NUMERIC(7,6),
    availability_confidence NUMERIC(4,3),
    classification TEXT,
    tier TEXT,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS soccer_training_dataset_builds (
    build_id TEXT PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    feature_count INTEGER NOT NULL DEFAULT 0,
    dataset_sha256 TEXT NOT NULL,
    selection_policy TEXT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS soccer_training_dataset_rows (
    build_id TEXT NOT NULL REFERENCES soccer_training_dataset_builds(build_id) ON DELETE CASCADE,
    row_number INTEGER NOT NULL,
    fixture_id BIGINT NOT NULL,
    snapshot_id BIGINT NOT NULL,
    snapshot_captured_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    league_id BIGINT,
    season INTEGER,
    stage TEXT,
    features JSONB NOT NULL,
    missingness JSONB NOT NULL,
    provenance JSONB,
    targets JSONB NOT NULL,
    row_payload JSONB,
    row_sha256 TEXT NOT NULL,
    PRIMARY KEY (build_id, row_number),
    UNIQUE (build_id, fixture_id)
);

CREATE TABLE IF NOT EXISTS soccer_results (
    fixture_id BIGINT PRIMARY KEY,
    final_status TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    final_score JSONB,
    match_stats JSONB,
    graded_at TIMESTAMPTZ,
    payload JSONB NOT NULL
);

-- FE-11: append-only sport-first research profiles, never betting outputs.
CREATE TABLE IF NOT EXISTS soccer_multimarket_history_profiles (
    fixture_id BIGINT NOT NULL REFERENCES soccer_fixtures (fixture_id),
    source_generated_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    source TEXT NOT NULL
        CHECK (source = 'SPORTS_EDGE_SCANNER_V15_HISTORICAL_CONTEXT'),
    model_version TEXT NOT NULL
        CHECK (model_version = 'SOCCER_SPORT_FIRST_MULTIMARKET_PROFILE_v1.0'),
    home_profile JSONB NOT NULL,
    away_profile JSONB NOT NULL,
    home_pattern JSONB NOT NULL,
    away_pattern JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classification TEXT NOT NULL DEFAULT 'WATCH'
        CHECK (classification = 'WATCH'),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (bet_eligible = FALSE),
    PRIMARY KEY (fixture_id, source_generated_at),
    CHECK (home_team_id <> away_team_id),
    CHECK (source_generated_at < kickoff)
);
CREATE INDEX IF NOT EXISTS idx_soccer_multimarket_history_profile_latest
    ON soccer_multimarket_history_profiles (fixture_id, source_generated_at DESC);

CREATE TABLE IF NOT EXISTS soccer_research_backfill_discovery (
    research_key TEXT NOT NULL,
    team_id BIGINT NOT NULL,
    cutoff TIMESTAMPTZ NOT NULL,
    lookback_days INTEGER NOT NULL,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider_status TEXT NOT NULL,
    provider_fixture_count INTEGER NOT NULL DEFAULT 0,
    persisted_fixture_count INTEGER NOT NULL DEFAULT 0,
    payload JSONB,
    PRIMARY KEY (research_key, team_id, cutoff)
);

CREATE TABLE IF NOT EXISTS soccer_alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stage TEXT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'INFO',
    classification TEXT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    payload JSONB,
    expires_at TIMESTAMPTZ,
    notification_ready BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_soccer_pipeline_runs_generated_at ON soccer_pipeline_runs (generated_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_fixtures_kickoff ON soccer_fixtures (kickoff);
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_generated ON soccer_refresh_events (generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_generated_event ON soccer_refresh_events (generated_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_fixture_generated ON soccer_refresh_events (fixture_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_team_totals_diversity
ON soccer_refresh_events (fixture_id, generated_at DESC)
WHERE COALESCE(payload -> 'team_totals_diversity_capture' ->> 'qualifies', 'false') = 'true';
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_team_totals_observed
ON soccer_refresh_events (fixture_id, generated_at DESC)
WHERE (
    CASE
        WHEN jsonb_typeof(COALESCE(payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb)) = 'array'
        THEN jsonb_array_length(COALESCE(payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb))
        ELSE 0
    END
) > 0;
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_player_props_generated
ON soccer_refresh_events (generated_at DESC)
INCLUDE (event_id, fixture_id, stage)
WHERE stage IN ('T-40','T-30','T-20','T-10')
  AND payload ? 'market'
  AND (
       payload ? 'player_shots_intelligence'
    OR payload ? 'player_sot_intelligence'
    OR payload ? 'player_goalscorer_intelligence'
    OR payload ? 'player_assists_intelligence'
    OR payload ? 'player_cards_intelligence'
    OR payload ? 'gk_saves_intelligence'
  );
CREATE INDEX IF NOT EXISTS idx_soccer_market_fixture_time ON soccer_market_snapshots (fixture_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_market_fixture_market_close
ON soccer_market_snapshots (
    fixture_id,
    (LOWER(TRIM(COALESCE(market, '')))),
    captured_at DESC,
    provider_update DESC
)
WHERE provider_update IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_soccer_market_ft_totals_coverage
ON soccer_market_snapshots (captured_at DESC, fixture_id, bookmaker_id, market_id)
WHERE LOWER(TRIM(COALESCE(market, ''))) IN (
    'goals over/under',
    'over/under',
    'goals over under'
);
CREATE INDEX IF NOT EXISTS idx_soccer_model_runs_fixture_time ON soccer_model_runs (fixture_id, run_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_lineup_fixture_time_confirmed ON soccer_lineup_snapshots (fixture_id, captured_at DESC) WHERE both_xi_confirmed IS TRUE;
CREATE INDEX IF NOT EXISTS idx_soccer_alerts_created ON soccer_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_alerts_notification ON soccer_alerts (notification_ready, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_feature_fixture_time ON soccer_feature_snapshots (fixture_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_training_rows_fixture ON soccer_training_dataset_rows (fixture_id);
CREATE INDEX IF NOT EXISTS idx_soccer_training_builds_asof ON soccer_training_dataset_builds (as_of DESC);

ALTER TABLE soccer_training_dataset_rows ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE soccer_training_dataset_rows ADD COLUMN IF NOT EXISTS row_payload JSONB;

CREATE INDEX IF NOT EXISTS idx_soccer_research_backfill_discovery_attempted
ON soccer_research_backfill_discovery (research_key, attempted_at DESC);

-- FE11 universal sport-first identity baseline. A baseline is frozen only
-- when the official fixture was observed as NS BEFORE its actual kickoff.
-- Not a projection, not a bookmaker quote, not a wager or OOS training label.
CREATE TABLE IF NOT EXISTS soccer_fe11_pregame_baselines (
    fixture_id BIGINT PRIMARY KEY REFERENCES soccer_fixtures (fixture_id),
    captured_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    league_id BIGINT NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    source TEXT NOT NULL
        CHECK (source = 'API_FOOTBALL_FIXTURES_DATE'),
    source_status TEXT NOT NULL DEFAULT 'NS'
        CHECK (source_status = 'NS'),
    snapshot JSONB NOT NULL,
    classification TEXT NOT NULL DEFAULT 'RESEARCH_ONLY'
        CHECK (classification = 'RESEARCH_ONLY'),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (bet_eligible = FALSE),
    CHECK (captured_at < kickoff),
    CHECK (home_team_id > 0 AND away_team_id > 0 AND home_team_id <> away_team_id)
);
CREATE INDEX IF NOT EXISTS idx_soccer_fe11_pregame_kickoff
    ON soccer_fe11_pregame_baselines (kickoff, league_id);

-- FE11 official finished-score observations are append-only. One fixture can
-- receive a later corrected official snapshot without editing the original.
-- Corners and other markets stay NULL/NOT VERIFIED unless a separate source
-- verifies the correct period and participant IDs.
CREATE TABLE IF NOT EXISTS soccer_fe11_official_outcome_snapshots (
    fixture_id BIGINT NOT NULL REFERENCES soccer_fe11_pregame_baselines (fixture_id),
    observed_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    final_status TEXT NOT NULL CHECK (final_status IN ('FT', 'AET', 'PEN')),
    home_goals INTEGER NOT NULL CHECK (home_goals BETWEEN 0 AND 50),
    away_goals INTEGER NOT NULL CHECK (away_goals BETWEEN 0 AND 50),
    halftime_home_goals INTEGER CHECK (halftime_home_goals BETWEEN 0 AND 50),
    halftime_away_goals INTEGER CHECK (halftime_away_goals BETWEEN 0 AND 50),
    regulation_home_goals INTEGER CHECK (regulation_home_goals BETWEEN 0 AND 50),
    regulation_away_goals INTEGER CHECK (regulation_away_goals BETWEEN 0 AND 50),
    second_half_home_goals INTEGER CHECK (second_half_home_goals BETWEEN 0 AND 50),
    second_half_away_goals INTEGER CHECK (second_half_away_goals BETWEEN 0 AND 50),
    corners_home INTEGER,
    corners_away INTEGER,
    corners_verified BOOLEAN NOT NULL DEFAULT FALSE CHECK (corners_verified = FALSE),
    source TEXT NOT NULL CHECK (source = 'API_FOOTBALL_FIXTURES_DATE'),
    payload JSONB NOT NULL,
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible = FALSE),
    PRIMARY KEY (fixture_id, observed_at),
    CHECK (observed_at >= kickoff),
    CHECK (corners_home IS NULL AND corners_away IS NULL)
);
CREATE INDEX IF NOT EXISTS idx_soccer_fe11_postgame_fixture_date
    ON soccer_fe11_official_outcome_snapshots (fixture_id, observed_at DESC);

-- A separate FT-only provider-statistics observation prevents invented
-- corners from entering the official-score payload. Corner values are
-- independently sourced, never inferred from fixture goals or team means.
CREATE TABLE IF NOT EXISTS soccer_fe11_verified_corner_snapshots (
    fixture_id BIGINT NOT NULL REFERENCES soccer_fe11_pregame_baselines (fixture_id),
    observed_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    home_corners INTEGER NOT NULL CHECK (home_corners BETWEEN 0 AND 60),
    away_corners INTEGER NOT NULL CHECK (away_corners BETWEEN 0 AND 60),
    period TEXT NOT NULL CHECK (period = 'FT_90'),
    source TEXT NOT NULL CHECK (source = 'API_FOOTBALL_FIXTURE_STATISTICS'),
    source_response_verified BOOLEAN NOT NULL CHECK (source_response_verified = TRUE),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible = FALSE),
    PRIMARY KEY (fixture_id, observed_at),
    CHECK (observed_at >= kickoff)
);
CREATE INDEX IF NOT EXISTS idx_soccer_fe11_corners_fixture_time
    ON soccer_fe11_verified_corner_snapshots (fixture_id, observed_at DESC);

-- Attempt throttling makes unavailable team stats explicit and prevents
-- repeatedly burning quota on the same match every scheduler tick.
CREATE TABLE IF NOT EXISTS soccer_fe11_corner_capture_attempts (
    fixture_id BIGINT PRIMARY KEY REFERENCES soccer_fe11_pregame_baselines (fixture_id),
    last_attempt_at TIMESTAMPTZ NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count BETWEEN 1 AND 20),
    last_status TEXT NOT NULL,
    provider_requests_added INTEGER NOT NULL DEFAULT 0 CHECK (provider_requests_added BETWEEN 0 AND 1),
    CHECK (length(last_status) <= 64)
);


-- Research-only input from the isolated, GitHub-runner canonical baseline.
-- NEVER add these rows to soccer_refresh_events or CLV signal anchors.
-- Original source timestamp is preserved separately from ingestion time.
CREATE TABLE IF NOT EXISTS soccer_isolated_sport_frozen_research_v1 (
    fixture_id BIGINT NOT NULL CHECK (fixture_id > 0),
    source_generated_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    home_team_id BIGINT NOT NULL CHECK (home_team_id > 0),
    away_team_id BIGINT NOT NULL CHECK (away_team_id > 0),
    source TEXT NOT NULL CHECK (source = 'LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS'),
    projection_model TEXT NOT NULL CHECK (projection_model = 'GOAL_RATE_POISSON_LIMITED_v0.1'),
    model_git_sha TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    raw_projection JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classification TEXT NOT NULL DEFAULT 'RESEARCH_ONLY' CHECK (classification = 'RESEARCH_ONLY'),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible = FALSE),
    production_promotion_allowed BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (production_promotion_allowed = FALSE),
    PRIMARY KEY (fixture_id, source_generated_at),
    CHECK (source_generated_at <= kickoff - INTERVAL '45 minutes'),
    CHECK (home_team_id <> away_team_id)
);


-- Every eligible fixture has a durable screening position independent of BET.
-- Do not turn absence of a run into a PASS; keep the first queue timestamp.
CREATE TABLE IF NOT EXISTS soccer_sport_screen_queue (
    fixture_id BIGINT PRIMARY KEY REFERENCES soccer_fixtures (fixture_id) ON DELETE CASCADE,
    kickoff TIMESTAMPTZ NOT NULL,
    sport_status TEXT NOT NULL DEFAULT 'QUEUED'
        CHECK (sport_status IN ('QUEUED','MODELED_LIMITED','DATA_INSUFFICIENT','RETRY_REQUIRED')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_attempt_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ,
    last_model_version TEXT,
    last_reason TEXT,
    first_queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_soccer_sport_screen_pending_kickoff
    ON soccer_sport_screen_queue (sport_status, kickoff, last_attempt_at)
    WHERE sport_status <> 'MODELED_LIMITED';
