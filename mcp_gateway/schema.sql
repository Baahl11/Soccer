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

CREATE INDEX IF NOT EXISTS idx_soccer_pipeline_runs_generated ON soccer_pipeline_runs (generated_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_fixtures_kickoff ON soccer_fixtures (kickoff);
CREATE INDEX IF NOT EXISTS idx_soccer_refresh_events_generated ON soccer_refresh_events (generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_market_fixture_time ON soccer_market_snapshots (fixture_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_alerts_created ON soccer_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_alerts_notification ON soccer_alerts (notification_ready, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_feature_fixture_time ON soccer_feature_snapshots (fixture_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_soccer_training_rows_fixture ON soccer_training_dataset_rows (fixture_id);
CREATE INDEX IF NOT EXISTS idx_soccer_training_builds_asof ON soccer_training_dataset_builds (as_of DESC);

ALTER TABLE soccer_training_dataset_rows ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE soccer_training_dataset_rows ADD COLUMN IF NOT EXISTS row_payload JSONB;
