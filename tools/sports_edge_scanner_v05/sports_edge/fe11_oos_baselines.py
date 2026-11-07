"""FE11 frozen walk-forward *baselines* for three-way +1 triple conditions.

Compare a naive product of independently estimated leg rates with the actual
same-fixture joint hit rate, using ONLY training fixtures whose labels were
available before the embargoed cutoff. This is RESEARCH ONLY: no fitted
opponent-adjusted model, no calibration claim, no executable quote, no BET.

No bookmaker odds, network/API calls, external packages or production writes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import log, sqrt
from typing import Any

from .fe11_oos_dataset import VERSION as DATASET_VERSION

VERSION = "SOCCER_FE11_FROZEN_HOLDOUT_BASELINES_v0.1"
SUPPORTED_MARKETS = {
    "home_three_way_plus_one_joint": "HOME",
    "away_three_way_plus_one_joint": "AWAY",
}
FORBIDDEN_FEATURES = frozenset({
    "odds", "price", "bookmaker", "market", "stake", "ev", "breakeven",
    "bet_eligible", "outcome", "label", "result", "actual",
})


def _dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        out = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return out.astimezone(timezone.utc) if out.tzinfo is not None else None
    except ValueError:
        return None


def _id(value: Any) -> int | None:
    return value if type(value) is int and 0 < value < 10**12 else None


def _safe_count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 80


def _wilson_95(hits: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    spread = z * sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [round(max(0, center-spread), 5), round(min(1, center+spread), 5)]


def _metrics(labels: list[int], p: float) -> dict[str, float]:
    p = min(1 - 1e-12, max(1e-12, p))
    n = len(labels)
    return {
        "brier": round(sum((p-y)**2 for y in labels)/n, 7),
        "log_loss": round(-sum(y*log(p)+(1-y)*log(1-p) for y in labels)/n, 7),
    }


def _validated_row(row: Any, market: str, goal_line: float,
                   corner_line: float, research_cutoff: datetime) -> dict[str, Any]:
    if (not isinstance(row, dict) or not isinstance(row.get("labels"), dict)
            or not isinstance(row.get("features"), dict)):
        raise ValueError("MALFORMED_FE11_ROW")
    if any(word in str(key).lower() for key in row["features"].keys()
           for word in FORBIDDEN_FEATURES):
        raise ValueError("MARKET_OR_TARGET_LEAKAGE_IN_FEATURES")
    fid = _id(row.get("fixture_id"))
    home, away = _id(row.get("home_team_id")), _id(row.get("away_team_id"))
    kick = _dt(row.get("kickoff_utc"))
    fin = _dt(row.get("target_source_available_at_utc"))
    corner_available = _dt(row.get("target_corners_available_at_utc"))
    if (fid is None or home is None or away is None or home == away
            or kick is None or fin is None or corner_available is None
            or kick > research_cutoff or fin > research_cutoff
            or corner_available > research_cutoff
            or fin < kick + timedelta(minutes=90)
            or corner_available < kick + timedelta(minutes=90)):
        raise ValueError("FE11_ROW_IDENTITY_OR_PROVENANCE_INVALID")
    lab = row["labels"]
    goals, corners, margin = lab.get("ft_goals"), lab.get("ft_corners"), lab.get("home_margin")
    if not (_safe_count(goals) and _safe_count(corners)
            and type(margin) is int and -40 <= margin <= 40):
        raise ValueError("FE11_LABELS_INVALID")
    if goals > 40 or corners > 80:
        raise ValueError("FE11_LABELS_OUT_OF_RANGE")
    g, c = goals > goal_line, corners > corner_line
    margin_plus = margin + 1 if market.startswith("home") else 1 - margin
    handicap = margin_plus > 0
    expected = "WIN" if g and c and handicap else "LOSS"
    if (type(lab.get("goals_over_hit")) is not bool
            or type(lab.get("corners_over_hit")) is not bool
            or lab["goals_over_hit"] != g or lab["corners_over_hit"] != c
            or lab.get(market) != expected):
        raise ValueError("FE11_LABELS_INCONSISTENT_OR_UNSETTLED")
    return {
        "fixture_id": fid, "kickoff": kick, "result_known": fin,
        "corners_known": corner_available, "league_id": _id(row.get("league_id")),
        "goals": int(g), "corners": int(c), "handicap": int(handicap),
        "joint": int(expected == "WIN"),
    }


def evaluate_fe11_frozen_baselines(
    dataset: dict[str, Any], *,
    cutoff_utc: str,
    selection: str = "home_three_way_plus_one_joint",
    league_id: int | None = None,
    embargo_hours: int = 72,
    min_training: int = 100,
    min_holdout: int = 30,
    laplace_alpha: float = 1.0,
) -> dict[str, Any]:
    """One untouched chronological holdout; never a trained production model.

    Independent: smoothed training P(goals)*P(corners)*P(+1 covers).
    Joint: smoothed P(all three in the SAME historic fixture).
    Report Brier/log-loss out-of-sample without promotion or implied EV.
    """
    cutoff = _dt(cutoff_utc)
    if (not isinstance(dataset, dict) or dataset.get("version") != DATASET_VERSION
            or dataset.get("market_inputs_used") is not False
            or dataset.get("source_provenance_required") is not True
            or dataset.get("calibrated_joint_probability") is not None
            or dataset.get("bet_eligible") is not False
            or selection not in SUPPORTED_MARKETS
            or cutoff is None or (league_id is not None and _id(league_id) is None)
            or type(embargo_hours) is not int or not 0 <= embargo_hours <= 720
            or type(min_training) is not int or min_training < 10
            or type(min_holdout) is not int or min_holdout < 5
            or type(laplace_alpha) not in (float, int) or not 0 < laplace_alpha <= 10):
        raise ValueError("FE11_OOS_CONFIGURATION_UNTRUSTED")
    research_cutoff = _dt(dataset.get("research_cutoff_utc"))
    if research_cutoff is None or cutoff >= research_cutoff:
        raise ValueError("FE11_CUTOFF_AFTER_RESEARCH_SNAPSHOT")
    lines = dataset.get("lines")
    if (not isinstance(lines, dict) or lines.get("handicap") != 1
            or type(lines.get("goals_over")) not in (float, int)
            or type(lines.get("corners_over")) not in (float, int)
            or lines["goals_over"] % 1 != .5 or lines["corners_over"] % 1 != .5):
        raise ValueError("FE11_LINES_UNVERIFIED")
    raw_rows = dataset.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("FE11_DATASET_ROWS_UNVERIFIED")
    rows, seen = [], set()
    for raw in raw_rows:
        r = _validated_row(raw, selection, lines["goals_over"],
                           lines["corners_over"], research_cutoff)
        if r["fixture_id"] in seen:
            raise ValueError("FE11_DUPLICATE_FIXTURE_PSEUDO_REPLICATION")
        seen.add(r["fixture_id"])
        if league_id is None or r["league_id"] == league_id:
            rows.append(r)
    train_end = cutoff - timedelta(hours=embargo_hours)
    train = [r for r in rows if r["kickoff"] < train_end
             and r["result_known"] < train_end and r["corners_known"] < train_end]
    holdout = [r for r in rows if r["kickoff"] >= cutoff]
    train.sort(key=lambda r: (r["kickoff"], r["fixture_id"]))
    holdout.sort(key=lambda r: (r["kickoff"], r["fixture_id"]))
    base = {
        "version": VERSION, "data_version": DATASET_VERSION,
        "selection": selection, "league_id": league_id,
        "goals_over": lines["goals_over"], "corners_over": lines["corners_over"],
        "handicap": "+1 THREE_WAY", "cutoff_utc": cutoff.isoformat(),
        "embargo_hours": embargo_hours,
        "n_training": len(train), "n_holdout": len(holdout),
        "market_inputs_used": False, "feature_adjusted": False,
        "frozen_fit": True, "out_of_sample_evaluation": True,
        "calibrated_joint_probability": None,
        "bookmaker_ticket_quote": None, "estimated_ev": None,
        "classification": "RESEARCH_ONLY", "bet_eligible": False,
        "production_promotion_allowed": False,
        "limitations": [
            "BASELINE_NOT_OPPONENT_ADJUSTED",
            "NO_PRODUCTION_CALIBRATION",
            "THREE_WAY_SETTLEMENT_ONLY",
            "DEPENDENT_TEAM_OR_SAME_DAY_SAMPLES_REQUIRE_CLUSTER_AWARE_INFERENCE",
            "NO_BOOKMAKER_SGP_QUOTE",
        ],
    }
    if len(train) < min_training or len(holdout) < min_holdout:
        return {
            **base, "status": "NOT_ENOUGH_OOS_DATA",
            "blockers": ["INSUFFICIENT_PUBLISHED_TRAINING_OR_HOLDOUT",
                         "NO_JOINT_MODEL_CALIBRATION"],
            "metrics": None,
        }
    n = len(train)
    alpha = float(laplace_alpha)
    marginals = {key: (sum(r[key] for r in train)+alpha)/(n+2*alpha)
                 for key in ("goals", "corners", "handicap")}
    independence = marginals["goals"] * marginals["corners"] * marginals["handicap"]
    joint = (sum(r["joint"] for r in train)+alpha)/(n+2*alpha)
    observed = [r["joint"] for r in holdout]
    m_ind, m_joint = _metrics(observed, independence), _metrics(observed, joint)
    return {
        **base, "status": "BASELINES_EVALUATED_RESEARCH_ONLY",
        "train_fixture_ids": [r["fixture_id"] for r in train],
        "holdout_fixture_ids": [r["fixture_id"] for r in holdout],
        "training_marginals_smoothed": {k: round(v, 7) for k, v in marginals.items()},
        "naive_independent_estimate": round(independence, 7),
        "same_fixture_joint_estimate": round(joint, 7),
        "training_dependence_difference_pp": round(100*(joint-independence), 4),
        "holdout_observed_joint_win_rate": round(sum(observed)/len(observed), 7),
        "holdout_observed_rate_wilson95": _wilson_95(sum(observed), len(observed)),
        "metrics": {
            "independent": m_ind, "same_fixture_joint": m_joint,
            "brier_joint_minus_independent": round(m_joint["brier"]-m_ind["brier"], 7),
            "log_loss_joint_minus_independent": round(m_joint["log_loss"]-m_ind["log_loss"], 7),
        },
        "blockers": ["NO_FEATURE_CONDITIONAL_MODEL",
                     "NO_PRODUCTION_CALIBRATION",
                     "NO_VERIFIED_BUILDER_SETTLEMENT_AND_PRICE"],
    }
