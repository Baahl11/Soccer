"""Read-only, multi-market validation registry from canonical Soccer research reports.

Model research status != calibrated, promotable, actionable wagering signal.
Source files are maintained by the pre-existing Soccer engine/state branch.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Market-specific evidence files in existing soccer-edge-state branch.
# Never transfer one family's calibration/status to a different market.
FAMILY_FILES = (
    ('1X2', 'Victoria/empate/derrota', 'v4_017_1x2_calibration_validation.json'),
    ('FT_TOTALS', 'Goles totales', 'v4_016_ft_totals_production_validation.json'),
    ('BTTS', 'Ambos anotan', 'v4_018_btts_calibration_validation.json'),
    ('TEAM_TOTALS', 'Goles por equipo', 'v4_019_team_totals_oos_validation.json'),
    ('FIRST_HALF', 'Primera mitad', 'v4_020_1h_oos_validation.json'),
    ('SECOND_HALF', 'Segunda mitad', 'v4_021_2h_oos_validation.json'),
    ('FT_CORNERS', 'Córners totales', 'v4_022_corners_oos_validation.json'),
    ('TEAM_CORNERS', 'Córners por equipo', 'v4_022_corners_oos_validation.json'),
    ('CARDS', 'Tarjetas / árbitro', 'phase14_cards_referee_validation.json'),
    ('PLAYER_PROPS', 'Tiros, asistencias y props', 'phase15_player_props_validation.json'),
    ('FORMATION_STYLE', 'Formación vs formación y estilos', 'formation_matchup_fm4_style_ablation_v1.json'),
    ('FOULS', 'Faltas cometidas', None),
)


def load_market_registry(source_dir: str | Path) -> dict[str, Any]:
    root = Path(source_dir)
    cache: dict[str, dict[str, Any]] = {}
    families = []
    for family, label, basename in FAMILY_FILES:
        report: dict[str, Any] = {}
        read_error = None
        if basename is not None:
            if basename not in cache:
                f = root / basename
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    cache[basename] = data if isinstance(data, dict) else {}
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    cache[basename] = {}
                    read_error = type(exc).__name__
            report = cache[basename]
        blockers = report.get('blockers')
        if not isinstance(blockers, list):
            health = report.get('health') or {}
            blockers = health.get('blockers') if isinstance(health, dict) else None
        status = report.get('status') if report else 'NOT_VERIFIED'
        if basename is None:
            blockers = ['NO_DEDICATED_OOS_VALIDATED_FOULS_MODEL_AND_PRICE_LINEAGE']
        elif not report:
            blockers = ['CANONICAL_MARKET_VALIDATION_REPORT_UNAVAILABLE']
        status = str(status or 'NOT_VERIFIED')
        families.append({
            'family': family, 'label': label, 'research_status': status,
            'model_version': report.get('model_version') if report else None,
            'evidence_file': basename,
            'blockers': [str(b) for b in (blockers or [])][:25],
            'production_promotion_allowed': False,  # Never inferred from research eligibility.
            'bet_eligible': False,
            'exact_bookmaker_price': 'NOT_VERIFIED',
            'data_class': 'RESEARCH_ONLY',
            'report_loaded': bool(report),
            'report_timestamp': report.get('generated_at_utc') or report.get('generated_at') or None,
        })
    return {
        'source': 'EXISTING_SOCCER_EDGE_STATE_ANALYSIS_REPORTS',
        'read_only': True, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'RESEARCH_VALIDATION_NOT_MATCH_SPECIFIC_PREDICTIONS',
        'families': families,
        'bet_eligible_families': 0,
        'source_freshness_verified': False,  # historical reports often lack captured timestamps
        'note': 'Original report status and blockers preserved; missing reports remain NOT_VERIFIED.',
    }