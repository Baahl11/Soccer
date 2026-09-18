from __future__ import annotations

import os
from datetime import datetime, timezone as dt_timezone
from typing import Any

DEFAULT_MAX_AGE_MINUTES = max(
    5, int(os.getenv("GALAXYPARLAY_ODDS_MAX_AGE_MINUTES", "20"))
)
MAX_FUTURE_SKEW_MINUTES = 5.0


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        out = value
    else:
        try:
            out = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out.astimezone(dt_timezone.utc)


def age_minutes(value: Any, now: datetime | None = None) -> float | None:
    stamp = parse_dt(value)
    if stamp is None:
        return None
    current = now or datetime.now(dt_timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt_timezone.utc)
    current = current.astimezone(dt_timezone.utc)
    return (current - stamp).total_seconds() / 60.0


def is_fresh(
    value: Any,
    now: datetime | None = None,
    max_age_minutes: float | None = None,
) -> bool:
    age = age_minutes(value, now)
    if age is None:
        return False
    limit = float(max_age_minutes or DEFAULT_MAX_AGE_MINUTES)
    return -MAX_FUTURE_SKEW_MINUTES <= age <= limit


def latest_timestamp(values: list[Any]) -> datetime | None:
    stamps = [parse_dt(value) for value in values]
    valid = [stamp for stamp in stamps if stamp is not None]
    return max(valid) if valid else None


def fresh_quotes(
    quotes: Any,
    now: datetime | None = None,
    max_age_minutes: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        if not is_fresh(quote.get("provider_update"), now, max_age_minutes):
            continue
        row = dict(quote)
        age = age_minutes(row.get("provider_update"), now)
        row["quote_age_minutes"] = round(age, 2) if age is not None else None
        row["quote_fresh"] = True
        row["quote_freshness_anchor"] = "PROVIDER_UPDATE"
        out.append(row)
    return out


def candidate_price_fresh(
    candidate: dict[str, Any],
    now: datetime,
    max_age_minutes: float | None = None,
) -> bool:
    if not isinstance(candidate, dict):
        return False

    # SGP references expose the exact same-book component quotes used to screen
    # the combination. Every referenced component must still be fresh.
    reference = candidate.get("component_price_reference")
    if isinstance(reference, dict):
        components = [
            item for item in (reference.get("components") or [])
            if isinstance(item, dict)
        ]
        if components:
            return all(
                is_fresh(item.get("provider_update"), now, max_age_minutes)
                for item in components
            )

    legs = [leg for leg in (candidate.get("legs") or []) if isinstance(leg, dict)]
    if not legs:
        return False

    # Rolling multis expose one selected same-book provider_update per public leg.
    if all(leg.get("provider_update") for leg in legs):
        return all(
            is_fresh(leg.get("provider_update"), now, max_age_minutes)
            for leg in legs
        )

    # Older candidate shapes can carry a quote list per leg. Require at least one
    # still-fresh quote for every leg; missing timestamps fail closed.
    for leg in legs:
        quotes = fresh_quotes(leg.get("quotes"), now, max_age_minutes)
        if not quotes:
            return False
    return True
