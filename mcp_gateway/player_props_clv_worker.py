import json
import sys

from mcp_gateway import player_props_clv_postgres_v4


def _bounded_int(raw: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def main() -> int:
    try:
        lookback_days = _bounded_int(
            sys.argv[1] if len(sys.argv) > 1 else "180",
            default=180,
            minimum=1,
            maximum=730,
        )
        max_rows = _bounded_int(
            sys.argv[2] if len(sys.argv) > 2 else "50000",
            default=50000,
            minimum=100,
            maximum=200000,
        )
        result = player_props_clv_postgres_v4.build_from_postgres(
            lookback_days=lookback_days,
            max_rows=max_rows,
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stderr.write(f"player_props_clv_worker_failed: {str(exc)[:1000]}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
