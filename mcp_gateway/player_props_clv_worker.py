from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp_gateway import player_props_clv_postgres_v4


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: player_props_clv_worker LOOKBACK_DAYS MAX_ROWS OUTPUT_PATH", file=sys.stderr)
        return 2
    try:
        lookback_days = int(sys.argv[1])
        max_rows = int(sys.argv[2])
    except ValueError:
        print("lookback_days and max_rows must be integers", file=sys.stderr)
        return 2

    output_path = Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    result = player_props_clv_postgres_v4.build_from_postgres(
        lookback_days=lookback_days,
        max_rows=max_rows,
    )
    temp_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str),
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)
    print(json.dumps({"status": "ok", "output_path": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
