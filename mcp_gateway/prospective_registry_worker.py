"""Run the prospective evaluator outside the persistent API process.

The evaluation method and all point-in-time selection criteria are unchanged.
A cgroup monitor in server.py can abort this child without OOM-killing Uvicorn.
"""
from __future__ import annotations

import json
import sys

from mcp_gateway.prospective_evaluation_registry_v1 import build_registry


def main() -> None:
    result = build_registry()
    # stdout is reserved for a single machine-readable report. If the worker
    # raises, the parent sees non-zero exit and never marks the run persisted.
    sys.stdout.write(json.dumps(result, separators=(",", ":"), allow_nan=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
