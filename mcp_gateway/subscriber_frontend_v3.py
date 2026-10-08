from __future__ import annotations

import gzip
from pathlib import Path

_PAYLOAD_PATH = Path(__file__).with_name("subscriber_frontend_v3.py.gz")
_SOURCE = gzip.decompress(_PAYLOAD_PATH.read_bytes()).decode("utf-8")
exec(compile(_SOURCE, str(Path(__file__).with_suffix(".source.py")), "exec"), globals(), globals())
