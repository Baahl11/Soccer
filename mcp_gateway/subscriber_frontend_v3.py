from __future__ import annotations

import gzip
from pathlib import Path

_PAYLOAD_PATH = Path(__file__).with_name("subscriber_frontend_v3.py.gz")
_SOURCE = gzip.decompress(_PAYLOAD_PATH.read_bytes()).decode("utf-8")
exec(compile(_SOURCE, str(Path(__file__).with_suffix(".source.py")), "exec"), globals(), globals())


# Compatibility fix: /app/api/v2/today wraps every full-slate row as
# {"fixture": {...}, "state": {...}, "coverage": {...}}.  The first V3 preview
# treated fixture identity as top-level, which rendered TBD/F placeholders and
# generated /match/undefined links.  Normalize that persisted contract shape in
# the V3 presentation layer only; model/market logic is untouched.
_v3_source_html = _html

def _html() -> str:
    rendered = _v3_source_html()
    needle = "function MatchRow({r}){const k=kick(r.kickoff);"
    replacement = "function MatchRow({r}){r={...r,...(r?.fixture||{})};const k=kick(r.kickoff);"
    if needle not in rendered:
        raise RuntimeError("V3 MatchRow contract-shape patch target not found")
    return rendered.replace(needle, replacement, 1)
