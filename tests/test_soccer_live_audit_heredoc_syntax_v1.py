"""Fail closed on invalid inline Python inside the LIVE audit workflow."""
import ast
import re
import textwrap
from pathlib import Path


def test_live_slate_audit_embedded_python_compiles():
    text = Path(
        ".github/workflows/soccer-live-slate-research-join-readonly.yml"
    ).read_text(encoding="utf-8")
    scripts = re.findall(
        r"(?ms)^ {10,}(?:if )?python - <<'PY'\n(.*?)^ {10}PY$",
        text,
    )
    assert len(scripts) == 2, "Audit's 2 expected Python scripts must be found"
    for script in scripts:
        ast.parse(textwrap.dedent(script))
