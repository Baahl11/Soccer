"""Offline checks: replay may only happen manually with verified OIDC provenance."""
from __future__ import annotations

import ast
import re
import subprocess
import textwrap
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/fe11-replay-latest-history.yml"


class Fe11ReplayPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_never_runs_automatically_or_from_pull_request(self):
        heading = self.workflow.split("\npermissions:\n", 1)[0]
        self.assertIn("  workflow_dispatch:", heading)
        self.assertNotIn("  push:", heading)
        self.assertNotIn("  pull_request:", heading)
        self.assertNotIn("  schedule:", heading)

    def test_hardcoded_original_provenance_and_fail_closed(self):
        source = self.workflow
        for required in (
            "SRC_RUN=42329502551",
            "SRC_SHA=fd9f3995e3c1ca1666176d8110045a79211f69fb",
            'REAL_SHA=$(gh api',
            '"$REAL_SHA" = "$SRC_SHA"',
            "gh run download",
            "soccer-paper-research",
            "SOURCE_ARTIFACT_MISSING_OR_AMBIGUOUS",
            "ORIGINAL_RESEARCH_SOURCE_TOO_OLD",
            "timedelta(hours=48)",
            'x.get("source")=="SPORTS_EDGE_SCANNER_V15_HISTORICAL_CONTEXT"',
            "assert x.get(\"bet_eligible\") is False",
            "assert x.get(\"calibrated_joint_probability\") is None",
        ):
            self.assertIn(required, source, required)

    def test_oidc_claim_preflight_and_production_authority(self):
        source = self.workflow
        for required in (
            "id-token: write",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
            "audience=soccer-edge-render",
            'echo "::add-mask::$TOKEN"',
            'claims.get("repository")=="Baahl11/Soccer"',
            'claims.get("ref")=="refs/heads/main"',
            'claims.get("workflow_ref","")',
            'claims.get("iss")=="https://token.actions.githubusercontent.com"',
            'claims.get("aud")=="soccer-edge-render"',
            "FE11_OIDC_LOCAL_CLAIMS_MATCH_EXPECTED_BACKEND_ALLOWLIST",
            "FE11_SERVER_REJECTED_VERIFIED_LOCAL_CLAIMS",
            "https://soccer-edge-api.onrender.com/internal/fe11-multimarket-history-import-v1",
        ):
            self.assertIn(required, source, required)
        self.assertIn('unset FE11_OIDC_JWT', source)
        self.assertNotIn("print(claims)", source)
        self.assertNotIn("print(token)", source)
        self.assertNotIn("API_FOOTBALL_KEY", source)
        executable = "\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("#"))\n        self.assertNotIn("/odds", executable)

    def test_embedded_shell_and_python_are_syntactically_valid(self):
        # Verify exact run scripts with bash -n: never execute remote calls.
        scripts = re.findall(r"(?m)^        run: \|\n((?:          .*\n|\n)+)", self.workflow)
        self.assertEqual(len(scripts), 2)
        for script in scripts:
            script = textwrap.dedent(script)
            p = subprocess.run(
                ["bash", "-n"], input=script, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
        scripts_python = re.findall(
            r"(?m)^          python - <<'PY'\n(.*?)^          PY$",
            self.workflow, re.DOTALL | re.MULTILINE)
        self.assertEqual(len(scripts_python), 2)
        for script in scripts_python:
            ast.parse(textwrap.dedent(script))


if __name__ == "__main__":
    unittest.main()
