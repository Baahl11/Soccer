from __future__ import annotations

import gzip
from pathlib import Path

_PAYLOAD_PATH = Path(__file__).with_name("subscriber_frontend_v3.py.gz")
_SOURCE = gzip.decompress(_PAYLOAD_PATH.read_bytes()).decode("utf-8")
exec(compile(_SOURCE, str(Path(__file__).with_suffix(".source.py")), "exec"), globals(), globals())

# V3 presentation compatibility shim.
# The V2 Today contract wraps registry identity under row.fixture while the
# transitional V3 preview expects a flattened row. Normalize only the browser
# payload before React consumes it. No model, market, provider, or persistence
# logic is changed.
_v3_source_html = _html

def _html() -> str:
    rendered = _v3_source_html()
    shim = r"""
<script>
(() => {
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const response = await originalFetch(input, init);
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (response.ok && url.includes('/app/api/v2/today')) {
      try {
        const payload = await response.clone().json();
        if (payload && payload.slate && Array.isArray(payload.slate.rows)) {
          payload.slate.rows = payload.slate.rows.map(row =>
            row && row.fixture ? Object.assign({}, row, row.fixture) : row
          );
          return new Response(JSON.stringify(payload), {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
          });
        }
      } catch (_) {}
    }
    return response;
  };
})();
</script>
"""
    marker = "</head>"
    if marker in rendered:
        return rendered.replace(marker, shim + marker, 1)
    return shim + rendered
