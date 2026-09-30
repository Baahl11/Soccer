from __future__ import annotations


def install(subscriber_app) -> None:
    """Presentation-only subscriber UI hotfix.

    Missing numeric evidence must render as N/V, never as a real zero. Fixture labels
    resolve all persisted team-name shapes before falling back to fixture_id.
    Canonical model, gates, thresholds and provider behavior remain untouched.
    """
    if hasattr(subscriber_app, "_v226_before_null_fixture_hotfix"):
        return

    base = subscriber_app._app_html
    subscriber_app._v226_before_null_fixture_hotfix = base

    def patched_html() -> str:
        html = base()
        # IMPORTANT: base() has already evaluated the Python f-string, therefore the
        # rendered JavaScript contains single braces. The previous hotfix searched
        # for doubled braces and consequently never matched production HTML.
        old = "V=(r,...k)=>{for(const x of k)if(r?.[x]!=null&&r[x]!=='')return r[x];return null},teams=r=>[V(r,'home_team','home'),V(r,'away_team','away')].filter(Boolean).join(' vs ')||V(r,'fixture_id')||'Fixture',market=r=>V(r,'market','selection','market_family')||'Market',status=r=>V(r,'execution_status','status','stage','classification')||'WATCH',pct=v=>{let n=Number(v);return Number.isFinite(n)?((Math.abs(n)<=1?n*100:n).toFixed(1)+'%'):'—'},edge=r=>{let n=Number(V(r,'edge_pp','edge','edge_pct','model_edge'));return Number.isFinite(n)?((Math.abs(n)<=1?n*100:n).toFixed(1)+' pp'):'—'};"
        new = "V=(r,...k)=>{for(const x of k)if(r?.[x]!=null&&r[x]!=='')return r[x];return null},N=(r,side)=>{let direct=V(r,side+'_team',side+'_team_name',side+'_name');if(direct)return direct;let x=r?.[side];if(typeof x==='string')return x;if(x&&typeof x==='object')return x.name||x.team_name||null;let t=r?.teams?.[side];if(typeof t==='string')return t;if(t&&typeof t==='object')return t.name||t.team_name||null;return null},teams=r=>[N(r,'home'),N(r,'away')].filter(Boolean).join(' vs ')||V(r,'fixture_name','match_name')||('Fixture '+(V(r,'fixture_id')||'N/V')),market=r=>V(r,'market','selection','market_family')||'Market',status=r=>V(r,'execution_status','status','stage','classification')||'WATCH',num=v=>{if(v==null||v==='')return null;let n=Number(v);return Number.isFinite(n)?n:null},pct=v=>{let n=num(v);return n==null?'N/V':((Math.abs(n)<=1?n*100:n).toFixed(1)+'%')},edge=r=>{let n=num(V(r,'edge_pp','edge','edge_pct','model_edge'));return n==null?'N/V':((Math.abs(n)<=1?n*100:n).toFixed(1)+' pp')},comparable=r=>num(V(r,'p_model_calibrated','model_probability','probability'))!=null&&num(V(r,'p_market_fair','p_market_devig','market_probability'))!=null&&num(V(r,'edge_pp','edge','edge_pct','model_edge'))!=null;"
        if old not in html:
            # Fail visibly rather than silently serving deceptive zeros again.
            html = html.replace("</body>", "<script>console.error('SOCCER_EDGE_UI_HOTFIX_PATTERN_MISS')</script></body>")
            return html
        html = html.replace(old, new, 1)
        html = html.replace(
            "let top=R(strong)[0]||R(value)[0]||R(slate)[0];",
            "let top=R(strong).find(comparable)||R(value).find(comparable)||R(slate).find(comparable)||null;",
            1,
        )
        return html

    subscriber_app._app_html = patched_html
