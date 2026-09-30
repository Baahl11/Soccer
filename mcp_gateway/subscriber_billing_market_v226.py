from __future__ import annotations

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_BILLING_MARKET_V226_1.0.0"

_MARKER = "SOCCER_V226_REGIONAL_BILLING"


def inject(html: str) -> str:
    """Add an explicit billing-market selector to the subscriber app.

    The browser only sends the enum US or MX_LATAM. Stripe Price IDs remain
    allowlisted server-side inside the Supabase Checkout Edge Function.
    No locale/IP/profile inference selects a billing market.
    """
    if not isinstance(html, str) or not html or _MARKER in html:
        return html

    fragment = r'''
<style id="SOCCER_V226_REGIONAL_BILLING">
#v226Billing{position:fixed;right:14px;bottom:14px;z-index:30;width:min(330px,calc(100vw - 28px));background:#07141d;border:1px solid #21445a;border-radius:12px;padding:12px;box-shadow:0 18px 50px #0008;font-family:Inter,system-ui,sans-serif}
#v226Billing .v226h{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px}#v226Billing .v226h b{font-size:12px}#v226Billing .v226tag{font-size:8px;color:#56e0ad;border:1px solid #285547;border-radius:999px;padding:4px 6px}
#v226Billing p{color:#8195a4;font-size:9px;line-height:1.45;margin:6px 0 9px}#v226Market{width:100%;background:#081923;color:#fff;border:1px solid #173448;border-radius:8px;padding:9px;margin-bottom:7px}
#v226Upgrade{width:100%;border:1px solid #2d7c69;background:#0c3a31;color:#d9fff1;border-radius:8px;padding:9px;font-weight:850;cursor:pointer}#v226Upgrade:disabled{opacity:.45;cursor:not-allowed}#v226BillingStatus{display:block;min-height:14px;color:#8195a4;font-size:9px;margin-top:7px}
#v226Billing.v226collapsed>*:not(.v226h){display:none}#v226Toggle{border:0;background:none;color:#8195a4;cursor:pointer;font-size:10px}
@media(max-width:800px){#v226Billing{bottom:62px}}
</style>
<div id="v226Billing" class="v226collapsed" data-schema="1.0.0">
  <div class="v226h"><div><b>Edge Pro · Founding Beta</b> <span class="v226tag">LIVE PRICING</span></div><button id="v226Toggle" type="button">Open</button></div>
  <p>Choose your billing market explicitly. Language and location are never used to choose a price.</p>
  <select id="v226Market" aria-label="Billing market">
    <option value="">Choose billing market</option>
    <option value="US">United States — US$14.99 / month</option>
    <option value="MX_LATAM">Mexico / LATAM — MX$249 / month</option>
  </select>
  <button id="v226Upgrade" type="button" disabled>Upgrade to Edge Pro</button>
  <span id="v226BillingStatus"></span>
</div>
<script>
(()=>{
  const panel=document.getElementById('v226Billing'),toggle=document.getElementById('v226Toggle'),market=document.getElementById('v226Market'),upgrade=document.getElementById('v226Upgrade'),status=document.getElementById('v226BillingStatus');
  if(!panel||!market||!upgrade)return;
  const MARKET_KEY='soccer_edge_billing_market', ACCESS_KEY='soccer_edge_access_token';
  const allowed=new Set(['US','MX_LATAM']);
  const prior=localStorage.getItem(MARKET_KEY)||''; if(allowed.has(prior))market.value=prior;
  const sync=()=>{upgrade.disabled=!allowed.has(market.value);}; sync();
  toggle?.addEventListener('click',()=>{panel.classList.toggle('v226collapsed');toggle.textContent=panel.classList.contains('v226collapsed')?'Open':'Close';});
  market.addEventListener('change',()=>{if(allowed.has(market.value))localStorage.setItem(MARKET_KEY,market.value);else localStorage.removeItem(MARKET_KEY);sync();});
  upgrade.addEventListener('click',async()=>{
    const billing_market=market.value; if(!allowed.has(billing_market))return;
    const token=localStorage.getItem(ACCESS_KEY); if(!token){status.textContent='Sign in before starting Checkout.';return;}
    let cfg={}; try{cfg=JSON.parse(document.getElementById('cfg')?.textContent||'{}')}catch(_){ }
    if(!cfg.supabase_url||!cfg.publishable_key){status.textContent='Billing configuration unavailable.';return;}
    upgrade.disabled=true; status.textContent='Opening secure Stripe Checkout…';
    try{
      const q=new URLSearchParams(location.search);
      const analytics={locale:(document.documentElement.lang||'').slice(0,2),utm_source:q.get('utm_source')||'',utm_medium:q.get('utm_medium')||'',utm_campaign:q.get('utm_campaign')||'',utm_content:q.get('utm_content')||''};
      const r=await fetch(`${cfg.supabase_url}/functions/v1/create-checkout-session`,{method:'POST',headers:{Authorization:`Bearer ${token}`,apikey:cfg.publishable_key,'Content-Type':'application/json'},body:JSON.stringify({billing_market,analytics})});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(d.error||'CHECKOUT_FAILED');
      if(!d.url)throw new Error('CHECKOUT_URL_MISSING');
      location.href=d.url;
    }catch(e){status.textContent=String(e?.message||e);upgrade.disabled=false;}
  });
})();
</script>
'''
    marker = "</body>"
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment


def contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "billing_markets": {
            "US": {"display_price": "US$14.99 / month", "currency": "usd"},
            "MX_LATAM": {"display_price": "MX$249 / month", "currency": "mxn"},
        },
        "billing_market_selection": "EXPLICIT_USER_CHOICE",
        "browser_sends_price_id": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
