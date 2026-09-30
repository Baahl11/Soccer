from __future__ import annotations

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_BILLING_V226_1.0.0"


def inject_billing(html: str) -> str:
    """Inject entitlement-safe Stripe controls into the subscriber shell.

    Browser code sends only a billing market enum. Stripe Price IDs remain
    authoritative inside the Supabase checkout Edge Function.
    """
    if "v226-billing-panel" in html:
        return html

    fragment = r'''
<style id="v226-billing-style">
#v226-billing-panel{position:fixed;right:18px;bottom:18px;z-index:30;width:min(360px,calc(100vw - 28px));background:#071923;border:1px solid #24506a;border-radius:14px;padding:14px;box-shadow:0 16px 50px #0009;color:#f3f8fb}
#v226-billing-panel h3{margin:0 0 5px;font-size:14px}#v226-billing-panel p{margin:0 0 10px;color:#8298a8;font-size:10px;line-height:1.45}
.v226-plans{display:grid;grid-template-columns:1fr 1fr;gap:7px}.v226-plan{border:1px solid #173448;background:#0a1822;color:#fff;border-radius:9px;padding:10px;text-align:left;cursor:pointer}.v226-plan:hover{border-color:#56e0ad}.v226-plan b,.v226-plan span{display:block}.v226-plan b{font-size:13px}.v226-plan span{color:#56e0ad;font-size:11px;margin-top:3px}
.v226-actions{display:flex;gap:7px;margin-top:8px}.v226-actions button{flex:1;border:1px solid #173448;border-radius:8px;background:#081923;color:#d8e5ed;padding:8px;cursor:pointer}.v226-actions button:hover{border-color:#64bfff}.v226-status{min-height:16px;margin-top:8px!important;color:#9eb4c3!important}
@media(max-width:800px){#v226-billing-panel{bottom:58px;right:10px}}
</style>
<section id="v226-billing-panel" aria-label="Soccer Edge Pro billing">
  <h3>EDGE PRO · FOUNDING BETA</h3>
  <p>Choose your billing market. The server selects the Stripe price; the browser cannot submit a Price ID.</p>
  <div class="v226-plans">
    <button class="v226-plan" data-billing-market="MX_LATAM"><b>México / LATAM</b><span>MX$249 / mes</span></button>
    <button class="v226-plan" data-billing-market="US"><b>United States</b><span>US$14.99 / month</span></button>
  </div>
  <div class="v226-actions"><button id="v226-portal" type="button">Manage subscription</button><button id="v226-hide" type="button">Hide</button></div>
  <p class="v226-status" id="v226-billing-status"></p>
</section>
<script id="v226-billing-script">
(()=>{
  const panel=document.getElementById('v226-billing-panel');
  if(!panel)return;
  const status=document.getElementById('v226-billing-status');
  const cfgEl=document.getElementById('cfg');
  let cfg={};try{cfg=JSON.parse(cfgEl?.textContent||'{}')}catch(_){cfg={}}
  const token=()=>localStorage.getItem('soccer_edge_access_token')||'';
  const headers=()=>({'Authorization':`Bearer ${token()}`,'apikey':cfg.publishable_key||'','Content-Type':'application/json'});
  const fn=(slug)=>`${String(cfg.supabase_url||'').replace(/\/$/,'')}/functions/v1/${slug}`;
  const setStatus=(m)=>{status.textContent=m||''};
  async function call(slug,body={}){
    if(!token()){setStatus('Sign in first / Inicia sesión primero.');return null}
    if(!cfg.supabase_url||!cfg.publishable_key){setStatus('Billing configuration unavailable.');return null}
    const r=await fetch(fn(slug),{method:'POST',headers:headers(),body:JSON.stringify(body)});
    let p={};try{p=await r.json()}catch(_){}
    if(!r.ok){setStatus(p.error==='BILLING_NOT_CONFIGURED'?'Billing activation pending secure Stripe keys.':(p.error||`Billing error ${r.status}`));return null}
    return p;
  }
  panel.querySelectorAll('[data-billing-market]').forEach(btn=>btn.addEventListener('click',async()=>{
    const market=btn.getAttribute('data-billing-market');
    setStatus('Opening secure Stripe Checkout…');
    const p=await call('create-checkout-session',{billing_market:market});
    if(p?.url&&/^https:\/\/checkout\.stripe\.com\//.test(p.url)){location.assign(p.url);return}
    if(p)setStatus('Checkout URL not returned.');
  }));
  document.getElementById('v226-portal')?.addEventListener('click',async()=>{
    setStatus('Opening Customer Portal…');
    const p=await call('create-customer-portal',{});
    if(p?.url&&/^https:\/\/billing\.stripe\.com\//.test(p.url)){location.assign(p.url);return}
    if(p)setStatus('Portal URL not returned.');
  });
  document.getElementById('v226-hide')?.addEventListener('click',()=>panel.remove());
})();
</script>
'''
    marker = "</body>"
    return html.replace(marker, fragment + marker, 1) if marker in html else html + fragment
