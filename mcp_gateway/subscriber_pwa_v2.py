from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PWA_V2_1.0.0"
CACHE_NAME = "soccer-edge-shell-v3"


def manifest_payload() -> dict[str, Any]:
    return {
        "name": "Soccer Edge",
        "short_name": "Soccer Edge",
        "description": "Sport-first soccer intelligence with transparent BET, LEAN and WATCH states.",
        "id": "/app",
        "start_url": "/app",
        "scope": "/",
        "display": "standalone",
        "background_color": "#050b11",
        "theme_color": "#06111a",
        "orientation": "any",
        "categories": ["sports", "finance"],
        "icons": [
            {
                "src": "/pwa/icon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any maskable",
            }
        ],
    }


def service_worker_script() -> str:
    return f"""const CACHE_NAME={json.dumps(CACHE_NAME)};
const SHELL_URL='/app';
const NETWORK_ONLY_PREFIXES=['/app/api/v2/','/app/fixture-identities','/functions/'];

self.addEventListener('install',event=>{{
  event.waitUntil((async()=>{{
    const cache=await caches.open(CACHE_NAME);
    try{{const response=await fetch(SHELL_URL,{{cache:'no-store'}});if(response.ok)await cache.put(SHELL_URL,response.clone())}}catch(_e){{}}
    self.skipWaiting();
  }})());
}});

self.addEventListener('activate',event=>{{
  event.waitUntil((async()=>{{
    const keys=await caches.keys();
    await Promise.all(keys.filter(key=>key!==CACHE_NAME).map(key=>caches.delete(key)));
    await self.clients.claim();
  }})());
}});

self.addEventListener('fetch',event=>{{
  const request=event.request;
  if(request.method!=='GET')return;
  const url=new URL(request.url);
  if(url.origin!==self.location.origin)return;

  if(NETWORK_ONLY_PREFIXES.some(prefix=>url.pathname.startsWith(prefix))){{
    event.respondWith(fetch(request,{{cache:'no-store'}}));
    return;
  }}

  if(request.mode==='navigate' && (url.pathname==='/app' || url.pathname==='/app/' || url.pathname.startsWith('/app/match/') || url.pathname==='/app-v2' || url.pathname==='/app-v2/')){{
    event.respondWith((async()=>{{
      try{{
        const response=await fetch(request,{{cache:'no-store'}});
        if(response.ok){{const cache=await caches.open(CACHE_NAME);await cache.put(SHELL_URL,response.clone())}}
        return response;
      }}catch(_e){{
        const cached=await caches.match(SHELL_URL);
        return cached || new Response(
          '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Soccer Edge offline</title><body style="background:#050b11;color:#eff7fb;font-family:system-ui;padding:32px"><h1>Soccer Edge</h1><p>The live snapshot is unavailable offline. Reconnect to load current picks, prices and verification state.</p></body>',
          {{headers:{{'Content-Type':'text/html; charset=utf-8'}}}}
        );
      }}
    }})());
  }}
}});

// Push plumbing is intentionally dormant until FE-9 notification subscriptions,
// consent, VAPID keys and server delivery are separately approved.
self.addEventListener('push',event=>{{
  if(!event.data)return;
  let payload={{}};
  try{{payload=event.data.json()}}catch(_e){{payload={{body:event.data.text()}}}}
  if(payload.enabled!==true)return;
  const title=String(payload.title||'Soccer Edge');
  const body=String(payload.body||'').slice(0,240);
  const url=String(payload.url||'/app');
  event.waitUntil(self.registration.showNotification(title,{{
    body,
    icon:'/pwa/icon.svg',
    badge:'/pwa/icon.svg',
    data:{{url}},
    tag:String(payload.tag||'soccer-edge'),
    renotify:false
  }}));
}});

self.addEventListener('notificationclick',event=>{{
  event.notification.close();
  const url=event.notification?.data?.url||'/app';
  event.waitUntil(clients.matchAll({{type:'window',includeUncontrolled:true}}).then(list=>{{
    for(const client of list){{if('focus' in client){{client.navigate(url);return client.focus()}}}}
    return clients.openWindow?clients.openWindow(url):undefined;
  }}));
}});
"""


def icon_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<rect width="512" height="512" rx="112" fill="#06111a"/>
<rect x="38" y="38" width="436" height="436" rx="94" fill="#0a2a3f" stroke="#23516b" stroke-width="12"/>
<text x="256" y="300" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="168" font-weight="900" fill="#64c1ff">SE</text>
<circle cx="401" cy="111" r="28" fill="#4de1ad"/>
</svg>"""


async def manifest(request: Request) -> JSONResponse:
    return JSONResponse(
        manifest_payload(),
        headers={
            "Cache-Control": "public, max-age=300",
            "Content-Type": "application/manifest+json",
        },
    )


async def service_worker(request: Request) -> Response:
    return Response(
        service_worker_script(),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Service-Worker-Allowed": "/",
        },
    )


async def icon(request: Request) -> Response:
    return Response(
        icon_svg(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "manifest_route": "/app.webmanifest",
        "service_worker_route": "/sw.js",
        "icon_route": "/pwa/icon.svg",
        "start_url": "/app",
        "api_cache_policy": "NETWORK_ONLY",
        "live_price_cache_allowed": False,
        "live_decision_cache_allowed": False,
        "push_subscription_enabled": False,
        "outbound_push_delivery_enabled": False,
        "push_requires_explicit_user_consent": True,
        "model_input_allowed": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
