"""Owner-only coverage preview; never writes forecasts, quotes or OOS records."""
import asyncio
import math
import time
from datetime import datetime, timezone
from html import escape
from zoneinfo import ZoneInfo

from starlette.responses import HTMLResponse, JSONResponse

from mcp_gateway import subscription_entitlements_v4, supabase_auth_v4

ZONE = ZoneInfo("America/Mexico_City")
FAMILIES = {
    "1X2": {"match winner"}, "Goles FT": {"goals over/under"},
    "BTTS": {"both teams score"}, "Doble oportunidad": {"double chance"},
    "Hándicap": {"asian handicap", "handicap result"},
    "Totales por equipo": {"total - home", "total - away", "home team total goals", "away team total goals"},
    "Marcador": {"exact score", "correct score"},
    "1T": set(), "2T": set(), "Córners": set(), "Córners por equipo": set(),
    "Tarjetas": set(), "Tiros jugador": set(), "Tiros a puerta jugador": set(),
}
PREMATCH = {"NS"}
_CACHE = {}
_LOCK = asyncio.Lock()


def goal_model(home, away):
    def split(row, side):
        return tuple(int(row[k][side]) for k in ("played", "gf", "ga"))
    try:
        n, gf, ga = split(home, "home")
        m, af, aa = split(away, "away")
    except (KeyError, TypeError, ValueError):
        return None
    if min(n, m) < 5 or min(gf, ga, af, aa) < 0:
        return None
    h, a = (gf + aa) / (n + m), (af + ga) / (n + m)
    if not (0 <= h <= 10 and 0 <= a <= 10):
        return None
    def pmf(lam):
        p = [math.exp(-lam)]
        for i in range(1, 61):
            p.append(p[-1] * lam / i)
        return p
    cells = [(i, j, p*q) for i,p in enumerate(pmf(h)) for j,q in enumerate(pmf(a))]
    def prob(fn):
        return round(sum(p for i,j,p in cells if fn(i,j)), 6)
    def total(side, line):
        return prob(lambda i,j: (i, j)[side] > line)
    def handicap(line):
        return {"win": prob(lambda i,j: i-j+line > 0),
                "push": prob(lambda i,j: i-j+line == 0),
                "loss": prob(lambda i,j: i-j+line < 0)}
    return {"lambda_home": h, "lambda_away": a, "input_matches": [n,m],
            "probabilities": {
                "1X2": {"home": prob(lambda i,j:i>j), "draw": prob(lambda i,j:i==j), "away": prob(lambda i,j:i<j)},
                "Goles FT": {str(x): prob(lambda i,j:i+j>x) for x in (0.5,1.5,2.5,3.5,4.5)},
                "BTTS": {"yes": prob(lambda i,j:i>0 and j>0)},
                "Doble oportunidad": {"1X": prob(lambda i,j:i>=j), "12": prob(lambda i,j:i!=j), "X2": prob(lambda i,j:i<=j)},
                "Hándicap": {str(x): handicap(x) for x in (-1.5,-1,-0.5,0,0.5,1,1.5)},
                "Totales por equipo": {side: {str(x):total(idx,x) for x in (0.5,1.5,2.5)} for idx,side in enumerate(("home","away"))},
                "Marcador": [{"score":f"{i}-{j}", "p":round(p,6)} for i,j,p in sorted(cells,key=lambda c:c[2],reverse=True)[:5]]
            }}


def index_slate(items, compact):
    out = {}
    for item in items:
        f = compact(item)
        key = f.get("fixture_id")
        if type(key) is not int or key <= 0:
            raise ValueError("INVALID_FIXTURE_ID")
        if key in out and out[key] != f:
            raise ValueError("CONFLICTING_FIXTURE_ID")
        out[key] = f
    return out


def stats(raw):
    d = raw["response"]
    return {"played": d["fixtures"]["played"], "gf": d["goals"]["for"]["total"], "ga": d["goals"]["against"]["total"]}


def quote_inventory(books):
    found = {k: [] for k in FAMILIES}
    for snapshot in books:
        for book in snapshot.get("bookmakers", []):
            for bet in book.get("bets", []):
                name = str(bet.get("name") or "").casefold()
                hits = [k for k,v in FAMILIES.items() if name in v]
                if not hits:
                    if "corner" in name: hits = ["Córners por equipo" if any(s in name for s in ("team","home","away")) else "Córners"]
                    elif "card" in name or "booking" in name: hits = ["Tarjetas"]
                    elif "shot" in name and "player" in name: hits = ["Tiros a puerta jugador" if "target" in name else "Tiros jugador"]
                    elif any(s in name for s in ("first half","1st half","1st half")): hits = ["1T"]
                    elif any(s in name for s in ("second half","2nd half")): hits = ["2T"]
                for family in hits:
                    found[family].append({"book":book.get("name"), "market":bet.get("name"),
                                          "source_updated_at":snapshot.get("update"), "prices":bet.get("values", [])})
    return found


def json_reply(payload, status=200):
    return JSONResponse(payload, status_code=status, headers={"Cache-Control":"private, no-store"})


async def handle(request, api, fixture_id=None):
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return json_reply({"error":"AUTH_REQUIRED"},401)
    ent = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not ent.get("ok") or not ent.get("authenticated"):
        return json_reply({"error":"AUTH_REQUIRED"},401)
    if not ent.get("owner"):
        return json_reply({"error":"OWNER_ONLY_PREVIEW"},403)
    day = datetime.now(ZONE).date().isoformat()
    async with _LOCK:
        try:
            if _CACHE.get("day") != day or time.monotonic() - _CACHE.get("at",0) > 180:
                raw = await api._get("fixtures", {"date":day,"timezone":str(ZONE)})
                if (raw.get("paging") or {}).get("total",1) != 1 or raw.get("results") != len(raw.get("response",[])):
                    return json_reply({"error":"PROVIDER_SLATE_INCOMPLETE"},503)
                indexed = index_slate(raw["response"], api._compact_fixture)
                old = _CACHE.get("cards",{}) if _CACHE.get("day") == day else {}
                _CACHE.update(day=day, at=time.monotonic(), slate=indexed, cards=old,
                              source_time=raw["retrieved_at_utc"], quota=raw.get("quota") or {})
            slate, cards = _CACHE["slate"], _CACHE["cards"]
            if fixture_id is None:
                return json_reply({"date":day,"source":"API-Football v3","retrieved_at":_CACHE["source_time"],
                    "counts":{"fixtures":len(slate),
                              "prematch":sum(f.get("status") in PREMATCH for f in slate.values()),
                              "researched":len(cards), "projections":sum(bool(c.get("model")) for _,c in cards.values()),
                              "provider_quote_fixtures":sum(any(c.get("quotes",{}).values()) for _,c in cards.values()),
                              "oos":None},
                    "fixtures":[{**f,"coverage_status": "RESEARCH_CAPTURED" if k in cards else
                        "PENDING_RESEARCH" if f.get("status") in PREMATCH else "NOT_PREMATCH"}
                        for k,f in slate.items()]})
            f = slate.get(fixture_id)
            if not f:
                return json_reply({"error":"FIXTURE_NOT_IN_TODAY_SLATE"},404)
            if fixture_id in cards:
                at, card = cards[fixture_id]
                if time.monotonic()-at < 300 or f["status"] not in PREMATCH:
                    return json_reply({**card,"fixture":f,"cached_capture":True})
            if f["status"] not in PREMATCH or datetime.now(timezone.utc).timestamp() >= f["timestamp"]:
                return json_reply({"fixture":f,"state":"NO_NEW_PREMATCH_PROJECTION","model":None,
                                   "families":list(FAMILIES),"bet_eligible":False})
            try:
                if int(_CACHE["quota"].get("daily_remaining") or 0) <= 1505:
                    return json_reply({"fixture":f,"state":"QUOTA_RESERVE","bet_eligible":False},503)
            except ValueError:
                return json_reply({"fixture":f,"state":"QUOTA_UNKNOWN","bet_eligible":False},503)
            model, inputs, quotes, notes = None, None, {}, []
            async def read(endpoint, args):
                result = await api._get(endpoint,args)
                _CACHE["quota"] = result.get("quota") or {}
                return result
            try:
                sides = []
                for team in (f["home_team_id"],f["away_team_id"]):
                    raw = await read("teams/statistics",{"team":team,"league":f["league_id"],"season":f["season"]})
                    sides.append(stats(raw))
                inputs = sides
                model = goal_model(*sides)
            except (api.APIFootballError, TypeError, KeyError, ValueError):
                notes.append("TEAM_STATS_NOT_VERIFIED")
            model_at = datetime.now(timezone.utc)
            if model_at.timestamp() >= f["timestamp"]:
                return json_reply({"fixture":f,"state":"KICKOFF_PASSED_DURING_CAPTURE","model":None,"bet_eligible":False})
            # Sporting model is finalized before odds access. Quote timestamp/freshness not certified.
            odds_meta = {}
            try:
                odds = await read("odds",{"fixture":fixture_id,"page":1})
                quotes = quote_inventory(odds["response"])
                odds_meta = {"retrieved_at":odds["retrieved_at_utc"],"paging":odds.get("paging"),"actionable":False}
                if (odds.get("paging") or {}).get("total",1) > 1:
                    notes.append("ODDS_PAGE_ONE_ONLY")
            except (api.APIFootballError, TypeError, KeyError, ValueError):
                notes.append("ODDS_NOT_VERIFIED")
            try:
                lineups = await read("fixtures/lineups",{"fixture":fixture_id})
                expected = {f["home_team_id"],f["away_team_id"]}
                complete = {row.get("team",{}).get("id") for row in lineups["response"]
                            if len(row.get("startXI") or []) == 11}
                xi = ("BOTH_XI_REPORTED" if complete == expected and
                      datetime.now(timezone.utc).timestamp() < f["timestamp"] else "NOT_VERIFIED")
            except (api.APIFootballError, TypeError, KeyError, ValueError):
                xi = "NOT_VERIFIED"
            card = {"fixture":f,"state":"RESEARCH_ONLY","model":model,"model_captured_at":model_at.isoformat(),
                    "model_version":"VENUE_GOALS_POISSON_RESEARCH_V1","model_calibrated":False,
                    "inputs":inputs,"quotes":quotes,"odds":odds_meta,"xi":xi,"notes":notes,
                    "families":{k:{"projection":"PROVISIONAL" if k in (model or {}).get("probabilities",{}) else "NOT_VERIFIED",
                                    "prices":"PROVIDER_REFERENCE" if quotes.get(k) else "NOT_VERIFIED"} for k in FAMILIES},
                    "bet_eligible":False,"oos":None}
            cards[fixture_id] = (time.monotonic(),card)
            return json_reply(card)
        except (api.APIFootballError, TypeError, KeyError, ValueError):
            return json_reply({"error":"PROVIDER_SLATE_NOT_VERIFIED"},503)


def page(token_key):
    return HTMLResponse("""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cobertura diaria · Soccer Edge</title>
<style>body{background:#0e1824;color:#edf3fb;font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:66rem}button{color:inherit;background:#263b50;padding:.5rem;border:1px solid #496681;border-radius:5px;cursor:pointer}li{margin:.35rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#a6dfff}</style>
<a href="/app">Volver a Soccer Edge</a><h1>Cobertura diaria</h1><p>Vista privada de investigación. Datos faltantes no son ceros; ninguna cuota es accionable aquí.</p>
<p id="status">Consultando calendario…</p><section id="detail" aria-live="polite"></section><ol id="slate"></ol>
<script>
const root="/app/api/coverage-v1", status=document.querySelector("#status"), slate=document.querySelector("#slate"), detail=document.querySelector("#detail");
const token=localStorage.getItem(TOKEN_KEY);
async function get(path){const r=await fetch(root+path,{headers:token?{Authorization:"Bearer "+token}:{},cache:"no-store"});const j=await r.json();if(!r.ok)throw Error(j.error||j.state||r.status);return j;}
async function openMatch(id){const pre=document.createElement("pre");detail.replaceChildren(pre);pre.textContent="Consultando ficha…";try{const d=await get("/"+id);pre.textContent=JSON.stringify(d,null,2);}catch(e){pre.textContent=e.message;}}
(async()=>{try{const d=await get("");const c=d.counts;status.textContent=d.date+" · "+c.fixtures+" partidos · "+c.prematch+" prepartido · "+c.projections+" proyecciones capturadas en esta vista · "+c.provider_quote_fixtures+" con precios de referencia · OOS no medido.";
for(const f of d.fixtures){const li=document.createElement("li"), b=document.createElement("button");b.textContent=f.home_team+" – "+f.away_team+" · "+f.league+" · "+f.status+" · "+f.coverage_status;b.onclick=()=>openMatch(f.fixture_id);li.append(b);slate.append(li);}const id=Number(new URLSearchParams(location.search).get("fixture_id"));if(Number.isSafeInteger(id)&&id>0)openMatch(id);
}catch(e){status.textContent=e.message==="AUTH_REQUIRED"?"Inicia sesión como propietario en Soccer Edge y vuelve a abrir esta página.":e.message;}})();
</script></html>""".replace("TOKEN_KEY",repr(token_key)),headers={"Cache-Control":"no-store"})
