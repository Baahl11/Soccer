"""As-of attested GitHub-state -> OIDC Render PostgreSQL research bridge.

A *real* Git commit containing the exact frozen raw prediction must predate the
kickoff. No backdated record, synthetic pregame stage, price or BET enters here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request
import urllib.error

MAX_BATCH=16
MAX_BODY=250_000
MAX_HISTORY_COMMITS=300
RELATIVE="soccer_edge_state/analysis/isolated_sport_first_pregame_v1.jsonl"
STATUS="FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION"


def _utc(text):
    if not isinstance(text,str):
        return None
    try:
        dt=datetime.fromisoformat(text.replace("Z","+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError,TypeError):
        return None


def _canonical(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True,timeout=45)


def origin_attestation(repo: Path, row: dict):
    """Prove the *identical source* was in an actual Git commit before kickoff."""
    fid=row.get("fixture_id")
    captured=_utc(row.get("captured_at_utc"))
    kickoff=_utc(row.get("kickoff"))
    if (type(fid) is not int or fid <= 0 or not captured or not kickoff
            or not captured <= kickoff-timedelta(minutes=45)):
        raise ValueError("SOURCE_ASOF_NOT_VERIFIED")
    needle='"fixture_id": '+str(fid)
    history=_git(repo,"log","--reverse","--format=%H|%cI",
                 "-S",needle,"--",RELATIVE).splitlines()
    if not history or len(history)>MAX_HISTORY_COMMITS:
        raise ValueError("ORIGINAL_PREGAME_COMMIT_NOT_FOUND_OR_UNBOUNDED")
    target=_canonical(row)
    first=None
    for line in history:
        try:
            sha,utc=line.split("|",1)
            commit_time=_utc(utc)
        except ValueError:
            continue
        if commit_time is None:
            continue
        try:
            source=_git(repo,"show",f"{sha}:{RELATIVE}")
        except subprocess.CalledProcessError:
            continue
        for data in source.splitlines():
            if not data or needle not in data:
                continue
            try:
                prior=json.loads(data)
            except ValueError:
                continue
            if isinstance(prior,dict) and prior.get("fixture_id")==fid and _canonical(prior)==target:
                first=(sha,commit_time)
                break
        if first:
            break
    if first is None:
        raise ValueError("MATCHING_FROZEN_SOURCE_NOT_IN_GIT_HISTORY")
    sha,when=first
    if when > kickoff or when < captured-timedelta(minutes=5):
        raise ValueError("ORIGINAL_GIT_COMMIT_NOT_PREGAME")
    return {"fixture_id":fid,"source_at":captured.isoformat(),
            "first_recorded_commit":sha,"first_recorded_at":when.isoformat(),
            "pregame_git_history_verified":True}


def prepare(state:Path, receipt:Path, max_rows:int=MAX_BATCH):
    if not 1<=max_rows<=MAX_BATCH:
        raise ValueError("UNBOUNDED_IMPORT_BATCH")
    source=state/RELATIVE
    if not source.is_file() or source.stat().st_size>25_000_000:
        raise ValueError("SOURCE_NOT_AVAILABLE_OR_TOO_LARGE")
    rows=[json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not isinstance(rows,list) or len(rows)>5000 or any(not isinstance(x,dict) for x in rows):
        raise ValueError("SOURCE_REGISTRY_INVALID")
    imported=set()
    if receipt.is_file():
        old=json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("status")!="RESEARCH_ONLY_FROZEN_PG_IMPORT":
            raise ValueError("IMPORT_MANIFEST_UNVERIFIED")
        previous=old.get("imported_keys")
        if not isinstance(previous,list) or len(previous)>5000:
            raise ValueError("IMPORT_MANIFEST_UNBOUNDED")
        imported=set(previous)
    candidates=[]
    for r in rows:
        if r.get("status")!=STATUS:
            raise ValueError("NON_FROZEN_RESEARCH_SOURCE")
        k=f"{r.get('fixture_id')}:{r.get('captured_at_utc')}"
        if k not in imported:
            candidates.append((k,r))
    chosen=candidates[:max_rows]
    attestations=[origin_attestation(state,r) for _,r in chosen]
    payload={"snapshots":[r for _,r in chosen]}
    b=json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode()
    if len(b)>MAX_BODY:
        raise ValueError("BOUNDED_SOURCE_BODY_TOO_LARGE")
    return payload,b,[k for k,_ in chosen],attestations,imported,len(rows)


def post_oidc(body,endpoint):
    oidc=os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
    token=os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not oidc or not token:
        raise RuntimeError("OIDC_ACTION_IDENTITY_NOT_AVAILABLE")
    joiner="&" if "?" in oidc else "?"
    url=oidc+joiner+urllib.parse.urlencode({"audience":"soccer-edge-render"})
    req=urllib.request.Request(url,headers={"Authorization":"bearer "+token})
    with urllib.request.urlopen(req,timeout=20) as response:
        jwt=json.load(response).get("value")
    if not isinstance(jwt,str) or not jwt:
        raise RuntimeError("SIGNED_OIDC_TOKEN_MISSING")
    req=urllib.request.Request(endpoint.rstrip("/")+"/internal/isolated-sport-frozen-pg-v1/import",
                               data=body,headers={"Content-Type":"application/json",
                               "Authorization":"Bearer "+jwt},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=38) as response:
            raw=response.read(12000)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("FROZEN_PG_IMPORT_HTTP_"+str(exc.code)) from None
    return json.loads(raw)


def run(state:Path,receipt:Path,endpoint:str):
    payload,b,keys,proofs,imported,registry_total=prepare(state,receipt)
    if keys:
        response=post_oidc(b,endpoint)
        if (response.get("status")!="AUTHENTIC_PREGAME_SPORT_RESEARCH_IMPORTED"
            or response.get("snapshots_validated")!=len(keys)
            or response.get("new_research_rows",0)+response.get("identical_existing_rows",0)!=len(keys)
            or response.get("production_promotion_allowed") is not False
            or response.get("bet_eligible") is not False
            or response.get("canonical_clv_rows_added")!=0
            or response.get("canonical_deep_dives_added")!=0
            or response.get("market_prices_added")!=0
            or response.get("provider_requests_added")!=0):
            raise RuntimeError("FROZEN_PG_IMPORT_RESPONSE_NOT_VERIFIED")
        imported.update(keys)
    else:
        response={"new_research_rows":0,"identical_existing_rows":0}
    out={"status":"RESEARCH_ONLY_FROZEN_PG_IMPORT",
         "generated_at_utc":datetime.now(timezone.utc).isoformat(),
         "total_original_pregame_records":registry_total,
         "verified_pre_kickoff_commits_this_run":proofs,
         "new_research_rows":response["new_research_rows"],
         "existing_identical_rows":response["identical_existing_rows"],
         "imported_keys":sorted(imported),
         "unimported_rows_remaining":registry_total-len(imported),
         "postgres_write_only_isolated_research_table":True,
         "canonical_clv_rows_added":0,"canonical_deep_dives_added":0,
         "provider_requests_added":0,"retroactive_predictions_created":False,
         "production_promotion_allowed":False,"bet_eligible":False}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ("imported_keys","verified_pre_kickoff_commits_this_run")},
                     sort_keys=True))
    return out


def self_test():
    example={"fixture_id":42,"captured_at_utc":"2026-11-10T10:00:00Z",
             "kickoff":"2026-11-10T11:00:00Z","raw":{"a":0.7}}
    assert _canonical(example)==_canonical(dict(reversed(list(example.items()))))
    assert _utc("2026-11-10T10:00:00Z")<_utc(example["kickoff"])
    assert not _utc("2026-11-10T10:00:00")
    assert MAX_BATCH<=16 and MAX_BODY<=250000
    print("SOCCER_FROZEN_PG_GIT_ASOF_PROVENANCE_SELF_TEST_PASS")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",type=Path)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--endpoint",default="https://soccer-edge-api.onrender.com")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--audit-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.state or not args.receipt:
        p.error("--state and --receipt required")
    if args.audit_only:
        _,_,keys,proofs,_,total=prepare(args.state,args.receipt)
        print(json.dumps({"status":"OFFLINE_ORIGINAL_GIT_ASOF_VERIFIED",
                          "eligible":len(keys),"total":total,
                          "commits":[x["first_recorded_commit"] for x in proofs],
                          "provider_requests_added":0}))
    else:
        run(args.state,args.receipt,args.endpoint)


if __name__=="__main__":
    main()
