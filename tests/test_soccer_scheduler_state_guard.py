from __future__ import annotations
from datetime import datetime, timedelta, timezone
import unittest

from scripts.soccer_scheduler_state_guard import verify_health

NOW = datetime(2026, 11, 7, 5, 0, tzinfo=timezone.utc)

def good():
    return {
        "status": "ok", "generated_at_utc": (NOW-timedelta(minutes=55)).isoformat(),
        "fixture_scan_count": 4491, "database_persisted": True,
        "database_error": None, "api_calls_this_tick": 17,
        "last_daily_remaining": 7335,
    }

class SchedulerStateContract(unittest.TestCase):
    def test_recent_canonical_tick_passes_without_bet(self):
        result=verify_health(good(), now_utc=NOW)
        self.assertEqual(result["status"],"VERIFIED_RECENT_CANONICAL_TICK")
        self.assertEqual(result["blockers"],[])
        self.assertEqual(result["provider_requests_added"],0)
        self.assertFalse(result["bets_authorized"])

    def test_stale_snapshot_fails(self):
        x=good();x["generated_at_utc"]=(NOW-timedelta(hours=4)).isoformat()
        out=verify_health(x,now_utc=NOW)
        self.assertIn("SCHEDULER_SNAPSHOT_STALE",out["blockers"])

    def test_false_db_persistence_never_passes(self):
        x=good();x["database_persisted"]=False
        out=verify_health(x,now_utc=NOW)
        self.assertIn("DATABASE_PERSISTENCE_NOT_VERIFIED",out["blockers"])

    def test_missing_or_invalid_slate_is_blocker(self):
        for count in (None,0,True,"100"):
            x=good();x["fixture_scan_count"]=count
            out=verify_health(x,now_utc=NOW)
            self.assertIn("OFFICIAL_FIXTURE_SCAN_NOT_VERIFIED",out["blockers"])

    def test_future_or_naive_timestamp_fails(self):
        for t in ((NOW+timedelta(minutes=10)).isoformat(),NOW.replace(tzinfo=None).isoformat()):
            x=good();x["generated_at_utc"]=t
            out=verify_health(x,now_utc=NOW)
            self.assertEqual(out["status"],"NOT_VERIFIED")

    def test_invalid_quota_telemetry_is_blocker(self):
        x=good();x["last_daily_remaining"]=-1
        out=verify_health(x,now_utc=NOW)
        self.assertIn("PROVIDER_QUOTA_TELEMETRY_INVALID",out["blockers"])

    def test_read_only_mode_is_unconditional(self):
        x=good()
        out=verify_health(x,now_utc=NOW)
        self.assertEqual(out["database_queries_added"],0)
        self.assertFalse(out["market_decisions_changed"])

if __name__ == "__main__":
    unittest.main()
