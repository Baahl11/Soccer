"""No wagering, no invented model probabilities, market-price discovery only."""
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sports_edge.parlay_radar import discover_two_leg_price_pairs,discover_saved_snapshot_pairs

NOW=datetime(2026,11,5,7,30,tzinfo=timezone.utc)

def quote(event, bookmaker='Example Book',market='Match Winner',selection='Home',point='',price=1.40,age=12):
    return {
       'sport':'soccer','event_id':str(event),'event_name':f'Fixture {event}',
       'bookmaker':bookmaker,'market':market,'selection':selection,'point':point,
       'source':'API-Football','price_decimal':price,
       'starts_at_utc':(NOW+timedelta(hours=10)).isoformat(),
       'source_updated_at':(NOW-timedelta(minutes=age)).isoformat(),
    }

class ParlayRadarTest(unittest.TestCase):
    def test_two_different_fixtures_same_book_price_only(self):
        r=discover_two_leg_price_pairs([quote(1),quote(2,price=1.43)],now=NOW)
        self.assertEqual(len(r['pairs']),1)
        pair=r['pairs'][0]
        self.assertEqual(pair['status'],'WATCH')
        self.assertEqual(pair['theoretical_combined_decimal'],2.002)
        self.assertIsNone(pair['joint_model_probability'])
        self.assertIsNone(pair['estimated_ev'])
        self.assertEqual(pair['dependence_evidence'],'NOT_VERIFIED')
        self.assertEqual(r['actionable_parlays'],0)
    def test_same_fixture_not_combinable(self):
        r=discover_two_leg_price_pairs([quote(1),quote(1,market='Both Teams to Score',selection='Yes')],now=NOW)
        self.assertEqual(r['pairs'],[])
    def test_different_books_do_not_mix(self):
        r=discover_two_leg_price_pairs([quote(1),quote(2,bookmaker='Different Book')],now=NOW)
        self.assertFalse(r['pairs'])
    def test_timestamp_must_be_provider_update_and_fresh(self):
        r=discover_two_leg_price_pairs([quote(1,age=121),quote(2)],now=NOW)
        self.assertEqual(r['rejected_quote_reasons']['QUOTE_STALE_OR_FUTURE'],1)
        self.assertFalse(r['pairs'])
    def test_scraped_missing_original_update_must_fail(self):
        a=quote(1);a.pop('source_updated_at')
        r=discover_two_leg_price_pairs([a,quote(2)],now=NOW)
        self.assertFalse(r['pairs'])
    def test_no_ghost_lines_for_derivatives(self):
        r=discover_two_leg_price_pairs([quote(1,market='Goals Over/Under',selection='Over',point=''),quote(2)],now=NOW)
        self.assertFalse(r['pairs'])
        self.assertEqual(r['rejected_quote_reasons']['UNSUPPORTED_OR_AMBIGUOUS_MARKET'],1)
    def test_supports_half_corners_cards_fouls_but_no_model(self):
        names=['Goals Over/Under First Half','Goals Over/Under - Second Half','Corners Over/Under','Fouls Over/Under','Cards Over/Under']
        for n in names:
            r=discover_two_leg_price_pairs([quote(1,market=n,selection='Over',point='4.5'),quote(2)],now=NOW)
            self.assertEqual(len(r['pairs']),1,n)
            self.assertIsNone(r['pairs'][0]['joint_model_probability'])
    def test_future_event_only(self):
        a=quote(1);a['starts_at_utc']=(NOW-timedelta(minutes=1)).isoformat()
        r=discover_two_leg_price_pairs([a,quote(2)],now=NOW)
        self.assertFalse(r['pairs'])
    def test_unverified_source_or_book_excluded(self):
        a=quote(1);a['bookmaker']=''
        r=discover_two_leg_price_pairs([a,quote(2)],now=NOW)
        self.assertFalse(r['pairs'])
    def test_reads_saved_snapshots_without_provider_request(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots=[]
            for e,p in ((1,1.4),(2,1.43)):
                path=Path(directory)/f'{e}.json'
                path.write_text(json.dumps({'fixture_id':e,'quotes':[quote(e,price=p)]}))
                snapshots.append({'file':str(path),'fixture_id':e})
            result=discover_saved_snapshot_pairs(snapshots,now=NOW)
            self.assertEqual(result['snapshots_read'],2)
            self.assertEqual(len(result['pairs']),1)
    def test_snapshot_fixture_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'a.json';p.write_text(json.dumps({'fixture_id':999,'quotes':[quote(1)]}))
            r=discover_saved_snapshot_pairs([{'file':str(p),'fixture_id':1}],now=NOW)
            self.assertEqual(r['snapshot_read_failures'],['SNAPSHOT_INVALID'])
    def test_reject_price_outside_window(self):
        r=discover_two_leg_price_pairs([quote(1,price=1.10),quote(2)],now=NOW)
        self.assertFalse(r['pairs'])
        self.assertEqual(r['rejected_quote_reasons']['LEG_PRICE_OUTSIDE_RESEARCH_WINDOW'],1)
    def test_source_cannot_be_silently_changed(self):
        a,b=quote(1),quote(2)
        b['source']='Other provider'
        r=discover_two_leg_price_pairs([a,b],now=NOW)
        self.assertFalse(r['pairs'])

if __name__=='__main__':unittest.main()

class BoardParlayRadarTest(unittest.TestCase):
    def test_display_explicit_watch_without_ev(self):
        from sports_edge.board import render_board
        radar=discover_two_leg_price_pairs([quote(1),quote(2,price=1.43)],now=NOW)
        html=render_board({'screened_fixtures':[],'match_date':'2026-11-05'},
                          {'match_date':'2026-11-05','sports_screened':0},parlay_radar=radar)
        self.assertIn('Radar parlay cercano a +100',html)
        self.assertIn('WATCH, EV NO VERIFICADO',html)
        self.assertIn('Caliente/Playdoit',html)
        self.assertNotIn('apuesta aprobada por parlay',html)
    def test_missing_provider_odds_does_not_invent_pairs(self):
        from sports_edge.board import render_board
        html=render_board({'screened_fixtures':[],'match_date':'2026-11-05'},
                          {'match_date':'2026-11-05','sports_screened':0},parlay_radar={'pairs':[]})
        self.assertIn('No se encontraron combinaciones',html)