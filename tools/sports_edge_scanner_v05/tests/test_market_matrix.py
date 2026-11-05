"""Unit tests for sport-first exact-market Bet365 review, no BET promotions."""
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest import TestCase
from sports_edge.market_matrix import examine_bet365_market_matrix,raw_exact_market_probability

NOW=datetime(2026,11,5,16,tzinfo=timezone.utc)

def q(fid,name,market,selection,point,price,ago=5):
    return {'sport':'soccer','event_id':str(fid),'event_name':name,
      'starts_at_utc':(NOW+timedelta(hours=3)).isoformat(),
      'market':market,'selection':selection,'point':point,
      'price_decimal':price,'bookmaker':'Bet365','source':'API-Football',
      'source_updated_at':(NOW-timedelta(minutes=ago)).isoformat()}

def raw(sample=8):
    return {'status':'MODELED_LIMITED','raw_total_goals':3.25,
        'raw_home_goal_rate':2.3,'raw_away_goal_rate':0.95,
        'raw_home_win_prob':.77,'raw_draw_prob':.15,'raw_away_win_prob':.08,
        'raw_btts_yes_prob':.76,'raw_over_1_5_prob':.84,
        'raw_over_2_5_prob':.65,'raw_over_3_5_prob':.41,
        'sample':{'minimum_split_sample':sample}}

class MarketMatrixTest(TestCase):
    def sample_inputs(self,base):
        names={1:'A vs B',2:'C vs D'}
        fixtures={'screened_fixtures':[{'fixture_id':i,'raw_sport_projection':raw(i+4)}
              for i in (1,2)]}
        quotes=[
          q(1,names[1],'Match Winner','Home','',1.42),
          q(1,names[1],'Match Winner','Draw','',4.0),
          q(1,names[1],'Match Winner','Away','',8.0),
          q(1,names[1],'Goals Over/Under','Over','2.5',1.6),
          q(1,names[1],'Goals Over/Under','Under','2.5',2.3),
          q(1,names[1],'Goal Line','Over','4.5',1.5),
          q(1,names[1],'Goal Line','Under','4.5',2.5),
          q(1,names[1],'Goals Over/Under - First Half','Over','0.5',1.33),
          q(2,names[2],'Both Teams to Score','Yes','',1.4),
          q(2,names[2],'Both Teams to Score','No','',2.8),
          q(2,names[2],'Corners','Over','6.5',1.34),
        ]
        for i in (1,2):
            out={'fixture_id':i,'quotes':[x for x in quotes if x['event_id']==str(i)]}
            (base/f'{i}.json').write_text(json.dumps(out))
        snapshots=[{'file':str(base/f'{i}.json'),'fixture_id':i} for i in (1,2)]
        return snapshots,fixtures
    def test_exact_market_and_poisson_derivative(self):
        direct={'family':'FULL_TIME_GOALS','point':'2.5','selection':'Over'}
        p,source=raw_exact_market_probability(direct,raw())
        self.assertEqual(p,.65)
        self.assertIn('UNCALIBRATED',source)
        derived={'family':'FULL_TIME_GOALS','point':'4.5','selection':'Under'}
        p,source=raw_exact_market_probability(derived,raw())
        self.assertTrue(.6<p<.9)
        self.assertIn('DERIVED',source)
    def test_refuses_half_and_corners_prediction(self):
        for family in ('FIRST_HALF_GOALS','SECOND_HALF_GOALS','CORNERS','FOULS'):
            p,_=raw_exact_market_probability({'family':family,'point':'2.5','selection':'Over'},raw())
            self.assertIsNone(p)
    def test_no_quarter_line_ev(self):
        p,_=raw_exact_market_probability({'family':'FULL_TIME_GOALS','point':'2.25','selection':'Over'},raw())
        self.assertIsNone(p)
    def test_market_table_and_novig_and_pairs(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        self.assertGreater(result['exact_or_poisson_derived_raw_quotes'],5)
        self.assertGreater(result['two_way_or_three_way_novig_quotes'],2)
        self.assertTrue(all(x['classification']=='WATCH' for x in result['all_quotes']))
        self.assertTrue(all(x['calibrated_probability'] is None for x in result['all_quotes']))
        self.assertGreaterEqual(result['sporting_screened_plus100_pairs_found'],1)
        self.assertEqual(result['actionable_parlays'],0)
        self.assertTrue(all(pair['joint_probability'] is None for pair in result['sporting_screened_plus100_pairs']))
        self.assertEqual(result['all_quotes'][0]['bookmaker'],'Bet365')
    def test_unknown_markets_never_claim_raw(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        corners=[x for x in result['all_quotes'] if x['family']=='CORNERS']
        self.assertTrue(corners and corners[0]['raw_sport_probability'] is None)
    def test_stale_price_excluded(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            p=Path(snaps[0]['file'])
            data=json.loads(p.read_text())
            for quote in data['quotes']:
                quote['source_updated_at']=(NOW-timedelta(hours=9)).isoformat()
            p.write_text(json.dumps(data))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        self.assertEqual(len([x for x in result['all_quotes'] if x['event_id']=='1']),0)
    def test_rejects_mismatched_snapshot(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            p=Path(snaps[0]['file'])
            data=json.loads(p.read_text())
            data['quotes'][0]['event_id']='999'
            p.write_text(json.dumps(data))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        self.assertIn('1',result['snapshot_read_failures'])
    def test_no_fair_if_other_book_or_incomplete_binary(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            for snap in snaps:
                p=Path(snap['file'])
                data=json.loads(p.read_text())
                data['quotes']=[x for x in data['quotes'] if x['selection'] in ('Home','Over')]
                p.write_text(json.dumps(data))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        self.assertEqual(result['two_way_or_three_way_novig_quotes'],0)

    def test_equivalent_goal_line_aliases_are_not_separate_parlays(self):
        with TemporaryDirectory() as d:
            snaps,fx=self.sample_inputs(Path(d))
            # Same exact 90m settlement (home fixture 1, Under 4.5).
            p=Path(snaps[0]['file'])
            data=json.loads(p.read_text())
            data['quotes']=[
                q(1,'A vs B','Goals Over/Under','Under','4.5',1.42),
                q(1,'A vs B','Goal Line','Under','4.5',1.44),
            ]
            p.write_text(json.dumps(data))
            p2=Path(snaps[1]['file'])
            data2=json.loads(p2.read_text())
            data2['quotes']=[q(2,'C vs D','Both Teams to Score','Yes','',1.40)]
            p2.write_text(json.dumps(data2))
            result=examine_bet365_market_matrix(snaps,fx,{'families':[]},now=NOW)
        self.assertEqual(result['fresh_quoted_fixtures'],2)
        self.assertEqual(result['same_outcome_alias_quotes_collapsed'],1)
        self.assertEqual(result['sporting_screened_plus100_pairs_found'],1)
        self.assertEqual(result['sporting_screened_plus100_pairs'][0]['minimum_venue_sample'],5)
