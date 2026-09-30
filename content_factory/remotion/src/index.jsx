import React from 'react';
import {Composition, registerRoot} from 'remotion';
import {SoccerEdgeShort, SoccerEdgeShortCard} from './short.jsx';

const TEST_DEFAULTS = {
  content_id: 'TEST_FIXTURE_ONLY',
  format: 'MODEL_VS_MARKET',
  locale: 'en',
  facts: {
    home: 'Home',
    away: 'Away',
    market: 'Market',
    selection: 'Selection',
    price: null,
    market_probability_display: 'N/V',
    model_probability_display: 'N/V',
    edge_display: 'N/V',
    execution_status: 'N/V'
  },
  copy: {
    hook: 'Soccer Edge · Model vs Market',
    voiceover: 'Test fixture only.',
    caption: 'Test fixture only.',
    x_post: 'Test fixture only.'
  }
};

const Root = () => (
  <>
    <Composition
      id="SoccerEdgeShort"
      component={SoccerEdgeShort}
      durationInFrames={900}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={TEST_DEFAULTS}
    />
    <Composition
      id="SoccerEdgeShortCard"
      component={SoccerEdgeShortCard}
      durationInFrames={1}
      fps={30}
      width={1600}
      height={900}
      defaultProps={TEST_DEFAULTS}
    />
  </>
);

registerRoot(Root);
