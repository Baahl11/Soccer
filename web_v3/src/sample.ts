import type { MatchCenterViewModel } from "./model";

const labels = ["0", "1", "2", "3", "4+"];

export const sampleMatch: MatchCenterViewModel = {
  sample: true,
  live: false,
  fixtureId: null,
  league: "Premier League",
  kickoff: "20:00",
  home: { name: "Arsenal", shortName: "ARS", side: "home", logoUrl: null },
  away: { name: "Brighton", shortName: "BHA", side: "away", logoUrl: null },
  dataQuality: "A",
  confidence: 84,
  lineup: "CONFIRMED",
  market: "FRESH",
  probability: { home: 67.1, draw: 20.3, away: 12.6 },
  xg: { home: 2.08, away: 0.91 },
  edge: {
    model: 71.8,
    market: 60.1,
    gap: 11.7,
    fairPrice: 1.39,
    marketPrice: 1.66,
    modelKind: "CALIBRATED MODEL",
  },
  sportProfile: [
    { label: "Attack strength", score: 83, status: "GOOD" },
    { label: "Defense strength", score: 58, status: "NEUTRAL" },
    { label: "Territorial control", score: 79, status: "GOOD" },
    { label: "Recent form", score: 86, status: "GOOD" },
    { label: "Home advantage", score: 81, status: "GOOD" },
    { label: "Opponent quality", score: 54, status: "NEUTRAL" },
  ],
  scoreMatrix: {
    labels,
    values: [
      [2, 5, 6, 3, 1],
      [5, 9, 11, 7, 2],
      [4, 10, 9, 6, 2],
      [2, 5, 6, 4, 1],
      [1, 2, 3, 2, 1],
    ],
    hot: [1, 2],
    mostLikely: { score: "2-1", probability: 11.4 },
  },
  scoringProfiles: [
    {
      team: "Arsenal",
      tone: "home",
      labels,
      values: [
        [1,3,4,2,1],[2,5,7,3,1],[1,4,6,2,1],[1,2,3,1,0],[0,1,1,0,0]
      ],
    },
    {
      team: "Brighton",
      tone: "away",
      labels,
      values: [
        [1,2,3,1,0],[2,4,5,2,1],[1,3,4,2,1],[0,2,2,1,0],[0,1,1,0,0]
      ],
    },
  ],
  over25: { model: 64.4, market: 55.3, edge: 8.9, modelKind: "RAW SPORT" },
  decision: {
    classification: null,
    displayBucket: "SAMPLE",
    tier: null,
    reason: "Illustrative Golden Master values only.",
  },
  sections: {
    probability: { state: "VERIFIED" },
    xg: { state: "VERIFIED" },
    edge: { state: "VERIFIED" },
    sportProfile: { state: "VERIFIED" },
    scoreMatrix: { state: "VERIFIED" },
    scoringProfiles: { state: "VERIFIED" },
    over25: { state: "VERIFIED" },
  },
  missingSections: [],
  disclosure: "SAMPLE DATA · NOT A LIVE PREDICTION",
};
