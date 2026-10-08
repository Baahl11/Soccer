export type VerificationState =
  | "VERIFIED"
  | "NOT_VERIFIED"
  | "INSUFFICIENT_DATA"
  | "MARKET_DATA_ONLY";

export interface TeamViewModel {
  name: string;
  shortName: string;
  side: "home" | "away";
}

export interface ScoreMatrix {
  labels: string[];
  values: number[][];
  hot: [number, number];
}

export interface TeamScoringProfile {
  team: string;
  tone: "home" | "away";
  labels: string[];
  values: number[][];
}

export interface MatchCenterViewModel {
  sample: boolean;
  league: string;
  kickoff: string;
  home: TeamViewModel;
  away: TeamViewModel;
  dataQuality: string;
  confidence: number;
  lineup: string;
  market: string;
  probability: { home: number; draw: number; away: number };
  xg: { home: number; away: number };
  edge: {
    model: number;
    market: number;
    gap: number;
    fairPrice: number;
    marketPrice: number;
  };
  sportProfile: Array<{ label: string; score: number; status: "GOOD" | "NEUTRAL" }>;
  scoreMatrix: ScoreMatrix;
  scoringProfiles: TeamScoringProfile[];
  over25: { model: number; market: number; edge: number };
}
