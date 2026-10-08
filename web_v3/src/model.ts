export type VerificationState =
  | "VERIFIED"
  | "NOT_VERIFIED"
  | "INSUFFICIENT_DATA"
  | "MARKET_DATA_ONLY"
  | "PREMIUM_REQUIRED"
  | "UNAVAILABLE";

export interface TeamViewModel {
  name: string;
  shortName: string;
  side: "home" | "away";
  logoUrl?: string | null;
}

export interface ScoreMatrix {
  labels: string[];
  values: Array<Array<number | null>>;
  hot: [number, number] | null;
  mostLikely?: { score: string; probability: number } | null;
}

export interface TeamScoringProfile {
  team: string;
  tone: "home" | "away";
  labels: string[];
  values: Array<Array<number | null>>;
}

export interface SectionState {
  state: VerificationState;
  note?: string;
}

export interface MatchCenterViewModel {
  sample: boolean;
  live: boolean;
  fixtureId?: number | null;
  league: string;
  kickoff: string;
  home: TeamViewModel;
  away: TeamViewModel;
  dataQuality: string;
  confidence: number | null;
  lineup: string;
  market: string;
  probability: { home: number; draw: number; away: number } | null;
  xg: { home: number | null; away: number | null } | null;
  edge: {
    model: number;
    market: number;
    gap: number;
    fairPrice: number | null;
    marketPrice: number | null;
    modelKind: string;
  } | null;
  sportProfile: Array<{ label: string; score: number; status: "GOOD" | "NEUTRAL" }>;
  scoreMatrix: ScoreMatrix | null;
  scoringProfiles: TeamScoringProfile[];
  over25: {
    model: number;
    market: number;
    edge: number;
    modelKind: string;
  } | null;
  decision: {
    classification: string | null;
    displayBucket: string | null;
    tier: string | null;
    reason: string | null;
  };
  sections: {
    probability: SectionState;
    xg: SectionState;
    edge: SectionState;
    sportProfile: SectionState;
    scoreMatrix: SectionState;
    scoringProfiles: SectionState;
    over25: SectionState;
  };
  missingSections: string[];
  disclosure: string;
}
