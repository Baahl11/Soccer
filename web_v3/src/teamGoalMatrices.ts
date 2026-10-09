/** Team-specific season goal profiles, never another copy of the match score matrix.
 * X: team's own goals for (GF). Y: that team's own goals against (GA).
 * Independent Poisson is an assumption: these are ESTIMATES from persisted
 * API-Football season rates, not empirical joint distributions or model bets.
 */
import type { EvidenceSection, TeamScoringProfile } from "./model";

const LABELS = ["0", "1", "2", "3", "4+"];
type EvidenceItem = EvidenceSection["items"][number];

function poissonBins(rate: number): number[] {
  const bins = [Math.exp(-rate)];
  for (let k = 1; k <= 3; k++) bins.push(bins[k - 1] * rate / k);
  bins.push(Math.max(0, 1 - bins.reduce((a, b) => a + b, 0)));
  return bins;
}

function seasonRate(evidence: EvidenceSection[], key: string): EvidenceItem | null {
  const row = evidence.flatMap(part => part.items).find(item => item.key === key);
  return row?.source === "API_FOOTBALL_TEAM_STATS"
    && row.status === "PERSISTED"
    && typeof row.value === "number" && Number.isFinite(row.value)
    && row.value >= 0 && row.sampleN !== null
    && Number.isInteger(row.sampleN) && row.sampleN > 0
    && !!row.capturedAt
    ? row : null;
}

function teamProfile(
  evidence: EvidenceSection[], team: string, tone: "home" | "away"
): TeamScoringProfile | null {
  const scored = seasonRate(evidence, `team_performance.${tone}_goals_for_avg`);
  const conceded = seasonRate(evidence, `team_performance.${tone}_goals_against_avg`);
  if (!scored || !conceded
      || scored.source !== conceded.source
      || scored.capturedAt !== conceded.capturedAt
      || scored.sampleN !== conceded.sampleN
      || scored.observationScope !== conceded.observationScope) return null;

  const gfRate = scored.value as number;
  const gaRate = conceded.value as number;
  const gf = poissonBins(gfRate);
  const ga = poissonBins(gaRate);
  return {
    team, tone, labels: LABELS, gfRate, gaRate,
    source: scored.source,
    capturedAt: scored.capturedAt,
    sampleN: scored.sampleN,
    observationScope: scored.observationScope,
    values: ga.map(pGa => gf.map(pGf => Number((100 * pGf * pGa).toFixed(1)))),
  };
}

/** Partial coverage is intentional: never borrow opponent rates for a missing team. */
export function makeTeamGoalMatrices(
  homeName: string, awayName: string, evidence: EvidenceSection[]
): TeamScoringProfile[] {
  return [
    teamProfile(evidence, homeName, "home"),
    teamProfile(evidence, awayName, "away"),
  ].filter((profile): profile is TeamScoringProfile => profile !== null);
}
