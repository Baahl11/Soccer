/** Model-implied team GF×GA grids, not observed or independently verified frequencies.
 * Only persisted RAW sport Poisson lambda inputs may authorize this visualization.
 * X-axis is goals scored (GF); Y-axis is goals conceded (GA).
 * Bin 4+ includes the entire Poisson tail so all 25 cells sum to 100%.
 */
import type { TeamScoringProfile } from "./model";

const LABELS = ["0", "1", "2", "3", "4+"];

function poissonBins(rate: number): number[] {
  const bins = [Math.exp(-rate)];
  for (let k = 1; k <= 3; k++) bins.push(bins[k - 1] * rate / k);
  bins.push(Math.max(0, 1 - bins.reduce((a, b) => a + b, 0)));
  return bins;
}

export function makeTeamGoalMatrices(
  homeName: string,
  awayName: string,
  rawHomeRate: number | null,
  rawAwayRate: number | null,
  goalRateSemantics: string | null
): TeamScoringProfile[] {
  if (goalRateSemantics !== "POISSON_LAMBDA_NOT_XG"
      || rawHomeRate === null || rawAwayRate === null
      || !Number.isFinite(rawHomeRate) || !Number.isFinite(rawAwayRate)
      || rawHomeRate < 0 || rawAwayRate < 0) return [];
  const home = poissonBins(rawHomeRate);
  const away = poissonBins(rawAwayRate);
  const build = (team: string, tone: "home" | "away", gf: number[], ga: number[]): TeamScoringProfile => ({
    team, tone, labels: LABELS,
    values: ga.map(pGa => gf.map(pGf => Number((100 * pGf * pGa).toFixed(1)))),
  });
  return [
    build(homeName, "home", home, away),
    build(awayName, "away", away, home),
  ];
}
