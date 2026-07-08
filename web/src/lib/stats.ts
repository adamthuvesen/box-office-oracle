import type { Movie } from "@/lib/types";

/** Aggregates for /stats. Pure functions over the catalog — no I/O, no framework. */

export interface YearlyGross {
  year: number;
  totalGross: number;
  releases: number;
}

/** Total gross and release count per year, continuous over the data's range. */
export function yearlyGross(movies: Movie[]): YearlyGross[] {
  const byYear = new Map<number, YearlyGross>();
  for (const m of movies) {
    const row = byYear.get(m.release_year) ?? {
      year: m.release_year,
      totalGross: 0,
      releases: 0,
    };
    row.totalGross += m.worldwide_gross ?? 0;
    row.releases += 1;
    byYear.set(m.release_year, row);
  }
  const years = [...byYear.keys()];
  const first = Math.min(...years);
  const last = Math.max(...years);
  const out: YearlyGross[] = [];
  for (let year = first; year <= last; year++) {
    out.push(byYear.get(year) ?? { year, totalGross: 0, releases: 0 });
  }
  return out;
}

export const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
] as const;

export const MONTH_ABBREV = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
] as const;

export interface Seasonality {
  years: number[];
  /** grid[monthIndex][yearIndex] = total worldwide gross for that cell. */
  grid: number[][];
  maxCell: number;
  bestMonth: { name: string; total: number };
}

/** Month × year totals for the heatmap, plus the strongest calendar month overall. */
export function seasonality(movies: Movie[]): Seasonality {
  const dated = movies.filter((m) => m.release_date != null);
  const yearSet = new Set(dated.map((m) => m.release_year));
  const first = Math.min(...yearSet);
  const last = Math.max(...yearSet);
  const years: number[] = [];
  for (let y = first; y <= last; y++) years.push(y);

  const grid: number[][] = MONTH_NAMES.map(() => years.map(() => 0));
  const monthTotals = MONTH_NAMES.map(() => 0);
  for (const m of dated) {
    const month = Number(m.release_date!.slice(5, 7)) - 1;
    const yearIdx = m.release_year - first;
    if (month < 0 || month > 11 || yearIdx < 0 || yearIdx >= years.length) continue;
    grid[month][yearIdx] += m.worldwide_gross ?? 0;
    monthTotals[month] += m.worldwide_gross ?? 0;
  }

  const maxCell = Math.max(...grid.flat());
  const bestIdx = monthTotals.indexOf(Math.max(...monthTotals));
  return {
    years,
    grid,
    maxCell,
    bestMonth: { name: MONTH_NAMES[bestIdx], total: monthTotals[bestIdx] },
  };
}

function median(sorted: number[]): number {
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface GenreEconomics {
  genre: string;
  count: number;
  medianGross: number;
  /** Median of known budgets; null if the genre has none. */
  medianBudget: number | null;
}

/** Median gross and budget per lead genre (a movie counts once, under its first genre). */
export function genreEconomics(movies: Movie[], minCount = 30): GenreEconomics[] {
  const byGenre = new Map<string, { grosses: number[]; budgets: number[] }>();
  for (const m of movies) {
    const lead = m.genres[0];
    if (!lead) continue;
    const bucket = byGenre.get(lead) ?? { grosses: [], budgets: [] };
    if (m.worldwide_gross != null) bucket.grosses.push(m.worldwide_gross);
    if (m.production_budget != null && m.production_budget > 0) {
      bucket.budgets.push(m.production_budget);
    }
    byGenre.set(lead, bucket);
  }
  return [...byGenre.entries()]
    .filter(([, b]) => b.grosses.length >= minCount)
    .map(([genre, b]) => ({
      genre,
      count: b.grosses.length,
      medianGross: median(b.grosses.sort((x, y) => x - y)),
      medianBudget:
        b.budgets.length > 0 ? median(b.budgets.sort((x, y) => x - y)) : null,
    }))
    .sort((a, b) => b.medianGross - a.medianGross);
}

export type ReturnBand = "over" | "mid" | "under";

export interface BudgetGrossPoint {
  title: string;
  year: number;
  budget: number;
  gross: number;
  logBudget: number;
  logGross: number;
  band: ReturnBand;
}

/** Scatter points for budget vs gross. Missing budgets or grosses excluded. */
export function budgetVsGross(movies: Movie[]): BudgetGrossPoint[] {
  const points: BudgetGrossPoint[] = [];
  for (const m of movies) {
    if (m.production_budget == null || m.production_budget <= 0) {
      continue;
    }
    if (m.worldwide_gross == null || m.worldwide_gross <= 0) continue;
    const budget = m.production_budget;
    const gross = m.worldwide_gross;
    points.push({
      title: m.title,
      year: m.release_year,
      budget,
      gross,
      logBudget: Math.log10(budget),
      logGross: Math.log10(gross),
      band: gross >= 2.5 * budget ? "over" : gross < budget ? "under" : "mid",
    });
  }
  return points;
}

export interface BandSummary {
  band: ReturnBand;
  count: number;
  share: number;
  medianReturn: number;
}

/** Per-band counts and median return, for the scatter's table fallback. */
export function returnBandSummary(points: BudgetGrossPoint[]): BandSummary[] {
  const bands: ReturnBand[] = ["over", "mid", "under"];
  return bands.map((band) => {
    const returns = points
      .filter((p) => p.band === band)
      .map((p) => p.gross / p.budget)
      .sort((x, y) => x - y);
    return {
      band,
      count: returns.length,
      share: returns.length / points.length,
      medianReturn: returns.length > 0 ? median(returns) : 0,
    };
  });
}

export interface GrossLeader {
  name: string;
  total: number;
  count: number;
}

/** Top studios or directors by total gross, with a minimum-films bar. */
export function grossLeaders(
  movies: Movie[],
  key: "production_company" | "director",
  minFilms = 3,
  top = 10,
): GrossLeader[] {
  const totals = new Map<string, GrossLeader>();
  for (const m of movies) {
    const name = m[key];
    if (!name) continue;
    const row = totals.get(name) ?? { name, total: 0, count: 0 };
    row.total += m.worldwide_gross ?? 0;
    row.count += 1;
    totals.set(name, row);
  }
  return [...totals.values()]
    .filter((r) => r.count >= minFilms)
    .sort((a, b) => b.total - a.total)
    .slice(0, top);
}
