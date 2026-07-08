import type { Movie, OraclePrediction, OraclePredictions } from "@/lib/types";

/** One movie as a particle in the budget×gross field. */
export interface Star {
  tmdbId: number;
  title: string;
  releaseYear: number;
  gross: number;
  budget: number;
  posterPath: string | null;
  /** Field layout, normalized to [-1, 1]: x = log budget, y = log gross. */
  fieldX: number;
  fieldY: number;
  /** Per-year error layout: x = year, y = log(pred/actual). 0-alpha if no prediction. */
  yearX: number;
  yearY: number;
  hasPrediction: boolean;
  predicted: number | null;
  /** Field-space y where the model's guess would sit (same log scale as fieldY). */
  predictedFieldY: number | null;
  /** 0..1 — how strongly the star glows (return on budget, log-scaled). */
  intensity: number;
  /** sqrt-scaled point size in CSS px at zoom 1. */
  size: number;
}

export interface StarField {
  stars: Star[];
  minYear: number;
  maxYear: number;
  /** Index of a famous, high-gross star with a prediction — the narrative subject. */
  featuredIndex: number;
}

const X_RANGE = 0.88;
const Y_RANGE = 0.78;

export function buildStarField(
  movies: Movie[],
  predictions: OraclePredictions | null,
): StarField {
  /** Honest, gradeable prediction: out-of-sample with a real gross to compare. */
  const honest = (m: Movie): OraclePrediction | null => {
    const p = predictions?.[String(m.tmdb_id)] ?? null;
    return p && p.prediction_kind === "out_of_sample" && p.actual_gross != null
      ? p
      : null;
  };

  const eligible = movies.filter(
    (m) =>
      m.production_budget != null &&
      m.production_budget > 100_000 &&
      m.worldwide_gross != null &&
      m.worldwide_gross > 10_000,
  );

  const logBudget = eligible.map((m) => Math.log10(m.production_budget!));
  const logGross = eligible.map((m) => Math.log10(m.worldwide_gross!));
  // x = budget rank (uniform spread — log dollars leave the lower half empty),
  // y = log gross clamped to percentiles (height stays a real magnitude).
  const budgetRank = rankPositions(logBudget);
  const [gMin, gMax] = percentileBounds(logGross, 0.01, 0.995);
  const years = eligible.map((m) => m.release_year);
  const predYears = eligible
    .filter((m) => honest(m) != null)
    .map((m) => m.release_year);
  const minYear = predYears.length ? Math.min(...predYears) : Math.min(...years);
  const maxYear = predYears.length ? Math.max(...predYears) : Math.max(...years);

  const maxGross = Math.max(...eligible.map((m) => m.worldwide_gross!));

  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

  const stars: Star[] = eligible.map((m, i) => {
    const pred = honest(m);
    const fieldX = budgetRank[i] * 2 * X_RANGE - X_RANGE;
    const fieldY =
      clamp01((logGross[i] - gMin) / (gMax - gMin)) * 2 * Y_RANGE - Y_RANGE;

    // Per-year error columns: only movies the model was graded on spread out.
    const yearSpan = Math.max(maxYear - minYear, 1);
    const yearX =
      ((m.release_year - minYear) / yearSpan) * 2 * X_RANGE - X_RANGE;
    const logRatio = pred
      ? Math.max(
          -1.2,
          Math.min(1.2, Math.log(pred.predicted_gross / pred.actual_gross!)),
        )
      : 0;
    const yearY = pred ? (logRatio / 1.2) * Y_RANGE * 0.9 : -0.95;

    const yScale = (2 * Y_RANGE) / (gMax - gMin);
    const predictedFieldY = pred
      ? fieldY +
        (Math.log10(pred.predicted_gross) - Math.log10(pred.actual_gross!)) *
          yScale
      : null;

    const roi = m.worldwide_gross! / m.production_budget!;
    const intensity = Math.max(0.3, Math.min(1, 0.45 + Math.log10(Math.max(roi, 0.05)) * 0.4));
    const size = 2.8 + Math.sqrt(m.worldwide_gross! / maxGross) * 5;

    return {
      tmdbId: m.tmdb_id,
      title: m.title,
      releaseYear: m.release_year,
      gross: m.worldwide_gross!,
      budget: m.production_budget!,
      posterPath: m.poster_path,
      fieldX,
      fieldY,
      yearX,
      yearY,
      hasPrediction: pred != null,
      predicted: pred?.predicted_gross ?? null,
      predictedFieldY,
      intensity,
      size,
    };
  });

  // Featured star: biggest actual gross among predicted movies — a name everyone knows.
  let featuredIndex = 0;
  let best = -1;
  stars.forEach((s, i) => {
    if (s.hasPrediction && s.gross > best) {
      best = s.gross;
      featuredIndex = i;
    }
  });

  return { stars, minYear, maxYear, featuredIndex };
}

/** Coarse spatial hash over normalized coords for O(1) hover hit-tests. */
export class StarGrid {
  private cells = new Map<string, number[]>();
  private cellSize: number;

  constructor(
    private stars: Star[],
    cellSize = 0.05,
  ) {
    this.cellSize = cellSize;
    stars.forEach((s, i) => {
      const key = this.key(s.fieldX, s.fieldY);
      const bucket = this.cells.get(key);
      if (bucket) bucket.push(i);
      else this.cells.set(key, [i]);
    });
  }

  private key(x: number, y: number): string {
    return `${Math.floor(x / this.cellSize)}:${Math.floor(y / this.cellSize)}`;
  }

  /** Nearest star index within `radius` of (x, y) in normalized coords, or -1. */
  nearest(x: number, y: number, radius = 0.04): number {
    const r = Math.ceil(radius / this.cellSize);
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    let bestIdx = -1;
    let bestD = radius * radius;
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        const bucket = this.cells.get(`${cx + dx}:${cy + dy}`);
        if (!bucket) continue;
        for (const i of bucket) {
          const s = this.stars[i];
          const d = (s.fieldX - x) ** 2 + (s.fieldY - y) ** 2;
          if (d < bestD) {
            bestD = d;
            bestIdx = i;
          }
        }
      }
    }
    return bestIdx;
  }

  /** Nearest star in a direction (unit dx/dy) from star `from` — keyboard walk. */
  neighbor(from: number, dx: number, dy: number): number {
    const origin = this.stars[from];
    let bestIdx = -1;
    let bestScore = Infinity;
    this.stars.forEach((s, i) => {
      if (i === from) return;
      const vx = s.fieldX - origin.fieldX;
      const vy = s.fieldY - origin.fieldY;
      const along = vx * dx + vy * dy;
      if (along <= 0.001) return;
      const ortho = Math.abs(vx * dy - vy * dx);
      const score = along + ortho * 3;
      if (score < bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    });
    return bestIdx;
  }
}

/** 0..1 rank position per value (ties keep insertion order — fine for layout). */
function rankPositions(values: number[]): number[] {
  const order = values
    .map((v, i) => [v, i] as const)
    .sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(values.length);
  order.forEach(([, originalIndex], rank) => {
    ranks[originalIndex] = rank / Math.max(values.length - 1, 1);
  });
  return ranks;
}

function percentileBounds(
  values: number[],
  lo: number,
  hi: number,
): [number, number] {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))];
  return [at(lo), at(hi)];
}

export function easeOutQuint(t: number): number {
  return 1 - Math.pow(1 - t, 5);
}

/** Map scroll progress p through [a, b] to eased 0..1. */
export function segment(p: number, a: number, b: number): number {
  if (p <= a) return 0;
  if (p >= b) return 1;
  return easeOutQuint((p - a) / (b - a));
}
