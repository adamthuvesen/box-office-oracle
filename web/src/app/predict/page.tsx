import type { Metadata } from "next";
import { Suspense } from "react";
import { loadMovies } from "@/lib/data";
import type { Movie } from "@/lib/types";
import type { ComparableMovie } from "@/lib/predict";
import { Oracle } from "@/components/oracle/oracle";

export const metadata: Metadata = {
  title: "Predict",
};

export default function PredictPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="title-caps text-2xl text-ink">Ask the Oracle</h1>
        <p className="mt-2 max-w-prose text-dim">
          Give the model what a studio knows before opening weekend. It will
          tell you what it thinks.
        </p>
      </header>
      <Suspense fallback={<OracleSkeleton />}>
        <OracleLive />
      </Suspense>
    </div>
  );
}

async function OracleLive() {
  const movies = await loadMovies();
  const liveApi = Boolean(
    process.env.INFERENCE_API_URL && process.env.INFERENCE_API_KEY,
  );
  // The oracle works without the catalog snapshot — comparables just stay empty.
  return (
    <Oracle catalog={movies ? slimCatalog(movies) : []} liveApi={liveApi} />
  );
}

const CATALOG_CAP = 800;

/**
 * Slim the catalog for the comparables strip: grossing movies only, posters
 * preferred, rank-sampled to ~800 so the client payload stays small while
 * still covering the whole gross range.
 */
function slimCatalog(movies: Movie[]): ComparableMovie[] {
  const eligible = movies.filter(
    (m): m is Movie & { worldwide_gross: number } =>
      m.worldwide_gross != null && m.worldwide_gross > 0,
  );
  const withPosters = eligible.filter((m) => m.poster_path);
  const pool = withPosters.length >= CATALOG_CAP ? withPosters : eligible;
  const sorted = [...pool].sort((a, b) => a.worldwide_gross - b.worldwide_gross);
  const sampled =
    sorted.length <= CATALOG_CAP
      ? sorted
      : Array.from(
          { length: CATALOG_CAP },
          (_, i) => sorted[Math.floor((i * sorted.length) / CATALOG_CAP)],
        );
  return sampled.map((m) => ({
    tmdb_id: m.tmdb_id,
    title: m.title,
    release_year: m.release_year,
    poster_path: m.poster_path,
    worldwide_gross: m.worldwide_gross,
  }));
}

function OracleSkeleton() {
  return (
    <div className="grid gap-10 lg:grid-cols-2 lg:gap-12">
      <div className="flex flex-col gap-6">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="h-16 animate-pulse rounded border border-hairline bg-surface"
          />
        ))}
      </div>
      <div className="min-h-72 animate-pulse rounded border border-hairline bg-surface" />
    </div>
  );
}
