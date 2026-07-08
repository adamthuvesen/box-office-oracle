import type { Metadata } from "next";
import { Suspense } from "react";
import { loadMovies } from "@/lib/data";
import { toCatalogMovie } from "@/lib/catalog";
import { MovieExplorer } from "@/components/movies/movie-explorer";
import { NoDataYet } from "@/components/empty-state";

export const metadata: Metadata = {
  title: "Movies",
};

export default function MoviesPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="title-caps text-2xl text-ink">The Catalog</h1>
        <p className="mt-2 text-dim">
          Every movie the model has studied — pre-release facts on the left,
          what actually happened on the right.
        </p>
      </header>
      <Suspense fallback={<ExplorerSkeleton />}>
        <ExplorerLive />
      </Suspense>
    </div>
  );
}

async function ExplorerLive() {
  const movies = await loadMovies();
  if (!movies) return <NoDataYet what="the movie catalog" />;
  return <MovieExplorer movies={movies.map(toCatalogMovie)} />;
}

function ExplorerSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          className="aspect-2/3 animate-pulse rounded border border-hairline bg-surface"
        />
      ))}
    </div>
  );
}
