import type { Metadata } from "next";
import { Suspense, ViewTransition } from "react";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { loadMovie, loadMovies, loadOraclePredictions } from "@/lib/data";
import { similarMovies } from "@/lib/catalog";
import { backdropUrl, posterUrl } from "@/lib/types";
import {
  dollarsCompact,
  releaseDateLabel,
  runtimeLabel,
} from "@/lib/format";
import { PosterFallback } from "@/components/movies/poster-card";
import { PredictionPanel } from "@/components/movies/prediction-panel";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const movie = await loadMovie(Number(id));
  return { title: movie?.title ?? "Movie" };
}

export default function MoviePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  return (
    <Suspense fallback={<MovieSkeleton />}>
      <MovieContent params={params} />
    </Suspense>
  );
}

function MovieSkeleton() {
  return (
    <div className="mx-auto flex max-w-6xl gap-6 px-6 pt-24 pb-10">
      <div className="aspect-2/3 w-36 animate-pulse rounded border border-hairline bg-surface sm:w-44" />
      <div className="flex flex-col gap-3 self-end pb-2">
        <div className="h-3 w-40 animate-pulse rounded bg-surface" />
        <div className="h-9 w-72 animate-pulse rounded bg-surface" />
        <div className="h-5 w-32 animate-pulse rounded bg-surface" />
      </div>
    </div>
  );
}

async function MovieContent({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const tmdbId = Number(id);
  if (!Number.isInteger(tmdbId)) notFound();

  const [movie, movies, oraclePredictions] = await Promise.all([
    loadMovie(tmdbId),
    loadMovies(),
    loadOraclePredictions(),
  ]);
  if (!movie || !movies) notFound();

  const oracle = oraclePredictions?.[String(tmdbId)] ?? null;
  const similar = similarMovies(movie, movies);
  const backdrop = backdropUrl(movie.backdrop_path);
  const poster = posterUrl(movie.poster_path, "w500");

  const facts: Array<[string, string | null]> = [
    ["Director", movie.director],
    ["Studio", movie.production_company],
    ["Released", releaseDateLabel(movie.release_date)],
    ["Runtime", runtimeLabel(movie.runtime)],
    ["Rating", movie.mpaa],
    [
      "Budget",
      movie.production_budget ? dollarsCompact(movie.production_budget) : "unknown",
    ],
  ];

  return (
    <article>
      {/* Title card: duotone-graded backdrop, wide-caps title. */}
      <header className="relative overflow-hidden border-b border-hairline">
        {backdrop && (
          <>
            <Image
              src={backdrop}
              alt=""
              fill
              priority
              sizes="100vw"
              className="object-cover object-top opacity-40 grayscale-60 saturate-80"
            />
            <div className="absolute inset-0 bg-linear-to-t from-screen via-screen/60 to-screen/30" />
            <div className="absolute inset-0 bg-actual-deep/10 mix-blend-color" />
          </>
        )}
        <div className="relative mx-auto flex max-w-6xl flex-col gap-6 px-6 pt-24 pb-10 sm:flex-row sm:items-end">
          <ViewTransition name={`poster-${movie.tmdb_id}`}>
            <div className="relative aspect-2/3 w-36 shrink-0 overflow-hidden rounded border border-hairline shadow-2xl sm:w-44">
              {poster ? (
                <Image
                  src={poster}
                  alt={`${movie.title} poster`}
                  fill
                  priority
                  sizes="176px"
                  className="object-cover"
                />
              ) : (
                <PosterFallback
                  title={movie.title}
                  releaseYear={movie.release_year}
                />
              )}
            </div>
          </ViewTransition>
          <div className="min-w-0">
            <p className="font-mono text-xs text-dim">
              {movie.release_year}
              {movie.genres.length > 0 && ` · ${movie.genres.join(" / ")}`}
            </p>
            <h1 className="title-caps mt-2 text-3xl leading-tight text-ink sm:text-4xl">
              {movie.title}
            </h1>
            {movie.tagline && (
              <p className="mt-3 max-w-xl italic text-dim">{movie.tagline}</p>
            )}
            <p className="mt-4 font-mono tabular text-lg text-actual">
              {movie.worldwide_gross != null ? (
                <>
                  {dollarsCompact(movie.worldwide_gross)}
                  <span className="ml-2 text-xs text-dim">worldwide gross</span>
                </>
              ) : (
                <span className="text-dim">no worldwide gross yet</span>
              )}
            </p>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-6xl gap-10 px-6 py-10 lg:grid-cols-[2fr_1fr]">
        <div className="flex min-w-0 flex-col gap-10">
          <PredictionPanel oracle={oracle} />

          {movie.overview && (
            <section>
              <h2 className="title-caps mb-3 text-sm text-dim">The pitch</h2>
              <p className="max-w-prose leading-relaxed text-ink">
                {movie.overview}
              </p>
            </section>
          )}

          {movie.actors.length > 0 && (
            <section>
              <h2 className="title-caps mb-3 text-sm text-dim">Billing</h2>
              <p className="max-w-prose text-dim">
                {movie.actors.slice(0, 8).join(" · ")}
              </p>
            </section>
          )}
        </div>

        <aside className="flex flex-col gap-6">
          <dl className="flex flex-col gap-3 rounded border border-hairline bg-surface p-4">
            {facts
              .filter(([, value]) => value != null)
              .map(([label, value]) => (
                <div key={label} className="flex justify-between gap-4 text-sm">
                  <dt className="text-dim">{label}</dt>
                  <dd className="text-right font-medium text-ink">{value}</dd>
                </div>
              ))}
          </dl>
          {movie.keywords.length > 0 && (
            <p className="text-xs leading-relaxed text-dim">
              {movie.keywords.slice(0, 12).join(" · ")}
            </p>
          )}
        </aside>
      </div>

      {similar.length > 0 && (
        <section className="mx-auto max-w-6xl px-6 pb-16">
          <h2 className="title-caps mb-4 text-sm text-dim">
            Similar bets{movie.genres[0] ? ` in ${movie.genres[0]}` : ""}
          </h2>
          <div className="grid grid-cols-3 gap-4 sm:grid-cols-6">
            {similar.map((m) => (
              <Link
                key={m.tmdb_id}
                href={`/movies/${m.tmdb_id}`}
                className="group block"
              >
                <div className="relative aspect-2/3 overflow-hidden rounded border border-hairline bg-surface transition-transform duration-150 ease-enter group-hover:-translate-y-0.5">
                  {posterUrl(m.poster_path, "w185") ? (
                    <Image
                      src={posterUrl(m.poster_path, "w185")!}
                      alt={`${m.title} poster`}
                      fill
                      sizes="150px"
                      className="object-cover"
                    />
                  ) : (
                    <PosterFallback
                      title={m.title}
                      releaseYear={m.release_year}
                    />
                  )}
                </div>
                <p className="mt-2 truncate text-xs text-dim group-hover:text-ink">
                  {m.title}
                </p>
              </Link>
            ))}
          </div>
        </section>
      )}
    </article>
  );
}
