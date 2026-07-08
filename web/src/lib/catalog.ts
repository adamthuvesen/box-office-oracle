import type { Movie } from "@/lib/types";

/** The slice of a movie the explorer needs — keeps the RSC→client payload small. */
export interface CatalogMovie {
  tmdb_id: number;
  title: string;
  release_year: number;
  genres: string[];
  director: string | null;
  mpaa: string | null;
  runtime: number | null;
  production_budget: number | null;
  worldwide_gross: number | null;
  poster_path: string | null;
}

export function toCatalogMovie(m: Movie): CatalogMovie {
  return {
    tmdb_id: m.tmdb_id,
    title: m.title,
    release_year: m.release_year,
    genres: m.genres,
    director: m.director,
    mpaa: m.mpaa,
    runtime: m.runtime,
    production_budget: m.production_budget,
    worldwide_gross: m.worldwide_gross,
    poster_path: m.poster_path,
  };
}

/** Return on budget: gross / budget. Null when either side is missing. */
export function roi(m: {
  production_budget: number | null;
  worldwide_gross: number | null;
}): number | null {
  if (!m.production_budget || m.production_budget <= 0 || m.worldwide_gross == null) {
    return null;
  }
  return m.worldwide_gross / m.production_budget;
}

/** Same lead genre, closest budget — the "if you liked this bet" strip. */
export function similarMovies(
  movie: Movie,
  all: Movie[],
  count = 6,
): Movie[] {
  const lead = movie.genres[0];
  const budget = movie.production_budget ?? 0;
  return all
    .filter((m) => m.tmdb_id !== movie.tmdb_id && (!lead || m.genres[0] === lead))
    .map((m) => ({
      m,
      distance:
        Math.abs(Math.log1p(m.production_budget ?? 0) - Math.log1p(budget)) +
        Math.abs(m.release_year - movie.release_year) * 0.05,
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, count)
    .map(({ m }) => m);
}
