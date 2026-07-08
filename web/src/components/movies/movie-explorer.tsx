"use client";

import { useDeferredValue, useMemo, useState } from "react";
import type { CatalogMovie } from "@/lib/catalog";
import { roi } from "@/lib/catalog";
import { PosterCard } from "@/components/movies/poster-card";
import { MoviesTable } from "@/components/movies/movies-table";

const SORTS = {
  "gross-desc": {
    label: "Gross, high to low",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      (b.worldwide_gross ?? -1) - (a.worldwide_gross ?? -1),
  },
  "budget-desc": {
    label: "Budget, high to low",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      (b.production_budget ?? 0) - (a.production_budget ?? 0),
  },
  "roi-desc": {
    label: "Return on budget",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      (roi(b) ?? -1) - (roi(a) ?? -1),
  },
  "year-desc": {
    label: "Newest first",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      b.release_year - a.release_year ||
      (b.worldwide_gross ?? -1) - (a.worldwide_gross ?? -1),
  },
  "year-asc": {
    label: "Oldest first",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      a.release_year - b.release_year ||
      (b.worldwide_gross ?? -1) - (a.worldwide_gross ?? -1),
  },
  "title-asc": {
    label: "Title, A to Z",
    compare: (a: CatalogMovie, b: CatalogMovie) =>
      a.title.localeCompare(b.title),
  },
} as const;

type SortKey = keyof typeof SORTS;

const GRID_PAGE = 60;

export function MovieExplorer({ movies }: { movies: CatalogMovie[] }) {
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState<string | null>(null);
  const [decade, setDecade] = useState<number | null>(null);
  const [sort, setSort] = useState<SortKey>("gross-desc");
  const [view, setView] = useState<"grid" | "table">("grid");
  const [visible, setVisible] = useState(GRID_PAGE);
  const deferredQuery = useDeferredValue(query);

  const genres = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of movies) {
      for (const g of m.genres) counts.set(g, (counts.get(g) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([g]) => g);
  }, [movies]);

  const decades = useMemo(() => {
    const set = new Set(movies.map((m) => Math.floor(m.release_year / 10) * 10));
    return [...set].sort((a, b) => b - a);
  }, [movies]);

  const filtered = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    return movies
      .filter((m) => {
        if (genre && !m.genres.includes(genre)) return false;
        if (decade !== null && Math.floor(m.release_year / 10) * 10 !== decade)
          return false;
        if (
          q &&
          !m.title.toLowerCase().includes(q) &&
          !(m.director ?? "").toLowerCase().includes(q)
        )
          return false;
        return true;
      })
      .sort(SORTS[sort].compare);
  }, [movies, deferredQuery, genre, decade, sort]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setVisible(GRID_PAGE);
          }}
          placeholder="Search titles and directors"
          aria-label="Search titles and directors"
          className="h-9 w-64 rounded border border-hairline bg-surface px-3 text-sm text-ink placeholder:text-dim focus:border-actual-deep focus:outline-none"
        />
        <select
          value={decade ?? ""}
          onChange={(e) => {
            setDecade(e.target.value === "" ? null : Number(e.target.value));
            setVisible(GRID_PAGE);
          }}
          aria-label="Filter by decade"
          className="h-9 rounded border border-hairline bg-surface px-2 text-sm text-ink"
        >
          <option value="">All decades</option>
          {decades.map((d) => (
            <option key={d} value={d}>
              {d}s
            </option>
          ))}
        </select>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as SortKey)}
          aria-label="Sort order"
          className="h-9 rounded border border-hairline bg-surface px-2 text-sm text-ink"
        >
          {Object.entries(SORTS).map(([key, { label }]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
        <div
          role="group"
          aria-label="View"
          className="ml-auto flex overflow-hidden rounded border border-hairline"
        >
          {(["grid", "table"] as const).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              aria-pressed={view === v}
              className={`px-3 py-1.5 text-sm capitalize transition-colors duration-150 ${
                view === v
                  ? "bg-surface-2 text-actual"
                  : "bg-surface text-dim hover:text-ink"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Genre">
        <GenreChip
          label="All genres"
          active={genre === null}
          onClick={() => {
            setGenre(null);
            setVisible(GRID_PAGE);
          }}
        />
        {genres.map((g) => (
          <GenreChip
            key={g}
            label={g}
            active={genre === g}
            onClick={() => {
              setGenre(genre === g ? null : g);
              setVisible(GRID_PAGE);
            }}
          />
        ))}
      </div>

      <p aria-live="polite" className="font-mono text-xs text-dim">
        {filtered.length.toLocaleString("en-US")} movies
      </p>

      {view === "grid" ? (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {filtered.slice(0, visible).map((m) => (
              <PosterCard
                key={m.tmdb_id}
                tmdbId={m.tmdb_id}
                title={m.title}
                releaseYear={m.release_year}
                posterPath={m.poster_path}
                worldwideGross={m.worldwide_gross}
              />
            ))}
          </div>
          {filtered.length > visible && (
            <button
              type="button"
              onClick={() => setVisible((v) => v + GRID_PAGE * 2)}
              className="mx-auto rounded border border-hairline px-4 py-2 text-sm text-ink transition-colors duration-150 hover:bg-surface"
            >
              Show more ({(filtered.length - visible).toLocaleString("en-US")}{" "}
              remaining)
            </button>
          )}
          {filtered.length === 0 && (
            <p className="py-12 text-center text-dim">
              No movies match. Loosen a filter — the model would.
            </p>
          )}
        </>
      ) : (
        <MoviesTable movies={filtered} />
      )}
    </div>
  );
}

function GenreChip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-full border px-3 py-1 text-xs transition-colors duration-150 ${
        active
          ? "border-actual-deep bg-actual/10 text-actual"
          : "border-hairline text-dim hover:border-actual-deep/50 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}
