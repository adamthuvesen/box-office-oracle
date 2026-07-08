"use client";

import { ViewTransition } from "react";
import Image from "next/image";
import Link from "next/link";
import { posterUrl } from "@/lib/types";
import { dollarsCompact } from "@/lib/format";

interface PosterCardProps {
  tmdbId: number;
  title: string;
  releaseYear: number;
  posterPath: string | null;
  worldwideGross: number | null;
  sizes?: string;
}

export function PosterCard({
  tmdbId,
  title,
  releaseYear,
  posterPath,
  worldwideGross,
  sizes = "(max-width: 640px) 45vw, (max-width: 1024px) 30vw, 190px",
}: PosterCardProps) {
  const src = posterUrl(posterPath, "w342");
  return (
    <Link
      href={`/movies/${tmdbId}`}
      className="group relative block overflow-hidden rounded border border-hairline bg-surface transition-transform duration-150 ease-enter hover:-translate-y-0.5"
    >
      <ViewTransition name={`poster-${tmdbId}`}>
        <div className="relative aspect-2/3 w-full">
          {src ? (
            <Image
              src={src}
              alt={`${title} poster`}
              fill
              sizes={sizes}
              className="object-cover"
            />
          ) : (
            <PosterFallback title={title} releaseYear={releaseYear} />
          )}
        </div>
      </ViewTransition>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 translate-y-1 bg-linear-to-t from-screen via-screen/80 to-transparent p-3 pt-10 opacity-0 transition-all duration-150 ease-enter group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100">
        <p className="truncate text-sm font-medium text-ink">{title}</p>
        <p className="font-mono tabular text-xs text-actual">
          {worldwideGross != null ? dollarsCompact(worldwideGross) : "—"}
          <span className="text-dim"> · {releaseYear}</span>
        </p>
      </div>
    </Link>
  );
}

/** Missing poster: a designed title card, not a gray box. */
export function PosterFallback({
  title,
  releaseYear,
}: {
  title: string;
  releaseYear: number;
}) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 border-b border-hairline bg-surface-2 px-4 text-center">
      <span
        aria-hidden
        className="h-px w-8 bg-actual-deep"
      />
      <span className="title-caps text-sm leading-snug text-ink [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:4] overflow-hidden">
        {title}
      </span>
      <span className="font-mono text-xs text-dim">{releaseYear}</span>
    </div>
  );
}
