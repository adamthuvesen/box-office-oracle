"use client";

import { useEffect, useMemo } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "motion/react";
import type { ComparableMovie } from "@/lib/predict";
import { posterUrl, type PredictionResult } from "@/lib/types";
import { dollarsCompact } from "@/lib/format";
import { PosterFallback } from "@/components/movies/poster-card";
import { ProjectorBeam } from "./projector-beam";
import { OracleError } from "./api";

const EASE_ENTER = [0.22, 1, 0.36, 1] as const;

export type OracleStatus = "idle" | "pending" | "result" | "error";

interface RevealStageProps {
  status: OracleStatus;
  result: PredictionResult | null;
  error: Error | null;
  revealKey: number;
  catalog: ComparableMovie[];
  liveApi: boolean;
  onRetry: () => void;
}

export function RevealStage({
  status,
  result,
  error,
  revealKey,
  catalog,
  liveApi,
  onRetry,
}: RevealStageProps) {
  return (
    <section aria-live="polite" className="lg:sticky lg:top-20 lg:self-start">
      {status === "idle" && (
        <div className="flex min-h-72 items-center justify-center rounded border border-dashed border-hairline p-8">
          <p className="text-dim">The oracle is waiting.</p>
        </div>
      )}
      {status === "pending" && (
        <div className="flex min-h-72 flex-col items-center justify-center gap-2 rounded border border-hairline bg-surface p-8">
          <p className="animate-pulse text-dim">Consulting…</p>
          {liveApi && (
            <p className="text-center text-xs text-dim">
              The model runs on a Lambda — a cold start can take a few seconds.
            </p>
          )}
        </div>
      )}
      {status === "error" && <OracleFault error={error} onRetry={onRetry} />}
      {status === "result" && result && (
        <Reveal key={revealKey} result={result} catalog={catalog} />
      )}
    </section>
  );
}

function OracleFault({
  error,
  onRetry,
}: {
  error: Error | null;
  onRetry: () => void;
}) {
  const fault = error instanceof OracleError ? error : null;
  const warming = fault?.status === 503 || fault?.code === "NO_MODEL_AVAILABLE";
  return (
    <div className="flex min-h-72 flex-col items-start justify-center gap-3 rounded border border-hairline bg-surface p-8">
      <h2 className="title-caps text-sm text-ink">
        The oracle didn&apos;t answer.
      </h2>
      <p className="text-sm text-dim">
        {warming
          ? "The projector is warming up — try again in a moment."
          : (error?.message ?? "Something went wrong.")}
      </p>
      {warming && error?.message && (
        <p className="font-mono text-xs text-dim">{error.message}</p>
      )}
      <button
        type="button"
        onClick={onRetry}
        className="mt-1 rounded border border-hairline px-4 py-2 text-sm text-ink transition-colors duration-150 hover:bg-surface-2"
      >
        Ask again
      </button>
    </div>
  );
}

const metaList = {
  hidden: {},
  show: { transition: { delayChildren: 0.35, staggerChildren: 0.06 } },
};

const metaItem = {
  hidden: { opacity: 0, y: 4 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: EASE_ENTER },
  },
};

/** The signature moment — mounts fresh per result (keyed), animates once. */
function Reveal({
  result,
  catalog,
}: {
  result: PredictionResult;
  catalog: ComparableMovie[];
}) {
  const interval = result.prediction_interval_heuristic;

  return (
    <div className="flex flex-col gap-6 rounded border border-hairline bg-surface p-6">
      <div>
        <p className="text-xs uppercase tracking-wider text-dim">
          The oracle says
        </p>
        <Odometer value={result.prediction} />
        <p className="sr-only">
          Predicted worldwide gross {dollarsCompact(result.prediction)}
        </p>
      </div>

      {interval && (
        <div>
          <ProjectorBeam
            prediction={result.prediction}
            lower={interval[0]}
            upper={interval[1]}
          />
          <p className="mt-1 text-xs text-dim">
            The interval is a heuristic, not a calibrated confidence bound.
          </p>
        </div>
      )}

      <motion.ul
        variants={metaList}
        initial="hidden"
        animate="show"
        className="flex flex-col gap-2 border-t border-hairline pt-4"
      >
        {interval && (
          <motion.li variants={metaItem} className="text-sm text-dim">
            Between{" "}
            <span className="font-mono tabular text-predicted">
              {dollarsCompact(interval[0])}
            </span>{" "}
            and{" "}
            <span className="font-mono tabular text-predicted">
              {dollarsCompact(interval[1])}
            </span>
            , probably.
          </motion.li>
        )}
        <motion.li variants={metaItem} className="font-mono text-xs text-dim">
          {result.mock ? (
            <span className="inline-block rounded-full border border-actual-deep px-2.5 py-0.5 text-actual">
              mock oracle — set INFERENCE_API_URL for the real one
            </span>
          ) : (
            <>
              {result.model_id} · v{result.model_version}
            </>
          )}
        </motion.li>
        <motion.li variants={metaItem} className="font-mono text-xs text-dim">
          answered in {Math.round(result.processing_time_ms)} ms
        </motion.li>
      </motion.ul>

      <Comparables catalog={catalog} prediction={result.prediction} />
    </div>
  );
}

/** Rolling count-up: animates the raw number, formats through dollarsCompact. */
function Odometer({ value }: { value: number }) {
  const reduced = useReducedMotion();
  const raw = useMotionValue(reduced ? value : 0);
  const display = useTransform(raw, (v) => dollarsCompact(v));

  useEffect(() => {
    if (reduced) {
      raw.set(value);
      return;
    }
    const controls = animate(raw, value, { duration: 0.6, ease: "easeOut" });
    return () => controls.stop();
  }, [raw, value, reduced]);

  return (
    <motion.p
      aria-hidden
      className="mt-1 font-mono tabular text-5xl text-predicted sm:text-6xl"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {display}
    </motion.p>
  );
}

const stripList = {
  hidden: {},
  show: { transition: { delayChildren: 0.6, staggerChildren: 0.06 } },
};

const stripItem = {
  hidden: { opacity: 0, y: 6 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: EASE_ENTER },
  },
};

function Comparables({
  catalog,
  prediction,
}: {
  catalog: ComparableMovie[];
  prediction: number;
}) {
  const picks = useMemo(
    () => nearestByGross(catalog, prediction, 5),
    [catalog, prediction],
  );
  if (picks.length === 0) return null;

  return (
    <div className="border-t border-hairline pt-4">
      <h3 className="title-caps text-xs text-dim">
        Movies that made about this much
      </h3>
      <motion.ul
        variants={stripList}
        initial="hidden"
        animate="show"
        className="mt-3 grid grid-cols-5 gap-2 sm:gap-3"
      >
        {picks.map((m) => {
          const src = posterUrl(m.poster_path, "w185");
          return (
            <motion.li key={m.tmdb_id} variants={stripItem}>
              <Link href={`/movies/${m.tmdb_id}`} className="group block">
                <div className="relative aspect-2/3 w-full overflow-hidden rounded border border-hairline bg-surface-2 transition-transform duration-150 ease-enter group-hover:-translate-y-0.5">
                  {src ? (
                    <Image
                      src={src}
                      alt={`${m.title} poster`}
                      fill
                      sizes="(max-width: 640px) 18vw, 100px"
                      className="object-cover"
                    />
                  ) : (
                    <PosterFallback title={m.title} releaseYear={m.release_year} />
                  )}
                </div>
                <p className="mt-1.5 truncate text-xs text-dim transition-colors duration-150 group-hover:text-ink">
                  {m.title} made{" "}
                  <span className="font-mono tabular text-actual">
                    {dollarsCompact(m.worldwide_gross)}
                  </span>
                </p>
              </Link>
            </motion.li>
          );
        })}
      </motion.ul>
    </div>
  );
}

/** The 5 catalog movies whose gross is nearest the prediction, in log space. */
function nearestByGross(
  catalog: ComparableMovie[],
  prediction: number,
  count: number,
): ComparableMovie[] {
  const target = Math.log(Math.max(prediction, 1));
  return catalog
    .map((m) => ({ m, d: Math.abs(Math.log(m.worldwide_gross) - target) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, count)
    .map(({ m }) => m);
}
