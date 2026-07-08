import type { Metadata } from "next";
import { Suspense } from "react";
import { loadModelMeta, loadMovies, loadOraclePredictions } from "@/lib/data";
import type { BacktestYear } from "@/lib/types";
import { percent, ratio } from "@/lib/format";
import { NoDataYet } from "@/components/empty-state";
import { BacktestBars } from "@/components/model/backtest-bars";
import { PredictionScatter } from "@/components/model/prediction-scatter";
import { LiveModelCard } from "@/components/model/live-model-card";

export const metadata: Metadata = {
  title: "Model",
};

export default function ModelPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-14">
        <h1 className="title-caps text-2xl text-ink">The Report Card</h1>
        <Suspense
          fallback={
            <div className="mt-2 h-5 w-full max-w-prose animate-pulse rounded bg-surface" />
          }
        >
          <Intro />
        </Suspense>
      </header>

      <Suspense fallback={<ReportSkeleton />}>
        <ReportLive />
      </Suspense>

      <section className="mt-20">
        <h2 className="title-caps text-sm text-dim">In the projector now</h2>
        <p className="mt-2 max-w-prose text-sm text-dim">
          The model currently loaded in the inference API — fetched live, not
          from the data snapshot.
        </p>
        <LiveModelCard />
      </section>
    </div>
  );
}

async function Intro() {
  const meta = await loadModelMeta();
  if (!meta || meta.per_year.length === 0) return null;
  const firstYear = Math.min(...meta.per_year.map((y) => y.year));
  return (
    <p className="mt-2 max-w-prose text-dim">
      Graded by an expanding-window backtest: for each year, the model trains
      only on earlier releases, then guesses that year&apos;s slate cold.
      Blind, out-of-fold predictions since {firstYear} — the model is never
      graded on a movie it had seen.
    </p>
  );
}

async function ReportLive() {
  const [meta, movies, oraclePredictions] = await Promise.all([
    loadModelMeta(),
    loadMovies(),
    loadOraclePredictions(),
  ]);
  if (!meta || meta.per_year.length === 0 || !movies || !oraclePredictions) {
    return <NoDataYet what="the model backtest" />;
  }

  // Honest guesses only: out-of-sample predictions with a real gross to grade.
  const guesses = movies.flatMap((m) => {
    const p = oraclePredictions[String(m.tmdb_id)];
    if (
      !p ||
      p.prediction_kind !== "out_of_sample" ||
      p.actual_gross == null ||
      p.ape == null
    ) {
      return [];
    }
    return [
      {
        tmdb_id: m.tmdb_id,
        title: m.title,
        release_year: m.release_year,
        y_true: p.actual_gross,
        y_pred: p.predicted_gross,
        ape: p.ape,
      },
    ];
  });

  const years = meta.per_year;
  const totalMovies = years.reduce((sum, y) => sum + y.n_movies, 0);
  const weighted = (pick: (y: BacktestYear) => number) =>
    years.reduce((sum, y) => sum + pick(y) * y.n_movies, 0) / totalMovies;

  const medianApe = weighted((y) => y.model_median_ape);
  const avgSpearman = weighted((y) => y.model_spearman);
  const pooledR2Log = weighted((y) => y.model_r2_log);
  const avgGain =
    years.reduce((sum, y) => sum + y.gain_r2_log, 0) / years.length;

  const lostYears = years.filter((y) => y.gain_r2_log < 0).map((y) => y.year);
  const covidYear = lostYears.includes(2020);

  return (
    <div className="flex flex-col gap-20">
      <section>
        <h2 className="title-caps text-sm text-dim">
          Model vs baseline, year by year
        </h2>
        <p className="mt-2 max-w-prose text-sm text-dim">
          R² on log dollars, one pair of bars per backtest year. The ghost
          outline is the baseline; the solid bar is the model.
        </p>
        <BacktestBars years={years} />
        <p className="mt-4 max-w-prose text-sm text-dim">
          The model beats the baseline in {years.length - lostYears.length} of{" "}
          {years.length} years, gaining{" "}
          <span className="font-mono tabular text-ink">
            +{ratio(avgGain)}
          </span>{" "}
          R² (log) on average.
          {covidYear && (
            <>
              {" "}
              The exception —{" "}
              <span className="font-mono text-under">2020</span>: COVID broke
              everyone&apos;s model.
            </>
          )}
        </p>
        <details className="mt-4">
          <summary className="cursor-pointer text-xs text-dim hover:text-ink">
            View as table
          </summary>
          <div className="mt-3 overflow-x-auto rounded border border-hairline">
            <BacktestTable years={years} />
          </div>
        </details>
      </section>

      <section>
        <h2 className="title-caps text-sm text-dim">
          Blind guesses vs reality
        </h2>
        <p className="mt-2 max-w-prose text-sm text-dim">
          Every out-of-fold prediction the model has made —{" "}
          {guesses.length.toLocaleString("en-US")} guesses, log scale on
          both axes, so misses read as multiples rather than millions.
        </p>
        <PredictionScatter guesses={guesses} />
      </section>

      <section>
        <h2 className="title-caps text-sm text-dim">How wrong, typically</h2>
        <div className="mt-6 grid gap-8 sm:grid-cols-3 sm:gap-0 sm:divide-x sm:divide-hairline">
          <StatBlock
            figure={percent(medianApe)}
            caption={`Typically off by ${percent(medianApe)} of the real gross — median APE, weighted across backtest years.`}
          />
          <StatBlock
            figure={`ρ = ${ratio(avgSpearman)}`}
            caption="Gets the ranking right — average Spearman rank correlation, weighted by year size."
          />
          <StatBlock
            figure={ratio(pooledR2Log)}
            caption="Pooled R² on log dollars — the scale the model trains and is graded on."
          />
        </div>
      </section>
    </div>
  );
}

function StatBlock({ figure, caption }: { figure: string; caption: string }) {
  return (
    <div className="sm:px-8 sm:first:pl-0 sm:last:pr-0">
      <p className="font-mono tabular text-3xl text-ink">{figure}</p>
      <p className="mt-2 max-w-60 text-sm text-dim">{caption}</p>
    </div>
  );
}

function BacktestTable({ years }: { years: BacktestYear[] }) {
  return (
    <table className="w-full min-w-[42rem] text-left text-sm">
      <thead className="text-xs text-dim">
        <tr>
          <th className="px-3 py-2 font-normal">Year</th>
          <th className="px-3 py-2 text-right font-normal">Movies</th>
          <th className="px-3 py-2 text-right font-normal">
            Baseline R² (log)
          </th>
          <th className="px-3 py-2 text-right font-normal">Model R² (log)</th>
          <th className="px-3 py-2 text-right font-normal">Gain</th>
          <th className="px-3 py-2 text-right font-normal">Model ρ</th>
          <th className="px-3 py-2 text-right font-normal">Median APE</th>
        </tr>
      </thead>
      <tbody>
        {years.map((y) => (
          <tr key={y.year} className="border-t border-hairline">
            <td className="px-3 py-1.5 font-mono tabular text-ink">
              {y.year}
            </td>
            <td className="px-3 py-1.5 text-right font-mono tabular text-dim">
              {y.n_movies}
            </td>
            <td className="px-3 py-1.5 text-right font-mono tabular text-actual">
              {ratio(y.baseline_r2_log)}
            </td>
            <td className="px-3 py-1.5 text-right font-mono tabular text-predicted">
              {ratio(y.model_r2_log)}
            </td>
            <td
              className={`px-3 py-1.5 text-right font-mono tabular ${
                y.gain_r2_log >= 0 ? "text-over" : "text-under"
              }`}
            >
              {y.gain_r2_log >= 0 ? "+" : ""}
              {ratio(y.gain_r2_log)}
            </td>
            <td className="px-3 py-1.5 text-right font-mono tabular text-dim">
              {ratio(y.model_spearman)}
            </td>
            <td className="px-3 py-1.5 text-right font-mono tabular text-dim">
              {percent(y.model_median_ape)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReportSkeleton() {
  return (
    <div className="flex flex-col gap-20">
      {[0, 1].map((i) => (
        <div key={i}>
          <div className="h-4 w-64 animate-pulse rounded bg-surface" />
          <div className="mt-6 h-80 animate-pulse rounded border border-hairline bg-surface" />
        </div>
      ))}
      <div className="grid gap-8 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-20 animate-pulse rounded bg-surface" />
        ))}
      </div>
    </div>
  );
}
