import type { Metadata } from "next";
import { Suspense, type ReactNode } from "react";
import { loadMovies } from "@/lib/data";
import {
  budgetVsGross,
  genreEconomics,
  grossLeaders,
  returnBandSummary,
  seasonality,
  yearlyGross,
  MONTH_NAMES,
  type GrossLeader,
} from "@/lib/stats";
import { dollarsCompact, percent, ratio } from "@/lib/format";
import { NoDataYet } from "@/components/empty-state";
import { YearlyGrossChart } from "@/components/charts/yearly-gross-chart";
import { SeasonalityStrip } from "@/components/charts/seasonality-strip";
import { GenreEconomicsChart } from "@/components/charts/genre-economics-chart";
import { BudgetGrossScatter } from "@/components/charts/budget-gross-scatter";
import { ViewAsTable } from "@/components/charts/view-as-table";

export const metadata: Metadata = {
  title: "Stats",
};

export default function StatsPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-12">
        <h1 className="title-caps text-2xl text-ink">The Numbers</h1>
        <p className="mt-2 text-dim">
          What actually happened at the box office — the catalog cut by year,
          season, genre, budget, and studio. No predictions on this page.
        </p>
      </header>
      <Suspense fallback={<StatsSkeleton />}>
        <StatsLive />
      </Suspense>
    </div>
  );
}

const BAND_LABEL = {
  over: "Returned ≥ 2.5× budget",
  mid: "In between",
  under: "Grossed less than budget",
} as const;

async function StatsLive() {
  const movies = await loadMovies();
  if (!movies) return <NoDataYet what="the movie catalog" />;

  const years = yearlyGross(movies);
  const season = seasonality(movies);
  const genres = genreEconomics(movies);
  const points = budgetVsGross(movies);
  const bands = returnBandSummary(points);
  const studios = grossLeaders(movies, "production_company");
  const directors = grossLeaders(movies, "director");

  const bestYear = years.reduce((a, b) => (b.totalGross > a.totalGross ? b : a));
  const topGenre = genres[0];
  const over = bands.find((b) => b.band === "over")!;
  const under = bands.find((b) => b.band === "under")!;
  const monthTotals = season.grid.map((row) => row.reduce((a, b) => a + b, 0));

  return (
    <div className="flex flex-col gap-14">
      <Section
        title="Total gross by year"
        takeaway={
          <>
            <Figure>{bestYear.year}</Figure> is the biggest year in this
            sample: <Figure>{dollarsCompact(bestYear.totalGross)}</Figure>{" "}
            across <Figure>{bestYear.releases}</Figure> releases.
          </>
        }
        note="This catalog is a sample of wide releases, not the whole industry — totals undercount the real market."
      >
        <YearlyGrossChart data={years} />
        <ViewAsTable
          caption="Total worldwide gross and number of releases per year"
          headers={["Year", "Worldwide gross", "Releases"]}
          rows={years.map((y) => [
            y.year,
            dollarsCompact(y.totalGross),
            y.releases,
          ])}
        />
      </Section>

      <Section
        title="The seasonality strip"
        takeaway={
          <>
            <Figure>{season.bestMonth.name}</Figure> is the strongest month of
            the release calendar —{" "}
            <Figure>{dollarsCompact(season.bestMonth.total)}</Figure> across
            the sample.
          </>
        }
      >
        <SeasonalityStrip data={season} />
        <ViewAsTable
          caption="Total worldwide gross per calendar month, summed across all years"
          headers={["Month", "Worldwide gross, all years"]}
          rows={MONTH_NAMES.map((name, i) => [
            name,
            dollarsCompact(monthTotals[i]),
          ])}
        />
      </Section>

      <Section
        title="Genre economics"
        takeaway={
          topGenre && (
            <>
              <Figure>{topGenre.genre}</Figure> earns the highest median gross
              — <Figure>{dollarsCompact(topGenre.medianGross)}</Figure>
              {topGenre.medianBudget != null && (
                <>
                  {" "}
                  on a median{" "}
                  <Figure>{dollarsCompact(topGenre.medianBudget)}</Figure>{" "}
                  budget
                </>
              )}
              .
            </>
          )
        }
        note="Each movie counts once, under its first-listed genre; genres with fewer than 30 movies are dropped. Movies without a documented budget are excluded from the budget medians."
      >
        <GenreEconomicsChart data={genres} />
        <ViewAsTable
          caption="Median worldwide gross and median budget per lead genre"
          headers={["Genre", "Films", "Median gross", "Median budget"]}
          rows={genres.map((g) => [
            g.genre,
            g.count,
            dollarsCompact(g.medianGross),
            g.medianBudget != null ? dollarsCompact(g.medianBudget) : "—",
          ])}
        />
      </Section>

      <Section
        title="Budget vs gross"
        takeaway={
          <>
            <Figure>{over.count.toLocaleString("en-US")}</Figure> of{" "}
            <Figure>{points.length.toLocaleString("en-US")}</Figure> movies (
            <Figure>{percent(over.share)}</Figure>) returned at least 2.5×
            their budget; <Figure>{under.count.toLocaleString("en-US")}</Figure>{" "}
            (<Figure>{percent(under.share)}</Figure>) never grossed their
            budget back.
          </>
        }
        note="Both axes are logarithmic. Movies without a documented budget or a final gross are excluded."
      >
        <BudgetGrossScatter data={points} />
        <ViewAsTable
          caption="Movies grouped by return on budget"
          headers={["Outcome", "Films", "Share", "Median return"]}
          rows={bands.map((b) => [
            BAND_LABEL[b.band],
            b.count,
            percent(b.share),
            `${ratio(b.medianReturn, 1)}×`,
          ])}
        />
      </Section>

      <Section
        title="The hit factory"
        takeaway={
          studios[0] && (
            <>
              <Figure>{studios[0].name}</Figure> leads the sample with{" "}
              <Figure>{dollarsCompact(studios[0].total)}</Figure> across{" "}
              <Figure>{studios[0].count}</Figure> films.
            </>
          )
        }
        note="Total worldwide gross, minimum three films in the catalog."
      >
        <div className="grid gap-10 sm:grid-cols-2">
          <LeaderList title="Studios" leaders={studios} />
          <LeaderList title="Directors" leaders={directors} />
        </div>
      </Section>
    </div>
  );
}

function Section({
  title,
  takeaway,
  note,
  children,
}: {
  title: string;
  takeaway?: ReactNode;
  note?: string;
  children: ReactNode;
}) {
  return (
    <section className="border-t border-hairline pt-8">
      <h2 className="title-caps text-sm text-dim">{title}</h2>
      {takeaway && <p className="mt-2 max-w-prose text-sm text-ink">{takeaway}</p>}
      <div className="mt-6 min-w-0">{children}</div>
      {note && <p className="mt-3 text-xs text-dim">{note}</p>}
    </section>
  );
}

/** A figure inside a takeaway sentence: mono, tabular, amber. */
function Figure({ children }: { children: ReactNode }) {
  return <span className="font-mono tabular text-actual">{children}</span>;
}

function LeaderList({
  title,
  leaders,
}: {
  title: string;
  leaders: GrossLeader[];
}) {
  return (
    <div>
      <h3 className="title-caps text-xs text-dim">{title}</h3>
      <ol className="mt-3 border-t border-hairline">
        {leaders.map((leader, i) => (
          <li
            key={leader.name}
            className="flex items-baseline gap-3 border-b border-hairline py-2"
          >
            <span className="w-5 shrink-0 font-mono tabular text-xs text-dim">
              {i + 1}
            </span>
            <span className="min-w-0 flex-1 truncate text-sm text-ink">
              {leader.name}
            </span>
            <span className="font-mono tabular text-sm text-actual">
              {dollarsCompact(leader.total)}
            </span>
            <span className="w-16 shrink-0 text-right text-xs text-dim">
              {leader.count} films
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="flex flex-col gap-14">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="border-t border-hairline pt-8">
          <div className="h-3 w-44 animate-pulse rounded bg-surface" />
          <div className="mt-2 h-4 w-80 animate-pulse rounded bg-surface" />
          <div className="mt-6 h-80 animate-pulse rounded bg-surface" />
        </div>
      ))}
    </div>
  );
}
