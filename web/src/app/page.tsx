import Link from "next/link";
import { Suspense } from "react";
import { loadMovies, loadOraclePredictions } from "@/lib/data";
import { buildStarField } from "@/lib/constellation";
import { ConstellationHero } from "@/components/constellation/hero";

export default function Home() {
  return (
    <Suspense fallback={<HeroFallback />}>
      <HomeLive />
    </Suspense>
  );
}

async function HomeLive() {
  const [movies, predictions] = await Promise.all([
    loadMovies(),
    loadOraclePredictions(),
  ]);

  if (!movies) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-24">
        <p className="font-mono text-sm text-predicted">
          XGBoost · Snowflake · SageMaker · Lambda
        </p>
        <h1 className="title-caps mt-4 max-w-3xl text-4xl leading-tight text-ink sm:text-5xl">
          The projector is dark
        </h1>
        <p className="mt-6 max-w-xl text-lg text-dim">
          Run <code className="font-mono text-actual">make web-data</code>{" "}
          from the repo root to export the catalog, then refresh. The house
          lights will do the rest.
        </p>
        <div className="mt-10">
          <Link
            href="/predict"
            className="rounded border border-hairline px-4 py-2 text-sm text-ink transition-colors duration-150 hover:bg-surface"
          >
            Ask the oracle anyway
          </Link>
        </div>
      </div>
    );
  }

  const field = buildStarField(movies, predictions);
  const totalGross = movies.reduce((sum, m) => sum + (m.worldwide_gross ?? 0), 0);

  return <ConstellationHero field={field} totalGross={totalGross} />;
}

function HeroFallback() {
  return (
    <div className="flex min-h-[70svh] items-end px-6 pb-16">
      <div className="mx-auto w-full max-w-6xl">
        <div className="h-4 w-64 animate-pulse rounded bg-surface" />
        <div className="mt-5 h-12 w-full max-w-2xl animate-pulse rounded bg-surface" />
        <div className="mt-4 h-6 w-80 animate-pulse rounded bg-surface" />
      </div>
    </div>
  );
}
