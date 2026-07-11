"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { BacktestYear } from "@/lib/types";
import { ratio } from "@/lib/format";
import {
  ChartTooltip,
  numericTick,
} from "@/components/charts/chart-tooltip";

/**
 * The signature chart: per-year R² (log dollars), model as solid cyan,
 * baseline as a ghost outline the model has to beat. Negative years
 * (2020) hang below the zero line — that is the point.
 */
export function BacktestBars({ years }: { years: BacktestYear[] }) {
  return (
    <figure className="mt-4">
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={years}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
            barGap={3}
            barCategoryGap="28%"
          >
            <CartesianGrid vertical={false} stroke="var(--color-hairline)" />
            <XAxis
              dataKey="year"
              tickLine={false}
              axisLine={{ stroke: "var(--color-hairline)" }}
              tick={numericTick}
            />
            <YAxis
              width={44}
              tickLine={false}
              axisLine={false}
              tick={numericTick}
              tickFormatter={(v: number) => v.toFixed(2)}
              domain={[
                (min: number) => Math.min(0, Math.floor(min * 10) / 10),
                (max: number) => Math.ceil(max * 10) / 10,
              ]}
            />
            <Tooltip
              content={YearTip}
              cursor={{ fill: "var(--color-surface-2)", fillOpacity: 0.5 }}
            />
            <ReferenceLine y={0} stroke="var(--color-dim)" />
            <Bar
              dataKey="baseline_r2_log"
              name="Baseline"
              fill="transparent"
              stroke="var(--color-actual-deep)"
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Bar
              dataKey="model_r2_log"
              name="Model"
              fill="var(--color-predicted)"
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption>
        <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-dim">
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-[2px] bg-predicted"
            />
            Model — R² on log dollars
          </li>
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-2.5 w-2.5 rounded-[2px] border border-actual-deep"
            />
            Baseline — the ghost to beat
          </li>
        </ul>
      </figcaption>
    </figure>
  );
}

function YearTip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const row: BacktestYear | undefined = payload[0]?.payload;
  if (!row) return null;
  return (
    <ChartTooltip
      label={`${row.year} · ${row.n_movies} movies`}
      rows={[
        {
          name: "Baseline R²",
          value: ratio(row.baseline_r2_log),
          color: "var(--color-actual)",
        },
        {
          name: "Model R²",
          value: ratio(row.model_r2_log),
          color: "var(--color-predicted)",
        },
        {
          name: "Gain",
          value: `${row.gain_r2_log >= 0 ? "+" : ""}${ratio(row.gain_r2_log)}`,
          color:
            row.gain_r2_log >= 0
              ? "var(--color-over)"
              : "var(--color-under)",
        },
      ]}
    />
  );
}
