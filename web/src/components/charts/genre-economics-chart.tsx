"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { GenreEconomics } from "@/lib/stats";
import { dollarsCompact } from "@/lib/format";
import {
  ChartLegend,
  ChartTooltip,
  GRID_STROKE,
  categoryTick,
  numericTick,
} from "@/components/charts/chart-tooltip";

function GenreTooltip({
  active,
  payload,
}: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as GenreEconomics;
  return (
    <ChartTooltip
      label={`${row.genre} · ${row.count} films`}
      rows={[
        {
          name: "Median gross",
          value: dollarsCompact(row.medianGross),
          color: "var(--color-actual)",
        },
        {
          name: "Median budget",
          value:
            row.medianBudget != null ? dollarsCompact(row.medianBudget) : "—",
          color: "var(--color-actual-deep)",
        },
      ]}
    />
  );
}

export function GenreEconomicsChart({ data }: { data: GenreEconomics[] }) {
  return (
    <div className="w-full">
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 12, left: 4, bottom: 0 }}
            barGap={2}
          >
            <CartesianGrid stroke={GRID_STROKE} horizontal={false} />
            <XAxis
              type="number"
              tick={numericTick}
              tickLine={false}
              axisLine={false}
              tickFormatter={dollarsCompact}
            />
            <YAxis
              type="category"
              dataKey="genre"
              tick={categoryTick}
              tickLine={false}
              axisLine={{ stroke: GRID_STROKE }}
              width={104}
            />
            <Tooltip
              content={GenreTooltip}
              cursor={{ fill: "var(--color-surface-2)", fillOpacity: 0.5 }}
            />
            <Bar
              dataKey="medianGross"
              fill="var(--color-actual)"
              barSize={12}
              radius={[0, 2, 2, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="medianBudget"
              fill="var(--color-actual-deep)"
              barSize={4}
              radius={[0, 2, 2, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ChartLegend
        items={[
          { label: "Median worldwide gross", color: "var(--color-actual)" },
          { label: "Median budget", color: "var(--color-actual-deep)" },
        ]}
      />
    </div>
  );
}
