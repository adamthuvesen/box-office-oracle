"use client";

import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { YearlyGross } from "@/lib/stats";
import { dollarsCompact } from "@/lib/format";
import {
  ChartLegend,
  ChartTooltip,
  GRID_STROKE,
  numericTick,
} from "@/components/charts/chart-tooltip";

function YearTooltip({
  active,
  payload,
  label,
}: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as YearlyGross;
  return (
    <ChartTooltip
      label={String(label)}
      rows={[
        {
          name: "Worldwide gross",
          value: dollarsCompact(row.totalGross),
          color: "var(--color-actual)",
        },
        {
          name: "Releases",
          value: String(row.releases),
          color: "var(--color-actual-deep)",
        },
      ]}
    />
  );
}

export function YearlyGrossChart({ data }: { data: YearlyGross[] }) {
  return (
    <div className="w-full">
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 16, right: 4, left: 4, bottom: 0 }}
          >
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis
              dataKey="year"
              tick={numericTick}
              tickLine={false}
              axisLine={{ stroke: GRID_STROKE }}
              interval={4}
            />
            <YAxis
              yAxisId="gross"
              tick={numericTick}
              tickLine={false}
              axisLine={false}
              tickFormatter={dollarsCompact}
              width={48}
            />
            <YAxis
              yAxisId="releases"
              orientation="right"
              tick={numericTick}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <Tooltip
              content={YearTooltip}
              cursor={{ fill: "var(--color-surface-2)", fillOpacity: 0.5 }}
            />
            <ReferenceLine
              yAxisId="gross"
              x={2020}
              stroke={GRID_STROKE}
              strokeDasharray="4 4"
              label={{
                value: "COVID shutdown",
                position: "top",
                fill: "var(--color-dim)",
                fontSize: 10,
              }}
            />
            <Bar
              yAxisId="gross"
              dataKey="totalGross"
              fill="var(--color-actual)"
              radius={[2, 2, 0, 0]}
              isAnimationActive={false}
            />
            <Line
              yAxisId="releases"
              dataKey="releases"
              stroke="var(--color-actual-deep)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <ChartLegend
        items={[
          { label: "Worldwide gross (left)", color: "var(--color-actual)" },
          {
            label: "Releases in sample (right)",
            color: "var(--color-actual-deep)",
            shape: "line",
          },
        ]}
      />
    </div>
  );
}
