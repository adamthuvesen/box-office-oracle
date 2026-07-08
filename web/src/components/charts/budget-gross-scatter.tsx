"use client";

import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { BudgetGrossPoint } from "@/lib/stats";
import { roi } from "@/lib/catalog";
import { dollarsCompact, ratio } from "@/lib/format";
import {
  ChartLegend,
  ChartTooltip,
  GRID_STROKE,
  numericTick,
} from "@/components/charts/chart-tooltip";

const BAND_COLOR = {
  over: "var(--color-over)",
  mid: "var(--color-actual)",
  under: "var(--color-under)",
} as const;

/** Fixed r=3 dot; Recharts clones this with cx/cy/fill per point. */
function PointDot({
  cx,
  cy,
  fill,
  opacity = 1,
}: {
  cx?: number;
  cy?: number;
  fill?: string;
  opacity?: number;
}) {
  if (cx == null || cy == null) return null;
  return <circle cx={cx} cy={cy} r={3} fill={fill} fillOpacity={opacity} />;
}

function ScatterTooltip({
  active,
  payload,
}: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as BudgetGrossPoint;
  const movieReturn = roi({
    production_budget: p.budget,
    worldwide_gross: p.gross,
  });
  return (
    <ChartTooltip
      label={`${p.title} (${p.year})`}
      rows={[
        { name: "Budget", value: dollarsCompact(p.budget) },
        {
          name: "Gross",
          value: dollarsCompact(p.gross),
          color: "var(--color-actual)",
        },
        {
          name: "Return",
          value: movieReturn != null ? `${ratio(movieReturn, 1)}×` : "—",
          color: BAND_COLOR[p.band],
        },
      ]}
    />
  );
}

const LOG_TICKS = [6, 7, 8, 9]; // $1M, $10M, $100M, $1B
const formatLogTick = (v: number) => dollarsCompact(10 ** v);

export function BudgetGrossScatter({ data }: { data: BudgetGrossPoint[] }) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of data) {
    lo = Math.min(lo, p.logBudget, p.logGross);
    hi = Math.max(hi, p.logBudget, p.logGross);
  }
  lo = Math.floor(lo * 2) / 2;
  hi = Math.ceil(hi * 2) / 2;

  const over = data.filter((p) => p.band === "over");
  const mid = data.filter((p) => p.band === "mid");
  const under = data.filter((p) => p.band === "under");

  return (
    <div className="w-full">
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 12, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={GRID_STROKE} />
            <XAxis
              type="number"
              dataKey="logBudget"
              domain={[lo, hi]}
              ticks={LOG_TICKS}
              tickFormatter={formatLogTick}
              tick={numericTick}
              tickLine={false}
              axisLine={{ stroke: GRID_STROKE }}
              name="Budget"
            />
            <YAxis
              type="number"
              dataKey="logGross"
              domain={[lo, hi]}
              ticks={LOG_TICKS}
              tickFormatter={formatLogTick}
              tick={numericTick}
              tickLine={false}
              axisLine={false}
              width={52}
              name="Gross"
            />
            <Tooltip
              content={ScatterTooltip}
              cursor={{ stroke: GRID_STROKE, strokeDasharray: "3 3" }}
            />
            <ReferenceLine
              segment={[
                { x: lo, y: lo },
                { x: hi, y: hi },
              ]}
              stroke={GRID_STROKE}
              strokeDasharray="4 4"
              label={{
                value: "break-even (gross = budget)",
                position: "insideTopLeft",
                fill: "var(--color-dim)",
                fontSize: 10,
              }}
            />
            <Scatter
              data={mid}
              shape={<PointDot opacity={0.6} />}
              fill="var(--color-actual)"
              isAnimationActive={false}
            />
            <Scatter
              data={over}
              shape={<PointDot />}
              fill="var(--color-over)"
              isAnimationActive={false}
            />
            <Scatter
              data={under}
              shape={<PointDot />}
              fill="var(--color-under)"
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <ChartLegend
        items={[
          { label: "Returned ≥ 2.5× budget", color: "var(--color-over)", shape: "dot" },
          { label: "In between", color: "var(--color-actual)", shape: "dot" },
          { label: "Grossed less than budget", color: "var(--color-under)", shape: "dot" },
        ]}
      />
    </div>
  );
}
