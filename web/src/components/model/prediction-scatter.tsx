"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  type TooltipContentProps,
} from "recharts";
import { dollarsCompact, percent } from "@/lib/format";
import {
  ChartTooltip,
  numericTick,
} from "@/components/charts/chart-tooltip";

/** Trimmed prediction row — just what the scatter needs over the wire. */
export interface BlindGuess {
  tmdb_id: number;
  title: string;
  release_year: number;
  y_true: number;
  y_pred: number;
  ape: number;
}

type Verdict = "within" | "over" | "under";

interface ScatterPoint extends BlindGuess {
  x: number; // log10(actual)
  y: number; // log10(predicted)
  verdict: Verdict;
}

const LOG_TICKS = [6, 7, 8, 9]; // $1M / $10M / $100M / $1B

function verdictOf(guess: BlindGuess): Verdict {
  const multiple = guess.y_pred / guess.y_true;
  if (multiple > 1.5) return "over";
  if (multiple < 0.5) return "under";
  return "within";
}

/** Every blind guess against reality, log-log, with the y=x prophecy line. */
export function PredictionScatter({ guesses }: { guesses: BlindGuess[] }) {
  const [tableOpen, setTableOpen] = useState(false);

  const { points, domain } = useMemo(() => {
    const pts: ScatterPoint[] = guesses.map((g) => ({
      ...g,
      x: Math.log10(g.y_true),
      y: Math.log10(g.y_pred),
      verdict: verdictOf(g),
    }));
    const top = Math.max(...pts.map((p) => Math.max(p.x, p.y)), 9);
    const bottom = Math.min(...pts.map((p) => Math.min(p.x, p.y)), 6.5);
    return {
      points: pts,
      domain: [
        Math.min(6, Math.floor(bottom * 10) / 10),
        Math.ceil((top + 0.1) * 10) / 10,
      ] as [number, number],
    };
  }, [guesses]);

  const within = points.filter((p) => p.verdict === "within");
  const over = points.filter((p) => p.verdict === "over");
  const under = points.filter((p) => p.verdict === "under");

  return (
    <figure className="mt-4">
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="var(--color-hairline)" />
            <XAxis
              type="number"
              dataKey="x"
              name="Actual"
              domain={domain}
              ticks={LOG_TICKS}
              tickFormatter={(v: number) => dollarsCompact(10 ** v)}
              tickLine={false}
              axisLine={{ stroke: "var(--color-hairline)" }}
              tick={numericTick}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Predicted"
              domain={domain}
              ticks={LOG_TICKS}
              tickFormatter={(v: number) => dollarsCompact(10 ** v)}
              tickLine={false}
              axisLine={false}
              tick={numericTick}
              width={52}
            />
            <ZAxis type="number" range={[28, 28]} />
            <ReferenceLine
              segment={[
                { x: domain[0], y: domain[0] },
                { x: domain[1], y: domain[1] },
              ]}
              stroke="var(--color-dim)"
              strokeDasharray="6 6"
              label={{
                value: "perfect prophecy",
                position: "insideTopRight",
                fill: "var(--color-dim)",
                fontSize: 11,
              }}
            />
            <Tooltip
              content={GuessTip}
              cursor={{ stroke: "var(--color-hairline)" }}
            />
            <Scatter
              name="Called it"
              data={within}
              fill="var(--color-predicted)"
              fillOpacity={0.7}
              isAnimationActive={false}
            />
            <Scatter
              name="Overshot"
              data={over}
              fill="var(--color-predicted-deep)"
              isAnimationActive={false}
            />
            <Scatter
              name="Undershot"
              data={under}
              fill="var(--color-under)"
              fillOpacity={0.6}
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <figcaption>
        <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-dim">
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-2 w-2 rounded-full bg-predicted opacity-70"
            />
            Called it — within ±50% of actual ({within.length})
          </li>
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-2 w-2 rounded-full bg-predicted-deep"
            />
            Overshot — more than 1.5× actual ({over.length})
          </li>
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-2 w-2 rounded-full bg-under opacity-60"
            />
            Undershot — less than half of actual ({under.length})
          </li>
          <li className="flex items-center gap-2">
            <span
              aria-hidden
              className="inline-block w-4 border-t border-dashed border-dim"
            />
            Perfect prophecy: guess = reality
          </li>
        </ul>
      </figcaption>

      <details
        className="mt-4"
        onToggle={(e: React.SyntheticEvent<HTMLDetailsElement>) =>
          setTableOpen(e.currentTarget.open)
        }
      >
        <summary className="cursor-pointer text-xs text-dim hover:text-ink">
          View as table
        </summary>
        {tableOpen && (
          <div className="mt-3 max-h-96 overflow-auto rounded border border-hairline">
            <table className="w-full min-w-[36rem] text-left text-sm">
              <thead className="sticky top-0 bg-surface text-xs text-dim">
                <tr>
                  <th className="px-3 py-2 font-normal">Title</th>
                  <th className="px-3 py-2 font-normal">Year</th>
                  <th className="px-3 py-2 text-right font-normal">Actual</th>
                  <th className="px-3 py-2 text-right font-normal">
                    Predicted
                  </th>
                  <th className="px-3 py-2 text-right font-normal">APE</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p) => (
                  <tr key={p.tmdb_id} className="border-t border-hairline">
                    <td className="px-3 py-1.5 text-ink">{p.title}</td>
                    <td className="px-3 py-1.5 font-mono tabular text-dim">
                      {p.release_year}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular text-actual">
                      {dollarsCompact(p.y_true)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular text-predicted">
                      {dollarsCompact(p.y_pred)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular text-dim">
                      {percent(p.ape)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </details>
    </figure>
  );
}

function GuessTip({ active, payload }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  const point: ScatterPoint | undefined = payload[0]?.payload;
  if (!point) return null;
  return (
    <ChartTooltip
      label={`${point.title} (${point.release_year})`}
      rows={[
        {
          name: "Actual",
          value: dollarsCompact(point.y_true),
          color: "var(--color-actual)",
        },
        {
          name: "Predicted",
          value: dollarsCompact(point.y_pred),
          color: "var(--color-predicted)",
        },
        { name: "APE", value: percent(point.ape) },
      ]}
    />
  );
}
