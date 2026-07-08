"use client";

/** Shared chart chrome: the one styled tooltip, plus axis/grid constants. */

export interface TooltipRow {
  name: string;
  value: string;
  /** CSS color for the value figure; defaults to ink. */
  color?: string;
}

export function ChartTooltip({
  label,
  rows,
}: {
  label?: string;
  rows: TooltipRow[];
}) {
  return (
    <div className="rounded border border-hairline bg-surface-2 px-3 py-2 text-xs shadow-lg">
      {label && <p className="mb-1.5 font-medium text-ink">{label}</p>}
      <dl className="flex flex-col gap-1">
        {rows.map((row) => (
          <div key={row.name} className="flex justify-between gap-6">
            <dt className="text-dim">{row.name}</dt>
            <dd
              className="font-mono tabular"
              style={{ color: row.color ?? "var(--color-ink)" }}
            >
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Numeric axis ticks: mono, dim, small — per the screening-room spec. */
export const numericTick = {
  fill: "var(--color-dim)",
  fontSize: 11,
  fontFamily: "var(--font-geist-mono)",
} as const;

/** Category axis ticks (names, not figures). */
export const categoryTick = {
  fill: "var(--color-dim)",
  fontSize: 11,
} as const;

export const GRID_STROKE = "var(--color-hairline)";

/** Inline legend under a chart — Recharts' own legend is too styled. */
export function ChartLegend({
  items,
}: {
  items: Array<{ label: string; color: string; shape?: "square" | "line" | "dot" }>;
}) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-dim">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          {item.shape === "line" ? (
            <span
              className="h-0.5 w-4 rounded-full"
              style={{ backgroundColor: item.color }}
            />
          ) : item.shape === "dot" ? (
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: item.color }}
            />
          ) : (
            <span
              className="size-2.5 rounded-[2px]"
              style={{ backgroundColor: item.color }}
            />
          )}
          {item.label}
        </span>
      ))}
    </div>
  );
}
