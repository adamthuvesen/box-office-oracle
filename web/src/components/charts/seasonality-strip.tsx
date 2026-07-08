import type { Seasonality } from "@/lib/stats";
import { MONTH_ABBREV, MONTH_NAMES } from "@/lib/stats";
import { dollarsCompact } from "@/lib/format";

/**
 * Month × year heatmap styled like a film-strip contact sheet: sprocket-hole
 * rows above and below, amber cells scaled by total gross. Pure CSS grid.
 */
export function SeasonalityStrip({ data }: { data: Seasonality }) {
  const { years, grid, maxCell } = data;
  const columns = `2.5rem repeat(${years.length}, minmax(0.875rem, 1fr))`;

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[640px]">
        <SprocketRow columns={columns} count={years.length} />
        <div
          className="grid gap-[2px]"
          style={{ gridTemplateColumns: columns }}
          role="img"
          aria-label="Heatmap of total worldwide gross by month and year. The same figures are in the table below."
        >
          {grid.map((row, monthIdx) => (
            <MonthRow
              key={MONTH_ABBREV[monthIdx]}
              monthIdx={monthIdx}
              row={row}
              years={years}
              maxCell={maxCell}
            />
          ))}
        </div>
        <SprocketRow columns={columns} count={years.length} />
        <div
          aria-hidden
          className="grid gap-[2px]"
          style={{ gridTemplateColumns: columns }}
        >
          <div />
          {years.map((year) => (
            <div
              key={year}
              className="pt-1 text-center font-mono text-[10px] text-dim"
            >
              {year % 5 === 0 ? `’${String(year).slice(2)}` : ""}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MonthRow({
  monthIdx,
  row,
  years,
  maxCell,
}: {
  monthIdx: number;
  row: number[];
  years: number[];
  maxCell: number;
}) {
  return (
    <>
      <div className="flex items-center pr-2 font-mono text-xs text-dim">
        {MONTH_ABBREV[monthIdx]}
      </div>
      {row.map((total, yearIdx) => (
        <div
          key={years[yearIdx]}
          title={`${MONTH_NAMES[monthIdx]} ${years[yearIdx]} · ${dollarsCompact(total)}`}
          className={`aspect-square rounded-[2px] ${
            total === 0 ? "border border-hairline" : ""
          }`}
          style={total > 0 ? { backgroundColor: cellColor(total, maxCell) } : undefined}
        />
      ))}
    </>
  );
}

/**
 * 5-step amber scale on the surface. Square-root scaling so mid-tier months
 * stay readable next to blockbuster peaks.
 */
function cellColor(total: number, maxCell: number): string {
  const step = Math.min(5, Math.max(1, Math.ceil(Math.sqrt(total / maxCell) * 5)));
  const mix = [16, 32, 52, 75, 100][step - 1];
  return `color-mix(in oklab, var(--color-actual) ${mix}%, var(--color-surface))`;
}

/** Decorative film-edge sprocket holes, one per year column. */
function SprocketRow({ columns, count }: { columns: string; count: number }) {
  return (
    <div
      aria-hidden
      className="grid gap-[2px] py-1.5"
      style={{ gridTemplateColumns: columns }}
    >
      <div />
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center justify-center">
          <div className="size-1.5 rounded-[2px] bg-hairline" />
        </div>
      ))}
    </div>
  );
}
