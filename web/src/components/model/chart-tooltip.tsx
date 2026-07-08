import type { ReactNode } from "react";

/** Axis ticks across all charts: small, dim, mono. */
export const AXIS_TICK = {
  fill: "var(--color-dim)",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
} as const;

/** The one dark tooltip every chart shares: surface-2, hairline, mono figures. */
export function TooltipCard({ children }: { children: ReactNode }) {
  return (
    <div className="rounded border border-hairline bg-surface-2 px-3 py-2 text-xs shadow-xl">
      {children}
    </div>
  );
}

export function TooltipRow({
  label,
  value,
  valueClass = "text-ink",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className="text-dim">{label}</span>
      <span className={`font-mono tabular ${valueClass}`}>{value}</span>
    </div>
  );
}
