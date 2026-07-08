const THIN_SPACE = " ";
const TRUE_MINUS = "−";

/** Compact dollars: $1.2B, $845M, $92K. The app-wide default for figures. */
export function dollarsCompact(value: number): string {
  const sign = value < 0 ? TRUE_MINUS : "";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${sign}$${trimmed(abs / 1e9)}B`;
  if (abs >= 1e6) return `${sign}$${trimmed(abs / 1e6)}M`;
  if (abs >= 1e3) return `${sign}$${trimmed(abs / 1e3)}K`;
  return `${sign}$${Math.round(abs)}`;
}

function trimmed(n: number): string {
  const s = n >= 100 ? n.toFixed(0) : n.toFixed(1);
  return s.endsWith(".0") ? s.slice(0, -2) : s;
}

/** Full dollars with thin-space grouping, for hero figures: $1 234 567 890. */
export function dollarsFull(value: number): string {
  const sign = value < 0 ? TRUE_MINUS : "";
  const grouped = Math.round(Math.abs(value))
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, THIN_SPACE);
  return `${sign}$${grouped}`;
}

/** Signed delta in compact dollars with a true minus: −$120M / +$85M. */
export function dollarsDelta(value: number): string {
  return value < 0
    ? dollarsCompact(value)
    : `+${dollarsCompact(value)}`;
}

export function percent(value: number, decimals = 0): string {
  const sign = value < 0 ? TRUE_MINUS : "";
  return `${sign}${Math.abs(value * 100).toFixed(decimals)}%`;
}

export function ratio(value: number, decimals = 2): string {
  const sign = value < 0 ? TRUE_MINUS : "";
  return `${sign}${Math.abs(value).toFixed(decimals)}`;
}

export function runtimeLabel(minutes: number | null): string | null {
  if (minutes == null) return null;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export function releaseDateLabel(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
