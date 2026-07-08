/** Accessible fallback for a chart: the same aggregates as a plain table. */
export function ViewAsTable({
  caption,
  headers,
  rows,
}: {
  caption: string;
  headers: string[];
  rows: Array<Array<string | number>>;
}) {
  return (
    <details className="mt-3 text-sm">
      <summary className="cursor-pointer text-xs text-dim transition-colors hover:text-ink">
        View as table
      </summary>
      <div className="mt-3 max-h-80 overflow-auto rounded border border-hairline">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr className="border-b border-hairline bg-surface text-left">
              {headers.map((h, i) => (
                <th
                  key={h}
                  scope="col"
                  className={`px-3 py-2 font-medium text-dim ${i > 0 ? "text-right" : ""}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-hairline last:border-b-0">
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className={
                      j > 0
                        ? "px-3 py-1.5 text-right font-mono tabular text-ink"
                        : "px-3 py-1.5 text-ink"
                    }
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
