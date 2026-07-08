"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import type { CatalogMovie } from "@/lib/catalog";
import { roi } from "@/lib/catalog";
import { dollarsCompact, ratio } from "@/lib/format";

const columnHelper = createColumnHelper<CatalogMovie>();

export function MoviesTable({ movies }: { movies: CatalogMovie[] }) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("title", {
        header: "Title",
        cell: (info) => (
          <span className="font-medium text-ink">{info.getValue()}</span>
        ),
      }),
      columnHelper.accessor("release_year", {
        header: "Year",
        cell: (info) => <Mono>{info.getValue()}</Mono>,
      }),
      columnHelper.accessor((m) => m.genres[0] ?? "—", {
        id: "genre",
        header: "Genre",
        cell: (info) => <span className="text-dim">{info.getValue()}</span>,
      }),
      columnHelper.accessor("director", {
        header: "Director",
        cell: (info) => (
          <span className="text-dim">{info.getValue() ?? "—"}</span>
        ),
      }),
      columnHelper.accessor("production_budget", {
        header: "Budget",
        cell: (info) => {
          const v = info.getValue();
          return <Mono>{v ? dollarsCompact(v) : "—"}</Mono>;
        },
        sortUndefined: "last",
      }),
      columnHelper.accessor("worldwide_gross", {
        header: "Gross",
        cell: (info) => {
          const v = info.getValue();
          return (
            <Mono className="text-actual">
              {v != null ? dollarsCompact(v) : "—"}
            </Mono>
          );
        },
        sortUndefined: "last",
      }),
      columnHelper.accessor((m) => roi(m), {
        id: "roi",
        header: "Return",
        cell: (info) => {
          const v = info.getValue();
          return <Mono>{v == null ? "—" : `${ratio(v, 1)}×`}</Mono>;
        },
        sortUndefined: "last",
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: movies,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 100 } },
  });

  const numeric = new Set(["release_year", "production_budget", "worldwide_gross", "roi"]);

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded border border-hairline">
        <table className="w-full min-w-160 border-collapse text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-hairline bg-surface">
                {headerGroup.headers.map((header) => {
                  const sorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      className={`px-3 py-2 font-normal ${
                        numeric.has(header.column.id) ? "text-right" : "text-left"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className="inline-flex items-center gap-1 text-xs uppercase tracking-wider text-dim hover:text-ink"
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                        <span aria-hidden className="w-3 text-actual">
                          {sorted === "asc" ? "↑" : sorted === "desc" ? "↓" : ""}
                        </span>
                      </button>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => router.push(`/movies/${row.original.tmdb_id}`)}
                className="cursor-pointer border-b border-hairline/50 transition-colors duration-150 last:border-b-0 hover:bg-surface"
              >
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={`px-3 py-2 ${
                      numeric.has(cell.column.id) ? "text-right" : "text-left"
                    }`}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.getPageCount() > 1 && (
        <div className="flex items-center justify-between text-sm text-dim">
          <button
            type="button"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="rounded border border-hairline px-3 py-1.5 transition-colors duration-150 enabled:hover:bg-surface enabled:hover:text-ink disabled:opacity-40"
          >
            Previous
          </button>
          <span className="font-mono tabular text-xs">
            Page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </span>
          <button
            type="button"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="rounded border border-hairline px-3 py-1.5 transition-colors duration-150 enabled:hover:bg-surface enabled:hover:text-ink disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function Mono({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span className={`font-mono tabular text-sm ${className}`}>{children}</span>
  );
}
