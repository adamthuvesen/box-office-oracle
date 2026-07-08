"use client";

import { useState } from "react";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import { z } from "zod";
import { modelInfoSchema } from "@/lib/types";

const errorBodySchema = z.object({ error: z.string() });

async function fetchModelInfo() {
  const res = await fetch("/api/model-info");
  const body: unknown = await res.json();
  if (!res.ok) {
    const parsed = errorBodySchema.safeParse(body);
    throw new Error(
      parsed.success ? parsed.data.error : `Request failed (${res.status})`,
    );
  }
  return modelInfoSchema.parse(body);
}

/** Live card for /model — provider is local so the root layout stays untouched. */
export function LiveModelCard() {
  const [client] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={client}>
      <ModelInfo />
    </QueryClientProvider>
  );
}

function ModelInfo() {
  const { data, isPending, error } = useQuery({
    queryKey: ["model-info"],
    queryFn: fetchModelInfo,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 60_000,
  });

  if (isPending) {
    return (
      <div className="mt-4 flex max-w-md flex-col gap-2 rounded border border-hairline bg-surface p-5">
        <div className="h-4 w-40 animate-pulse rounded bg-surface-2" />
        <div className="h-4 w-56 animate-pulse rounded bg-surface-2" />
        <div className="h-4 w-48 animate-pulse rounded bg-surface-2" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="mt-4 max-w-md rounded border border-dashed border-hairline p-5 text-sm text-dim">
        <p>
          The projector is warming up — live model info needs{" "}
          <code className="font-mono">INFERENCE_API_URL</code> in{" "}
          <code className="font-mono">web/.env.local</code>.
        </p>
        {error && (
          <p className="mt-2 font-mono text-xs">{error.message}</p>
        )}
      </div>
    );
  }

  const rows: Array<[string, string]> = [];
  if (data.model_id) rows.push(["model_id", data.model_id]);
  if (data.version != null) rows.push(["version", String(data.version)]);
  if (data.status) rows.push(["status", data.status]);
  for (const [key, value] of Object.entries(data.metrics ?? {})) {
    rows.push([key, metricLabel(value)]);
  }

  if (rows.length === 0) {
    return (
      <div className="mt-4 max-w-md rounded border border-dashed border-hairline p-5 text-sm text-dim">
        The inference API answered, but reported nothing about the loaded
        model.
      </div>
    );
  }

  return (
    <dl className="mt-4 flex max-w-md flex-col gap-2 rounded border border-hairline bg-surface p-5">
      {rows.map(([key, value]) => (
        <div
          key={key}
          className="flex items-baseline justify-between gap-6 text-sm"
        >
          <dt className="font-mono text-dim">{key}</dt>
          <dd className="text-right font-mono tabular text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function metricLabel(value: unknown): string {
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(3);
  }
  return String(value);
}
