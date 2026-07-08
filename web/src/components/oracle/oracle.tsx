"use client";

import { useState } from "react";
import { MotionConfig } from "motion/react";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
} from "@tanstack/react-query";
import type { ComparableMovie } from "@/lib/predict";
import { askOracle } from "./api";
import { OracleForm } from "./oracle-form";
import { RevealStage, type OracleStatus } from "./reveal-stage";

interface OracleProps {
  catalog: ComparableMovie[];
  liveApi: boolean;
}

export function Oracle({ catalog, liveApi }: OracleProps) {
  // Local provider: /predict is the only subtree that talks to the API.
  const [queryClient] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={queryClient}>
      <OracleInner catalog={catalog} liveApi={liveApi} />
    </QueryClientProvider>
  );
}

function OracleInner({ catalog, liveApi }: OracleProps) {
  // Bumped per successful answer so the reveal remounts and animates once.
  const [revealKey, setRevealKey] = useState(0);
  const mutation = useMutation({
    mutationFn: askOracle,
    onSuccess: () => setRevealKey((k) => k + 1),
  });

  const status: OracleStatus = mutation.isPending
    ? "pending"
    : mutation.isError
      ? "error"
      : mutation.data
        ? "result"
        : "idle";

  return (
    <MotionConfig reducedMotion="user">
      <div className="grid gap-10 lg:grid-cols-2 lg:gap-12">
        <OracleForm
          pending={mutation.isPending}
          onAsk={(request) => mutation.mutate(request)}
        />
        <RevealStage
          status={status}
          result={mutation.data ?? null}
          error={mutation.error}
          revealKey={revealKey}
          catalog={catalog}
          liveApi={liveApi}
          onRetry={() => {
            if (mutation.variables) mutation.mutate(mutation.variables);
          }}
        />
      </div>
    </MotionConfig>
  );
}
