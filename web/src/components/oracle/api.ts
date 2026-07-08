import type { PredictRequestInput } from "@/lib/predict";
import type { PredictionResult } from "@/lib/types";

export class OracleError extends Error {
  constructor(
    message: string,
    readonly code: string | null,
    readonly status: number,
  ) {
    super(message);
    this.name = "OracleError";
  }
}

export async function askOracle(
  request: PredictRequestInput,
): Promise<PredictionResult> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  const payload: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const fault = (payload ?? {}) as { error?: string; message?: string };
    throw new OracleError(
      fault.message ?? `The request failed with status ${res.status}.`,
      fault.error ?? null,
      res.status,
    );
  }
  return payload as PredictionResult;
}
