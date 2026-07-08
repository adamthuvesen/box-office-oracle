import { predictRequestSchema, type PredictRequest } from "@/lib/predict";
import type { PredictionResult } from "@/lib/types";

const UPSTREAM_TIMEOUT_MS = 15_000;

export async function POST(request: Request) {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return Response.json(
      { error: "INVALID_JSON", message: "Request body must be JSON." },
      { status: 400 },
    );
  }

  const parsed = predictRequestSchema.safeParse(raw);
  if (!parsed.success) {
    const message = parsed.error.issues
      .map((issue) => `${issue.path.join(".") || "body"}: ${issue.message}`)
      .join("; ");
    return Response.json(
      { error: "VALIDATION_ERROR", message },
      { status: 400 },
    );
  }

  const apiUrl = process.env.INFERENCE_API_URL;
  const apiKey = process.env.INFERENCE_API_KEY;
  if (apiUrl && apiKey) {
    return proxyPredict(parsed.data, apiUrl, apiKey);
  }
  return Response.json(mockPredict(parsed.data));
}

async function proxyPredict(
  body: PredictRequest,
  apiUrl: string,
  apiKey: string,
): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "TimeoutError";
    return Response.json(
      {
        error: timedOut ? "UPSTREAM_TIMEOUT" : "UPSTREAM_UNREACHABLE",
        message: timedOut
          ? "The model API did not answer within 15 seconds."
          : "Could not reach the model API.",
      },
      { status: 503 },
    );
  }

  let payload: unknown = null;
  try {
    payload = await upstream.json();
  } catch {
    // Non-JSON upstream body — fall through to the generic error below.
  }

  if (!upstream.ok) {
    const detail = (
      payload as { detail?: { error?: string; message?: string } } | null
    )?.detail;
    return Response.json(
      {
        error: detail?.error ?? "UPSTREAM_ERROR",
        message:
          detail?.message ?? `The model API answered ${upstream.status}.`,
      },
      { status: upstream.status },
    );
  }

  return Response.json(payload);
}

/**
 * Mock mode: no INFERENCE_API_URL/INFERENCE_API_KEY set. Deterministic fake —
 * base = budget × (1.1 + 0.3·sin(hash of the text-ish inputs)), scaled by a
 * per-genre multiplier, clamped to ≥ $1M.
 */
function mockPredict(input: PredictRequest): PredictionResult {
  const genres = Array.isArray(input.genre) ? input.genre : [input.genre];
  const seed = hashString(
    [
      ...genres,
      input.director ?? "",
      ...(input.actors ?? []),
      input.production_company ?? "",
      input.mpaa,
      String(input.release_month),
      String(input.release_year),
      String(input.runtime),
      String(input.ip_tier),
      String(input.prior_franchise_gross),
      String(input.is_franchise_followup),
    ].join("|"),
  );
  const genreMultiplier =
    genres.reduce((sum, g) => sum + (GENRE_MULTIPLIER[g] ?? 1), 0) /
    genres.length;
  // Invented, not learned: tier 5 (no IP) = 1x, tier 1 (top IP) = 1.8x, plus
  // a small bump for franchise follow-ups with real prior gross.
  const ipMultiplier =
    1 + 0.2 * (5 - input.ip_tier) +
    (input.is_franchise_followup && input.prior_franchise_gross > 0 ? 0.15 : 0);
  const base = input.budget * (1.1 + 0.3 * Math.sin(seed));
  const prediction = Math.max(
    base * genreMultiplier * ipMultiplier,
    1e6,
  );
  return {
    prediction,
    model_id: "mock-oracle",
    model_version: 0,
    prediction_interval_heuristic: [prediction * 0.55, prediction * 1.8],
    timestamp: new Date().toISOString(),
    processing_time_ms: 0,
    mock: true,
  };
}

/** Rough theatrical pull per genre — mock mode only, invented, not learned. */
const GENRE_MULTIPLIER: Record<string, number> = {
  Action: 1.35,
  Adventure: 1.5,
  Animation: 1.6,
  Comedy: 1.05,
  Crime: 0.85,
  Drama: 0.8,
  Family: 1.3,
  Fantasy: 1.35,
  History: 0.75,
  Horror: 1.25,
  Music: 0.9,
  Mystery: 0.95,
  Romance: 0.95,
  "Science Fiction": 1.4,
  Thriller: 1.0,
  War: 0.8,
  Western: 0.7,
};

function hashString(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(h, 33) + s.charCodeAt(i)) >>> 0;
  }
  return h;
}
