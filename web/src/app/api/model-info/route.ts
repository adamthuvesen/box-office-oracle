/**
 * Server-side proxy for the inference API's model/info endpoint.
 * The API key stays in server env — never shipped to the browser.
 * GET route handlers are dynamic by default, so nothing here is cached.
 */
export async function GET() {
  const url = process.env.INFERENCE_API_URL;
  const key = process.env.INFERENCE_API_KEY;

  if (!url) {
    return Response.json(
      { error: "INFERENCE_API_URL is not configured" },
      { status: 503 },
    );
  }

  const base = url.endsWith("/") ? url : `${url}/`;
  try {
    const res = await fetch(`${base}model/info`, {
      headers: key ? { "X-API-Key": key } : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      return Response.json(
        { error: `Inference API responded with ${res.status}` },
        { status: 503 },
      );
    }
    return Response.json(await res.json());
  } catch {
    return Response.json(
      { error: "Inference API did not answer within 5 seconds" },
      { status: 503 },
    );
  }
}
