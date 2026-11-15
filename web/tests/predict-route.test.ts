import { afterEach, beforeEach, describe, expect, it } from "vitest";

import contract from "@/generated/predict-contract.json";
import { POST } from "@/app/api/predict/route";

const validBody = {
  budget: 50_000_000,
  runtime: 110,
  genre: ["Action"],
  release_month: 7,
  release_year: 2027,
};

function postJson(body: unknown): Request {
  return new Request("http://localhost/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("POST /api/predict", () => {
  beforeEach(() => {
    delete process.env.INFERENCE_API_URL;
    delete process.env.INFERENCE_API_KEY;
  });

  afterEach(() => {
    delete process.env.INFERENCE_API_URL;
    delete process.env.INFERENCE_API_KEY;
  });

  it("answers in mock mode when INFERENCE_API_URL is unset", async () => {
    const response = await POST(postJson(validBody));

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      mock: true,
      model_id: "mock-oracle",
    });
  });

  it("rejects a runtime above the contract maximum", async () => {
    const maximum = contract.predict_request.properties.runtime.maximum;

    const response = await POST(postJson({ ...validBody, runtime: maximum + 1 }));

    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({
      error: "VALIDATION_ERROR",
      message: expect.stringContaining("runtime"),
    });
  });
});
