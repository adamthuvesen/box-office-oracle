"""Post-deploy probe of the inference API: /health with retries, then /model/info.

Runs in CD after a plain `uv sync --frozen`, so it uses only the standard library.
Reads API_ENDPOINT (required) and INFERENCE_API_KEY (optional) from the environment.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

HEALTH_ATTEMPTS = 3
HEALTH_TIMEOUT_SECONDS = 60.0
HEALTH_RETRY_DELAY_SECONDS = 5
INFO_TIMEOUT_SECONDS = 30.0


def _get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _health(url: str) -> tuple[int, str]:
    last_error: Exception | None = None
    for attempt in range(1, HEALTH_ATTEMPTS + 1):
        try:
            return _get(url, {}, HEALTH_TIMEOUT_SECONDS)
        except OSError as error:
            last_error = error
            print(f"  attempt {attempt}/{HEALTH_ATTEMPTS} failed: {error}")
            if attempt < HEALTH_ATTEMPTS:
                time.sleep(HEALTH_RETRY_DELAY_SECONDS)
    print(f"Health check failed after {HEALTH_ATTEMPTS} attempts: {last_error}")
    sys.exit(1)


def main() -> None:
    api_endpoint = os.environ.get("API_ENDPOINT", "").strip()
    if not api_endpoint or api_endpoint == "null":
        print("No API endpoint found")
        sys.exit(1)
    print(f"Testing API endpoint: {api_endpoint}")

    api_key = os.environ.get("INFERENCE_API_KEY", "").strip()
    authed_headers = {"X-API-Key": api_key} if api_key else {}
    base_url = api_endpoint.rstrip("/")

    try:
        health_url = f"{base_url}/health"
        print(f"Testing health endpoint (with retries for cold start): {health_url}")
        status, body = _health(health_url)
        if status != 200:
            print(f"Health check failed: {status} - {body}")
            sys.exit(1)
        print(f"Health check passed: {json.loads(body)}")

        print("Testing model info endpoint...")
        status, body = _get(f"{base_url}/model/info", authed_headers, INFO_TIMEOUT_SECONDS)
        if status == 200:
            print(f"Model info endpoint working: {json.loads(body)}")
        else:
            print(
                f"Model info endpoint returned: {status} (may be expected if no model deployed yet)"
            )
    except (OSError, ValueError) as error:
        print(f"API validation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
