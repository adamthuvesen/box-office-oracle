# Serving

The trained artifacts load into the inference app and answer over ASGI with no network. The API key is enforced, request bounds are enforced, and `/model/info` reports the training-data fingerprint.

Source: box_office/inference/app/main.py, box_office/inference/app/runtime.py, box_office/inference/app/predictor.py
Inventory-id: POST /predict, GET /model/info

## Sub-features

- `predict` returns a positive revenue number for a valid request.
- `auth` rejects a missing key and a wrong key with 401.
- `bounds` rejects `runtime` 9999 and `release_month` 13 with 400.
- `model-info` reports a fingerprint equal to the one in `metadata.json`.

## How to get to it (user POV)

- `POST /predict` with an `X-API-Key` header and the request body in `box_office/inference/app/predictor.py` (`PredictionRequest`).
- `GET /model/info`.

## Driving it with verify

Preconditions: none; the drive trains first when it runs alone.

- **Probe.** `make verify ARGS=serve`. `serve_probe.py` runs five requests through `httpx.ASGITransport`. A `ModelLoader` stand-in points the runtime at the local artifact directory, because the real loader downloads from S3.
- **Proof.** `serve.*-probe.json` and a manifest block. The stand-in replaces only the S3 download.

## Gotchas

- The app logs to stdout, so the probe writes its answers to a file.
- The app reads its settings at import, so `API_KEY` must be in the child environment before the import.
