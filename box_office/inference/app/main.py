"""
FastAPI application for serverless ML inference.
Optimized for AWS Lambda with container support.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any
from pydantic import ValidationError
from .config import get_settings
from .predictor import PredictionResponse
from .model_loader import ModelLoadError
from .integrity import ArtifactIntegrityError
from .runtime import get_runtime
from box_office.ml.registry_constants import FeatureSchemaVersionMismatch
from box_office.ml.text_utils import LiteralEvalTooLarge

MAX_REQUEST_BODY_BYTES = 1024 * 1024  # 1 MiB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

settings = get_settings()
environment = settings.environment.lower()
is_production = environment in {"prod", "production"}

app = FastAPI(
    title=settings.api_title,
    description="Cost-efficient serverless inference API for ML model predictions",
    version=settings.api_version,
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
)

allowed_origins = settings.cors_origins
allow_credentials = True

if "*" in allowed_origins:
    logger.warning(
        "CORS wildcard '*' detected with credentials - disabling credentials for CORS compliance"
    )
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=["*"],
)


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Authenticate requests using API key when authentication is enabled."""
    if request.url.path == "/health":
        return await call_next(request)

    if not settings.enable_api_key_auth:
        return await call_next(request)

    api_key_header_name = settings.api_key_header.lower().replace("_", "-")
    provided_key = request.headers.get(api_key_header_name) or request.headers.get(
        settings.api_key_header
    )
    expected_key = settings.api_key

    if not expected_key:
        logger.warning(
            "API key authentication enabled but API_KEY environment variable not set"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AUTHENTICATION_CONFIGURATION_ERROR",
                "message": "API key authentication is enabled but not configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    if not provided_key:
        logger.warning(
            "Authentication failed - missing API key header",
            extra={
                "path": request.url.path,
                "method": request.method,
                "expected_header": settings.api_key_header,
            },
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "UNAUTHORIZED",
                "message": f"Missing {settings.api_key_header} header",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if provided_key != expected_key:
        logger.warning(
            "Authentication failed - invalid API key",
            extra={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "UNAUTHORIZED",
                "message": "Invalid API key",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    correlation_id = f"req_{int(start_time * 1000000)}"

    logger.info(
        f"Request started - {correlation_id}",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
        },
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"Request completed - {correlation_id}",
        extra={
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "processing_time_ms": round(process_time * 1000, 2),
        },
    )

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Processing-Time"] = str(round(process_time * 1000, 2))

    return response


@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for service monitoring."""
    lambda_context = {
        "memory_limit": os.getenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
        "function_name": os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
        "function_version": os.getenv("AWS_LAMBDA_FUNCTION_VERSION"),
        "log_group": os.getenv("AWS_LAMBDA_LOG_GROUP_NAME"),
    }
    lambda_context = {k: v for k, v in lambda_context.items() if v is not None}

    health_data: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "serverless-inference-api",
        "version": settings.api_version,
        "environment": settings.environment,
        "aws_region": settings.aws_region,
    }

    if lambda_context:
        health_data["lambda"] = lambda_context

    return health_data


@app.get("/", tags=["info"])
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    return {
        "message": settings.api_title,
        "version": settings.api_version,
        "environment": settings.environment,
        "docs": "/docs" if not is_production else "disabled",
        "health": "/health",
        "predict": "/predict",
        "model_info": "/model/info",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: Request) -> PredictionResponse:
    """Main prediction endpoint for ML model inference."""
    start_time = time.time()
    correlation_id = request.headers.get(
        "X-Correlation-ID", f"pred_{int(start_time * 1000000)}"
    )

    try:
        # Enforce the body cap before parsing — Content-Length is a fast
        # path but not authoritative, so re-check actual body length too.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_BODY_BYTES:
                    logger.warning(
                        f"Request body too large (Content-Length={content_length}) "
                        f"- {correlation_id}"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": "REQUEST_TOO_LARGE",
                            "message": (
                                f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"
                            ),
                            "correlation_id": correlation_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except ValueError:
                # Non-integer Content-Length: fall through to body-length check.
                pass

        raw_body = await request.body()
        if len(raw_body) > MAX_REQUEST_BODY_BYTES:
            logger.warning(
                f"Request body too large (actual={len(raw_body)}) - {correlation_id}"
            )
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "REQUEST_TOO_LARGE",
                    "message": f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Malformed JSON is 400, not 500 — don't page for client mistakes.
        try:
            request_data = json.loads(raw_body) if raw_body else None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON body - {correlation_id}: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_JSON",
                    "message": "Request body is not valid JSON",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Reject non-object payloads at the boundary; downstream code does
        # `'budget' in request_data` which would TypeError → 500.
        if not isinstance(request_data, dict):
            logger.warning(
                f"Non-object JSON payload ({type(request_data).__name__}) - {correlation_id}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_PAYLOAD_SHAPE",
                    "message": "Request body must be a JSON object",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        logger.info(
            f"Prediction request received - {correlation_id}",
            extra={
                "correlation_id": correlation_id,
                "features_count": len(request_data),
                "has_budget": "budget" in request_data,
                "has_runtime": "runtime" in request_data,
            },
        )

        runtime = get_runtime()

        try:
            await asyncio.to_thread(runtime.ensure_ready)
        except (
            ModelLoadError,
            ArtifactIntegrityError,
            FeatureSchemaVersionMismatch,
        ) as e:
            logger.error(f"Model refresh failed - {correlation_id}: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "SERVICE_UNAVAILABLE",
                    "message": "Model refresh failed",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        try:
            validated_request = runtime.validate_input(request_data)
        except ValidationError as e:
            logger.warning(f"Input validation failed - {correlation_id}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_INPUT",
                    "message": "Input validation failed",
                    "details": e.errors(),
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        try:
            response = await asyncio.to_thread(runtime.predict, validated_request)

            logger.info(
                f"Prediction completed - {correlation_id}",
                extra={
                    "correlation_id": correlation_id,
                    "prediction": response.prediction,
                    "model_id": response.model_id,
                    "model_version": response.model_version,
                    "processing_time_ms": response.processing_time_ms,
                },
            )

            return response

        except LiteralEvalTooLarge as e:
            # Bounded literal-eval rejected an oversize field. This is a client
            # input problem (a 400), not a server bug.
            logger.warning(f"Oversize input rejected - {correlation_id}: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INPUT_TOO_LARGE",
                    "message": str(e),
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except RuntimeError as e:
            # Don't leak internals to clients; correlation_id points ops at the logs.
            logger.error(
                f"Prediction failed - {correlation_id}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "PREDICTION_FAILED",
                    "message": (
                        f"Prediction failed; check logs for correlation_id "
                        f"{correlation_id}"
                    ),
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    except HTTPException:
        raise


@app.get("/model/info", tags=["model"])
async def get_model_info() -> Dict[str, Any]:
    """Get information about the currently loaded model."""
    try:
        runtime = get_runtime()
        engine = runtime._engine

        if not engine.is_loaded():
            model_info = runtime._loader.get_latest_approved_model_info()

            if not model_info:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "NO_MODEL_AVAILABLE",
                        "message": "No approved model available",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            return {
                "model_id": model_info.model_id,
                "version": model_info.version,
                "status": model_info.status,
                "created_at": model_info.created_at.isoformat(),
                "metrics": model_info.metrics,
                "loaded": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        model_info = engine.get_model_info()
        return {
            "model_id": model_info.model_id,
            "version": model_info.version,
            "status": model_info.status,
            "created_at": model_info.created_at,
            "metrics": model_info.metrics,
            "framework": model_info.framework,
            "loaded": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise


# Module-scope Mangum adapter. Built once on cold start; reused across every
# warm invocation. Building it per-call (the previous behavior) defeated
# Lambda's warm-start advantage.
try:
    from mangum import Mangum

    _mangum_handler = Mangum(
        app,
        lifespan="off",  # Disable lifespan for Lambda
        api_gateway_base_path=None,  # Auto-detect base path
        text_mime_types=[
            "application/json",
            "application/javascript",
            "application/xml",
            "application/vnd.api+json",
        ],
    )
except ImportError:
    # Mangum is only required in the Lambda runtime; local dev can run the app
    # via uvicorn without it. lambda_handler will surface the error if invoked.
    _mangum_handler = None


def lambda_handler(event, context):
    """AWS Lambda entry point.

    Delegates to the module-scope Mangum adapter so each invocation pays only
    the request cost, not the adapter-construction cost.
    """
    if _mangum_handler is None:
        logger.error("Mangum not installed. Required for Lambda deployment.")
        return {
            "statusCode": 500,
            "body": '{"error": "Lambda runtime configuration error"}',
            "headers": {"Content-Type": "application/json"},
        }

    logger.info(
        "Lambda invocation started",
        extra={
            "request_id": context.aws_request_id if context else "unknown",
            "function_name": context.function_name if context else "unknown",
            "remaining_time_ms": (
                context.get_remaining_time_in_millis() if context else "unknown"
            ),
            "memory_limit": context.memory_limit_in_mb if context else "unknown",
        },
    )
    response = _mangum_handler(event, context)
    logger.info("Lambda invocation completed successfully")
    return response


if __name__ == "__main__":
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info("For local testing, use: python -m uvicorn app.main:app --reload")
