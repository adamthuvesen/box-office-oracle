# Multi-stage build for Lambda inference container
# Build context: project root
# Pin to a manifest-list digest; buildx selects arm64/amd64.
# Resolved 2026-04-26 via: docker buildx imagetools inspect public.ecr.aws/lambda/python:3.12
FROM public.ecr.aws/lambda/python@sha256:4fe0754d2499f7c4234fb55cf408a390af7099651e8fbf4fca2999d2be911253 AS builder

# Install uv from the official image instead of bootstrapping it with pip.
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /uvx /bin/

# Copy dependency files and package code
COPY pyproject.toml uv.lock README.md ./
COPY box_office/ ./box_office/

# Install dependencies into a dedicated target directory so the runtime stage
# copies only runtime deps.
RUN uv pip install --target /opt/python ".[inference]"

# Stage 2: Runtime — copy only the dedicated target tree, not the builder's
# system site-packages or /var/lang/bin.
FROM public.ecr.aws/lambda/python@sha256:4fe0754d2499f7c4234fb55cf408a390af7099651e8fbf4fca2999d2be911253

# Copy installed runtime packages from the dedicated target dir.
COPY --from=builder /opt/python /opt/python

# Copy application code (inference API)
COPY box_office/inference/app/ ${LAMBDA_TASK_ROOT}/app/
COPY box_office/inference/__init__.py ${LAMBDA_TASK_ROOT}/

# Set Python path and environment variables
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}:/opt/python"
ENV PYTHONUNBUFFERED=1

# Run as non-root. The Lambda Python base image runs as root by default;
# any code-exec bug in the inference handler would otherwise execute with
# root inside the container. UID 1001 is a conventional non-root choice; chown
# ensures the application can read its task root and runtime deps.
RUN chown -R 1001:1001 ${LAMBDA_TASK_ROOT} \
 && chown -R 1001:1001 /opt/python
USER 1001

# Set the CMD to your handler
CMD ["app.main.lambda_handler"]
