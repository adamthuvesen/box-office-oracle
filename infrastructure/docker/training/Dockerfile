# SageMaker training image: the repository's training code and the locked
# `train` dependencies on the same Python 3.14 base as the Lambda image.
# Build context: project root. SageMaker runs `docker run <image> train`.
# Resolved 2026-11-12 via: docker buildx imagetools inspect public.ecr.aws/lambda/python:3.14
FROM public.ecr.aws/lambda/python@sha256:e6e52fb3a611579eac34a9b97c8970700552334c94d5ce6a045d4cdf833889b1

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /opt/program
COPY pyproject.toml uv.lock README.md ./
COPY box_office/ ./box_office/

RUN uv export --frozen --no-dev --extra train --no-emit-project --output-file /tmp/requirements.txt \
 && uv pip install --system --requirement /tmp/requirements.txt \
 && uv pip install --system --no-deps . \
 && rm /tmp/requirements.txt

COPY infrastructure/docker/training/train /usr/local/bin/train
RUN chmod 0755 /usr/local/bin/train

# The SageMaker container contract paths, read by box_office.ml.model_training.
ENV SM_MODEL_DIR=/opt/ml/model \
    SM_CHANNEL_TRAIN=/opt/ml/input/data/train \
    SM_OUTPUT_DATA_DIR=/opt/ml/output/data \
    PYTHONUNBUFFERED=1

# The Lambda base sets a Lambda entrypoint; SageMaker needs `train` on PATH instead.
ENTRYPOINT []
CMD ["train"]
