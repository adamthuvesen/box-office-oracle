#!/usr/bin/env bash
# Build the SageMaker training image and push it to ECR. Spends: it writes to ECR.
# Refuses unless the approval flag is passed. Needs TRAINING_IMAGE_URI, the full
# ECR image URI with tag (docs/training-image.md).
set -euo pipefail

cd "$(dirname "$0")/.."

approved=false
for arg in "$@"; do
  case "$arg" in
    --approved) approved=true ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$approved" != true ]]; then
  echo "refused: building and pushing the training image spends money or writes to production. Pass --approved." >&2
  exit 1
fi

: "${TRAINING_IMAGE_URI:?set TRAINING_IMAGE_URI to <account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>}"
registry="${TRAINING_IMAGE_URI%%/*}"
region="$(cut -d. -f4 <<<"$registry")"

aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry"
docker build --platform linux/amd64 --file infrastructure/docker/training/Dockerfile --tag "$TRAINING_IMAGE_URI" .
docker push "$TRAINING_IMAGE_URI"
echo "pushed $TRAINING_IMAGE_URI; set SAGEMAKER_TRAINING_IMAGE_URI to it"
