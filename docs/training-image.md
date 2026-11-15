# SageMaker training image

SageMaker training runs `box_office/ml/model_training.py` in the image that
[`infrastructure/docker/training/Dockerfile`](../infrastructure/docker/training/Dockerfile)
builds. The image uses the same Python 3.14 base as the Lambda image and installs
the locked `train` extra from `uv.lock`. The stock SageMaker XGBoost container is
not used, because its Python cannot run this code ([ADR 0003](adr/0003-custom-python-3-14-training-image.md)).

The estimator reads the image URI from `SAGEMAKER_TRAINING_IMAGE_URI`. When the
variable is not set, `box-office-pipeline` stops before it launches a training job.

## Build and push

Building and pushing writes to ECR, so only Adam runs these commands. Run them
from the repository root, with AWS credentials for the account that owns the
SageMaker role.

1. Create the ECR repository. Do this once:

   ```bash
   aws ecr create-repository --region eu-north-1 \
     --repository-name box-office-training \
     --image-scanning-configuration scanOnPush=true
   ```

2. Build the image for SageMaker's `x86_64` instances, then push it:

   ```bash
   export TRAINING_IMAGE_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-north-1.amazonaws.com/box-office-training:$(git rev-parse --short HEAD)"
   aws ecr get-login-password --region eu-north-1 \
     | docker login --username AWS --password-stdin "${TRAINING_IMAGE_URI%%/*}"
   docker build --platform linux/amd64 \
     --file infrastructure/docker/training/Dockerfile \
     --tag "$TRAINING_IMAGE_URI" .
   docker push "$TRAINING_IMAGE_URI"
   ```

3. Set `SAGEMAKER_TRAINING_IMAGE_URI` to the value of `$TRAINING_IMAGE_URI` in
   `.env` and in the `ml-pipeline.yml` environment.

`make training-image` runs step 2 through `scripts/build_training_image.sh`. It
reads `TRAINING_IMAGE_URI` and refuses until `ARGS` carries the approval flag.

## Container contract

- SageMaker runs `docker run <image> train`. The `train` script on `PATH` starts
  `python -m box_office.ml.model_training`.
- Hyperparameters arrive in `/opt/ml/input/config/hyperparameters.json`. The CV
  window (`cv_folds`, `start_eval_year`, `end_year`) and the preprocessing
  artifact URIs arrive as environment variables.
- The image sets `SM_MODEL_DIR`, `SM_CHANNEL_TRAIN`, and `SM_OUTPUT_DATA_DIR` to
  the standard `/opt/ml` paths.
