# Train on a custom Python 3.14 image, not the stock SageMaker XGBoost container

Status: accepted (2026-11-12)

The training job used the stock SageMaker XGBoost container (`framework_version="1.7-1"`). Its Python cannot run this code, which needs Python 3.14, numpy 2, and XGBoost 3.2. We build our own image from `infrastructure/docker/training/Dockerfile`. It uses the same Python 3.14 Lambda base as the serving image and installs the locked `train` extra, so training and serving run the same dependency versions. The estimator reads the image URI from `SAGEMAKER_TRAINING_IMAGE_URI`, and `box-office-pipeline` stops before it launches a job when the variable is not set.

## Considered options

- Pin the code to what the stock container runs (old Python, numpy 1.x, XGBoost 1.7). Rejected: it would split the training stack from the serving and local stacks, and a model pickled on one XGBoost version can fail to load on another.

## Consequences

- We own an ECR repository and an image to rebuild when the lock changes or the base image moves. A rebuild writes to ECR, so it needs approval (`make training-image` refuses without it).
- The `train` script on `PATH` is the container contract. SageMaker runs `docker run <image> train` ([docs/training-image.md](../training-image.md)).
