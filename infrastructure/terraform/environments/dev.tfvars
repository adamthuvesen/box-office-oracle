# Development environment configuration
environment = "dev"
aws_region  = "eu-north-1"

# Project configuration
project_name = "box-office"

# Lambda configuration
# Note: lambda_image_uri will be provided dynamically by CI/CD pipeline
# The pipeline builds and pushes the image, then passes the URI via -var
lambda_image_uri   = ""
lambda_memory_size = 3008 # More memory = more vCPU, needed for cold-start model download + load
lambda_timeout     = 120  # Cold start downloads + loads the model from the registry; 30s is too tight
log_level          = "DEBUG"

# Use Lambda Function URL
enable_function_url = true

# Monitoring
alarm_sns_topic_arn = ""

# Example value only. Use an untracked tfvars override for real deployments
# (never commit the real bucket name — it embeds the AWS account ID).
terraform_state_bucket_name = "example-box-office-dev-terraform-state"

# Optional features (disabled for dev)
enable_xray_tracing  = false
reserved_concurrency = null

# api_key authenticates /predict and /model/info (/health is always open). It is
# a secret — inject via TF_VAR_api_key, never commit it. Only non-secret config below.
#
# box_office/__init__.py eagerly builds config = Settings() on any `import
# box_office`, and Settings marks these fields required (enforced by
# tests/test_config.py). The inference serving path never touches Snowflake or
# the SageMaker role — it reads S3/model-registry with the Lambda role — so these
# are inert placeholders that only satisfy import-time validation.
additional_environment_variables = {
  DEBUG_MODE        = "true"
  SAGEMAKER_ROLE_ARN = "unused-by-inference"
  SNOWFLAKE_USER     = "unused-by-inference"
  SNOWFLAKE_ACCOUNT  = "unused-by-inference"
  SNOWFLAKE_DATABASE = "unused-by-inference"
}
