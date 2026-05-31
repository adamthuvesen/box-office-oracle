# Development environment configuration
environment = "dev"
aws_region  = "eu-north-1"

# Project configuration
project_name = "box-office"

# Lambda configuration
# Note: lambda_image_uri will be provided dynamically by CI/CD pipeline
# The pipeline builds and pushes the image, then passes the URI via -var
lambda_image_uri   = ""
lambda_memory_size = 2048 # Lower memory for dev to save costs
lambda_timeout     = 30
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
additional_environment_variables = {
  DEBUG_MODE = "true"
}
