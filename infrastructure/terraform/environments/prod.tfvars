# Production environment configuration
environment = "prod"
aws_region  = "eu-north-1"

# Project configuration
project_name = "box-office"

# Lambda configuration
lambda_memory_size = 3008 # Higher memory for better performance
lambda_timeout     = 30
log_level          = "INFO"

# Use Lambda Function URL
enable_function_url = true

# Monitoring
# Left empty so the committed config applies cleanly. For real prod, create the
# SNS topic with a confirmed subscription and set this via an untracked override
# (or TF_VAR_alarm_sns_topic_arn); an empty value disables alarm actions:
#   aws sns create-topic --name box-office-prod-alarms --region eu-north-1
#   aws sns subscribe --topic-arn <arn> --protocol email --notification-endpoint ops@example.com
alarm_sns_topic_arn = ""

# Example value only. Use an untracked tfvars override for real deployments
# (never commit the real bucket name — it embeds the AWS account ID).
terraform_state_bucket_name = "example-box-office-prod-terraform-state"

# Optional features
enable_xray_tracing  = true
reserved_concurrency = null

# api_key is required (enable_function_url = true) but is a secret — inject it via
# TF_VAR_api_key from a secret store, never here. Only non-secret config below.
additional_environment_variables = {
  ENABLE_METRICS = "true"
}
