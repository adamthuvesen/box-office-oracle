# terraform/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be either 'dev' or 'prod'."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "box-office"
}

variable "sagemaker_execution_role_name" {
  description = "Name for the SageMaker execution role"
  type        = string
  default     = "sagemaker-execution-role"
}

variable "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  type        = string
  default     = "box-office-dev-box-office-models"
}

variable "github_org" {
  description = "GitHub organization name for CI/CD"
  type        = string
  default     = "adamthuvesen"
}

variable "github_repo" {
  description = "GitHub repository name for CI/CD"
  type        = string
  default     = "box-office-oracle"
}

variable "terraform_state_bucket_name" {
  # No default: account-specific bucket names belong in tfvars, not in the
  # shared variable file.
  description = "The name of the S3 bucket for Terraform state. Must be supplied per environment via tfvars."
  type        = string
}

variable "terraform_state_key" {
  description = "The key (path) for the Terraform state file in S3."
  type        = string
  default     = "box-office/terraform.tfstate"
}

variable "terraform_dynamodb_table" {
  description = "The name of the DynamoDB table for Terraform state locking."
  type        = string
  default     = "terraform-state-lock"
}

# Serverless Inference API Variables
variable "lambda_image_uri" {
  description = "URI of the container image for Lambda function"
  type        = string
  default     = ""
}

variable "lambda_memory_size" {
  description = "Memory size for Lambda function in MB"
  type        = number
  default     = 3008
  validation {
    condition     = var.lambda_memory_size >= 128 && var.lambda_memory_size <= 10240
    error_message = "Lambda memory size must be between 128 and 10240 MB."
  }
}

variable "lambda_timeout" {
  description = "Timeout for Lambda function in seconds"
  type        = number
  default     = 30
  validation {
    condition     = var.lambda_timeout >= 1 && var.lambda_timeout <= 900
    error_message = "Lambda timeout must be between 1 and 900 seconds."
  }
}

variable "log_level" {
  description = "Log level for the application"
  type        = string
  default     = "INFO"
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
  }
}

variable "enable_function_url" {
  description = "Whether to create Lambda function URL"
  type        = bool
  default     = true
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms"
  type        = string
  default     = ""
}

variable "vpc_config" {
  description = "VPC configuration for Lambda function"
  type = object({
    subnet_ids         = list(string)
    security_group_ids = list(string)
  })
  default = null
}

variable "reserved_concurrency" {
  description = "Reserved concurrent executions for the Lambda function (null = unreserved)"
  type        = number
  default     = null
}

variable "enable_xray_tracing" {
  description = "Whether to enable X-Ray tracing for Lambda function"
  type        = bool
  default     = false
}

variable "dead_letter_config" {
  description = "Dead letter queue configuration"
  type = object({
    target_arn = string
  })
  default = null
}

variable "api_key" {
  description = "API key for the inference X-API-Key auth. Required when enable_function_url is true; inject via TF_VAR_api_key from a secret, never commit it."
  type        = string
  default     = ""
  sensitive   = true
}

variable "additional_environment_variables" {
  description = "Extra non-secret environment variables merged into the inference Lambda (e.g. ENABLE_METRICS)."
  type        = map(string)
  default     = {}
}
