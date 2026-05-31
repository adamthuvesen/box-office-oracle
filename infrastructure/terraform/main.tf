terraform {
  # >= 1.2 for lifecycle precondition/postcondition blocks (used by the
  # serverless-inference module to require an API key for the public URL).
  required_version = ">= 1.2"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local values
locals {
  account_id  = data.aws_caller_identity.current.account_id
  region      = data.aws_region.current.name
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# S3 Bucket for SageMaker artifacts
resource "aws_s3_bucket" "sagemaker_artifacts" {
  bucket = "${local.name_prefix}-sagemaker-artifacts-${local.account_id}"

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-sagemaker-artifacts"
    Purpose = "SageMaker training data and model artifacts"
  })
}

resource "aws_s3_bucket_versioning" "sagemaker_artifacts" {
  bucket = aws_s3_bucket.sagemaker_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Expire noncurrent versions so deleted or overwritten artifacts do not
# accumulate indefinitely alongside bucket versioning.
resource "aws_s3_bucket_lifecycle_configuration" "sagemaker_artifacts" {
  bucket = aws_s3_bucket.sagemaker_artifacts.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  depends_on = [aws_s3_bucket_versioning.sagemaker_artifacts]
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sagemaker_artifacts" {
  bucket = aws_s3_bucket.sagemaker_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "sagemaker_artifacts" {
  bucket = aws_s3_bucket.sagemaker_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# SageMaker Model Package Group
resource "aws_sagemaker_model_package_group" "box_office_models" {
  model_package_group_name        = "${local.name_prefix}-box-office-models"
  model_package_group_description = "Box office prediction models for ${var.environment}"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-model-package-group"
  })
}

# CloudWatch Log Group for SageMaker
resource "aws_cloudwatch_log_group" "sagemaker_logs" {
  name              = "/aws/sagemaker/${local.name_prefix}"
  retention_in_days = var.environment == "prod" ? 30 : 14

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-sagemaker-logs"
  })
}

# ------------------------------------------------------------------------------
# Serverless Inference Module
# ------------------------------------------------------------------------------

module "serverless_inference" {
  source = "./modules/serverless-inference"

  project_name              = var.project_name
  environment               = var.environment
  model_registry_group_name = aws_sagemaker_model_package_group.box_office_models.model_package_group_name
  s3_bucket_name            = aws_s3_bucket.sagemaker_artifacts.bucket

  # Lambda configuration
  lambda_image_uri   = var.lambda_image_uri
  lambda_memory_size = var.lambda_memory_size
  lambda_timeout     = var.lambda_timeout
  log_level          = var.log_level

  # Function URL configuration
  enable_function_url = var.enable_function_url

  # Monitoring
  alarm_sns_topic_arn = var.alarm_sns_topic_arn

  # Optional configurations
  api_key                          = var.api_key
  vpc_config                       = var.vpc_config
  reserved_concurrency             = var.reserved_concurrency
  enable_xray_tracing              = var.enable_xray_tracing
  dead_letter_config               = var.dead_letter_config
  additional_environment_variables = var.additional_environment_variables
}
