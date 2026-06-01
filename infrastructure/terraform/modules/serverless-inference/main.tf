# terraform/modules/serverless-inference/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
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
    Component   = "serverless-inference"
    ManagedBy   = "terraform"
  }
}

# ECR repository for storing inference Docker images.
resource "aws_ecr_repository" "inference_api" {
  name                 = "${local.name_prefix}-inference-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-inference-api-ecr"
    Purpose = "Container images for serverless inference API"
  })
}

resource "aws_ecr_lifecycle_policy" "inference_api" {
  repository = aws_ecr_repository.inference_api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "inference_api" {
  name              = "/aws/lambda/${local.name_prefix}-inference-api"
  retention_in_days = var.environment == "prod" ? 30 : 14

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-inference-api-logs"
  })
}

# Lambda function (container-based with simplified Dockerfile)
resource "aws_lambda_function" "inference_api" {
  count = var.lambda_image_uri != "" ? 1 : 0

  function_name = "${local.name_prefix}-inference-api"
  role          = aws_iam_role.lambda_execution.arn

  package_type = "Image"
  image_uri    = var.lambda_image_uri

  memory_size                    = var.lambda_memory_size
  timeout                        = var.lambda_timeout
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    # api_key is merged last so a stray API_KEY in additional_environment_variables
    # can never shadow the injected secret.
    variables = merge(
      {
        MODEL_REGISTRY_GROUP_NAME = "${var.project_name}-${var.environment}-box-office-models"
        S3_BUCKET_NAME            = var.s3_bucket_name
        LOG_LEVEL                 = var.log_level
        ENVIRONMENT               = var.environment
        PROJECT_NAME              = var.project_name
      },
      var.additional_environment_variables,
      var.api_key != "" ? { API_KEY = var.api_key } : {}
    )
  }

  tracing_config {
    mode = var.enable_xray_tracing ? "Active" : "PassThrough"
  }

  dynamic "vpc_config" {
    for_each = var.vpc_config != null ? [var.vpc_config] : []
    content {
      subnet_ids         = vpc_config.value.subnet_ids
      security_group_ids = vpc_config.value.security_group_ids
    }
  }

  dynamic "dead_letter_config" {
    for_each = var.dead_letter_config != null ? [var.dead_letter_config] : []
    content {
      target_arn = dead_letter_config.value.target_arn
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_logs,
    aws_cloudwatch_log_group.inference_api,
  ]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-inference-api-lambda"
  })
}

# Public Lambda function URL.
#
# authorization_type is NONE at the network layer; the application enforces an
# X-API-Key check on every route except /health (fail-closed: it returns 500 if
# the key is enabled but unset). The precondition below refuses to create a public
# URL unless an API_KEY is actually provided, so a forgotten key fails the apply
# instead of shipping a reachable endpoint.
resource "aws_lambda_function_url" "inference_api" {
  count              = var.enable_function_url && var.lambda_image_uri != "" ? 1 : 0
  function_name      = aws_lambda_function.inference_api[0].function_name
  authorization_type = "NONE"

  lifecycle {
    precondition {
      condition     = var.api_key != ""
      error_message = "enable_function_url is set but no api_key was provided. A public inference URL must be protected by an API key — inject one via TF_VAR_api_key."
    }
  }

  cors {
    allow_credentials = false
    allow_origins     = ["*"]
    allow_methods     = ["GET", "POST"]
    allow_headers     = ["date", "keep-alive", "content-type", "x-api-key"]
    expose_headers    = ["date", "keep-alive"]
    max_age           = 86400
  }
}

# CloudWatch Alarms for monitoring (conditional based on Lambda function existence)
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  count               = var.lambda_image_uri != "" ? 1 : 0
  alarm_name          = "${local.name_prefix}-inference-api-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors lambda errors"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    FunctionName = aws_lambda_function.inference_api[0].function_name
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  count               = var.lambda_image_uri != "" ? 1 : 0
  alarm_name          = "${local.name_prefix}-inference-api-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Average"
  threshold           = "25000" # 25 seconds (close to 30s timeout)
  alarm_description   = "This metric monitors lambda duration"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    FunctionName = aws_lambda_function.inference_api[0].function_name
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  count               = var.lambda_image_uri != "" ? 1 : 0
  alarm_name          = "${local.name_prefix}-inference-api-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "This metric monitors lambda throttles"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    FunctionName = aws_lambda_function.inference_api[0].function_name
  }

  tags = local.common_tags
}
