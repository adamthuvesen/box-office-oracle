# terraform/modules/serverless-inference/iam.tf

# Lambda execution role
resource "aws_iam_role" "lambda_execution" {
  name = "${local.name_prefix}-inference-lambda-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-inference-lambda-execution-role"
    Purpose = "Lambda execution role for serverless inference API"
  })
}

# Basic Lambda execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# CloudWatch Logs policy
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Custom policy for Lambda function
resource "aws_iam_role_policy" "lambda_inference_policy" {
  name = "${local.name_prefix}-inference-lambda-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions for model artifacts (read-only)
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      },
      # SageMaker Model Registry permissions (read-only)
      {
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackages",
          "sagemaker:DescribeModelPackage"
        ]
        Resource = [
          "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package-group/${var.model_registry_group_name}",
          "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package/${var.model_registry_group_name}/*"
        ]
      },
      # CloudWatch Logs permissions
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/lambda/${local.name_prefix}-inference-api:*"
      },
      # CloudWatch Metrics permissions
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = [
              "AWS/Lambda",
              "${var.project_name}/InferenceAPI"
            ]
          }
        }
      },
      # Systems Manager Parameter Store permissions (for configuration)
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = [
          "arn:aws:ssm:${local.region}:${local.account_id}:parameter/${var.project_name}/${var.environment}/*"
        ]
      },
      # KMS permissions for encrypted parameters/S3 objects
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = [
              "s3.${local.region}.amazonaws.com",
              "ssm.${local.region}.amazonaws.com"
            ]
          }
        }
      }
    ]
  })
}
