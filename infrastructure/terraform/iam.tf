# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${local.name_prefix}-${var.sagemaker_execution_role_name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-sagemaker-execution-role"
    Purpose = "SageMaker training and inference execution"
  })
}

# SageMaker execution policy
resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "${local.name_prefix}-sagemaker-execution-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions for artifacts and data
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_artifacts.arn,
          "${aws_s3_bucket.sagemaker_artifacts.arn}/*"
        ]
      },
      # CloudWatch Logs permissions
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"
      },
      # ECR permissions for custom containers
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      # CloudWatch metrics permissions
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      # STS permissions for account ID retrieval
      {
        Effect = "Allow"
        Action = [
          "sts:GetCallerIdentity"
        ]
        Resource = "*"
      },
      # Comprehensive experiment tracking permissions
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateExperiment",
          "sagemaker:DescribeExperiment",
          "sagemaker:ListExperiments",
          "sagemaker:DeleteExperiment",
          "sagemaker:CreateTrial",
          "sagemaker:DescribeTrial",
          "sagemaker:ListTrials",
          "sagemaker:DeleteTrial",
          "sagemaker:CreateTrialComponent",
          "sagemaker:DescribeTrialComponent",
          "sagemaker:ListTrialComponents",
          "sagemaker:DeleteTrialComponent",
          "sagemaker:UpdateTrialComponent",
          "sagemaker:AssociateTrialComponent",
          "sagemaker:DisassociateTrialComponent",
          "sagemaker:BatchPutMetrics",
          "sagemaker:Search"
        ]
        Resource = "*"
      },
      # Model registry permissions
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackageGroups",
          "sagemaker:DeleteModelPackageGroup",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages",
          "sagemaker:UpdateModelPackage",
          "sagemaker:DeleteModelPackage"
        ]
        Resource = "*"
      },
      # Comprehensive training and inference permissions
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:ListTrainingJobs",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:ListModels",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:ListEndpointConfigs",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:ListEndpoints",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "*"
      },
      # Resource management and cleanup permissions
      {
        Effect = "Allow"
        Action = [
          "sagemaker:AddTags",
          "sagemaker:ListTags",
          "sagemaker:DeleteTags",
          "sagemaker:GetSearchSuggestions"
        ]
        Resource = "*"
      }
    ]
  })
}

# Do NOT attach AmazonSageMakerFullAccess: it grants iam:PassRole on "*"
# and is a known privilege-escalation surface. The scoped inline policy
# above is the intended replacement; tests/infrastructure/test_iam_posture.py
# fails if the managed policy comes back.

# GitHub Actions Role for CI/CD
resource "aws_iam_role" "github_actions_role" {
  count = var.github_org != "" && var.github_repo != "" ? 1 : 0

  name = "${local.name_prefix}-github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_org}/${var.github_repo}:*"
          }
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-github-actions-role"
    Purpose = "GitHub Actions CI/CD execution"
  })
}

locals {
  github_actions_sagemaker_arns = [
    "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package-group/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package/${local.name_prefix}-*/*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:training-job/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:model/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:endpoint-config/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:endpoint/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:experiment/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:trial/${local.name_prefix}-*",
    "arn:aws:sagemaker:${local.region}:${local.account_id}:trial-component/${local.name_prefix}-*",
  ]

  github_actions_lambda_arns = [
    "arn:aws:lambda:${local.region}:${local.account_id}:function:${local.name_prefix}-*",
  ]

  github_actions_log_group_arns = [
    "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/${local.name_prefix}*",
    "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/${local.name_prefix}*:*",
    "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/lambda/${local.name_prefix}-*",
    "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/lambda/${local.name_prefix}-*:*",
  ]

  github_actions_cloudwatch_alarm_arns = [
    "arn:aws:cloudwatch:${local.region}:${local.account_id}:alarm:${local.name_prefix}-*",
  ]
}

# GitHub Actions policy for ML pipeline execution
resource "aws_iam_role_policy" "github_actions_policy" {
  count = var.github_org != "" && var.github_repo != "" ? 1 : 0

  name = "${local.name_prefix}-github-actions-policy"
  role = aws_iam_role.github_actions_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Specific permissions for the SageMaker artifacts bucket
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_artifacts.arn,
          "${aws_s3_bucket.sagemaker_artifacts.arn}/*"
        ]
      },
      # Permissions for the Terraform state file
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = "arn:aws:s3:::${var.terraform_state_bucket_name}/${var.terraform_state_key}"
      },
      # Permissions for the Terraform lock table
      {
        Effect = "Allow",
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem"
        ],
        Resource = "arn:aws:dynamodb:${local.region}:${local.account_id}:table/${var.terraform_dynamodb_table}"
      },
      # Permission to pass execution roles to the services that consume them:
      # SageMaker training/endpoints, and the serverless inference Lambda.
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.sagemaker_execution_role.arn,
          module.serverless_inference.lambda_execution_role_arn,
        ]
      },
      # IAM read-only here on purpose. Granting any IAM mutation
      # (Create/Delete/PutRolePolicy, OIDC provider mgmt, etc.) would let
      # anyone with workflow-write escalate to account admin — that lives
      # in a separate, manually-applied admin stack. iam:PassRole is also
      # intentionally absent; the statement above scopes it explicitly.
      {
        Effect = "Allow"
        Action = [
          "iam:ListRoles",
          "iam:ListPolicies",
          "iam:GetRole",
          "iam:GetRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:GetOpenIDConnectProvider",
          "iam:ListOpenIDConnectProviders",
          "iam:GetServiceLinkedRoleDeletionStatus",
        ]
        Resource = "*"
      },
      # Account-wide list/discovery verbs that AWS only supports with "*".
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets",
          "sagemaker:ListExperiments",
          "sagemaker:ListTrials",
          "sagemaker:ListTrialComponents",
          "sagemaker:ListTrainingJobs",
          "sagemaker:ListModels",
          "sagemaker:ListEndpointConfigs",
          "sagemaker:ListEndpoints",
          "sagemaker:ListModelPackages",
          "sagemaker:Search",
          "sagemaker:GetSearchSuggestions",
          "logs:DescribeLogGroups",
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:DescribeResourcePolicies",
          "logs:PutResourcePolicy",
          "logs:DeleteResourcePolicy",
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "ecr:GetAuthorizationToken",
          "ecr:CreateRepository",
          "lambda:ListFunctions",
          "apigateway:GET",
          "apigateway:POST",
          "apigateway:PUT",
          "apigateway:DELETE",
          "apigateway:PATCH",
          "apigateway:UpdateRestApiPolicy",
          "apigateway:TagResource",
          "apigateway:UntagResource",
          "apigateway:UpdateAccount",
          "apigateway:GetAccount",
          "application-autoscaling:RegisterScalableTarget",
          "application-autoscaling:DeregisterScalableTarget",
          "application-autoscaling:DescribeScalableTargets",
          "application-autoscaling:PutScalingPolicy",
          "application-autoscaling:DeleteScalingPolicy",
          "application-autoscaling:DescribeScalingPolicies",
          "dynamodb:ListTables",
          "dynamodb:DescribeTable",
          "sts:GetCallerIdentity"
        ]
        Resource = "*"
      },
      # S3 bucket management for the project-owned SageMaker artifact bucket.
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning",
          "s3:GetBucketAcl",
          "s3:GetBucketPolicy",
          "s3:GetBucketCORS",
          "s3:GetBucketWebsite",
          "s3:GetBucketRequestPayment",
          "s3:GetBucketLogging",
          "s3:CreateBucket",
          "s3:DeleteBucket",
          "s3:PutBucketVersioning",
          "s3:PutEncryptionConfiguration",
          "s3:PutBucketPublicAccessBlock",
          "s3:PutBucketNotification",
          "s3:PutLifecycleConfiguration",
          "s3:GetLifecycleConfiguration",
          "s3:GetReplicationConfiguration",
          "s3:GetEncryptionConfiguration",
          "s3:GetBucketObjectLockConfiguration",
          "s3:GetBucketPublicAccessBlock",
          "s3:GetBucketNotification",
          "s3:PutBucketTagging",
          "s3:GetBucketTagging",
          "s3:DeleteBucketTagging",
          "s3:PutBucketLifecycleConfiguration",
          "s3:GetBucketLifecycleConfiguration",
          "s3:DeleteBucketLifecycleConfiguration",
          "s3:GetObjectVersion",
          "s3:DeleteObjectVersion",
          "s3:GetObjectAttributes"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_artifacts.arn,
          "${aws_s3_bucket.sagemaker_artifacts.arn}/*"
        ]
      },
      # SageMaker operations are scoped to project-prefixed resources.
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DeleteModelPackageGroup",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:CreateExperiment",
          "sagemaker:DescribeExperiment",
          "sagemaker:DeleteExperiment",
          "sagemaker:CreateTrial",
          "sagemaker:DescribeTrial",
          "sagemaker:DeleteTrial",
          "sagemaker:CreateTrialComponent",
          "sagemaker:DescribeTrialComponent",
          "sagemaker:DeleteTrialComponent",
          "sagemaker:UpdateTrialComponent",
          "sagemaker:AssociateTrialComponent",
          "sagemaker:DisassociateTrialComponent",
          "sagemaker:BatchPutMetrics",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",
          "sagemaker:UpdateModelPackage",
          "sagemaker:DeleteModelPackage",
          "sagemaker:AddTags",
          "sagemaker:ListTags",
          "sagemaker:DeleteTags",
          "sagemaker:InvokeEndpoint"
        ]
        Resource = local.github_actions_sagemaker_arns
      },
      # CloudWatch Logs are limited to this project's SageMaker and Lambda logs.
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:DeleteLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents",
          "logs:PutRetentionPolicy",
          "logs:TagResource",
          "logs:UntagResource",
          "logs:ListTagsForResource"
        ]
        Resource = local.github_actions_log_group_arns
      },
      # CloudWatch alarm mutations are scoped to project-prefixed alarms.
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:ListTagsForResource",
          "cloudwatch:TagResource",
          "cloudwatch:UntagResource"
        ]
        Resource = local.github_actions_cloudwatch_alarm_arns
      },
      # ECR image publishing is limited to the inference repository.
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:DeleteRepository",
          "ecr:PutLifecyclePolicy",
          "ecr:GetLifecyclePolicy",
          "ecr:DeleteLifecyclePolicy",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:TagResource",
          "ecr:UntagResource",
          "ecr:ListTagsForResource",
          "ecr:BatchDeleteImage",
          "ecr:SetRepositoryPolicy",
          "ecr:GetRepositoryPolicy",
          "ecr:DeleteRepositoryPolicy"
        ]
        Resource = module.serverless_inference.ecr_repository_arn
      },
      # Lambda mutations are limited to project-prefixed functions.
      {
        Effect = "Allow"
        Action = [
          "lambda:CreateFunction",
          "lambda:DeleteFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateFunctionConfiguration",
          "lambda:GetFunction",
          "lambda:GetFunctionConfiguration",
          "lambda:InvokeFunction",
          "lambda:CreateFunctionUrlConfig",
          "lambda:DeleteFunctionUrlConfig",
          "lambda:UpdateFunctionUrlConfig",
          "lambda:GetFunctionUrlConfig",
          "lambda:AddPermission",
          "lambda:RemovePermission",
          "lambda:GetPolicy",
          "lambda:TagResource",
          "lambda:UntagResource",
          "lambda:ListTags",
          "lambda:PublishVersion",
          "lambda:CreateAlias",
          "lambda:DeleteAlias",
          "lambda:UpdateAlias",
          "lambda:GetAlias",
          "lambda:ListAliases",
          "lambda:ListVersionsByFunction",
          "lambda:PutConcurrency",
          "lambda:DeleteConcurrency",
          "lambda:GetProvisionedConcurrencyConfig",
          "lambda:PutProvisionedConcurrencyConfig",
          "lambda:DeleteProvisionedConcurrencyConfig"
        ]
        Resource = local.github_actions_lambda_arns
      }
    ]
  })

  # The CI role is intentionally IAM-read-only and cannot rewrite its own
  # policy (iam:PutRolePolicy is withheld by design). This policy is therefore
  # managed out-of-band with elevated credentials; the CI pipeline must not try
  # to modify it, so ignore drift here to keep deploys from failing on
  # PutRolePolicy. Edit + apply this with admin creds when the policy changes.
  lifecycle {
    ignore_changes = [policy]
  }
}
