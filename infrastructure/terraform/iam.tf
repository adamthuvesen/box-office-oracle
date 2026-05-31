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
      # TODO: scope Lambda/ECR/SageMaker/CloudWatch actions below to
      # ${local.name_prefix}-* ARNs. Account-wide list verbs
      # (s3:ListAllMyBuckets, sts:GetCallerIdentity) will always need "*".
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets",
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
          "s3:GetObjectAttributes",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DeleteModelPackageGroup",
          "sagemaker:DescribeModelPackageGroup",
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
          "sagemaker:Search",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:ListTrainingJobs",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:ListModels",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:ListEndpoints",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages",
          "sagemaker:UpdateModelPackage",
          "sagemaker:DeleteModelPackage",
          "sagemaker:AddTags",
          "sagemaker:ListTags",
          "sagemaker:DeleteTags",
          "sagemaker:GetSearchSuggestions",
          "sagemaker:InvokeEndpoint",
          "logs:CreateLogGroup",
          "logs:DeleteLogGroup",
          "logs:DescribeLogGroups",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
          "logs:GetLogEvents",
          "logs:FilterLogEvents",
          "logs:PutRetentionPolicy",
          "logs:TagResource",
          "logs:UntagResource",
          "logs:ListTagsForResource",
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
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:ListTagsForResource",
          "cloudwatch:TagResource",
          "cloudwatch:UntagResource",
          # ECR permissions for inference container registry
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:CreateRepository",
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
          "ecr:DeleteRepositoryPolicy",
          # Lambda permissions for inference function
          "lambda:CreateFunction",
          "lambda:DeleteFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateFunctionConfiguration",
          "lambda:GetFunction",
          "lambda:GetFunctionConfiguration",
          "lambda:ListFunctions",
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
          "lambda:DeleteProvisionedConcurrencyConfig",
          # API Gateway v2 permissions for inference API
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
          # Application Auto Scaling (for Lambda concurrency)
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
