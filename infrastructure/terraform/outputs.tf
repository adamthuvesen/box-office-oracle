# terraform/outputs.tf

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = local.account_id
}

output "sagemaker_execution_role_arn" {
  description = "The ARN of the SageMaker execution role."
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_artifacts_bucket" {
  description = "Name of the S3 bucket for SageMaker artifacts"
  value       = aws_s3_bucket.sagemaker_artifacts.bucket
}

output "sagemaker_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for SageMaker artifacts"
  value       = aws_s3_bucket.sagemaker_artifacts.arn
}

output "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.box_office_models.model_package_group_name
}

output "model_package_group_arn" {
  description = "ARN of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.box_office_models.arn
}

output "github_actions_role_arn" {
  description = "The ARN of the IAM role for GitHub Actions CI/CD."
  value       = var.github_org != "" && var.github_repo != "" ? aws_iam_role.github_actions_role[0].arn : null
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for SageMaker"
  value       = aws_cloudwatch_log_group.sagemaker_logs.name
}

# Serverless Inference API Outputs
output "inference_api_lambda_function_name" {
  description = "Name of the inference API Lambda function"
  value       = module.serverless_inference.lambda_function_name
}

output "inference_api_lambda_function_arn" {
  description = "ARN of the inference API Lambda function"
  value       = module.serverless_inference.lambda_function_arn
}

output "inference_api_ecr_repository_url" {
  description = "URL of the ECR repository for inference API"
  value       = module.serverless_inference.ecr_repository_url
}

output "inference_api_endpoints" {
  description = "Available endpoints for the inference API"
  value       = module.serverless_inference.endpoints
}

output "inference_api_function_url" {
  description = "URL of the Lambda function for inference API"
  value       = module.serverless_inference.lambda_function_url
}

output "inference_api_cloudwatch_alarms" {
  description = "CloudWatch alarm names and ARNs for inference API"
  value       = module.serverless_inference.cloudwatch_alarms
}
