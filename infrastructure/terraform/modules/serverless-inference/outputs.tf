# terraform/modules/serverless-inference/outputs.tf

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = var.lambda_image_uri != "" ? aws_lambda_function.inference_api[0].function_name : null
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = var.lambda_image_uri != "" ? aws_lambda_function.inference_api[0].arn : null
}

output "lambda_function_invoke_arn" {
  description = "Invoke ARN of the Lambda function"
  value       = var.lambda_image_uri != "" ? aws_lambda_function.inference_api[0].invoke_arn : null
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_execution.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.inference_api.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.inference_api.arn
}

output "lambda_function_url" {
  description = "URL of the Lambda function (if enabled)"
  value       = var.enable_function_url && var.lambda_image_uri != "" ? aws_lambda_function_url.inference_api[0].function_url : null
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for Lambda"
  value       = aws_cloudwatch_log_group.inference_api.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for Lambda"
  value       = aws_cloudwatch_log_group.inference_api.arn
}

output "cloudwatch_alarms" {
  description = "CloudWatch alarm names and ARNs"
  value = {
    lambda_errors = var.lambda_image_uri != "" ? {
      name = aws_cloudwatch_metric_alarm.lambda_errors[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.lambda_errors[0].arn
    } : null
    lambda_duration = var.lambda_image_uri != "" ? {
      name = aws_cloudwatch_metric_alarm.lambda_duration[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.lambda_duration[0].arn
    } : null
    lambda_throttles = var.lambda_image_uri != "" ? {
      name = aws_cloudwatch_metric_alarm.lambda_throttles[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.lambda_throttles[0].arn
    } : null
  }
}

output "endpoints" {
  description = "Available endpoints for the inference API"
  value = {
    predict    = var.enable_function_url && var.lambda_image_uri != "" ? "${aws_lambda_function_url.inference_api[0].function_url}predict" : null
    health     = var.enable_function_url && var.lambda_image_uri != "" ? "${aws_lambda_function_url.inference_api[0].function_url}health" : null
    model_info = var.enable_function_url && var.lambda_image_uri != "" ? "${aws_lambda_function_url.inference_api[0].function_url}model/info" : null
  }
}
