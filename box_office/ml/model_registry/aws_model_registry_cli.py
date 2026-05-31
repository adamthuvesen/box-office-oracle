#!/usr/bin/env python3
"""
AWS SageMaker Model Registry CLI tool.

Simple command-line interface for managing models in AWS SageMaker Model Registry.

To list all models in the registry:
    - python box_office/ml/model_registry/aws_model_registry_cli.py list-models
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Dict

from botocore.exceptions import BotoCoreError, ClientError

from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.config import config
from box_office.utils.cli_prompts import confirm


logger = logging.getLogger(__name__)


def load_config() -> Dict[str, str]:
    """Load AWS configuration from config system."""
    return {"aws_region": config.aws.region}


def _cli_exit(message: str, exc: BaseException | None = None) -> None:
    text = f"{message}: {exc}" if exc is not None else message
    logger.error(text)
    print(f"Error: {text}", file=sys.stderr)
    sys.exit(1)


def cmd_list_groups(args: argparse.Namespace) -> None:
    """List model package groups."""
    aws_config = load_config()
    registry = AWSModelRegistry(region_name=aws_config["aws_region"])

    group_name = args.group_name or AWSModelRegistry.get_model_group_name()

    if group_name:
        try:
            response = registry.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            print(f"Group '{group_name}' exists")
            print(f"Created: {response['CreationTime']}")
            print(
                f"Description: {response.get('ModelPackageGroupDescription', 'No description')}"
            )
            print(f"ARN: {response['ModelPackageGroupArn']}")
        except ClientError as e:
            logger.info("describe_model_package_group failed: %s", e)
            print(f"Group '{group_name}' not found or AWS error: {e}")
        except BotoCoreError as e:
            _cli_exit("AWS transport error while describing group", e)
        return

    print("Use --group-name to check if a specific group exists")
    print(f"Default group name: {AWSModelRegistry.get_model_group_name()}")
    print(f"Dev group name: {AWSModelRegistry.get_model_group_name('dev')}")
    print(f"Prod group name: {AWSModelRegistry.get_model_group_name('prod')}")


def cmd_list_models(args: argparse.Namespace) -> None:
    """List model packages."""
    aws_config = load_config()
    registry = AWSModelRegistry(region_name=aws_config["aws_region"])

    group_name = args.group_name or AWSModelRegistry.get_model_group_name()

    try:
        models = registry.list_model_packages(
            model_package_group_name=group_name,
            approval_status=args.status,
            max_results=args.limit,
        )

        if not models:
            print("No models found")
            print(f"Group: {group_name}")
            if args.status:
                print(f"Status filter: {args.status}")
            return

        print(f"Found {len(models)} model package(s)")
        print(f"Group: {group_name}")
        if args.status:
            print(f"Status filter: {args.status}")

        print()
        for i, model in enumerate(models, 1):
            model_id = model["ModelPackageArn"].split("/")[-1]
            print(f"{i}. Model ID: {model_id}")
            print(f"Full ARN: {model['ModelPackageArn']}")
            print(f"Created: {model['CreationTime']}")
            print(f"Status: {model['ModelApprovalStatus']}")
            if "ModelPackageDescription" in model:
                print(f"Description: {model['ModelPackageDescription']}")
            print()

    except (ClientError, BotoCoreError) as e:
        _cli_exit("Failed to list model packages", e)


def cmd_get_model(args: argparse.Namespace) -> None:
    """Get detailed model package information."""
    aws_config = load_config()
    registry = AWSModelRegistry(region_name=aws_config["aws_region"])

    try:
        response = registry.sagemaker_client.describe_model_package(
            ModelPackageName=args.model_arn
        )

        print("Model Package Details:")
        print(f"ARN: {response['ModelPackageArn']}")
        print(f"Created: {response['CreationTime']}")
        print(f"Status: {response['ModelApprovalStatus']}")

        if "ModelPackageDescription" in response:
            print(f"Description: {response['ModelPackageDescription']}")

        if "ModelPackageGroupName" in response:
            print(f"Group: {response['ModelPackageGroupName']}")

        if "CustomerMetadataProperties" in response:
            print("Custom Metadata:")
            for key, value in response["CustomerMetadataProperties"].items():
                print(f"{key}: {value}")

        if "InferenceSpecification" in response:
            inference_spec = response["InferenceSpecification"]
            print("Inference Containers:")
            for i, container in enumerate(inference_spec.get("Containers", []), 1):
                print(f"{i}. Image: {container.get('Image', 'Unknown')}")
                if "Framework" in container:
                    print(
                        f"         Framework: {container['Framework']} {container.get('FrameworkVersion', '')}"
                    )

    except (ClientError, BotoCoreError) as e:
        _cli_exit("Failed to describe model package", e)


def cmd_promote_model(args: argparse.Namespace) -> None:
    """Promote a model package to Approved status."""
    aws_config = load_config()
    registry = AWSModelRegistry(region_name=aws_config["aws_region"])

    try:
        if not args.yes:
            print("About to promote model package:")
            print(f"ARN: {args.model_arn}")
            print("New Status: Approved")

            if not confirm("\nProceed with promotion? (y/N): "):
                print("Promotion cancelled")
                return

        result = registry.update_model_approval_status(
            model_package_arn=args.model_arn,
            approval_status="Approved",
            approval_description=args.description
            or f"Manual promotion via CLI - {datetime.now(timezone.utc).isoformat()}",
        )

        if result["status"] == "success":
            print("Model package promoted successfully!")
            print(f"ARN: {args.model_arn}")
            print("Status: Approved")
        else:
            print(f"Promotion failed: {result.get('error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPromotion cancelled", file=sys.stderr)
        sys.exit(130)
    except (ClientError, BotoCoreError) as e:
        _cli_exit("Failed to promote model package", e)


def cmd_get_latest_approved(args: argparse.Namespace) -> None:
    """Get the latest approved model from a group."""
    aws_config = load_config()
    registry = AWSModelRegistry(region_name=aws_config["aws_region"])

    group_name = args.group_name or AWSModelRegistry.get_model_group_name()

    try:
        model = registry.get_latest_approved_model(group_name)

        if not model:
            print(f"No approved models found in group '{group_name}'")
            return

        print("Latest Approved Model:")
        print(f"ARN: {model['ModelPackageArn']}")
        print(f"Created: {model['CreationTime']}")
        print(f"Status: {model['ModelApprovalStatus']}")

        if "ModelPackageDescription" in model:
            print(f"Description: {model['ModelPackageDescription']}")

    except (ClientError, BotoCoreError) as e:
        _cli_exit("Failed to get latest approved model", e)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AWS SageMaker Model Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list-groups --group-name box-office-dev-box-office-models
  %(prog)s list-models --group-name box-office-dev-box-office-models
  %(prog)s list-models --group-name box-office-dev-box-office-models --status Approved
  %(prog)s list-models  # Uses default group name based on ENVIRONMENT
  %(prog)s get-model arn:aws:sagemaker:region:account:model-package/package-name
  %(prog)s promote-model arn:aws:sagemaker:region:account:model-package/package-name
  %(prog)s get-latest-approved --group-name box-office-dev-box-office-models
  %(prog)s get-latest-approved  # Uses default group name based on ENVIRONMENT

Environment Variables:
  ENVIRONMENT    - Environment name (dev, prod, etc.) - defaults to 'dev'
  PROJECT_NAME   - Project name - defaults to 'box-office'

  Default group name: {PROJECT_NAME}-{ENVIRONMENT}-box-office-models
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_list_groups = subparsers.add_parser(
        "list-groups", help="Check if model package group exists"
    )
    parser_list_groups.add_argument(
        "--group-name", help="Model package group name to check (optional)"
    )
    parser_list_groups.set_defaults(func=cmd_list_groups)

    parser_list_models = subparsers.add_parser(
        "list-models", help="List model packages with full ARNs"
    )
    parser_list_models.add_argument(
        "--group-name",
        help="Filter by model package group name (optional, uses default)",
    )
    parser_list_models.add_argument(
        "--status",
        choices=["Approved", "Rejected", "PendingManualApproval"],
        help="Filter by approval status",
    )
    parser_list_models.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )
    parser_list_models.set_defaults(func=cmd_list_models)

    parser_get_model = subparsers.add_parser(
        "get-model", help="Get detailed model package information"
    )
    parser_get_model.add_argument("model_arn", help="Model package ARN")
    parser_get_model.set_defaults(func=cmd_get_model)

    parser_promote = subparsers.add_parser(
        "promote-model", help="Promote model package to Approved status"
    )
    parser_promote.add_argument("model_arn", help="Model package ARN to promote")
    parser_promote.add_argument("--description", help="Approval description")
    parser_promote.add_argument(
        "--yes", action="store_true", help="Skip confirmation prompt"
    )
    parser_promote.set_defaults(func=cmd_promote_model)

    parser_latest = subparsers.add_parser(
        "get-latest-approved", help="Get latest approved model from group"
    )
    parser_latest.add_argument(
        "--group-name", help="Model package group name (optional, uses default)"
    )
    parser_latest.set_defaults(func=cmd_get_latest_approved)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
