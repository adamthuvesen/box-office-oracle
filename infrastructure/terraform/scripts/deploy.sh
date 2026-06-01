#!/bin/bash
set -e

# Default values
ENVIRONMENT=${1:-dev}
ACTION=${2:-apply}
AUTO_APPROVE=false

# Parse additional arguments
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --auto-approve)
      AUTO_APPROVE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [ENVIRONMENT] [ACTION] [OPTIONS]"
      echo "Arguments:"
      echo "  ENVIRONMENT    Environment to deploy (dev, prod) [default: dev]"
      echo "  ACTION         Terraform action (plan, apply, destroy) [default: apply]"
      echo "Options:"
      echo "  --auto-approve Auto approve terraform apply/destroy"
      echo "  -h, --help     Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 dev plan                    # Plan dev environment"
      echo "  $0 prod apply --auto-approve   # Apply prod with auto-approve"
      echo "  $0 dev destroy                 # Destroy dev environment"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

TERRAFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Box Office ML Infrastructure Deployment"
echo "Working directory: $TERRAFORM_DIR"
echo "Environment: $ENVIRONMENT"
echo "Action: $ACTION"
echo "Auto-approve: $AUTO_APPROVE"
echo ""

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod)$ ]]; then
  echo "Error: Environment must be 'dev' or 'prod'"
  exit 1
fi

# Validate action
if [[ ! "$ACTION" =~ ^(plan|apply|destroy)$ ]]; then
  echo "Error: Action must be 'plan', 'apply', or 'destroy'"
  exit 1
fi

# Check if environment file exists
if [ ! -f "$TERRAFORM_DIR/environments/$ENVIRONMENT.tfvars" ]; then
    echo "Environment file not found: $TERRAFORM_DIR/environments/$ENVIRONMENT.tfvars"
    exit 1
fi

cd "$TERRAFORM_DIR"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Select or create workspace
echo "Selecting workspace: $ENVIRONMENT"
terraform workspace select "$ENVIRONMENT" 2>/dev/null || terraform workspace new "$ENVIRONMENT"

# Validate configuration
echo "Validating Terraform configuration..."
terraform validate

# Execute the requested action
case $ACTION in
  plan)
    echo "Planning deployment..."
    terraform plan -var-file="environments/$ENVIRONMENT.tfvars" -out="$ENVIRONMENT.tfplan"
    echo ""
    echo "Plan completed successfully."
    echo "To apply these changes, run: $0 $ENVIRONMENT apply"
    ;;
  apply)
    echo "Planning deployment..."
    terraform plan -var-file="environments/$ENVIRONMENT.tfvars" -out="$ENVIRONMENT.tfplan"

    if [[ "$AUTO_APPROVE" == true ]]; then
      echo "Applying changes with auto-approve..."
      terraform apply -auto-approve "$ENVIRONMENT.tfplan"
    else
      echo ""
      read -p "Do you want to apply these changes? (yes/no): " confirm
      if [ "$confirm" != "yes" ]; then
        echo "Deployment cancelled"
        rm -f "$ENVIRONMENT.tfplan"
        exit 0
      fi
      echo "Applying changes..."
      terraform apply "$ENVIRONMENT.tfplan"
    fi

    # Clean up plan file
    rm -f "$ENVIRONMENT.tfplan"

    echo ""
    echo "Deployment completed successfully."
    echo ""
    echo "=== Infrastructure Outputs ==="
    terraform output
    ;;
  destroy)
    echo "WARNING: This will destroy all infrastructure for the $ENVIRONMENT environment!"
    echo "This action cannot be undone."

    if [[ "$AUTO_APPROVE" == true ]]; then
      echo "Destroying infrastructure with auto-approve..."
      terraform destroy -var-file="environments/$ENVIRONMENT.tfvars" -auto-approve
    else
      echo ""
      read -p "Are you absolutely sure you want to destroy everything? Type 'destroy' to confirm: " confirm
      if [ "$confirm" != "destroy" ]; then
        echo "Destruction cancelled"
        exit 0
      fi
      echo "Destroying infrastructure..."
      terraform destroy -var-file="environments/$ENVIRONMENT.tfvars"
    fi

    echo "Infrastructure destroyed successfully."
    ;;
esac
