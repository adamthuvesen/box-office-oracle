terraform {
  # Public-safe backend declaration: no bucket/account-specific values committed.
  # Supply real state values at init time, e.g.:
  #   terraform init \
  #     -backend-config="bucket=<state-bucket>" \
  #     -backend-config="key=box-office/terraform.tfstate" \
  #     -backend-config="region=eu-north-1" \
  #     -backend-config="dynamodb_table=terraform-state-lock" \
  #     -backend-config="encrypt=true"
  # CI passes these from the TERRAFORM_STATE_BUCKET secret + workflow env.
  backend "s3" {}
}
