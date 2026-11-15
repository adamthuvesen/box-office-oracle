VERIFY_SKILL := .agents/skills/verify-box-office-oracle

.PHONY: help check fix verify slow install install-dev test test-unit test-integration test-inference-lean training-image lint format clean docker-build docker-test setup datasets web-data web-install web-build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

check: setup ## Fast gate: locked sync, ruff, basedpyright, import contracts, fast tests, web lint/format/typecheck/tests (CI runs this)
	@scripts/check.sh

fix: install-dev ## Auto-fix lint and format with ruff and oxfmt; regenerate the web contract
	uv run ruff check --fix box_office tests scripts $(VERIFY_SKILL) $(VERIFY_SKILL)/verify
	uv run ruff format box_office tests scripts $(VERIFY_SKILL) $(VERIFY_SKILL)/verify
	uv run python scripts/generate_web_contract.py
	pnpm --dir web format

verify: setup ## Drive the app on synthetic fixtures (verify-box-office-oracle); ARGS=all|list|control|<drive>
	@uv sync --locked --extra dev --extra inference --extra train --quiet
	@uv run --no-sync python $(VERIFY_SKILL)/verify $(ARGS)

slow: setup ## Full test suite, pinned infra linters (actionlint, checkov), offline dbt parse, web build
	@scripts/slow.sh

setup: ## Verify uv is installed (no curl | sh)
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed. Install it via your package manager, e.g.:"; \
		echo "  brew install uv        # macOS"; \
		echo "  pipx install uv        # cross-platform"; \
		echo "  see https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

install: setup ## Install the serving dependencies (what the Lambda image gets)
	uv sync --extra inference --frozen

install-dev: setup ## Install development, serving, and training dependencies
	uv sync --extra dev --extra inference --extra train --frozen

test: setup ## Run all tests
	uv sync --extra dev --extra inference --extra train --frozen
	uv run pytest tests/ -v

test-unit: setup ## Run unit tests only (markers)
	uv sync --extra dev --extra inference --extra train --frozen
	uv run pytest tests/ -m "unit and not slow" -v

test-integration: setup ## Run integration tests only
	uv sync --extra dev --extra inference --extra train --frozen
	uv run pytest tests/ -m "integration and not slow" -v

test-coverage: setup ## Run tests with coverage
	uv sync --extra dev --extra inference --extra train --frozen
	uv run pytest tests/ -m "not integration and not slow" --cov=box_office --cov-report=xml --cov-report=term-missing

test-inference-lean: setup ## Inference tests with only the serving dependencies (what the Lambda image gets)
	UV_PROJECT_ENVIRONMENT=.venv-inference uv sync --locked --extra dev --extra inference --quiet
	UV_PROJECT_ENVIRONMENT=.venv-inference uv run --no-sync pytest tests/inference

lint: install-dev ## Run linting checks
	uv run ruff check box_office tests scripts $(VERIFY_SKILL) $(VERIFY_SKILL)/verify

format: install-dev ## Format code with ruff
	uv run ruff format box_office tests scripts $(VERIFY_SKILL) $(VERIFY_SKILL)/verify

format-check: install-dev ## Check formatting without making changes
	uv run ruff format --check box_office tests scripts $(VERIFY_SKILL) $(VERIFY_SKILL)/verify

clean: ## Clean build artifacts and caches
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image for inference
	docker build \
		--platform linux/amd64 \
		--tag box-office-inference:latest \
		--file infrastructure/docker/inference/Dockerfile \
		.

docker-test: docker-build ## Test Docker image
	docker run --rm \
		--platform linux/amd64 \
		--entrypoint python \
		box-office-inference:latest \
		-c "import fastapi, mangum, boto3, pandas, numpy, sklearn, xgboost, pydantic; print('All dependencies available')"

training-image: setup ## Build and push the SageMaker training image (spends: refuses until ARGS carries the approval flag; docs/training-image.md)
	@scripts/build_training_image.sh $(ARGS)

pipeline-run: install-dev ## Run ML pipeline (spends: refuses until ARGS carries the approval flag)
	uv run box-office-pipeline --environment dev $(ARGS)

web-install: ## Install the web app from its lockfile (required before check and slow)
	pnpm --dir web install --frozen-lockfile

web-build: ## Next.js production build (CI web-build job)
	pnpm --dir web build

web-data: install-dev ## Export web/data JSON snapshots for the Next.js app (local dataset, no Snowflake)
	uv run python scripts/score_all_movies.py
	uv run python scripts/export_web_data.py

datasets: install-dev ## Refresh the rich local TMDB dataset
	uv run box-office-rich-backfill
