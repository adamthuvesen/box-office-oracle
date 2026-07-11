.PHONY: help install install-dev test test-unit test-integration lint format clean docker-build docker-test setup datasets web-data

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Verify uv is installed (no curl | sh)
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed. Install it via your package manager, e.g.:"; \
		echo "  brew install uv        # macOS"; \
		echo "  pipx install uv        # cross-platform"; \
		echo "  see https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

install: setup ## Install production dependencies
	uv sync --frozen

install-dev: setup ## Install development dependencies
	uv sync --extra dev --frozen

test: setup ## Run all tests
	uv sync --extra dev --extra inference --frozen
	uv run pytest tests/ -v

test-unit: setup ## Run unit tests only (markers)
	uv sync --extra dev --extra inference --frozen
	uv run pytest tests/ -m "unit and not slow" -v

test-integration: setup ## Run integration tests only
	uv sync --extra dev --extra inference --frozen
	uv run pytest tests/ -m "integration and not slow" -v

test-coverage: setup ## Run tests with coverage
	uv sync --extra dev --extra inference --frozen
	uv run pytest tests/ --cov=box_office --cov-report=xml --cov-report=term-missing

lint: install-dev ## Run linting checks
	uv run ruff check box_office tests scripts

format: install-dev ## Format code with ruff
	uv run ruff format box_office tests scripts

format-check: install-dev ## Check formatting without making changes
	uv run ruff format --check box_office tests scripts

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

pipeline-run: install-dev ## Run ML pipeline (requires environment variables)
	uv run box-office-pipeline --environment dev

web-data: install-dev ## Export web/data JSON snapshots for the Next.js app (local dataset, no Snowflake)
	uv run python scripts/score_all_movies.py
	uv run python scripts/export_web_data.py

datasets: install-dev ## Refresh the rich local TMDB dataset
	uv run box-office-rich-backfill
