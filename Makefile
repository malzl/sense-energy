.PHONY: help setup install lint format test clean data features train evaluate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install the project in editable mode with dev extras
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

install: ## Install the package (editable, dev extras) into the active environment
	pip install -e ".[dev]"

lint: ## Run ruff and mypy
	ruff check code/
	ruff format --check code/
	mypy

format: ## Auto-format and auto-fix
	ruff format code/
	ruff check --fix code/

test: ## Run the test suite with coverage
	pytest --cov=sense_energy --cov-report=term-missing

data: ## Build the processed dataset from raw inputs
	python -m sense_energy.cli build-dataset --config code/configs/data.yaml

features: ## Build the model-ready feature table
	python -m sense_energy.cli build-features --config code/configs/features.yaml

train: ## Train a model
	python -m sense_energy.cli train --config code/configs/model_baseline.yaml

evaluate: ## Evaluate the latest trained model
	python -m sense_energy.cli evaluate --config code/configs/model_baseline.yaml

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
