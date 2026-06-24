.PHONY: help build check clean fmt lint test benchmark docs

default: check

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  help        Show this help message"
	@echo "  build       Build the project"
	@echo "  check       Check the code (formatting, linting, and tests)"
	@echo "  clean       Clean the build artifacts"
	@echo "  fmt         Format the code"
	@echo "  lint        Lint the code"
	@echo "  test        Run tests"
	@echo "  benchmark   Run benchmark test matrix"
	@echo "  docs        Generate documentation"	

build:
	@echo "Building the project..."
	# Add your build commands here, e.g., compiling source code

check: fmt lint test
	@echo "All checks passed successfully!"

clean:
	@echo "Cleaning build artifacts..."
	# Add your clean commands here, e.g., removing compiled files

fmt:
	@echo "Formatting the code..."
	@uvx ruff format
# 	@prettier -w "docs/**/*.{json,yaml,yml,md}" "tests/**/*.{json,yaml,yml,md}" "*.{json,yaml,yml,md}"

lint:
	@echo "Linting the code..."
	@uvx ruff check --fix

test:
	@echo "Running tests..."
	@uv run pytest --cov=./src tests/

benchmark:
	@echo "Running benchmark matrix..."
	@uv run pytest -n=0 --run-benchmarks -s tests/benchmarks/ -v

docs: fmt
	@echo "Generating documentation..."
	@uvx mkdocs build