# Runner Makefile
# Manages build, test, and deployment of the runner service

.PHONY: all build test lint start deploy proto clean help

all: build test

env:
	@echo "Setting up runner environment variables..."
	@. ./env.sh
	@echo "Runner environment variables set"
	@env | grep RUNNER_ || true

# =============================================================================
# Build
# =============================================================================

build:
	@echo "Building .."
	@. .venv/bin/activate && python -m compileall .
	@echo "Runner build complete"

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running runner tests..."
	@. .venv/bin/activate && pytest test/ -v
	@echo "Runner tests complete"

test-unit:
	@echo "Running runner unit tests..."
	@. .venv/bin/activate && pytest test/unit/ -v
	@echo "Runner unit tests complete"

test-integration:
	@echo "Running runner integration tests..."
	@. .venv/bin/activate && pytest test/integration/ -v
	@echo "Runner integration tests complete"

test-cover:
	@echo "Running runner tests with coverage..."
	@. .venv/bin/activate && pytest test/ --cov=runner --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# =============================================================================
# Linting
# =============================================================================

lint:
	@echo "Running runner linting..."
	@. .venv/bin/activate && pylint runner/ --disable=C,R,W0622,W0611,W0613,W0718,W0511 --max-line-length=120 --jobs=4
	@echo "Runner linting complete"

lint-fix:
	@echo "Auto-fixing runner linting issues..."
	@. .venv/bin/activate && pylint runner/ --disable=C,R,W0622,W0611,W0613,W0718,W0511 --max-line-length=120 --jobs=4 --fix
	@echo "Runner linting auto-fix complete"

# =============================================================================
# Development
# =============================================================================

start: env
	@echo "Starting runner in development mode..."
	@. .venv/bin/activate && PYTHONPATH="gen/python:." python -m server.grpc

start-debug: env
	@echo "Starting runner with debug mode..."
	@. .venv/bin/activate && python -m server.grpc --log-level debug

# =============================================================================
# Proto Generation
# =============================================================================

proto:
	@echo "Generating runner proto code..."
	@cd ../proto && make generate-runner
	@echo "Runner proto code generated"

# =============================================================================
# Deployment
# =============================================================================

deploy:
	@echo "Deploying runner to k8s..."
	@chmod +x ./k8s/build.sh
	@$(eval BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD | tr '/' '.'))
	@echo "Using branch: $(BRANCH_NAME) for image tag"
	@DOCKER_TAG=$(BRANCH_NAME) ./k8s/build.sh
	@DOCKER_TAG=$(BRANCH_NAME) ./k8s/apply.sh
	@kubectl rollout restart deployment llmmll-runner -n llmmll --wait=true
	@echo "Runner deployed successfully"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning runner artifacts..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf __pycache__/
	rm -rf runner/__pycache__/
	rm -rf runner/*/__pycache__/
	rm -rf runner/*/*/__pycache__/
	rm -rf runner/*/*/*/__pycache__/
	rm -rf runner/*/*/*/*/__pycache__/
	rm -rf runner/*/*/*/*/*/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Runner artifacts cleaned"

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Runner Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build           - Build the runner"
	@echo "  test            - Run all runner tests"
	@echo "  test-unit       - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-cover      - Run tests with coverage report"
	@echo "  lint            - Run linting"
	@echo "  lint-fix        - Auto-fix linting issues"
	@echo "  start           - Start runner in dev mode"
	@echo "  start-debug     - Start runner with debug logging"
	@echo "  proto           - Generate proto code"
	@echo "  deploy          - Deploy to k8s"
	@echo "  clean           - Clean artifacts"
	@echo "  help            - Show this help message"