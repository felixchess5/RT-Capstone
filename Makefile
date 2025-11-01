# 🧪 Intelligent-Assignment-Grading-System Testing Makefile
# Simplified commands for running tests and development tasks

.PHONY: help install test test-unit test-integration test-e2e test-all test-cov test-fast lint format security clean docs setup

# Default target
help: ## 📋 Show this help message
	@echo "🧪 Intelligent-Assignment-Grading-System Testing Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and installation
install: ## 📦 Install all dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup: install ## 🔧 Setup development environment
	@echo "🔧 Setting up development environment..."
	pip install pre-commit black isort flake8 mypy
	pre-commit install
	@echo "✅ Development environment setup complete"

# Testing commands
test: ## 🧪 Run all tests
	@echo "🧪 Running all tests..."
	pytest tests/ -v

test-unit: ## 🧪 Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## 🔗 Run integration tests only
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v

test-e2e: ## 🎯 Run end-to-end tests only
	@echo "🎯 Running end-to-end tests..."
	pytest tests/e2e/ -v

test-fast: ## ⚡ Run fast tests only (exclude slow tests)
	@echo "⚡ Running fast tests..."
	pytest tests/ -v -m "not slow"

test-cov: ## 📊 Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📁 Coverage report generated in htmlcov/"

test-cov-xml: ## 📊 Run tests with XML coverage report
	@echo "📊 Running tests with XML coverage..."
	pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing

test-parallel: ## 🚀 Run tests in parallel
	@echo "🚀 Running tests in parallel..."
	pytest tests/ -n auto -v

test-failed: ## 🔄 Re-run only failed tests
	@echo "🔄 Re-running failed tests..."
	pytest --lf -v

test-watch: ## 👀 Run tests in watch mode (requires pytest-watch)
	@echo "👀 Running tests in watch mode..."
	pip install pytest-watch
	ptw tests/ src/

# Quality and formatting
lint: ## 🔍 Run linting checks
	@echo "🔍 Running linting checks..."
	flake8 src tests --count --show-source --statistics
	@echo "✅ Linting complete"

format: ## 🖤 Format code with black and isort
	@echo "🖤 Formatting code..."
	black src tests
	isort src tests
	@echo "✅ Code formatting complete"

format-check: ## ✔️ Check code formatting
	@echo "✔️ Checking code formatting..."
	black --check src tests
	isort --check-only src tests

type-check: ## 🏷️ Run type checking with mypy
	@echo "🏷️ Running type checks..."
	mypy src --ignore-missing-imports

# Security
security: ## 🔒 Run security checks
	@echo "🔒 Running security checks..."
	pip install bandit safety
	bandit -r src/
	safety check
	@echo "✅ Security checks complete"

# Specific test categories
test-math: ## 📐 Run math processor tests
	@echo "📐 Running math processor tests..."
	pytest tests/ -v -k "math"

test-spanish: ## 🇪🇸 Run Spanish processor tests
	@echo "🇪🇸 Running Spanish processor tests..."
	pytest tests/ -v -k "spanish"

test-gradio: ## 🌐 Run Gradio interface tests
	@echo "🌐 Running Gradio interface tests..."
	pytest tests/ -v -m "gradio"

test-workflow: ## ⚡ Run workflow tests
	@echo "⚡ Running workflow tests..."
	pytest tests/ -v -k "workflow"

test-performance: ## 📊 Run performance tests
	@echo "📊 Running performance tests..."
	pytest tests/ -v -m "performance"

# Debugging and development
test-debug: ## 🐛 Run tests with debugging
	@echo "🐛 Running tests with debugging..."
	pytest tests/ -v -s --pdb

test-verbose: ## 📢 Run tests with maximum verbosity
	@echo "📢 Running tests with maximum verbosity..."
	pytest tests/ -vvv -s --tb=long

test-quiet: ## 🤫 Run tests quietly
	@echo "🤫 Running tests quietly..."
	pytest tests/ -q

# Reporting
test-html: ## 📄 Generate HTML test report
	@echo "📄 Generating HTML test report..."
	pytest tests/ --html=reports/pytest_report.html --self-contained-html
	@echo "📁 HTML report generated in reports/"

test-json: ## 📄 Generate JSON test report
	@echo "📄 Generating JSON test report..."
	mkdir -p reports
	pytest tests/ --json-report --json-report-file=reports/pytest_report.json
	@echo "📁 JSON report generated in reports/"

# Cleanup
clean: ## 🧹 Clean up test artifacts
	@echo "🧹 Cleaning up test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf reports/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

clean-all: clean ## 🗑️ Deep clean (includes virtual environment)
	@echo "🗑️ Deep cleaning..."
	rm -rf venv/
	rm -rf .venv/
	@echo "✅ Deep cleanup complete"

# Documentation
docs: ## 📚 Generate documentation
	@echo "📚 Generating documentation..."
	pip install sphinx sphinx-rtd-theme
	sphinx-build -b html docs/ docs/_build/
	@echo "📁 Documentation generated in docs/_build/"

docs-serve: ## 🌐 Serve documentation locally
	@echo "🌐 Serving documentation..."
	cd docs/_build && python -m http.server 8000

# Continuous Integration simulation
ci: lint test-cov security ## 🤖 Run CI pipeline locally
	@echo "🤖 Running CI pipeline locally..."
	@echo "✅ CI pipeline complete"

ci-fast: lint test-fast ## ⚡ Run fast CI pipeline
	@echo "⚡ Running fast CI pipeline..."
	@echo "✅ Fast CI pipeline complete"

# Benchmarking
benchmark: ## 📊 Run performance benchmarks
	@echo "📊 Running performance benchmarks..."
	pip install pytest-benchmark
	pytest tests/ --benchmark-only --benchmark-json=benchmark.json
	@echo "📁 Benchmark results saved to benchmark.json"

# Development helpers
install-dev: ## 🔧 Install development dependencies
	@echo "🔧 Installing development dependencies..."
	pip install pytest pytest-cov pytest-mock pytest-asyncio pytest-html
	pip install black isort flake8 mypy bandit safety
	pip install pre-commit pytest-watch pytest-benchmark
	@echo "✅ Development dependencies installed"

pre-commit: ## 🔄 Run pre-commit hooks
	@echo "🔄 Running pre-commit hooks..."
	pre-commit run --all-files

# Environment information
info: ## ℹ️ Show environment information
	@echo "ℹ️ Environment Information"
	@echo "========================="
	@echo "Python version: $$(python --version)"
	@echo "Pytest version: $$(pytest --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Python path: $$(which python)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not in git repo')"
	@echo "Test files found: $$(find tests -name '*.py' | wc -l)"

# Test data management
test-data: ## 📁 Setup test data
	@echo "📁 Setting up test data..."
	mkdir -p tests/fixtures/assignments
	mkdir -p tests/fixtures/responses
	mkdir -p Assignments
	mkdir -p output
	@echo "✅ Test data directories created"

# Database-related (if applicable)
test-db: ## 🗃️ Setup test database
	@echo "🗃️ Setting up test database..."
	# Add database setup commands here if needed
	@echo "✅ Test database setup complete"

# Performance monitoring
monitor: ## 📈 Monitor test performance
	@echo "📈 Monitoring test performance..."
	pytest tests/ --durations=10
	@echo "✅ Performance monitoring complete"

# Memory profiling
profile: ## 🧠 Profile memory usage
	@echo "🧠 Profiling memory usage..."
	pip install memory-profiler
	python -m memory_profiler tests/test_memory_usage.py || echo "No memory test file found"

# Integration with external tools
sonar: ## 📊 Run SonarQube analysis (requires SonarQube)
	@echo "📊 Running SonarQube analysis..."
	sonar-scanner || echo "SonarQube not available"

# Makefile validation
validate: ## ✅ Validate Makefile syntax
	@echo "✅ Validating Makefile..."
	@make -n help > /dev/null && echo "Makefile syntax is valid"

# Show test structure
test-structure: ## 📁 Show test directory structure
	@echo "📁 Test Directory Structure"
	@echo "=========================="
	@tree tests/ 2>/dev/null || find tests/ -type f -name "*.py" | head -20

# Quick commands for daily development
dev: format lint test-fast ## 🚀 Quick development workflow
	@echo "🚀 Development workflow complete"

commit: format lint test-unit ## 📝 Pre-commit workflow
	@echo "📝 Ready to commit"

deploy: test-all security ## 🚀 Pre-deployment workflow
	@echo "🚀 Ready for deployment"
