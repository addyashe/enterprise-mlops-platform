# Build and test
test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src tests

format:
	black src tests

type-check:
	mypy src --ignore-missing-imports

# Development
dev-up:
	docker-compose up -d

dev-down:
	docker-compose down

dev-logs:
	docker-compose logs -f

# Training
train:
	python scripts/train_model.py

# Serving
serve:
	python scripts/serve_model.py

# Orchestration
airflow-up:
	docker-compose up -d airflow

airflow-ui:
	open http://localhost:8080

# Utilities
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

help:
	@echo "Available targets:"
	@echo "  test       - Run tests with coverage"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"
	@echo "  type-check - Run type checking"
	@echo "  dev-up     - Start development environment"
	@echo "  dev-down   - Stop development environment"
	@echo "  train      - Train a model"
	@echo "  serve      - Serve model"
	@echo "  clean      - Clean build artifacts"

.PHONY: test lint format type-check dev-up dev-down train serve airflow-up clean help
