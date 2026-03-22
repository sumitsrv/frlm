# =============================================================================
# FRLM - Factual Retrieval Language Model
# Makefile for common operations
# =============================================================================

PYTHON      ?= python
PIP         ?= pip
PYTEST      ?= pytest
CONFIG      ?= config/default.yaml
SHELL       := /bin/bash
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

.PHONY: setup
setup: ## Install all dependencies and the package in editable mode
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,notebook]"
	$(PIP) install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz || true
	@echo "Setup complete."

.PHONY: setup-gpu
setup-gpu: ## Install with GPU-specific dependencies (CUDA 11.8)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,notebook]"
	@echo "GPU setup complete."

.PHONY: setup-neo4j
setup-neo4j: ## Setup Neo4j via Docker (recommended)
	bash scripts/setup_neo4j.sh docker

.PHONY: setup-neo4j-native
setup-neo4j-native: ## Setup Neo4j via native apt install
	bash scripts/setup_neo4j.sh native

.PHONY: neo4j-status
neo4j-status: ## Check Neo4j status
	bash scripts/setup_neo4j.sh status

.PHONY: neo4j-stop
neo4j-stop: ## Stop Neo4j
	bash scripts/setup_neo4j.sh stop

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

.PHONY: download-corpus
download-corpus: ## Download PMC Open Access corpus
	$(PYTHON) scripts/01_download_corpus.py --config $(CONFIG)

.PHONY: extract-entities
extract-entities: ## Extract biomedical entities with SciSpacy
	$(PYTHON) scripts/02_extract_entities.py --config $(CONFIG)

.PHONY: extract-relations
extract-relations: ## Extract relations with Claude API
	$(PYTHON) scripts/03_extract_relations.py --config $(CONFIG)

.PHONY: build-kg
build-kg: ## Populate Neo4j knowledge graph
	$(PYTHON) scripts/04_populate_kg.py --config $(CONFIG)

.PHONY: build-index
build-index: ## Build FAISS vector index over KG embeddings
	$(PYTHON) scripts/05_build_faiss_index.py --config $(CONFIG)

.PHONY: label-data
label-data: ## Generate router training labels with Claude API
	$(PYTHON) scripts/06_generate_router_labels.py --config $(CONFIG)

.PHONY: prepare-training-data
prepare-training-data: ## Convert span labels → tokenized training data for all 3 phases
	$(PYTHON) scripts/06b_prepare_training_data.py --config $(CONFIG)

.PHONY: data-pipeline
data-pipeline: download-corpus extract-entities extract-relations build-kg build-index label-data prepare-training-data ## Run full data pipeline end-to-end
	@echo "Data pipeline complete."

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

.PHONY: train-router
train-router: ## Phase 1: Train router head
	$(PYTHON) scripts/07_train_router.py --config $(CONFIG)

.PHONY: train-retrieval
train-retrieval: ## Phase 2: Train retrieval head
	$(PYTHON) scripts/08_train_retrieval.py --config $(CONFIG)

.PHONY: train-joint
train-joint: ## Phase 3: Joint fine-tuning
	$(PYTHON) scripts/09_train_joint.py --config $(CONFIG)

.PHONY: train-all
train-all: train-router train-retrieval train-joint ## Run all three training phases sequentially
	@echo "All training phases complete."

# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

STEP ?= 1

.PHONY: full-pipeline
full-pipeline: ## Run entire 11-step pipeline end-to-end
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG) --yes

.PHONY: resume
resume: ## Resume pipeline from step STEP (e.g. make resume STEP=7)
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG) --start-from $(STEP) --yes

.PHONY: dry-run
dry-run: ## Dry-run: print pipeline commands without executing
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG) --dry-run

.PHONY: status
status: ## Show pipeline completion status (scans artifacts + persisted state)
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG) --status

.PHONY: status-json
status-json: ## Dump pipeline status as JSON (machine-readable)
	$(PYTHON) -c "import sys; sys.path.insert(0,'.'); from src.status import PipelineStatusTracker, scan_artifacts_into_status; from config.config import load_config; cfg=load_config('$(CONFIG)'); t=PipelineStatusTracker(); scan_artifacts_into_status(t, cfg); import json; print(json.dumps(t.get_all(), indent=2))"

.PHONY: estimate-costs estimate-cost
estimate-costs estimate-cost: ## Print estimated resource costs for the full pipeline
	$(PYTHON) scripts/estimate_costs.py --config $(CONFIG)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

.PHONY: evaluate
evaluate: ## Run full evaluation suite
	$(PYTHON) scripts/10_evaluate.py --config $(CONFIG)

# ---------------------------------------------------------------------------
# Inference / Serving
# ---------------------------------------------------------------------------

.PHONY: serve
serve: ## Start FastAPI inference server
	$(PYTHON) scripts/11_run_inference.py --config $(CONFIG) --mode serve

.PHONY: inference
inference: ## Run batch inference
	$(PYTHON) scripts/11_run_inference.py --config $(CONFIG) --mode batch

# ---------------------------------------------------------------------------
# Testing & Quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run unit tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov

.PHONY: test-fast
test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v -x

.PHONY: lint
lint: ## Run linters (ruff, mypy)
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

.PHONY: format
format: ## Auto-format code with black and isort
	black src/ tests/ scripts/ config/
	isort src/ tests/ scripts/ config/

.PHONY: check
check: lint test ## Run linters and tests

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t frlm:latest .

.PHONY: docker-run
docker-run: ## Run training in Docker container
	docker run --gpus all --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/logs:/app/logs \
		--env-file .env \
		frlm:latest

.PHONY: docker-serve
docker-serve: ## Run inference server in Docker
	docker run --gpus all --rm -p 8000:8000 \
		-v $(PWD)/checkpoints:/app/checkpoints \
		--env-file .env \
		frlm:latest python scripts/11_run_inference.py --config config/default.yaml --mode serve

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

.PHONY: clean-data
clean-data: ## Remove all processed data (keeps raw corpus)
	rm -rf data/processed/* data/kg/* data/labels/* data/faiss_indices/*
	@echo "Processed data cleaned."

.PHONY: clean-all
clean-all: clean clean-data ## Remove everything (artifacts + processed data)
	rm -rf checkpoints/* logs/*
	@echo "Full clean complete."

# ---------------------------------------------------------------------------
# Backup & Persistence
# ---------------------------------------------------------------------------

.PHONY: backup
backup: ## Backup all pipeline data (local compressed archive)
	bash scripts/backup_data.sh local

.PHONY: backup-critical
backup-critical: ## Backup only critical data (processed + labels — costly to regenerate)
	@mkdir -p backups
	tar -czf backups/frlm_critical_$$(date +%Y%m%d_%H%M%S).tar.gz \
		data/processed data/labels data/kg
	@echo "Critical data backed up to backups/"

.PHONY: backup-status
backup-status: ## Show current data sizes and backup inventory
	bash scripts/backup_data.sh status
	@echo ""; echo "Existing backups:"; ls -lh backups/ 2>/dev/null || echo "  (none)"

.PHONY: restore
restore: ## Restore data from a backup archive (usage: make restore ARCHIVE=backups/file.tar.gz)
	bash scripts/backup_data.sh restore $(ARCHIVE)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'