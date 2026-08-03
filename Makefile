PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
POETRY ?= poetry

TOKENIZER_MODELS ?= distilbert-base-uncased bert-base-uncased google/bert_uncased_L-4_H-512_A-8 microsoft/xtremedistil-l6-h256-uncased
FORMAT_PATHS ?= .
DATASETS := amazon ag_news imdb dbpedia yelp
DATASET ?= amazon
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help install-poetry install install-pip install-poetry-env install-poetry-project install[poetry] init tokenizers download-datasets load-ftpfn run run-all check fix test clean

help: ## Show this help message.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} /^[^[:space:]][^:]*:.*##/ {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:
	$(PIP) install -e .

uninstall-existing-torch:
	$(PIP) uninstall -y torch torchaudio torchvision

init: download-datasets tokenizers load-ftpfn ## Download datasets, tokenizers, and FTPFN dependency/model.

download-datasets: ## Download and extract all exam datasets into ./data.
	$(PYTHON) scripts/download-datasets.py

tokenizers: ## Download tokenizer files into ./tokenizers.
	@for model in $(TOKENIZER_MODELS); do \
		$(PYTHON) scripts/save_tokenizer.py --model-name "$$model"; \
	done

load-ftpfn: ## Import/load FTPFN once so required assets are initialized.
	$(PYTHON) scripts/load-ftpfn.py

run: ## Run AutoML for one dataset. Usage: make run DATASET=amazon ARGS="--seed 42"
	@if ! echo "$(DATASETS)" | grep -wq "$(DATASET)"; then \
		echo "Invalid DATASET='$(DATASET)'. Valid values: $(DATASETS)"; \
		exit 1; \
	fi
	$(PYTHON) -m automl --dataset $(DATASET) $(ARGS)

run-%: ## Run AutoML for one dataset via shorthand. Usage: make run-amazon ARGS="--seed 42"
	$(MAKE) run DATASET=$* ARGS="$(ARGS)"

run-all: ## Run AutoML sequentially for all datasets. Usage: make run-all ARGS="--seed 42"
	@for dataset in $(DATASETS); do \
		echo "Running AutoML for dataset: $$dataset"; \
		$(PYTHON) -m automl --dataset "$$dataset" $(ARGS) || exit $$?; \
	done

check: ## Check formatting with Black.
	$(PYTHON) -m black --check $(FORMAT_PATHS)

fix: ## Format code with Black.
	$(PYTHON) -m black $(FORMAT_PATHS)

test: ## Run the test suite.
	$(PYTHON) -m pytest

clean: ## Remove common local Python cache files.
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

streamline: uninstall-existing-torch install init
	$(PYTHON) -m automl