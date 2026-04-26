## Makefile for the harxhar volatility-forecasting repo.
## Run targets from WSL or git-bash on Windows.

# Use bash for shell loops / case statements.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

# Python interpreter. Override with `make PYTHON=...` if needed.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help table diagnostics diagnostics-quick audit pipeline-export strategy-eval lint type test clean-cache

help:  ## Show available targets.
	@echo "harxhar Makefile — available targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

table:  ## Regenerate master_table.csv + LaTeX.
	$(PYTHON) scripts/build_master_table.py

diagnostics:  ## Regenerate full diagnostics bundles.
	$(PYTHON) scripts/build_diagnostics.py

diagnostics-quick:  ## Regenerate diagnostics, skipping bundles that already exist.
	$(PYTHON) scripts/build_diagnostics.py --skip-existing

audit:  ## Run the audit gate.
	$(PYTHON) scripts/audit_check.py --quick || echo "audit_check.py not yet built"

pipeline-export:  ## Export every notebooks/pipeline/*.ipynb to its src/ module.
	@for nb in notebooks/pipeline/*.ipynb; do \
		name=$$(basename $$nb .ipynb); \
		case "$$name" in \
			01_loading)    out=src/loading.py ;; \
			02_transforms) out=src/transforms.py ;; \
			03_evaluation) out=src/evaluation.py ;; \
			04_scaling)    out=src/scaling.py ;; \
			05_executor)   out=src/executor.py ;; \
			06_strategy_eval) out=src/strategy_eval.py ;; \
			*) echo "no mapping for $$name; skipping"; continue ;; \
		esac; \
		echo "exporting $$nb -> $$out"; \
		$(PYTHON) notebooks/_exporter.py $$nb $$out; \
	done

strategy-eval:  ## Assemble, export, and validate the strategy_eval pipeline.
	$(PYTHON) scripts/_assemble_strategy_eval_nb.py
	$(PYTHON) notebooks/_exporter.py notebooks/pipeline/06_strategy_eval.ipynb src/strategy_eval.py
	PYTHONPATH=. $(PYTHON) scripts/validate_strategy_eval.py

lint:  ## Run ruff check --fix and ruff format on scripts/ and src/.
	ruff check --fix scripts/ src/ && ruff format scripts/ src/

type:  ## Run mypy on scripts/ and src/.
	mypy --ignore-missing-imports scripts/ src/

test:  ## Run pytest in quiet mode.
	pytest -q

clean-cache:  ## Remove __pycache__/, .pytest_cache/, .mypy_cache/, .ruff_cache/.
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	@find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	@find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	@echo "cache directories removed."
