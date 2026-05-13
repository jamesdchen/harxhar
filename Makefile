## Makefile for the harxhar volatility-forecasting repo.
## Run targets from WSL or git-bash on Windows.

# Use bash for shell loops / case statements.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

# Python interpreter. Override with `make PYTHON=...` if needed.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help table diagnostics diagnostics-quick audit pipeline-export scripts-export executors-export export strategy-eval lint type test clean-cache repro repro-fixture

help:  ## Show available targets.
	@echo "harxhar Makefile — available targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

table:  ## Regenerate master_table.csv + LaTeX.
	$(PYTHON) scripts/build_master_table.py

diagnostics:  ## Regenerate full diagnostics bundles.
	PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace notebooks/audits/build_diagnostics.ipynb

diagnostics-quick:  ## Regenerate diagnostics, skipping bundles that already exist.
	SKIP_EXISTING=1 PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace notebooks/audits/build_diagnostics.ipynb

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
			05b_dl_executor) out=src/dl_executor.py ;; \
			06_strategy_eval) out=src/strategy_eval.py ;; \
			07_tune_tree) out=src/tune_tree.py ;; \
			*) echo "no mapping for $$name; skipping"; continue ;; \
		esac; \
		echo "exporting $$nb -> $$out"; \
		PYTHONUTF8=1 $(PYTHON) notebooks/_exporter.py $$nb $$out; \
	done

scripts-export:  ## Re-export every notebook in notebooks/scripts/ to scripts/.
	@for nb in notebooks/scripts/*.ipynb; do \
		name=$$(basename $$nb .ipynb); \
		out=scripts/$$name.py; \
		echo "exporting $$nb -> $$out"; \
		PYTHONUTF8=1 $(PYTHON) notebooks/_exporter.py $$nb $$out; \
	done

executors-export:  ## Re-export every notebook in notebooks/executors/ to src/.
	@for nb in notebooks/executors/*.ipynb; do \
		name=$$(basename $$nb .ipynb); \
		out=src/$$name.py; \
		echo "exporting $$nb -> $$out"; \
		PYTHONUTF8=1 $(PYTHON) notebooks/_exporter.py $$nb $$out; \
	done

export: pipeline-export executors-export scripts-export  ## Re-export every notebook in the project.

strategy-eval:  ## Assemble and export the strategy_eval pipeline.
	$(PYTHON) scripts/_assemble_strategy_eval_nb.py
	PYTHONUTF8=1 $(PYTHON) notebooks/_exporter.py notebooks/pipeline/06_strategy_eval.ipynb src/strategy_eval.py

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

repro:  ## Reproduce a run end-to-end (local sequential): executor -> finalize -> table -> audit. Required: RUN, METHOD. Optional: REPRO_ARGS.
	@: $${RUN:?must set RUN=<name>} $${METHOD:?must set METHOD=<name>}
	$(PYTHON) -c "from src.executor import CONFIGS; assert '$(METHOD)' in CONFIGS, 'unknown method: $(METHOD) (registered: ' + ','.join(CONFIGS) + ')'"
	@mkdir -p results/$(RUN)
	PYTHONPATH=. $(PYTHON) src/ml_$(METHOD).py --output-file results/$(RUN)/results.csv $(REPRO_ARGS)
	PYTHONPATH=. $(PYTHON) scripts/finalize_run.py --run-dir results/$(RUN) --method $(METHOD) --update-manifest results/MANIFEST.json
	$(MAKE) table
	$(MAKE) audit

repro-fixture:  ## Run a tiny CI-friendly Ridge repro (--train-window 10 --end 1000) for the audit gate.
	$(MAKE) repro RUN=ci_fixture METHOD=ridge REPRO_ARGS="--train-window 10 --end 1000"
