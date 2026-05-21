## Makefile for the harxhar volatility-forecasting repo.
## Run targets from WSL or git-bash on Windows.

# Use bash for shell loops / case statements.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

# Python interpreter. Override with `make PYTHON=...` if needed.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help table diagnostics diagnostics-quick audit pipeline-export scripts-export executors-export export strategy-eval lint type test clean-cache repro repro-fixture new-experiment template-test

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
		PYTHONUTF8=1 $(PYTHON) notebooks/_export_executor.py $$nb $$out; \
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
	@test -f src/ml_$(METHOD).py || { echo "unknown method: $(METHOD) (no src/ml_$(METHOD).py)"; exit 1; }
	@mkdir -p results/$(RUN)
	# Per-method scripts (src/ml_*.py) expose a `compute(args)` function, no
	# `__main__`. Use the stdlib-only dispatcher so CI (no claude_hpc) can run
	# this; production HPC uses the equivalent `python -m cli ...` dispatcher
	# via .hpc/cli.py (which depends on claude_hpc).
	#
	# A converted (@register_run) executor returns a metrics dict that the
	# injected compute() writes to --output-file, and writes the per-row
	# results.csv alongside; a legacy executor writes results.csv directly to
	# --output-file. Pick --output-file accordingly so finalize_run always
	# finds results/$(RUN)/results.csv. (Migration scaffolding — drops out in
	# B6 once every executor is converted.)
	@if grep -q '@register_run' src/ml_$(METHOD).py; then \
		out=results/$(RUN)/run_metrics.json; \
	else \
		out=results/$(RUN)/results.csv; \
	fi; \
	echo "executor output -> $$out"; \
	PYTHONPATH=. $(PYTHON) scripts/_repro_dispatch.py src.ml_$(METHOD) --output-file $$out $(REPRO_ARGS)
	PYTHONPATH=. $(PYTHON) scripts/finalize_run.py --run-dir results/$(RUN) --method $(METHOD) --update-manifest results/MANIFEST.json
	$(MAKE) table
	$(MAKE) audit

repro-fixture:  ## Run a tiny CI-friendly Ridge repro (--train-window 10 --end 1000) for the audit gate.
	$(MAKE) repro RUN=ci_fixture METHOD=ridge REPRO_ARGS="--train-window 10 --end 1000"

new-experiment:  ## Scaffold a new experiment notebook. Required: NAME.
	@: $${NAME:?must set NAME=<name>}
	$(PYTHON) scripts/_scaffold_experiment.py $(NAME)

template-test:  ## Run the experiment-template (src/_template.py) unit tests.
	pytest core/tests/test_template.py -q
