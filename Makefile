## Makefile for the harxhar volatility-forecasting repo.
## Run targets from WSL or git-bash on Windows.

# Use bash for shell loops / case statements.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

# Python interpreter. Override with `make PYTHON=...` if needed.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help table diagnostics diagnostics-quick audit export strategy-eval lint type test clean-cache repro repro-fixture new-experiment

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

# `export` builds the generated src/ package from notebooks/. src/ is a build
# artifact — not committed (see .gitignore). notebooks/_build_package.py is a
# convention-driven local stand-in for hpc-agent's `export-package` primitive;
# when that ships, this target swaps to `hpc-agent export-package`. The
# assemble step rebuilds the strategy_eval notebook from fragments first.
export:  ## Build the generated src/ package from notebooks/ (convention-driven).
	$(PYTHON) scripts/_assemble_strategy_eval_nb.py
	PYTHONUTF8=1 $(PYTHON) notebooks/_build_package.py

strategy-eval:  ## Assemble and export the strategy_eval pipeline.
	$(PYTHON) scripts/_assemble_strategy_eval_nb.py
	PYTHONUTF8=1 $(PYTHON) notebooks/_exporter.py notebooks/pipeline/06_strategy_eval.ipynb src/strategy_eval.py

lint: export  ## Run ruff check --fix and ruff format on scripts/ and src/.
	ruff check --fix scripts/ src/ && ruff format scripts/ src/

type: export  ## Run mypy on scripts/ and src/.
	mypy --ignore-missing-imports scripts/ src/

test:  ## Run pytest in quiet mode.
	pytest -q

clean-cache:  ## Remove __pycache__/, .pytest_cache/, .mypy_cache/, .ruff_cache/.
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	@find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	@find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	@echo "cache directories removed."

repro: export  ## Reproduce a run end-to-end (local sequential): executor -> finalize -> table -> audit. Required: RUN, METHOD. Optional: REPRO_ARGS.
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
