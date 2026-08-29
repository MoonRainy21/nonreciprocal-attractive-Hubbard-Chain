SHELL := /bin/bash
PYTHON ?= python
PYTHONPATH := src
THREADS := OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PAPER := configs/paper.yaml
SMOKE := configs/smoke.yaml
LOG_DIR := logs

.PHONY: test test-code lint smoke clean-output run-paper finish process initial-process \
	paired-route-audit figures archive revised-figures reproduce run-fig2 run-fig3 \
	run-fig4 run-branch-audit run-supplement fig2 fig3 fig4 supplement _reproduce

test:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

# These tests do not consume generated publication tables and can run before
# a clean production campaign.  The complete suite runs after processing.
test-code:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q --ignore=tests/test_publication_figure_data.py

lint:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src scripts tests

smoke:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(SMOKE) --study all
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(SMOKE)
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure all

clean-output:
	$(PYTHON) scripts/clean_outputs.py

run-fig2:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study fig2

run-fig3:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study fig3

run-fig4:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study fig4

run-branch-audit:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study branch_audit

run-supplement:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study conditioning
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run.py --config $(PAPER) --study green

run-paper: run-fig2 run-supplement run-fig3 run-fig4 run-branch-audit

# Resume only missing paper branches, reusing compatible validated raw data
# declared in configs/paper.yaml.  Unlike reproduce, this never cleans output.
finish: run-fig3 run-fig4 process figures

process:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)

initial-process:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)

paired-route-audit:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/paired_route_audit.py

figures:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure all

archive:
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/archive.py

fig2: run-fig2
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure fig02

fig3: run-fig3
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure fig03

fig4: run-fig4
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure fig04

supplement: run-supplement
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/process.py --config $(PAPER)
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure figS1
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure figS2
	$(THREADS) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/figures.py --figure figS3

reproduce:
	@$(MAKE) --no-print-directory clean-output
	@mkdir -p $(LOG_DIR)
	@set -o pipefail; $(MAKE) --no-print-directory _reproduce 2>&1 | tee $(LOG_DIR)/reproduce.log

_reproduce: test-code lint run-paper initial-process paired-route-audit process figures test
