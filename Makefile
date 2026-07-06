.PHONY: help install verify reproduce figures tables paper clean
export PYTHONPATH := .
SCALE ?= standard

help:
	@echo "make verify     - run the pytest suite (14 tests)"
	@echo "make reproduce  - simulate all stages + render figures/tables (SCALE=$(SCALE))"
	@echo "make figures    - render Monte-Carlo + game-theory figures"
	@echo "make tables     - render tables"
	@echo "make paper      - build the manuscript PDF (needs pdflatex/bibtex)"
	@echo "make clean      - remove caches and simulation outputs"

install:
	pip install -r requirements.txt

verify:
	python -m pytest tests/ -q

reproduce:
	python scripts/run_all.py $(SCALE)
	python scripts/make_figures.py $(SCALE)
	python scripts/make_gt_figures.py
	python scripts/make_tables.py $(SCALE)

figures:
	python scripts/make_figures.py $(SCALE)
	python scripts/make_gt_figures.py

tables:
	python scripts/make_tables.py $(SCALE)

paper:
	cd paper && pdflatex -interaction=nonstopmode legitimacy_fairness.tex && \
	  bibtex legitimacy_fairness && \
	  pdflatex -interaction=nonstopmode legitimacy_fairness.tex && \
	  pdflatex -interaction=nonstopmode legitimacy_fairness.tex

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache results
