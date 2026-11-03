PYTHON ?= .venv/bin/python
PIP_CACHE_DIR := $(CURDIR)/.cache/pip
PIP_TMP_DIR := $(CURDIR)/.cache/tmp
export OMP_NUM_THREADS := 2
export OPENBLAS_NUM_THREADS := 2
export MKL_NUM_THREADS := 2

.PHONY: install-dev install-lock fetch validate-data train evaluate test quality smoke-app app

install-dev:
	mkdir -p $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -e '.[dev]'

install-lock:
	mkdir -p $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -r requirements.lock.txt
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install --no-deps --no-build-isolation -e .

fetch:
	$(PYTHON) -m civilcomments_robustness.data fetch

validate-data:
	$(PYTHON) -m civilcomments_robustness.data validate

train:
	$(PYTHON) -m civilcomments_robustness.train

evaluate:
	$(PYTHON) -m civilcomments_robustness.evaluate

test:
	$(PYTHON) -m pytest

quality:
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke-app:
	$(PYTHON) scripts/smoke_app.py

app:
	$(PYTHON) -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true
