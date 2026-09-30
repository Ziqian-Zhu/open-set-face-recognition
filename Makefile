PYTHON ?= python3
VENV := face_compare_system/.venv-ui
VENV_PYTHON := $(VENV)/bin/python

.PHONY: bootstrap setup models dev run test preview

bootstrap: setup models

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r face_compare_system/requirements.txt

models:
	$(VENV_PYTHON) face_compare_system/scripts/download_models.py

dev: setup
	$(VENV_PYTHON) -m pip install -r face_compare_system/requirements-dev.txt

run:
	$(VENV_PYTHON) face_compare_system/main.py

test: dev
	$(VENV_PYTHON) -m pytest -q face_compare_system/tests face_research/tests -p no:cacheprovider

preview:
	$(VENV_PYTHON) face_compare_system/scripts/check_ui.py --preview
