PY ?= python

.PHONY: setup test fixture eval-fixture bundle check fetch

setup:
	$(PY) -m pip install -r requirements-research.txt

test:
	$(PY) -m pytest -q tests

fixture:
	$(PY) scripts/make_fixture.py

eval-fixture: fixture
	$(PY) scripts/run_local.py solution --cases data/fixtures/cases.jsonl --no-internals --time-limit 600 && $(PY) scripts/run_local.py solution --small --cases data/fixtures/cases.jsonl --no-internals --time-limit 600

bundle:
	$(PY) scripts/build_bundle.py

check: test eval-fixture bundle

fetch:
	$(PY) research/fetch_data.py
