PY = .venv/bin/python

.PHONY: setup test eval

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m evals.run_evals
