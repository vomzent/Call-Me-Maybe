PY = python3

install:
	uv sync
run:
	uv run $(PY) -m src
debug:
	uv run $(PY) -m pdb -m src
clean:
	rm -rf .mypy_cashe .venv llm_sdk/llm_sdk/__pycache__ src/__pycache__
lint:
	uv run $(PY) -m flake8 src
	uv run $(PY) -m mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
lint-strict:
	uv run $(PY) -m flake8 src
	uv run $(PY) -m mypy src --strict
