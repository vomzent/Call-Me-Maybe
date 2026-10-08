PY = python3

install:
	uv sync
run:
	uv run $(PY) -m src
debug:
	uv run $(PY) -m pdb -m src

