install:
	pip install .

install-dev:
	pip install .[dev]

install-editable:
	pip install -e . --no-deps

commit:
	cz commit -l 72
