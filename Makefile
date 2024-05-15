install:
	pip install .

install-dev:
	pip install .[dev]

commit:
	cz commit -l 72
	cz bump
