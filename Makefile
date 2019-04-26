all: test


test:
	python -m unittest discover -v -f -c

install:
	python -m pip install -e .
