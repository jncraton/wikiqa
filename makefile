all:

test:
	python3 -m doctest qa.py

lint:
	flake8 --max-line-length 88 qa.py

format:
	black qa.py

clean:
	rm -rf __pycache__
