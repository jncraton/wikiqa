all:

lint:
	flake8 qa.py

format:
	black qa.py

clean:
	rm -rf __pycache__
