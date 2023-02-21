all:

test:
	python3 -m doctest wikiqabot.py

lint:
	flake8 --max-line-length 88 wikiqabot.py

format:
	black wikiqabot.py

ci:
	pip3 install -r requirements.txt
	python3 -m spacy download en_core_web_sm
	make lint
	make test

clean:
	rm -rf __pycache__
