.PHONY: install run test test-taxonomy clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make run            Run classifier on all conversations"
	@echo "  make run-sample     Run classifier on first 10 conversations"
	@echo "  make test           Run all tests"
	@echo "  make test-taxonomy  Run taxonomy validation tests only"
	@echo "  make clean          Remove generated files"

install:
	pip install -r requirements.txt

run:
	python main.py

run-sample:
	python main.py --limit 10

test:
	pytest tests/ -v

test-taxonomy:
	pytest tests/test_taxonomy.py -v

clean:
	rm -f data/labels.csv
	rm -rf __pycache__ .pytest_cache tests/__pycache__
