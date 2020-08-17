## MAKEFILE

.DEFAULT: help


## HELP

.PHONY: help
help:
	@echo "    black"
	@echo "        Format code using black, the Python code formatter"
	@echo "    black-check"
	@echo "        Check if source code complies with black"
	@echo "    lint"
	@echo "        Check source code with flake8"
	@echo "    check-codestyle"
	@echo "        Perform a complete codestyle checking"


## CODE STYLE RELATED

.PHONY: black
black:
	# run black code formatter
	black model/ scripts/

.PHONY: black-check
black-check:
	# dry run black code formatter
	black --check model/ scripts/

.PHONY: lint
lint:
	# run flake linter
	flake8 --max-line-length 100 --ignore E203,W503,E402 model/ scripts/


## TESTS

.PHONY: check-codestyle
check-codestyle: black-check lint
