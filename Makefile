PYTHON_VERSION = 3.9.15

.PHONY: install
install: check_pyenv check_poetry pyenv_setup poetry_setup
	@VENV_PATH=$$(poetry env info --path); \
	SITE_PACKAGES_DIR="$${VENV_PATH}/lib/python3.9/site-packages"; \
	PROJECT_ROOT=$$(echo $${VENV_PATH} | rev | cut -d'/' -f2- | rev); \
	VENV_NAME=$$(basename `dirname $${VENV_PATH}`); \
	echo $${PROJECT_ROOT} > $${SITE_PACKAGES_DIR}/$${VENV_NAME}.pth

.PHONY: check_pyenv
check_pyenv:
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "Error: pyenv is not installed. please visit https://github.com/pyenv/pyenv#installation in details."; \
		exit 1; \
	fi

.PHONY: check_poetry
check_poetry:
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "Error: poetry is not installed. please visit https://python-poetry.org/docs/#installation in details."; \
		exit 1; \
	fi

.PHONY: pyenv_setup
pyenv_setup:
	pyenv install -s $(PYTHON_VERSION)
	pyenv global $(PYTHON_VERSION)

.PHONY: poetry_setup
poetry_setup:
	poetry env use $(PYTHON_VERSION)
	poetry install

.PHONY: freeze
freeze:
	poetry run pip freeze > requirements.txt
