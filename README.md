.PHONY: help poetry-init poetry-config install activate data train api env-info clean-cache

help:
	@echo "Available commands:"
	@echo "  make poetry-init    Initialize Poetry pyproject.toml"
	@echo "  make poetry-config  Configure Poetry to use .venv in this repo"
	@echo "  make install        Install dependencies into .venv"
	@echo "  make activate       Print the command to activate the Poetry environment"
	@echo "  make run            Run Python through Poetry"
	@echo "  make env-info       Show Poetry environment info"
	@echo "  make clean          Remove Python/tool caches"
	@echo "  make clean-cache    Remove Python/tool caches"
	@echo "  make clean-build    Remove build/package artifacts"
	@echo "  make clean-venv     Remove local virtual environment"
	@echo "  make clean-all      Remove caches, build artifacts, and .venv"

poetry-init:
	poetry init

install:
	poetry config virtualenvs.in-project true --local
	poetry install

activate:
	poetry env activate

data:
	poetry run python -m scripts.run_data

train:
	poetry run python -m scripts.run_train

env-info:
	poetry env info
	poetry env list

clean-cache:
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Force -Directory -Include '__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.tox','.nox' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"