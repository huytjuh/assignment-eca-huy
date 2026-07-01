.PHONY: help poetry-init poetry-config install activate data train test monitor api env-info clean clean-cache clean-build clean-venv clean-all

help:
	@echo "Available commands:"
	@echo "  make help          Show this help"
	@echo "  make poetry-init   Initialize pyproject.toml with Poetry"
	@echo "  make poetry-config Configure Poetry to use .venv in this repo"
	@echo "  make install       Install dependencies into .venv"
	@echo "  make activate      Print the command to activate the Poetry environment"
	@echo "  make data          Run the data preparation pipeline"
	@echo "  make train         Run model training"
	@echo "  make test          Run tests"
	@echo "  make monitor       Run the monitoring pipeline"
	@echo "  make api           Run the API"
	@echo "  make env-info      Show Poetry environment info"
	@echo "  make clean-cache   Remove Python cache directories"
	@echo "  make clean-build   Remove build/package artifacts"
	@echo "  make clean-venv    Remove the local virtual environment"
	@echo "  make clean-all     Remove caches, build artifacts, and .venv"

poetry-init:
	poetry init

poetry-config:
	poetry config virtualenvs.in-project true --local

install:
	poetry install

activate:
	poetry env activate

data:
	poetry run python -m scripts.run_data

train:
	poetry run python -m scripts.run_train --sample 20000 --saved_model

test:
	poetry run python -m scripts.run_eval

monitor:
	poetry run python -m scripts.run_monitor

api:
	poetry run python -m scripts.run_uvicorn

env-info:
	poetry env info
	poetry env list

clean-cache:
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Force -Directory -Include '__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.tox','.nox' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
