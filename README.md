# Accelrod


## Installation

> pip install -U accelrod

[all] includes nbformat to visualize the result in notebook
> pip install -U accelrod[all]


## Developer guide

### Setup working env / dependencies

Sync the project's dependencies with the environment.
> uv sync

lock dependencies declared in a pyproject.toml
> uv pip compile pyproject.toml -o requirements.txt



### Add dev dependencies
>uv add --dev ruff pytest

### Editable Installs
>uv pip install -e .[all,dev]

### Test package with local build
uv pip install dist/accelrod-0.1.0-py3-none-any.whl
uv run --with accelrod --no-project --refresh-package accelrod -- python -c "import accelrod"