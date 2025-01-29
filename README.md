# accelrod



# developer guide

Setup working env / dependencies
Sync the project's dependencies with the environment.
> uv sync

lock dependencies declared in a pyproject.toml
> uv pip compile pyproject.toml -o requirements.txt



# add dev dependencies
uv add --dev ruff pytest