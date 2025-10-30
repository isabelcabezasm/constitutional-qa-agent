from pathlib import Path


# I need to extract this function due to circular import issues
def root() -> Path:
    return Path(__file__).parents[2].resolve()
