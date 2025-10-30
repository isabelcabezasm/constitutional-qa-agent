import os

from dotenv import dotenv_values


def env_values() -> dict[str, str | None]:
    """Load environment variables from .env file and system environment."""
    return {**dotenv_values(), **os.environ}
