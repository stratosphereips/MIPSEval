# env_loader.py

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_project_env(env_file: str = "../.env") -> dict:
    """
    Loads environment variables from a .env file and returns them as a dict.
    Falls back to existing environment if python-dotenv is not installed.
    """

    env_path = Path(env_file)

    # If python-dotenv is available, use it
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)
    else:
        # Optional: warn the user if .env exists but dotenv is missing
        if env_path.exists() and load_dotenv is None:
            print("[env_loader] python-dotenv not installed; environment file not loaded")

    # Return a copy of the full environment
    return dict(os.environ)
