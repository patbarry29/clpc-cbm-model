import os
import subprocess
from pathlib import Path

def get_git_root():
    """
    Get the root directory of the git repository.
    Returns a Path object pointing to the git repository root.
    Falls back to current directory if not in a git repo.
    """
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return Path(git_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to environment variable if set
        env_path = os.environ.get("PROJECT_ROOT")
        if env_path:
            return Path(env_path)

        # Otherwise, fall back to current directory
        print("Warning: Not in a git repository and PROJECT_ROOT environment variable not set. Using current directory.")
        return Path(os.getcwd())

# Get project root path
PROJECT_ROOT = get_git_root()

# Dataset configurations
CUB_CONFIG = {
    "N_CONCEPTS": 312,
    "N_TRIMMED_CONCEPTS": 112,
    "N_CLASSES": 200,
    'N_IMAGES': 11788
}

DERM7PT_CONFIG = {
    "N_CONCEPTS": 19,
    "N_TRIMMED_CONCEPTS": 19,
    "N_CLASSES": 5,
    'N_IMAGES': 2013
}

RIVAL10_CONFIG = {
    "N_CONCEPTS": 18,
    "N_TRIMMED_CONCEPTS": 18,
    "N_CLASSES": 10,
    'N_IMAGES': 26384
}

# Helper functions for common paths
def get_data_path(relative_path=""):
    """Get path to data directory or file within data directory"""
    data_dir = PROJECT_ROOT / "data"
    if relative_path:
        return data_dir / relative_path
    return data_dir

def get_src_path(relative_path=""):
    """Get path to src directory or file within src directory"""
    src_dir = PROJECT_ROOT / "src"
    if relative_path:
        return src_dir / relative_path
    return src_dir